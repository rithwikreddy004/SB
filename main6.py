#main3


import os
import google.generativeai as genai
import httpx
import asyncio
import time
import re
from bs4 import BeautifulSoup
from ddgs import DDGS
from readability import Document
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import nltk



# --- NEW: NLTK Data Check on Startup ---
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' data found.")
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' data not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' data downloaded successfully.")
# ----------------------------------------
from nltk.tokenize import sent_tokenize




# --- Setup and Configuration (Using your specified models) ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Using the models from your working code



pro_model = genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')

flash_model = genai.GenerativeModel('models/gemini-flash-latest')

content_genmodel=genai.GenerativeModel('models/gemini-2.5-flash-preview-09-2025')


embedding_model = 'models/text-embedding-004'

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not found in .env file")
supabase: Client = create_client(supabase_url, supabase_key)
print("Clients for Google AI and Supabase initialized successfully.")


# --- Helper Functions (Your existing, working code) ---
def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[str]:
    # (This function is unchanged)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_words = current_chunk.split()[-chunk_overlap:]
            current_chunk = " ".join(overlap_words) + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def add_scraped_data_to_db(article_title: str, article_text: str, article_url: str):
    # (This function is unchanged)
    print(f"BACKGROUND TASK: Starting to upload '{article_title[:30]}...'")
    try:
        raw_chunks = chunk_text(article_text)
        chunks = [chunk for chunk in raw_chunks if chunk and not chunk.isspace()]
        if not chunks:
            print("BACKGROUND TASK: No valid chunks to process.")
            return
        embedding_result = genai.embed_content(model=embedding_model, content=chunks, task_type="retrieval_document")
        embeddings = embedding_result['embedding']
        documents_to_insert = [{"content": chunk, "embedding": embeddings[i], "source_title": article_title, "source_url": article_url} for i, chunk in enumerate(chunks)]
        supabase.table('documents').insert(documents_to_insert).execute()
        print(f"BACKGROUND TASK: Successfully uploaded {len(documents_to_insert)} chunks.")
    except Exception as e:
        print(f"BACKGROUND TASK: Failed to add data to DB. Error: {e}")

async def scrape_url(client: httpx.AsyncClient, url: str, scraped_urls: set):
    # (This function is unchanged)
    if url in scraped_urls:
        return None
    print(f"Scraping: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = await client.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        scraped_urls.add(url)
        doc = Document(response.text)
        title = doc.title()
        article_html = doc.summary()
        soup = BeautifulSoup(article_html, 'html.parser')
        article_text = soup.get_text(separator='\n', strip=True)
        return {"url": url, "title": title, "text": article_text}
    except Exception as e:
        print(f"An error occurred while processing {url}: {e}")
        return None

async def deep_search_and_scrape(keywords: list[str], scraped_urls: set) -> list[dict]:
    # (This function is unchanged)
    print("--- DEEP WEB SCRAPE: Starting full search... ---")
    urls_to_scrape = set()
    with DDGS(timeout=20) as ddgs:
        for keyword in keywords:
            search_results = list(ddgs.text(keyword, region='wt-wt', max_results=3))
            if search_results:
                urls_to_scrape.add(search_results[0]['href'])
    async with httpx.AsyncClient() as client:
        tasks = [scrape_url(client, url, scraped_urls) for url in urls_to_scrape]
        results = await asyncio.gather(*tasks)
        return [res for res in results if res and res.get("text")]

async def get_latest_news_context(topic: str, scraped_urls: set) -> list[dict]:
    # (This function is unchanged)
    print("--- LIGHT WEB SCRAPE: Starting lightweight news search... ---")
    try:
        keyword = f"{topic} latest news today"
        urls_to_scrape = set()
        with DDGS(timeout=10) as ddgs:
            search_results = list(ddgs.text(keyword, region='wt-wt', max_results=2))
            for result in search_results:
                urls_to_scrape.add(result['href'])
        async with httpx.AsyncClient() as client:
            tasks = [scrape_url(client, url, scraped_urls) for url in urls_to_scrape]
            results = await asyncio.gather(*tasks)
            return [res for res in results if res and res.get("text")]
    except Exception as e:
        print(f"--- WEB TASK: Error during news scraping: {e} ---")
        return []

async def get_db_context(topic: str) -> list[dict]:
    # (This function is unchanged)
    print("--- DB TASK: Starting HyDE database search... ---")
    try:
        hyde_prompt = f"""
        Write a short, factual, encyclopedia-style paragraph that provides a direct answer to the following topic.
        This will be used to find similar documents, so be concise and include key terms.
        
        Topic: "{topic}"
        """
        hyde_response = await flash_model.generate_content_async(hyde_prompt)
        query_embedding = genai.embed_content(model=embedding_model, content=hyde_response.text, task_type="retrieval_query")['embedding']
        db_results = supabase.rpc('match_documents', {'query_embedding': query_embedding, 'match_threshold': 0.65, 'match_count': 5}).execute()
        return db_results.data
    except Exception as e:
        print(f"--- DB TASK: Error during database search: {e} ---")
        return []

# --- FastAPI App ---
app = FastAPI()
class PromptRequest(BaseModel):
    topic: str

@app.get("/")
async def read_root(): return {"status": "Welcome"}

@app.post("/process-topic")
async def process_topic(request: PromptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"Received topic from user: {request.topic}")
    
    try:
        db_task = asyncio.create_task(get_db_context(request.topic))
        
        await asyncio.sleep(11) # Your working sleep time

        db_results = []
        new_articles = []
        scraped_urls = set()
        # --- NEW: Initialize these here to ensure they exist for the final return ---
        base_keywords = []
        source_of_context = ""
        # --------------------------------------------------------------------------

        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        if len(db_results) >= 3:
            source_of_context = "DATABASE_WITH_NEWS"
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            source_of_context = "DEEP_SCRAPE"
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")
            keyword_prompt =  f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            # This parsing logic is from your older, working version
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes:
                base_keywords = keywords_in_quotes
            else:
                base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        db_context, web_context = "", ""
        source_urls = []
        
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
            source_urls.extend(list(set([item['source_url'] for item in db_results if item['source_url']])))

        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            source_urls.extend([art['url'] for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any information."}

        

        # --- THIS IS THE UPGRADED PROMPT ---
        final_prompt = f"""
        You are an expert YouTube title strategist and scriptwriter.
        Your mission is to generate 4 distinct, attention-grabbing video titles for the topic: "{request.topic}", AND a corresponding description for each.

        Use the provided research material to inform your output:
        - Use the 'FOUNDATIONAL KNOWLEDGE' for deep context, facts, and historical background.
        - Use the 'LATEST NEWS' to find a fresh, timely, or surprising angle, especially considering the current date is October 15, 2025.

        RULES FOR YOUR OUTPUT:
        1.  For each of the 4 ideas, provide a 'TITLE' and a 'DESCRIPTION'.
        2.  Each 'DESCRIPTION' MUST be between 90 and 110 words.
        3.  Separate each complete idea (title + description) with '---'.
        4.  DO NOT add any introductory sentences, explanations, or any text other than the titles and descriptions in the specified format.

        EXAMPLE OUTPUT FORMAT:
        TITLE: This Is Why Everyone Is Suddenly Talking About [Topic]
        DESCRIPTION: In this video, we uncover the shocking truth behind [Topic]. For years, experts have believed one thing, but new data from October 2025 reveals a completely different story. We'll break down the historical context, analyze the latest reports, and explain exactly why this topic is about to become the biggest conversation on the internet. You'll learn about the key players, the secret history, and what this means for the future. Don't miss this deep dive into one of the most misunderstood subjects of our time, it will change everything you thought you knew.
        ---
        TITLE: The Hidden Truth Behind [Related Concept]
        DESCRIPTION: Everyone thinks they understand [Related Concept], but they're wrong. We've dug through the archives and analyzed the latest breaking news to bring you the untold story. This video explores the forgotten origins, the powerful figures who shaped its narrative, and the surprising new developments that are challenging everything we know. We connect the dots from the foundational knowledge to the fresh web updates to give you a complete picture you won't find anywhere else. Get ready to have your mind blown by the real story behind [Related Concept].
        ---
        
        RESEARCH FOR TOPIC: "{request.topic}"
        ---
        FOUNDATIONAL KNOWLEDGE (from our database):
        {db_context}
        ---
        LATEST NEWS UPDATES (from the web):
        {web_context}
        ---
        """
        step3_start_time = time.time()
        final_response = await pro_model.generate_content_async(final_prompt)
        step3_end_time = time.time()
        print(f"--- PROFILING: Step 3 (Final Idea Gen) took {step3_end_time - step3_start_time:.2f} seconds ---")

        # --- THIS IS THE NEW, SMARTER PARSING LOGIC ---
        response_text = final_response.text
        
        final_ideas = []
        final_descriptions = []
        
        # Split the entire response into blocks, one for each idea
        idea_blocks = response_text.strip().split('---')
        
        for block in idea_blocks:
            title = ""
            description = ""
            lines = block.strip().split('\n')
            
            for line in lines:
                if line.startswith('TITLE:'):
                    # Extract text after "TITLE:"
                    title = line.replace('TITLE:', '', 1).strip()
                elif line.startswith('DESCRIPTION:'):
                    # Extract text after "DESCRIPTION:"
                    description = line.replace('DESCRIPTION:', '', 1).strip()
            
            # Only add the pair if both title and description were found
            if title and description:
                final_ideas.append(title)
                final_descriptions.append(description)

        print(f"Final generated ideas: {final_ideas}")
        print(f"Final generated descriptions: {len(final_descriptions)} descriptions found.")
        total_end_time = time.time()
        print(f"--- PROFILING: Total request time was {total_end_time - total_start_time:.2f} seconds ---")
        
        # --- THIS IS THE UPDATED RETURN STATEMENT ---
        return {
            "source_of_context": source_of_context,
            "ideas": final_ideas,
            "descriptions": final_descriptions, # The new descriptions list
            "generated_keywords": base_keywords,
            "source_urls": list(set(source_urls)),
            "scraped_text_context": f"DB CONTEXT:\n{db_context}\n\nWEB CONTEXT:\n{web_context}"
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": "An error occurred in the processing pipeline."}
    
# --- Add this new Pydantic model with your other one ---
class ScriptRequest(BaseModel):
    topic: str

# --- Add this new endpoint, now with YOUR smart conditional logic ---
@app.post("/generate-script")
async def generate_script(request: ScriptRequest, background_tasks: BackgroundTasks):
    total_start_time = time.time()
    print(f"SCRIPT GENERATION: Received request for topic: {request.topic}")

    try:
        # --- Step 1: Start the DB search in the background (Your Logic) ---
        db_task = asyncio.create_task(get_db_context(request.topic))
        
        # Give the DB task a head start
        await asyncio.sleep(11) 

        db_results = []
        new_articles = []
        scraped_urls = set()
        base_keywords = []

        # Check if the DB task finished early and was successful
        if db_task.done():
            db_results = db_task.result()
            print(f"--- DB task finished early. Found {len(db_results)} documents. ---")

        # --- Step 2: The Conditional Scrape (Your Logic) ---
        if len(db_results) >= 3:
            # If DB has enough data, just get the latest news
            print("--- DB HIT: Performing LIGHT web scrape for latest news. ---")
            new_articles = await get_latest_news_context(request.topic, scraped_urls)
        else:
            # If DB is slow or has no data, perform a full, deep scrape
            print("--- DB MISS or SLOW: Initiating DEEP web scrape. ---")
            keyword_prompt = f"""
            Your ONLY task is to generate 3 diverse search engine keyword phrases for the topic: '{request.topic}'.
            Follow these rules STRICTLY:
            1. Return ONLY the 3 phrases.
            2. DO NOT add numbers, markdown, explanations, or any introductory text.
            3. Each phrase must be on a new line.
            EXAMPLE INPUT: Is coding dead?
            EXAMPLE OUTPUT:
            future of programming jobs automation
            AI replacing software developers
            demand for software engineers 2025
            """
            response = await flash_model.generate_content_async(keyword_prompt)
            raw_text = response.text
            keywords_in_quotes = re.findall(r'"(.*?)"', raw_text)
            if keywords_in_quotes:
                base_keywords = keywords_in_quotes
            else:
                base_keywords = [kw.strip() for kw in raw_text.strip().split('\n') if kw.strip()]
            
            targeted_keywords = [kw for kw in base_keywords] + [f"{kw} site:reddit.com" for kw in base_keywords]
            new_articles = await deep_search_and_scrape(targeted_keywords, scraped_urls)

        # --- Step 3: Wait for DB task if it's still running ---
        if not db_task.done():
            print("--- Waiting for DB task to complete... ---")
            db_results = await db_task
            print(f"--- DB task finished. Found {len(db_results)} documents. ---")

        # --- Step 4: The Merge ---
        db_context, web_context = "", ""
        if db_results:
            db_context = "\n\n".join([item['content'] for item in db_results])
        if new_articles:
            web_context = "\n\n".join([f"Source: {art['title']}\n{art['text']}" for art in new_articles])
            for article in new_articles:
                background_tasks.add_task(add_scraped_data_to_db, article['title'], article['text'], article['url'])

        if not db_context and not web_context:
            return {"error": "Could not find any research material to write the script."}
            
        # --- Step 5: Use YOUR detailed prompt to generate the script ---
        print("SCRIPT GENERATION: Generating full script with content_gen_model...")
        script_prompt = f"""
        You are an expert YouTube scriptwriter who creates engaging and natural long-form content.

        Your task is to generate a complete YouTube video script of **10 minutes** in length, totaling around **1300 words**, based on the provided **main idea or topic**.

        Follow this exact structure and divide both **time and word count** proportionally across sections:

        1. **Hook & Introduction** (Approx. 1 minute / 130–150 words)
           - Begin with a powerful hook that grabs attention immediately.
           - Briefly introduce the topic and explain why it matters to viewers.
           - End with a line that builds curiosity for what’s coming next.

        2. **Problem Statement** (Approx. 1.5 minutes / 180–200 words)
           - Clearly define the main issue or challenge related to the topic.
           - Explain how it affects people, industries, or society.
           - Keep it relatable and emotionally engaging.

        3. **Evidence & Data** (Approx. 2 minutes / 250–270 words)
           - Present relevant research findings, stats, or scientific explanations from the provided context.
           - Reference credible sources, reports, or studies mentioned in the research.
           - Explain the underlying logic in simple, conversational language.

        4. **Real-world Examples** (Approx. 2.5 minutes / 300–320 words)
           - Use 2–3 case studies, events, or real-world stories from the research to illustrate the topic.
           - Connect these examples to the audience’s understanding.
           - Maintain storytelling tone and flow.

        5. **Potential Solutions** (Approx. 2.5 minutes / 300–320 words)
           - Discuss practical or innovative solutions to the problem found in the research.
           - Include expert opinions or emerging technologies if available.
           - Offer a balanced view of pros and cons.

        6. **Call to Action** (Approx. 0.5 minute / 100–120 words)
           - End with a strong conclusion and call to action.
           - Inspire the audience to think, share, or take meaningful steps.
           - Maintain an optimistic or thought-provoking tone.

        Additional Requirements:
        - Tone: natural, engaging, and conversational (like a smart storyteller).
        - Style: Mix of facts, storytelling, and insights — no fluff or filler.
        - Avoid repetition, and ensure smooth transitions between sections.
        - Keep the total around **1300 words** ±50 words.
        
        ---
        MAIN TOPIC/IDEA: "{request.topic}"

        RESEARCH CONTEXT:
        FOUNDATIONAL KNOWLEDGE (from database): {db_context}
        LATEST NEWS (from web): {web_context}
        ---
        """
        
        script_response = await content_genmodel.generate_content_async(script_prompt)
        
        total_end_time = time.time()
        print(f"--- PROFILING: Script generation took {total_end_time - total_start_time:.2f} seconds ---")
        
        return {"script": script_response.text}

    except Exception as e:
        print(f"SCRIPT GENERATION: An error occurred: {e}")
        return {"error": "An error occurred during the script generation pipeline."}