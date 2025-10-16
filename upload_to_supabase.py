import os
import json
import time
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
import nltk
from nltk.tokenize import sent_tokenize

# --- CONFIGURATION ---
INPUT_FILE = 'news_history_data.json'
PROGRESS_FILE = 'progress.log' # File to save our progress
EMBEDDING_MODEL = 'models/text-embedding-004'
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
BATCH_SIZE = 100
MAX_ARTICLES = 5000
# --------------------

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # (This function is unchanged)
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        words = sentence.split()
        if len(current_chunk.split()) + len(words) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_words = current_chunk.split()[-chunk_overlap:]
            current_chunk = " ".join(overlap_words) + " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def main():
    """Main function to run the ingestion pipeline."""
    print("--- Starting Supabase Ingestion Script ---")
    
    load_dotenv()
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        return
    genai.configure(api_key=google_api_key)

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        print("ERROR: Supabase credentials not found.")
        return
    supabase: Client = create_client(supabase_url, supabase_key)
    print("Supabase client initialized.")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        articles_to_process = articles[:MAX_ARTICLES]
    except FileNotFoundError:
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    # --- FIX 2: Resume Feature ---
    start_index = 0
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            start_index = int(f.read())
        print(f"--- Resuming from article #{start_index + 1} ---")
    # -----------------------------

    batch_to_upload = []
    total_chunks = 0
    
    # Use the start_index to skip already processed articles
    for i, article in enumerate(articles_to_process[start_index:], start=start_index):
        print(f"Processing article {i + 1}/{len(articles_to_process)}: {article['title'][:50]}...")
        
        article_chunks = chunk_text(article['text'], CHUNK_SIZE, CHUNK_OVERLAP)
        
        for chunk_content in article_chunks:
            data_object = {
                "content": chunk_content,
                "source_title": article['title'],
                "source_url": None 
            }
            batch_to_upload.append(data_object)

            if len(batch_to_upload) >= BATCH_SIZE:
                success = process_batch(batch_to_upload, supabase)
                if not success:
                    print("!!! Critical error in batch processing. Aborting script. !!!")
                    return # Stop the script if a batch fails completely
                
                total_chunks += len(batch_to_upload)
                batch_to_upload = []
                
                # --- FIX 2: Save progress after each successful batch ---
                with open(PROGRESS_FILE, 'w') as f:
                    f.write(str(i + 1)) # Save the index of the next article to start from
                # ----------------------------------------------------
                time.sleep(1)

    if batch_to_upload:
        process_batch(batch_to_upload, supabase)
        total_chunks += len(batch_to_upload)

    # Clean up progress file on successful completion
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    print(f"\n--- Ingestion process complete! ---")
    print(f"Total chunks uploaded to Supabase: {total_chunks}")

def process_batch(batch: list[dict], supabase: Client) -> bool:
    """Embeds and uploads a single batch of data with a retry mechanism."""
    print(f"  Processing batch of {len(batch)} chunks...")
    
    contents_to_embed = [item['content'] for item in batch]
    
    # --- FIX 1: Retry Mechanism for Embedding ---
    embeddings = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"    Generating {len(contents_to_embed)} embeddings (Attempt {attempt + 1}/{max_retries})...")
            embedding_result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=contents_to_embed,
                task_type="retrieval_document"
            )
            embeddings = embedding_result['embedding']
            print("    Embeddings generated successfully.")
            break # Exit the loop on success
        except Exception as e:
            print(f"    ERROR generating embeddings: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) # Exponential backoff: 1, 2, 4 seconds
                print(f"    Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("    Max retries reached. Failed to generate embeddings for this batch.")
                return False # Signal failure
    # ---------------------------------------------
    
    if not embeddings:
        return False

    for i, item in enumerate(batch):
        item['embedding'] = embeddings[i]

    try:
        print(f"    Uploading {len(batch)} records to Supabase...")
        supabase.table('documents').insert(batch).execute()
        print("    Batch uploaded successfully.")
        return True # Signal success
    except Exception as e:
        print(f"    CRITICAL ERROR uploading batch to Supabase: {e}")
        return False # Signal failure

if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt')
    
    main()