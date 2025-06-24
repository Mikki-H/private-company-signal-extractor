import sys
import os
from tqdm import tqdm

# Adjust path to import from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.database.mongo_db import MongoDB
from src.ml.signal_extractor import SignalExtractor

def process_and_store_signals():
    """
    Processes articles from MongoDB, extracts signals, and stores them back.
    """
    db_handler = MongoDB()
    extractor = SignalExtractor()
    
    # Fetch all processed articles
    articles_collection = db_handler.get_collection('processed_articles')
    articles = list(articles_collection.find())
    
    if not articles:
        print("No processed articles found in the database.")
        return

    print(f"Found {len(articles)} processed articles. Starting signal extraction...")
    
    all_signals = []
    for article in tqdm(articles, desc="Extracting Signals"):
        article_text = article.get("clean_text", "")
        if not article_text:
            continue
            
        signals = extractor.extract_signals_from_text(article_text)
        
        # Add article context to each signal
        for signal in signals:
            signal["article_url"] = article.get("url")
            signal["article_title"] = article.get("title")
            all_signals.append(signal)

    print(f"\nExtraction complete. Found {len(all_signals)} signals in total.")

    if all_signals:
        signals_collection = db_handler.get_collection('signals')
        # Clear existing signals to avoid duplicates
        signals_collection.delete_many({})
        signals_collection.insert_many(all_signals)
        print(f"Successfully stored {len(all_signals)} signals in the 'signals' collection.")
    else:
        print("No signals were found to store.")

if __name__ == '__main__':
    process_and_store_signals() 