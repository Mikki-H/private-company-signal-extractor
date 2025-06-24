import spacy
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Any
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.database.mongo_db import MongoDB

class TextProcessor:
    def __init__(self):
        """Initialize spaCy with the English language model."""
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.db = MongoDB()
        print("TextProcessor initialized")

    def clean_html(self, html_content: str) -> str:
        """Remove HTML tags and extract clean text."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        
        # Remove extra whitespace and normalize
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text through the NLP pipeline."""
        doc = self.nlp(text)
        
        return {
            'tokens': [token.text for token in doc if not token.is_space],
            'lemmas': [token.lemma_ for token in doc if not token.is_space and not token.is_stop],
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
        }

    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article document."""
        # Extract clean text from HTML
        clean_text = self.clean_html(article['html_content'])
        
        # Process the clean text
        processed_data = self.process_text(clean_text)
        
        # Create processed article document
        processed_article = {
            'url': article['url'],
            'source': article['source'],
            'scraped_at': article['scraped_at'],
            'clean_text': clean_text,
            'processed_data': processed_data
        }
        
        return processed_article

    def process_all_articles(self):
        """Process all unprocessed articles in the database."""
        articles_collection = self.db.get_collection('articles')
        processed_collection = self.db.get_collection('processed_articles')
        
        # Get all articles that haven't been processed yet
        for article in articles_collection.find():
            # Check if already processed
            if processed_collection.find_one({'url': article['url']}):
                print(f"Skipping {article['url']} - already processed")
                continue
                
            try:
                print(f"Processing {article['url']}")
                processed_article = self.process_article(article)
                processed_collection.insert_one(processed_article)
                print(f"Processed and saved {article['url']}")
            except Exception as e:
                print(f"Error processing {article['url']}: {str(e)}")

if __name__ == '__main__':
    # Download spaCy model if not already downloaded
    if not spacy.util.is_package("en_core_web_sm"):
        print("Downloading spaCy English model...")
        os.system("python -m spacy download en_core_web_sm")
    
    processor = TextProcessor()
    processor.process_all_articles() 