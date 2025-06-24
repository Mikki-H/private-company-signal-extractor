import os
import sys
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import re
from datetime import datetime

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.database.mongo_db import MongoDB

BASE_URL = "https://techcrunch.com/category/venture/"

def scrape_techcrunch():
    """
    Scrapes articles from TechCrunch and saves them to MongoDB.
    """
    # Initialize MongoDB connection
    db = MongoDB()

    # Clear existing articles
    deleted_articles_count = db.clear_collection("articles")
    print(f"Cleared {deleted_articles_count} documents from 'articles' collection.")
    deleted_processed_count = db.clear_collection("processed_articles")
    print(f"Cleared {deleted_processed_count} documents from 'processed_articles' collection.")

    articles_collection = db.get_collection("articles")

    response = requests.get(BASE_URL)
    if response.status_code != 200:
        print(f"Failed to fetch homepage: {response.status_code}")
        return

    soup = BeautifulSoup(response.content, "html.parser")

    # Find all links that look like article links
    article_links = soup.find_all("a", href=re.compile(r"/\d{4}/\d{2}/\d{2}/"))

    print(f"Found {len(article_links)} article links")

    # Use a set to store unique URLs
    unique_links = set()
    for link in article_links:
        unique_links.add(link["href"])

    print(f"Found {len(unique_links)} unique article links")

    articles_to_insert = []
    for article_url in unique_links:
        article_url = urljoin(BASE_URL, article_url)
        
        # Check if article already exists in MongoDB
        if articles_collection.find_one({"url": article_url}):
            print(f"Skipping {article_url} - already in database")
            continue

        try:
            print(f"Fetching {article_url}")
            article_response = requests.get(article_url)
            
            if article_response.status_code == 200:
                # Create article document
                article_data = {
                    'url': article_url,
                    'html_content': article_response.text,
                    'source': 'techcrunch',
                    'scraped_at': datetime.now().isoformat()
                }
                
                articles_to_insert.append(article_data)
                print(f"Queued {article_url}")
                time.sleep(1)  # Be nice to their servers
            else:
                print(f"Failed to fetch {article_url}: {article_response.status_code}")
        
        except Exception as e:
            print(f"Error processing {article_url}: {str(e)}")
    
    # Bulk insert all new articles
    if articles_to_insert:
        result = db.insert_many("articles", articles_to_insert)
        print(f"Inserted {len(articles_to_insert)} new articles into MongoDB")
    else:
        print("No new articles to insert")

if __name__ == '__main__':
    scrape_techcrunch()