import os
import sys
import json
from typing import List, Dict, Any
import spacy
from spacy.tokens import Doc
import re

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.database.mongo_db import MongoDB

class NERDatasetBuilder:
    def __init__(self):
        """Initialize spaCy and MongoDB connection."""
        self.nlp = spacy.load("en_core_web_sm")
        self.db = MongoDB()
        
        # Define our custom entity types
        self.custom_entities = {
            "COMPANY": ["Inc", "Corp", "LLC", "Ltd", "Limited", "Technologies", "Tech"],
            "FINANCIAL": [
                "million", "billion", "funding", "investment", "revenue", "valuation",
                "Series A", "Series B", "Series C", "IPO", "acquisition", "merger"
            ],
            "EVENT": [
                "launch", "release", "announce", "unveil", "partnership",
                "acquisition", "merger", "funding round", "IPO"
            ]
        }
        
        # Compile regex patterns for each entity type
        self.patterns = {
            "COMPANY": re.compile(r'[A-Z][a-zA-Z]*(?:\s+(?:' + '|'.join(self.custom_entities["COMPANY"]) + r'))+'),
            "FINANCIAL": re.compile(r'\$?\d+(?:\.\d+)?(?:\s+(?:' + '|'.join(self.custom_entities["FINANCIAL"]) + r'))+'),
            "EVENT": re.compile('|'.join(self.custom_entities["EVENT"]))
        }

    def extract_custom_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract custom entities using regex patterns."""
        entities = []
        
        # Find all matches for each entity type
        for ent_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "label": ent_type
                })
        
        return sorted(entities, key=lambda x: x["start"])

    def merge_entities(self, spacy_ents: List[Dict], custom_ents: List[Dict]) -> List[Dict]:
        """Merge and deduplicate entities from spaCy and custom extraction."""
        all_entities = []
        
        # Add relevant spaCy entities
        for ent in spacy_ents:
            if ent.label_ in ["ORG", "PERSON", "MONEY", "DATE"]:
                all_entities.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "label": "COMPANY" if ent.label_ == "ORG" else ent.label_
                })
        
        # Add custom entities
        all_entities.extend(custom_ents)
        
        # Sort by start position and remove overlaps
        all_entities.sort(key=lambda x: x["start"])
        merged = []
        
        for ent in all_entities:
            if not merged or ent["start"] >= merged[-1]["end"]:
                merged.append(ent)
        
        return merged

    def create_training_data(self) -> List[Dict[str, Any]]:
        """Create training data from processed articles."""
        processed_collection = self.db.get_collection('processed_articles')
        training_data = []
        
        for article in processed_collection.find():
            text = article['clean_text']
            doc = self.nlp(text)
            
            # Get entities from spaCy
            spacy_entities = list(doc.ents)
            
            # Get custom entities
            custom_entities = self.extract_custom_entities(text)
            
            # Merge entities
            merged_entities = self.merge_entities(spacy_entities, custom_entities)
            
            # Create training example
            if merged_entities:
                training_data.append({
                    "text": text,
                    "entities": [
                        [ent["start"], ent["end"], ent["label"]]
                        for ent in merged_entities
                    ]
                })
        
        return training_data

    def save_training_data(self, output_dir: str = "data/processed"):
        """Save the training data to a JSON file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        training_data = self.create_training_data()
        
        output_path = os.path.join(output_dir, "ner_training_data.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"Created {len(training_data)} training examples")
        print(f"Saved training data to {output_path}")

if __name__ == "__main__":
    builder = NERDatasetBuilder()
    builder.save_training_data() 