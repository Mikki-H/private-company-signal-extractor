import os
import json
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, AutoModelForTokenClassification
from typing import List, Dict, Any
import sys
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

class NERPredictor:
    def __init__(self, model_dir: str = "models/ner"):
        """Initialize the NER predictor with our fine-tuned model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load label mappings
        with open(os.path.join(model_dir, 'labels.json'), 'r') as f:
            label_data = json.load(f)
            self.id2label = {int(k): v for k, v in label_data['id2label'].items()}
            self.label2id = {v: int(k) for k, v in label_data['id2label'].items()}
            print(f"Loaded label mappings: {self.label2id}")
        
        # Load model and tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
        self.id2label = self.model.config.id2label
        self.model.to(self.device)
        self.model.eval()
        print("Model and tokenizer loaded successfully")

    def merge_subword_tokens(self, tokens: List[str], predictions: List[int], confidences: List[float], offsets: List[List[int]]) -> List[Dict[str, Any]]:
        """Merge subword tokens into complete words with their predictions."""
        merged = []
        current_word_tokens = []
        current_word_preds = []
        current_word_confs = []
        current_word_offsets = []

        for token, pred, conf, offset in zip(tokens, predictions, confidences, offsets):
            if offset == [0, 0]: # Skip padding tokens
                continue

            if not current_word_tokens or not token.startswith('##'):
                if current_word_tokens:
                    # Finalize the previous word
                    start = current_word_offsets[0][0]
                    end = current_word_offsets[-1][1]
                    
                    # Get most common non-'O' label
                    non_o_preds = [p for p in current_word_preds if self.id2label[p] != 'O']
                    if non_o_preds:
                        from collections import Counter
                        final_pred = Counter(non_o_preds).most_common(1)[0][0]
                    else:
                        final_pred = current_word_preds[0] # Default to first token's pred if all 'O'
                    
                    mean_conf = np.mean(current_word_confs)

                    merged.append({
                        'label': self.id2label[final_pred],
                        'confidence': float(mean_conf),
                        'start': start,
                        'end': end
                    })
                
                # Start a new word
                current_word_tokens = [token]
                current_word_preds = [pred]
                current_word_confs = [conf]
                current_word_offsets = [offset]
            else:
                # Continue current word
                current_word_tokens.append(token)
                current_word_preds.append(pred)
                current_word_confs.append(conf)
                current_word_offsets.append(offset)
        
        # Add the last word
        if current_word_tokens:
            start = current_word_offsets[0][0]
            end = current_word_offsets[-1][1]
            non_o_preds = [p for p in current_word_preds if self.id2label[p] != 'O']
            if non_o_preds:
                from collections import Counter
                final_pred = Counter(non_o_preds).most_common(1)[0][0]
            else:
                final_pred = current_word_preds[0]
            mean_conf = np.mean(current_word_confs)
            merged.append({
                'label': self.id2label[final_pred],
                'confidence': float(mean_conf),
                'start': start,
                'end': end
            })

        return merged

    def merge_adjacent_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Merge adjacent entities of the same type."""
        if not entities:
            return entities
        
        merged = []
        current = entities[0].copy()
        
        for next_entity in entities[1:]:
            # Check if entities are of the same type and either adjacent or separated by spaces/punctuation
            text_between = text[current['end']:next_entity['start']].strip()
            if (current['label'] == next_entity['label'] and 
                (not text_between or text_between in ['.', ',', '-', '&', ' '])):
                # Merge entities
                current['end'] = next_entity['end']
                current['text'] = text[current['start']:current['end']]
                current['confidence'] = min(current['confidence'], next_entity['confidence'])
            else:
                merged.append(current)
                current = next_entity.copy()
        
        merged.append(current)
        return merged

    def predict_entities(self, text: str, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Predicts named entities in a given text."""
        encoding = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'][0]
        special_tokens_mask = encoding['special_tokens_mask'][0].bool()

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0].cpu()
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).numpy()
            confidence_scores = torch.max(probabilities, dim=1)[0].numpy()

        # First, merge subword tokens
        non_special_indices = ~special_tokens_mask
        merged_words = self.merge_subword_tokens(
            [t for t, m in zip(self.tokenizer.convert_ids_to_tokens(input_ids[0]), special_tokens_mask) if not m],
            predictions[non_special_indices],
            confidence_scores[non_special_indices],
            [o.tolist() for o, m in zip(offset_mapping, special_tokens_mask) if not m]
        )

        # Now, group consecutive words with the same entity label
        entities = []
        if not merged_words:
            return []

        current_entity = None
        for word in merged_words:
            # If word is 'O', finalize previous entity and skip
            if word['label'] == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue

            # If we have a current entity and the new word has the same label
            if current_entity and current_entity['label'] == word['label']:
                # Extend the current entity
                current_entity['end'] = word['end']
                current_entity['confidence'] = min(current_entity['confidence'], word['confidence'])
            else:
                # Finalize the previous entity and start a new one
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    'start': word['start'],
                    'end': word['end'],
                    'label': word['label'],
                    'confidence': word['confidence']
                }
        
        # Add the last entity if it exists
        if current_entity:
            entities.append(current_entity)

        # Add text to entities and filter by confidence
        final_entities = []
        for ent in entities:
            if ent['confidence'] > confidence_threshold:
                ent['text'] = text[ent['start']:ent['end']]
                final_entities.append(ent)
                
        return final_entities

if __name__ == "__main__":
    # Example usage
    predictor = NERPredictor()
    
    # Test text
    test_text = """
    Tesla Inc. announced a new $5 billion investment in AI research today. 
    CEO Elon Musk revealed the company's plans to launch their next-generation 
    autonomous driving system in partnership with NVIDIA Corp.
    """
    
    results = predictor.predict_entities(test_text)
    print("\nTest Results:")
    print("Text:", test_text)
    print("\nExtracted Entities:")
    for entity in results:
        print(f"{entity['label']} ({entity['confidence']:.2f}): {entity['text']}") 