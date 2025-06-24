import os
import json
import numpy as np
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)
from seqeval.metrics import classification_report
from tqdm import tqdm
from .ner_dataset_builder import NERDatasetBuilder
from . import ner_dataset_builder

class NERDataset(Dataset):
    def __init__(self, texts, entities, tokenizer, label2id):
        self.texts = texts
        self.entities = entities
        self.tokenizer = tokenizer
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        entities = self.entities[idx]

        # Tokenize text and align labels
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt',
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )

        # Create labels of the same length as tokens
        labels = torch.ones(encoding['input_ids'].shape) * -100
        offset_mapping = encoding['offset_mapping'][0]
        special_tokens_mask = encoding['special_tokens_mask'][0]

        # Create a mapping of character positions to entity labels
        char_to_label = {}
        for start, end, label in entities:
            for pos in range(start, end):
                char_to_label[pos] = self.label2id[label]

        # Label each token
        for idx, (start, end) in enumerate(offset_mapping):
            # Skip special tokens
            if special_tokens_mask[idx]:
                continue

            # Skip empty tokens
            if start == end:
                continue

            # Find the most common entity label in this token's character span
            token_labels = [char_to_label.get(pos, self.label2id['O']) 
                          for pos in range(start, end)]
            if token_labels:
                # Use the most common label for this token
                label = max(set(token_labels), key=token_labels.count)
                labels[0, idx] = label

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten().long()
        }

class NERTrainer:
    def __init__(self, model_name: str = "bert-base-cased"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_name = model_name
        
        # Define label scheme
        self.labels = ['O', 'COMPANY', 'PERSON', 'FINANCIAL', 'EVENT', 'DATE']
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        # Add spaCy's MONEY label as an alias for FINANCIAL
        self.label2id['MONEY'] = self.label2id['FINANCIAL']
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        print(f"Label mappings: {self.label2id}")
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)

    def prepare_data(self, data_path: str):
        """Load and prepare training data."""
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        texts = []
        entities = []

        for example in raw_data:
            texts.append(example['text'])
            entities.append(example['entities'])

        return texts, entities

    def train(self, train_data_path: str, output_dir: str, epochs: int = 10):
        """Train the NER model."""
        # Prepare data
        texts, entities = self.prepare_data(train_data_path)
        print(f"Loaded {len(texts)} training examples")
        
        # Create dataset
        dataset = NERDataset(texts, entities, self.tokenizer, self.label2id)
        train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,  # 10% of total steps for warmup
            num_training_steps=total_steps
        )

        # Training loop
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="Training")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            print(f"Average loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save the best model
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                
                # Save label mappings
                with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
                    json.dump({
                        'labels': self.labels,
                        'label2id': self.label2id,
                        'id2label': self.id2label
                    }, f)
                
                print(f"Saved best model with loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs")
                    break

if __name__ == "__main__":
    # Create dataset
    print("Building training dataset...")
    builder = NERDatasetBuilder()
    builder.save_training_data()
    
    # Train model
    print("\nTraining NER model...")
    trainer = NERTrainer()
    trainer.train(
        train_data_path="data/processed/ner_training_data.json",
        output_dir="models/ner",
        epochs=10
    ) 