import spacy
from typing import List, Dict, Any
import re

from src.ml.ner_predictor import NERPredictor

class SignalExtractor:
    def __init__(self):
        """Initializes the SignalExtractor with an NER predictor and NLP model."""
        print("Initializing Signal Extractor...")
        self.ner_predictor = NERPredictor()
        self.nlp = spacy.load("en_core_web_sm")
        print("Signal Extractor initialized.")

        # Define rule-based patterns for signals
        self.signal_rules = {
            "FUNDING": {
                "keywords": [
                    "raise", "funding", "round", "seed", "series", 
                    "investment", "backing", "raised", "invested"
                ],
                # We will look for at least one company and a monetary value (either as FINANCIAL entity or regex match)
                "entities": ["COMPANY"] 
            },
            "ACQUISITION": {
                "keywords": ["acquire", "acquires", "acquired", "acquisition"],
                "entities": ["COMPANY"]
            }
        }

    def _find_company_fallback(self, sent: Any) -> List[str]:
        """Fallback to find company names using simple NLP patterns."""
        # Pattern: Look for proper nouns at the beginning of a sentence
        # This is a simple heuristic and can be improved.
        companies = []
        for token in sent:
            if token.pos_ == "PROPN":
                companies.append(token.text)
            else:
                # Stop at the first non-proper-noun
                break 
        return companies

    def extract_signals_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts signals (e.g., funding, acquisition) from a block of text.

        Args:
            text: The input text to analyze.

        Returns:
            A list of signals found in the text.
        """
        signals = []
        entities = self.ner_predictor.predict_entities(text)
        doc = self.nlp(text)

        entity_map = {ent['text']: ent['label'] for ent in entities}

        for sent in doc.sents:
            sent_text = sent.text.lower()
            sent_entities = [ent for ent in entities if ent['start'] >= sent.start_char and ent['end'] <= sent.end_char]

            for signal_type, rules in self.signal_rules.items():
                # Check if any keyword for the signal type is in the sentence
                if any(re.search(r'\b' + keyword + r'\b', sent_text) for keyword in rules["keywords"]):
                    
                    # Check for required entities or fallbacks
                    companies_found = [ent for ent in sent_entities if ent['label'] == 'COMPANY']
                    
                    if signal_type == "FUNDING":
                        # Rule: Requires keywords + a monetary value + a company
                        has_monetary_value = any(ent['label'] == 'FINANCIAL' for ent in sent_entities) or \
                                             re.search(r'\$\d+(\.\d+)?\s*(million|billion|thousand)?', sent_text)
                        
                        if not companies_found:
                            fallback_companies = self._find_company_fallback(sent)
                            if fallback_companies:
                                companies_found.append({'text': ' '.join(fallback_companies), 'label': 'COMPANY (Fallback)'})

                        if not (has_monetary_value and companies_found):
                            continue

                    elif signal_type == "ACQUISITION":
                        # Rule: Requires at least two companies (acquirer and acquired)
                        if len(companies_found) < 2:
                            continue
                    
                    signal = {
                        "type": signal_type,
                        "sentence": sent.text,
                        "entities": sent_entities,
                        "context": {
                            "companies": companies_found
                        }
                    }
                    signals.append(signal)

        return self._deduplicate_signals(signals)

    def _deduplicate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates signals based on the sentence."""
        unique_signals = []
        seen_sentences = set()
        for signal in signals:
            if signal["sentence"] not in seen_sentences:
                unique_signals.append(signal)
                seen_sentences.add(signal["sentence"])
        return unique_signals

if __name__ == '__main__':
    # Example usage
    extractor = SignalExtractor()
    
    example_text = (
        "ProjectDiscovery, a startup building open source security tools, today "
        "announced that it has raised $25 million in a Series A funding round. "
        "Separately, it was reported that TechCrunch was acquired by AOL for $30 million. "
        "The company also launched a new product last week."
    )

    print(f"\nAnalyzing text: \"{example_text}\"")
    extracted_signals = extractor.extract_signals_from_text(example_text)

    print("\n--- Extracted Signals ---")
    if extracted_signals:
        for signal in extracted_signals:
            print(f"Type: {signal['type']}")
            print(f"  Sentence: \"{signal['sentence'].strip()}\"")
            print(f"  Entities: {signal['entities']}")
            print("-" * 20)
    else:
        print("No signals found.") 