# src/evaluate_and_split.py
import random
import logging
import os
from typing import List, Tuple
import regex as re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='evaluation.log'
)

class DataEvaluator:
    def __init__(self, 
                 input_path: str = 'data/labeled/labeled_data.conll',
                 train_path: str = 'data/labeled/train.conll',
                 val_path: str = 'data/labeled/val.conll',
                 test_path: str = 'data/labeled/test.conll',
                 report_path: str = 'data/evaluation/labeling_report.txt'):
        """Initialize with input and output paths."""
        self.input_path = input_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.report_path = report_path
        self.sentences = []
        self.price_pattern = r'^\d+[,\d]*ብር?$|^ዋጋ$|^price$|^birr$'
        self.location_indicators = {'አድራሻ', 'ለቡ', 'መገናኛ', 'ሞል', 'ፎቅ', 'ቢሮ', 'ሲቲ', 'ቅርንጫፍ'}
        self.product_keywords = {
            'spatulas', 'slicer', 'guard', 'strip', 'warmer', 'sterilizer', 'steamer',
            'pan', 'clipper', 'bottle', 'box', 'bag', 'mop', 'pad', 'tape', 'diffuser'
        }

    def load_conll(self) -> List[List[Tuple[str, str]]]:
        """Load CoNLL data into list of sentences."""
        try:
            sentences = []
            current_sentence = []
            with open(self.input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('-DOCSTART-'):
                        continue
                    elif line == '':
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                    else:
                        try:
                            parts = line.split('\t')
                            if len(parts) != 2 or not parts[0].strip():
                                logging.warning(f"Skipping invalid line: {line}")
                                continue
                            token, label = parts
                            current_sentence.append((token, label))
                        except ValueError:
                            logging.warning(f"Skipping malformed line: {line}")
                            continue
            if current_sentence:
                sentences.append(current_sentence)
            logging.info(f"Loaded {len(sentences)} sentences from {self.input_path}")
            self.sentences = sentences
            return sentences
        except Exception as e:
            logging.error(f"Error loading CoNLL data: {str(e)}")
            raise

    def evaluate_labeling(self, sample_size: int = 100) -> str:
        """Evaluate labeling quality on a random sample."""
        try:
            if not self.sentences:
                self.load_conll()
            
            sample_sentences = random.sample(self.sentences, min(sample_size, len(self.sentences)))
            correct_labels = 0
            total_tokens = 0
            errors = []

            for sentence in sample_sentences:
                for token, label in sentence:
                    expected_label = 'O'
                    
                    if re.match(self.price_pattern, token) or token in {'ብር', 'ዋጋ', 'price', 'birr'}:
                        expected_label = 'I-PRICE'
                    elif token in self.location_indicators or re.match(r'^\d+[.\d]*$|^[ቁምስሪለቡ#]+$', token):
                        expected_label = 'I-LOC'
                    elif token.lower() in self.product_keywords:
                        expected_label = 'B-PRODUCT'
                    elif re.match(r'^[ፀጉራለብስምግብ]+$', token):
                        expected_label = 'I-PRODUCT'

                    if label == expected_label:
                        correct_labels += 1
                    else:
                        errors.append(f"Token: {token}, Expected: {expected_label}, Got: {label}")
                    total_tokens += 1

            accuracy = correct_labels / total_tokens if total_tokens > 0 else 0
            report = [
                f"Labeling Evaluation Report",
                f"Sample Size: {len(sample_sentences)} sentences",
                f"Total Tokens Evaluated: {total_tokens}",
                f"Correct Labels: {correct_labels}",
                f"Accuracy: {accuracy:.2%}",
                f"\nSample Errors (up to 10):",
                *(errors[:10] if errors else ["No errors found"])
            ]
            return "\n".join(report)
        except Exception as e:
            logging.error(f"Error evaluating labeling: {str(e)}")
            raise

    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split data into training, validation, and test sets."""
        try:
            if not self.sentences:
                self.load_conll()
            
            random.shuffle(self.sentences)
            total = len(self.sentences)
            train_end = int(total * train_ratio)
            val_end = int(total * (train_ratio + val_ratio))
            
            train_data = self.sentences[:train_end]
            val_data = self.sentences[train_end:val_end]
            test_data = self.sentences[val_end:]

            logging.info(f"Split data: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            return train_data, val_data, test_data
        except Exception as e:
            logging.error(f"Error splitting data: {str(e)}")
            raise

    def save_conll(self, data: List[List[Tuple[str, str]]], output_path: str):
        """Save data to CoNLL format."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('-DOCSTART-\t-X-\n\n')
                for sentence in data:
                    for token, label in sentence:
                        f.write(f"{token}\t{label}\n")
                    f.write('\n')
            logging.info(f"Saved data to {output_path}")
        except Exception as e:
            logging.error(f"Error saving CoNLL file: {str(e)}")
            raise

    def save_report(self, report: str):
        """Save evaluation report."""
        try:
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logging.info(f"Saved evaluation report to {self.report_path}")
        except Exception as e:
            logging.error(f"Error saving report: {str(e)}")
            raise

    def process(self):
        """Run evaluation and data splitting."""
        try:
            report = self.evaluate_labeling()
            self.save_report(report)
            train_data, val_data, test_data = self.split_data()
            self.save_conll(train_data, self.train_path)
            self.save_conll(val_data, self.val_path)
            self.save_conll(test_data, self.test_path)
        except Exception as e:
            logging.error(f"Error in process: {str(e)}")
            raise

def main():
    evaluator = DataEvaluator()
    evaluator.process()

if __name__ == "__main__":
    main()