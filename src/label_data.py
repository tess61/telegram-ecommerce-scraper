# src/label_data.py
import pandas as pd
import regex as re
import logging
import ast
from typing import List, Tuple
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='labeling.log'
)

class DataLabeler:
    def __init__(self, input_path: str = 'data/processed/processed_data.csv', 
                 output_path: str = 'data/labeled/labeled_data.conll'):
        """Initialize with input and output paths."""
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.product_keywords = {
            'spatulas', 'slicer', 'guard', 'strip', 'warmer', 'sterilizer', 'steamer',
            'pan', 'clipper', 'bottle', 'box', 'bag', 'mop', 'pad', 'tape', 'diffuser',
            'humidifier', 'wire', 'corrector', 'blender', 'chopper', 'whisk', 'magnifier',
            'speaker', 'base', 'refrigerator', 'cutter', 'earbuds', 'backpack', 'roller',
            'machine', 'sticker', 'dryer', 'styler', 'vibrator', 'oximeter', 'shaper'
        }
        self.price_indicators = {'ዋጋ', 'ብር', 'price', 'birr'}
        self.location_indicators = {'አድራሻ', 'ለቡ', 'መገናኛ', 'ሞል', 'ፎቅ', 'ቢሮ', 'ሲቲ', 'ቅርንጫፍ'}

    def load_data(self):
        """Load preprocessed data from CSV."""
        try:
            self.df = pd.read_csv(self.input_path, encoding='utf-8')
            logging.info(f"Loaded preprocessed data from {self.input_path}")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def label_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Label tokens with B-PRODUCT, I-PRODUCT, I-PRICE, I-LOC, or O."""
        labeled_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            label = 'O'

            # Price labeling
            if token in self.price_indicators or re.match(r'^\d+[,\d]*ብር$', token) or re.match(r'^\d+[,\d]*$', token):
                label = 'I-PRICE'
            elif i > 0 and labeled_tokens[-1][1] == 'I-PRICE' and token in {'ብር', 'birr'}:
                label = 'I-PRICE'

            # Location labeling
            elif token in self.location_indicators or re.match(r'^[ቁምስሪለቡ#]+[.\d]*$', token):
                label = 'I-LOC'

            # Product labeling
            elif token.lower() in self.product_keywords:
                label = 'B-PRODUCT'
                # Look ahead for multi-word products
                j = i + 1
                while j < len(tokens) and (
                    tokens[j].lower() in {'stainless', 'steel', 'double', 'rechargable', 'isolated',
                                          'portable', 'menstural', 'heating', 'convenience', 'wireless',
                                          'micro', 'personal', 'electric'} or
                    re.match(r'^[ፀጉርልብስምግብ]+$', tokens[j])
                ):
                    labeled_tokens.append((tokens[j], 'I-PRODUCT'))
                    j += 1
                i = j - 1

            labeled_tokens.append((token, label))
            i += 1

        return labeled_tokens

    def process_data(self):
        """Process and label all messages."""
        try:
            if self.df is None:
                self.load_data()
            
            labeled_data = []
            for idx, row in self.df.iterrows():
                try:
                    # Parse tokens (stored as string representation of list)
                    tokens = ast.literal_eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
                    labeled_tokens = self.label_tokens(tokens)
                    # Add sentence metadata
                    labeled_data.append(('-DOCSTART-', '-X-'))
                    for token, label in labeled_tokens:
                        labeled_data.append((token, label))
                    labeled_data.append(())  # Empty line for sentence boundary
                except Exception as e:
                    logging.warning(f"Error labeling message ID {row['ID']}: {str(e)}")
                    continue
            
            return labeled_data
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise

    def save_labeled_data(self):
        """Save labeled data in CoNLL format."""
        try:
            labeled_data = self.process_data()
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for item in labeled_data:
                    if isinstance(item, tuple) and len(item) == 2:
                        f.write(f"{item[0]}\t{item[1]}\n")
                    else:
                        f.write('\n')
            logging.info(f"Labeled data saved to {self.output_path}")
        except Exception as e:
            logging.error(f"Error saving labeled data: {str(e)}")
            raise

def main():
    labeler = DataLabeler()
    labeler.save_labeled_data()

if __name__ == "__main__":
    main()