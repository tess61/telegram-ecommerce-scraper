# src/preprocess.py
import pandas as pd
import regex as re
import unidecode
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='preprocess.log'
)

class DataPreprocessor:
    def __init__(self, input_path: str = 'data/raw/telegram_data.csv', output_path: str = 'data/processed/processed_data.csv'):
        """Initialize with input and output paths."""
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        """Load raw data from CSV."""
        try:
            self.df = pd.read_csv(self.input_path, encoding='utf-8')
            logging.info(f"Loaded data from {self.input_path}")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and normalize Amharic text."""
        try:
            if not isinstance(text, str) or not text:
                return []
            
            # Normalize text: remove extra spaces, convert to lowercase
            text = re.sub(r'\s+', ' ', text.strip())
            text = text.lower()
            
            # Tokenize: split on spaces and punctuation, preserve Amharic characters
            tokens = re.findall(r'\b[\p{Script=Ethiopic}\w]+\b', text, re.UNICODE)
            
            # Normalize Unicode characters
            tokens = [unidecode.unidecode(token) for token in tokens]
            
            return tokens
        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            return []

    def process_data(self):
        """Process all messages and structure the data."""
        try:
            if self.df is None:
                self.load_data()
            
            # Apply preprocessing to Message column
            self.df['tokens'] = self.df['Message'].apply(self.preprocess_text)
            
            # Clean metadata
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Media Path'] = self.df['Media Path'].fillna('')
            
            # Select relevant columns
            processed_df = self.df[[
                'Channel Title', 'Channel Username', 'ID', 'Message', 'tokens',
                'Date', 'Media Path'
            ]]
            
            return processed_df
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            raise

    def save_processed_data(self):
        """Save preprocessed data to CSV."""
        try:
            processed_df = self.process_data()
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            processed_df.to_csv(self.output_path, index=False, encoding='utf-8')
            logging.info(f"Preprocessed data saved to {self.output_path}")
        except Exception as e:
            logging.error(f"Error saving preprocessed data: {str(e)}")
            raise

def main():
    preprocessor = DataPreprocessor()
    preprocessor.save_processed_data()

if __name__ == "__main__":
    main()