# src/analyze_vendors.py
import pandas as pd
import logging
import os
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import regex as re
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='analytics.log'
)

class VendorAnalyzer:
    def __init__(self,
                 data_path: str = 'data/raw/telegram_data.csv',
                 model_path: str = 'models/ner_model/ner/model',
                 output_csv: str = 'data/analytics/vendor_scorecard.csv'):
        """Initialize with input and output paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.output_csv = output_csv
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
            logging.info(f"Loaded NER model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading NER model: {str(e)}")
            raise
        self.df = None

    def load_data(self):
        """Load raw Telegram data."""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8')
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df = self.df.dropna(subset=['Date'])  # Drop invalid dates
            logging.info(f"Loaded {len(self.df)} messages from {self.data_path}")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def extract_entities(self, text: str) -> Dict[str, List]:
        """Extract entities using NER model."""
        try:
            if not isinstance(text, str) or not text.strip():
                return {'products': [], 'prices': [], 'locations': []}
            predictions = self.pipeline(text)
            entities = {'products': [], 'prices': [], 'locations': []}
            for pred in predictions:
                entity = pred['entity'].split('-')[-1] if '-' in pred['entity'] else pred['entity']
                if entity == 'PRODUCT':
                    entities['products'].append(pred['word'])
                elif entity == 'PRICE':
                    if re.match(r'^\d+[,\d]*ብር?$', pred['word']):
                        price = float(pred['word'].replace('ብር', '').replace(',', ''))
                        entities['prices'].append(price)
                elif entity == 'LOC':
                    entities['locations'].append(pred['word'])
            return entities
        except Exception as e:
            logging.warning(f"Error extracting entities from text: {str(e)}")
            return {'products': [], 'prices': [], 'locations': []}

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate vendor metrics."""
        try:
            if self.df is None:
                self.load_data()

            # Initialize metrics
            vendors = self.df['Channel Username'].unique()
            metrics = []

            # Time range for frequency calculation
            min_date = self.df['Date'].min()
            max_date = self.df['Date'].max()
            weeks = max((max_date - min_date).days / 7, 1)  # Avoid division by zero

            for vendor in vendors:
                vendor_df = self.df[self.df['Channel Username'] == vendor]
                
                # Posting Frequency: Posts per week
                posts_per_week = len(vendor_df) / weeks
                
                # Average Views per Post
                avg_views = vendor_df['Views'].mean() if 'Views' in vendor_df.columns and not vendor_df.empty else 0
                
                # Top Performing Post
                top_views, top_product, top_price = 0, 'Unknown', 0
                if 'Views' in vendor_df.columns and not vendor_df.empty:
                    top_post = vendor_df.loc[vendor_df['Views'].idxmax()]
                    top_views = top_post['Views']
                    top_text = top_post['Message']
                    entities = self.extract_entities(top_text)
                    top_product = entities['products'][0] if entities['products'] else 'Unknown'
                    top_price = entities['prices'][0] if entities['prices'] else 0
                
                # Average Price Point
                prices = []
                for _, row in vendor_df.iterrows():
                    entities = self.extract_entities(row['Message'])
                    prices.extend(entities['prices'])
                avg_price = sum(prices) / len(prices) if prices else 0
                
                # Lending Score: 0.5 * Normalized Avg Views + 0.5 * Posts/Week
                norm_views = avg_views / 1000 if avg_views > 0 else 0  # Normalize views
                lending_score = 0.5 * norm_views + 0.5 * posts_per_week
                
                metrics.append({
                    'Vendor': vendor,
                    'Avg Views/Post': round(avg_views, 2),
                    'Posts/Week': round(posts_per_week, 2),
                    'Avg Price (ETB)': round(avg_price, 2),
                    'Top Post Views': top_views,
                    'Top Product': top_product,
                    'Top Price (ETB)': top_price,
                    'Lending Score': round(lending_score, 2)
                })
            
            metrics_df = pd.DataFrame(metrics)
            logging.info(f"Calculated metrics for {len(vendors)} vendors")
            return metrics_df
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            raise

    def save_scorecard(self):
        """Save vendor scorecard to CSV."""
        try:
            metrics_df = self.calculate_metrics()
            os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
            metrics_df.to_csv(self.output_csv, index=False, encoding='utf-8')
            logging.info(f"Saved scorecard to {self.output_csv}")
        except Exception as e:
            logging.error(f"Error saving scorecard: {str(e)}")
            raise

def main():
    analyzer = VendorAnalyzer()
    analyzer.save_scorecard()

if __name__ == "__main__":
    main()