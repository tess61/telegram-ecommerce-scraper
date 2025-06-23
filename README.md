# Telegram E-commerce Scraper

This project scrapes messages and images from Ethiopian e-commerce Telegram channels, preprocesses the data, and prepares it for entity extraction (product, price, location) in CoNLL format.

## Directory Structure

- `src/`: Contains Python scripts for scraping and preprocessing.
- `data/raw/`: Stores raw scraped data (`telegram_data.csv`).
- `data/processed/`: Stores preprocessed data (`processed_data.csv`).
- `data/images/`: Stores downloaded images.
- `data/labeled/`: Stores labeled data in CoNLL format.
- `.env`: Stores Telegram API credentials (not tracked by Git).
- `telegram_ingestion.log`: Logs for scraping process.
- `preprocess.log`: Logs for preprocessing.

## Setup

1. Clone the repository:
   ```bash
        git clone https://github.com/tess61/telegram-ecommerce-scraper.git
        cd telegram-ecommerce-scraper
   ```
2. Install dependencies:
   ```bash
       pip install -r src/requirements.txt
   ```
3. Set up Telegram API credentials in .env:
   ```bash
   TG_API_ID=your_api_id
   TG_API_HASH=your_api_hash
   phone=your_phone_number
   ```
4. Run the scraper:
   ```bash
       python src/telegram_scraper.py
   ```
5. Run preprocessing:
   ```bash
       python src/preprocess.py
   ```
6. Run labeling:
   ```bash
       python src/label_data.py
   ```
