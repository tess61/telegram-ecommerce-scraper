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
