# src/telegram_scraper.py
from telethon.sync import TelegramClient
from telethon.tl.types import MessageMediaPhoto
import csv
import os
from dotenv import load_dotenv
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='telegram_ingestion.log'
)

class TelegramScraper:
    def __init__(self, api_id: str, api_hash: str, phone: str, channels: List[str]):
        """Initialize Telegram client and parameters."""
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.channels = channels
        self.client = TelegramClient('scraping_session', api_id, api_hash)
        self.media_dir = 'data/images'
        self.output_file = 'data/raw/telegram_data.csv'

    async def authenticate(self):
        """Authenticate with Telegram API."""
        try:
            await self.client.start(phone=self.phone)
            logging.info("Successfully authenticated with Telegram.")
        except Exception as e:
            logging.error(f"Authentication failed: {str(e)}")
            raise

    async def scrape_channel(self, channel: str, writer):
        """Scrape data from a single channel."""
        try:
            entity = await self.client.get_entity(channel)
            channel_title = entity.title
            logging.info(f"Scraping channel: {channel} ({channel_title})")
            async for message in self.client.iter_messages(entity, limit=1000):
                media_path = None
                if isinstance(message.media, MessageMediaPhoto):
                    filename = f"{channel.replace('@', '')}_{message.id}.jpg"
                    media_path = os.path.join(self.media_dir, filename)
                    await self.client.download_media(message.media, media_path)
                
                writer.writerow([
                    channel_title,
                    channel,
                    message.id,
                    message.message or '',
                    message.date,
                    media_path or ''
                ])
        except Exception as e:
            logging.error(f"Error scraping {channel}: {str(e)}")

    async def scrape_all(self):
        """Scrape all specified channels."""
        try:
            await self.client.connect()
            if not await self.client.is_user_authorized():
                await self.authenticate()

            os.makedirs(self.media_dir, exist_ok=True)
            with open(self.output_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
                
                for channel in self.channels:
                    await self.scrape_channel(channel, writer)
                    logging.info(f"Completed scraping {channel}")
        except Exception as e:
            logging.error(f"Error in scrape_all: {str(e)}")
            raise
        finally:
            await self.client.disconnect()

def main():
    # Load environment variables
    load_dotenv('.env')
    api_id = os.getenv('TG_API_ID')
    api_hash = os.getenv('TG_API_HASH')
    phone = os.getenv('phone')

    # List of channels to scrape
    channels = [
        '@ZemenExpress',
        '@nevacomputer',
        '@meneshayeofficial',
        '@ethio_brand_collection',
        '@Leyueqa',
        '@sinayelj',
        '@Shewabrand',
        '@helloomarketethiopia',
        '@modernshoppingcenter',
        '@qnashcom',
        '@Fashiontera',
        '@kuruwear',
        '@gebeyaadama',
        '@MerttEka',
        '@forfreemarket',
        '@classybrands',
        '@marakibrand',
        '@aradabrand2',
        '@marakisat2',
        '@belaclassic',
        '@AwasMart'
    ]

    # Remove duplicates
    channels = list(dict.fromkeys(channels))

    scraper = TelegramScraper(api_id, api_hash, phone, channels)
    with scraper.client:
        scraper.client.loop.run_until_complete(scraper.scrape_all())

if __name__ == "__main__":
    main()