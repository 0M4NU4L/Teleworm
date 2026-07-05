import asyncio
import signal
import argparse
from telethon import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError

from utils.logger import Logger
from utils.helpers import NLTKHelper
from utils.config_manager import ConfigManager
from core.scraper import TelegramScraper

def main():
    Logger.banner()
    NLTKHelper.ensure_nltk_data()
    
    parser = argparse.ArgumentParser(description='Telegram Content Crawler')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--message-depth', type=int, default=1000, help='Number of messages to crawl per channel')
    parser.add_argument('--channel-depth', type=int, default=2, help='Depth of channel crawling')
    parser.add_argument('--api-id', type=str, help='API ID for Telegram client')
    parser.add_argument('--api-hash', type=str, help='API hash for Telegram client')
    parser.add_argument('--phone-number', type=str, help='Phone number for Telegram client')
    args = parser.parse_args()

    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    
    if config is None:
        user_input = input(f"Config file '{args.config}' not found. Create a default config? (y/n): ")
        if user_input.lower() == 'y':
            config = config_manager.create_default_config()
        else:
            Logger.error("Please provide a valid config file. Exiting.")
            exit(1)

    API_ID = args.api_id or config_manager.get('api_id')
    API_HASH = args.api_hash or config_manager.get('api_hash')
    PHONE_NUMBER = args.phone_number or config_manager.get('phone_number')
    channel_depth = args.channel_depth or config_manager.get('channel_depth', 2)

    if not API_ID or not API_HASH or not PHONE_NUMBER:
        Logger.error("API credentials are missing. Please provide them either as command-line arguments or in the config file.")
        exit(1)

    print("\nSelect an option:")
    print("1. Crawl channels from initial list.")
    print("2. Generate and search for channels related to keywords.")
    choice = input("Enter choice (1/2): ")

    if choice not in ['1', '2']:
        Logger.error("Invalid choice. Exiting.")
        exit(1)

    client = TelegramClient('session_name', API_ID, API_HASH)
    scraper = TelegramScraper(client, config, args.message_depth, channel_depth)

    def shutdown(signal_received, frame):
        Logger.warning("\nReceived interrupt signal. Cleaning up...")
        asyncio.create_task(client.disconnect())
        exit(0)

    signal.signal(signal.SIGINT, shutdown)

    try:
        asyncio.run(scraper.run(choice))
    except KeyboardInterrupt:
        Logger.warning("\nReceived keyboard interrupt. Cleaning up...")
    except FloodWaitError as e:
        wait_time = min(e.seconds, 30)
        Logger.warning(f"FloodWaitError during execution: {e}. Waiting for {wait_time} seconds.")
        asyncio.run(asyncio.sleep(wait_time))
    except ChannelPrivateError:
        Logger.warning("Encountered a private channel during execution.")
    except Exception as e:
        Logger.error(f"An unexpected error occurred: {e}")
    finally:
        if client.is_connected():
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(client.disconnect())

    Logger.success("Scraping completed successfully!")

if __name__ == "__main__":
    main()