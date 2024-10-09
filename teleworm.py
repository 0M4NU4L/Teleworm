import asyncio
import json
import re
import signal
import argparse
import os
from datetime import datetime

from telethon import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError
from telethon.tl.functions.channels import JoinChannelRequest, LeaveChannelRequest
from telethon.tl.types import Channel, User, Chat

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from colorama import init, Fore, Style

init(autoreset=True)

PURPLE_BLUE = '\033[38;2;100;100;255m'
LIGHT_PURPLE = '\033[38;2;200;180;255m'
BOLD_WHITE = '\033[1;37m'
RESET = Style.RESET_ALL

keywords = ['hacking', 'scam', 'drugs', 'vulnerability', 'exploit', 'malware']

def print_info(message):
    print(f"{PURPLE_BLUE}ℹ {BOLD_WHITE}{message}{RESET}")

def print_success(message):
    print(f"{LIGHT_PURPLE}✔ {BOLD_WHITE}{message}{RESET}")

def print_warning(message):
    print(f"{Fore.YELLOW}{Style.BRIGHT}⚠ {BOLD_WHITE}{message}{RESET}")

def print_error(message):
    print(f"{Fore.RED}✘ {message}{RESET}")

def print_header(message):
    print(f"\n{PURPLE_BLUE}{Style.BRIGHT}{message}")
    print(f"{PURPLE_BLUE}{'-' * len(message)}{RESET}")

def print_subheader(message):
    print(f"\n{LIGHT_PURPLE}{Style.BRIGHT}{message}")
    print(f"{LIGHT_PURPLE}{'-' * len(message)}{RESET}")

def banner():
    print(f"""
{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}
                _______  _______
             _/       --'       \\_
            /                     \\
       _____/          O O          \\_____
      /                                o  \\
      \\_o     __     _______     __       /
        \\_   /  \\   /       \\   /  \\_____/
          \\__/    \\_/         \\_/
  
         _______             _______
      _/       \\___________/       \\_
     /                                 \\
    /___________________________________\\
{RESET}
    """)

class CybersecuritySentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.cybersecurity_lexicon = {
            'vulnerability': 2.0,
            'exploit': -3.0,
            'patch': 2.0,
            'hack': -2.0,
            'secure': 3.0,
            'breach': -4.0,
            'protect': 3.0,
            'malware': -3.0,
            'ransomware': -4.0,
            'encryption': 2.0,
            'backdoor': -3.0,
            'firewall': 2.0,
            'phishing': -3.0,
            'authentication': 2.0,
            'threat': -2.0,
            'zero-day': -4.0,
            'security': 1.0,
            'attack': -2.0,
            'defense': 2.0,
            'compromise': -3.0
        }
        self.sia.lexicon.update(self.cybersecurity_lexicon)

    def polarity_scores(self, text):
        return self.sia.polarity_scores(text)

class ChannelManager:
    def __init__(self):
        self.joined_channels = set()
        self.processed_channels = set()
        self.channel_affiliations = {}
        self.initial_channels = set()

    def add_channel(self, link, depth, source_channel=None):
        cleaned_link = clean_link(link)
        if cleaned_link and cleaned_link not in self.joined_channels and cleaned_link not in self.processed_channels:
            channel_queue.put_nowait((cleaned_link, depth))
            if source_channel:
                self.channel_affiliations[cleaned_link] = source_channel
            else:
                self.initial_channels.add(cleaned_link)
            print_info(f"Discovered new channel: {cleaned_link} at depth {depth}")

    def mark_as_joined(self, link):
        cleaned_link = clean_link(link)
        if cleaned_link:
            self.joined_channels.add(cleaned_link)

    def mark_as_processed(self, link):
        cleaned_link = clean_link(link)
        if cleaned_link:
            self.processed_channels.add(cleaned_link)

    def has_unprocessed_channels(self):
        return not channel_queue.empty()

    def display_status(self):
        print_subheader("Channel Status")
        print(f"  Channels joined: {len(self.joined_channels)}")
        print(f"  Channels processed: {len(self.processed_channels)}")

class BatchProcessor:
    def __init__(self, batch_size=1000, cybersecurity_sia=None):
        self.batch = []
        self.batch_size = batch_size
        self.batch_counter = 1
        self.total_messages = 0
        self.cybersecurity_sia = cybersecurity_sia or CybersecuritySentimentAnalyzer()
        self.all_messages_df = pd.DataFrame(columns=['Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment', 'Channel Name', 'Affiliated Channel'])

    def add_messages(self, messages, channel_name, affiliated_channel):
        messages_with_info = [
            message + [channel_name, affiliated_channel if affiliated_channel else "Initial Config"]
            for message in messages
        ]
        self.batch.extend(messages_with_info)
        self.total_messages += len(messages)
        if len(self.batch) >= self.batch_size:
            self.save_batch()

    def save_batch(self):
        if self.batch:
            df = pd.DataFrame(self.batch, columns=['Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment', 'Channel Name', 'Affiliated Channel'])
            df['Sentiment'] = df['Message'].apply(self.cybersecurity_sia.polarity_scores)
            df['Compound_Sentiment'] = df['Sentiment'].apply(lambda x: x['compound']).astype(float)
            
            batch_filename = f"telegram_scraped_messages_batch_{self.batch_counter}.csv"
            df.to_csv(batch_filename, index=False)
            print_success(f"Saved batch {self.batch_counter} with {len(self.batch)} messages to {batch_filename}")
            
            self.all_messages_df = pd.concat([self.all_messages_df, df], ignore_index=True)
            
            self.batch = []
            self.batch_counter += 1

    def generate_final_report(self):
        if self.all_messages_df.empty:
            print_warning("No messages to generate report from.")
            return
        
        generate_sentiment_report(self.all_messages_df)

    def finalize(self):
        self.save_batch()
        self.generate_final_report()

def extract_channel_links(text):
    if not text or not isinstance(text, str):
        return []
    pattern = r'(?:https?://)?t\.me/(?:joinchat/)?[a-zA-Z0-9_-]+'
    return re.findall(pattern, text)

def clean_link(link):
    if not link or not isinstance(link, str):
        return None
    
    link = link.rstrip('.,!?;:')
    
    link = re.sub(r'^https?://', '', link)
    
    match = re.match(r't\.me/(?:joinchat/)?([a-zA-Z0-9_-]+)', link)
    if match:
        username_or_hash = match.group(1)
        if 'joinchat' in link:
            return f"joinchat/{username_or_hash}"
        return username_or_hash
    else:
        if re.match(r'^[a-zA-Z0-9_]{5,}$', link):
            return link
    return None

async def send_notification(client, target, message):
    try:
        await client.send_message(target, message)
        print_success(f"Notification sent to {target}")
    except FloodWaitError as e:
        wait_time = min(e.seconds, 30)
        print_warning(f"FloodWaitError when sending notification. Waiting for {wait_time} seconds.")
        await asyncio.sleep(wait_time)
        await send_notification(client, target, message)
    except ChannelPrivateError:
        print_warning(f"Cannot send notification to private channel {target}.")
    except Exception as e:
        print_error(f"Failed to send notification: {e}")

async def join_channel(client, channel_manager, link, max_retries=3):
    cleaned_link = clean_link(link)
    if not cleaned_link:
        print_warning(f"Invalid link format: {link}")
        return False

    retries = 0
    while retries < max_retries:
        try:
            entity = await client.get_entity(cleaned_link)
            entity_name = await get_entity_name(entity)
            
            if isinstance(entity, (Channel, Chat)):
                if hasattr(entity, 'username') and entity.username:
                    await client(JoinChannelRequest(entity))
                else:
                    print_warning(f"Cannot join private channel {entity_name} without an invite link")
                    return False
            elif isinstance(entity, User):
                print_info(f"Entity {entity_name} is a user, no need to join")
                return False
            else:
                print_warning(f"Unknown entity type for {entity_name}")
                return False
            
            print_success(f"Successfully joined and processed entity: {entity_name}")
            channel_manager.mark_as_joined(cleaned_link)
            
            joined_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            total_members = getattr(entity, 'participants_count', 'N/A')
            description = getattr(entity, 'about', 'No description')
            channel_info = {
                'Channel Name': entity.title,
                'Joined At': joined_time,
                'Total Members': total_members,
                'Description': description
            }
            df_channel = pd.DataFrame([channel_info])
            df_channel.to_csv('channels_info.csv', mode='a', header=not os.path.exists('channels_info.csv'), index=False)
            print_success(f"Saved channel info for {entity.title} to 'channels_info.csv'")
            
            return True

        except FloodWaitError as e:
            wait_time = min(e.seconds, 30)
            print_warning(f"FloodWaitError encountered. Waiting for {wait_time} seconds. (Attempt {retries + 1}/{max_retries})")
            await asyncio.sleep(wait_time)
        except ChannelPrivateError:
            print_warning(f"Private channel {cleaned_link} cannot be accessed.")
            return False
        except Exception as e:
            print_error(f"Failed to process entity {cleaned_link}: {e}")
        
        retries += 1
        await asyncio.sleep(1)
    
    print_warning(f"Max retries exceeded. Failed to process entity: {cleaned_link}")
    return False

async def scrape_messages(client, entity, message_limit, keywords, channel_manager, affiliated_channel=None, notification_target=None, current_depth=1):
    messages = []
    match_found = False
    try:
        entity_name = await get_entity_name(entity)
        async for message in client.iter_messages(entity, limit=message_limit):
            if message.text:
                if affiliated_channel:
                    print_info(f"Message from {Fore.CYAN}{Style.BRIGHT}{entity_name}{RESET}.{Fore.YELLOW}{Style.BRIGHT} <-- {affiliated_channel}{RESET}: {message.text}")
                else:
                    print_info(f"Message from {Fore.CYAN}{Style.BRIGHT}{entity_name}{RESET}: {message.text}")
                messages.append([message.sender_id, message.date.strftime('%Y-%m-%d %H:%M:%S'), message.text, None, None])
                
                if any(keyword.lower() in message.text.lower() for keyword in keywords):
                    match_found = True
                    try:
                        sender = await message.get_sender()
                        user_info = get_user_public_info(sender)
                    except Exception as e:
                        user_info = "Could not retrieve user info."
                        print_warning(f"Error fetching sender info: {e}")
                    
                    notification_message = (
                        f"**Keyword Match Found**\n\n"
                        f"**Message ID**: {message.id}\n"
                        f"**Username**: @{sender.username if sender.username else 'N/A'} (ID: {sender.id})\n"
                        f"**Date**: {message.date.strftime('%Y-%m-%d')}\n"
                        f"**Time**: {message.date.strftime('%H:%M:%S')}\n"
                        f"**Channel**: {entity_name}\n"
                        f"**User Info**: {user_info}\n\n"
                        f"**Message Content**:\n{message.text}"
                    )
                    await send_notification(client, notification_target, notification_message)
                
                links = extract_channel_links(message.text)
                for link in links:
                    channel_manager.add_channel(link, depth=current_depth + 1, source_channel=entity_name)
            
            await asyncio.sleep(0.1)
    except FloodWaitError as e:
        print_warning(f"FloodWaitError in scrape_messages: {e}")
        await asyncio.sleep(min(e.seconds, 30))
    except ChannelPrivateError:
        print_warning(f"Cannot access messages from private channel {entity_name}.")
    except Exception as e:
        print_error(f"Error scraping entity {entity_name}: {e}")
    
    if messages:
        df_messages = pd.DataFrame(messages, columns=['Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment'])
        df_messages['Sentiment'] = df_messages['Message'].apply(lambda x: CybersecuritySentimentAnalyzer().polarity_scores(x))
        df_messages['Compound_Sentiment'] = df_messages['Sentiment'].apply(lambda x: x['compound']).astype(float)
        df_messages['Channel Name'] = entity_name
        df_messages['Affiliated Channel'] = affiliated_channel if affiliated_channel else "Initial Config"
        
        df_messages.to_csv('messages.csv', mode='a', header=not os.path.exists('messages.csv'), index=False)
        print_success(f"Saved {len(messages)} messages from {entity_name} to 'messages.csv'")

    user_ids = [msg[0] for msg in messages if msg[0] is not None]
    unique_user_ids = set(user_ids)
    users_info = []
    for user_id in unique_user_ids:
        try:
            sender = await client.get_entity(user_id)
            user_info = {
                'User ID': sender.id,
                'Username': sender.username or 'N/A',
                'First Name': sender.first_name or '',
                'Last Name': sender.last_name or '',
                'Phone': sender.phone or 'N/A'
            }
            users_info.append(user_info)
        except FloodWaitError as e:
            wait_time = min(e.seconds, 30)
            print_warning(f"FloodWaitError when fetching user info. Waiting for {wait_time} seconds.")
            await asyncio.sleep(wait_time)
            try:
                sender = await client.get_entity(user_id)
                user_info = {
                    'User ID': sender.id,
                    'Username': sender.username or 'N/A',
                    'First Name': sender.first_name or '',
                    'Last Name': sender.last_name or '',
                    'Phone': sender.phone or 'N/A'
                }
                users_info.append(user_info)
            except Exception as ex:
                print_warning(f"Could not fetch user info for user ID {user_id}: {ex}")
        except ChannelPrivateError:
            print_warning(f"Cannot access user ID {user_id} as they are in a private channel.")
        except Exception as e:
            print_warning(f"Could not fetch user info for user ID {user_id}: {e}")
    
    if users_info:
        df_users = pd.DataFrame(users_info)
        df_users.to_csv('users_info.csv', mode='a', header=not os.path.exists('users_info.csv'), index=False)
        print_success(f"Saved {len(users_info)} user infos to 'users_info.csv'")
    
    return messages, entity_name, match_found

def generate_channel_names(keywords, num_channels=5):
    generated = []
    for keyword in keywords:
        for i in range(1, num_channels + 1):
            name = f"{keyword.capitalize()}Hub{i}"
            generated.append(f"https://t.me/{name}")
    return generated

def generate_sentiment_report(df):
    try:
        df['Compound_Sentiment'] = pd.to_numeric(df['Compound_Sentiment'], errors='coerce')
        
        avg_sentiment = df['Compound_Sentiment'].mean()
        
        df['Sentiment_Category'] = df['Compound_Sentiment'].apply(lambda x: 
            'High Alert' if x <= -0.5 else
            'Potential Threat' if -0.5 < x <= -0.1 else
            'Neutral' if -0.1 < x < 0.1 else
            'Potentially Positive' if 0.1 <= x < 0.5 else
            'Very Positive'
        )
        sentiment_counts = df['Sentiment_Category'].value_counts()
        total_messages = len(df)

        overall_score = avg_sentiment * 100 if not pd.isna(avg_sentiment) else 0

        report = f"""
Sentiment Analysis Report
{'-' * 50}
Total messages analyzed: {total_messages}

Overall Sentiment Score: {overall_score:.1f}/100
Interpretation: 
{interpret_overall_score(overall_score)}

Message Sentiment Breakdown:
"""

        categories = [
            ('High Alert', "Severe Threats"),
            ('Potential Threat', "Potential Threats"),
            ('Neutral', "Neutral Messages"),
            ('Potentially Positive', "Potentially Positive"),
            ('Very Positive', "Strong Security Indicators")
        ]

        for category, description in categories:
            count = sentiment_counts.get(category, 0)
            percentage = (count / total_messages) * 100 if total_messages > 0 else 0
            report += f"{category} ({description}): {count} messages ({percentage:.1f}%)\n"

        report += f"\nTop 5 Most Concerning Messages (Potential Threats):\n"
        for _, row in df.nsmallest(5, 'Compound_Sentiment').iterrows():
            threat_level = abs(row['Compound_Sentiment']) * 100
            report += f"- {row['Message'][:100]}... (Threat Level: {threat_level:.1f}/100)\n"

        report += f"\nTop 5 Most Positive Messages (Potential Security Improvements):\n"
        for _, row in df.nlargest(5, 'Compound_Sentiment').iterrows():
            positivity_level = row['Compound_Sentiment'] * 100
            report += f"- {row['Message'][:100]}... (Positivity Level: {positivity_level:.1f}/100)\n"

        with open('sentiment_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print_success("Sentiment analysis report generated and saved to 'sentiment_report.txt'")
        
        print_info("Sentiment Category Counts:")
        for category, description in categories:
            count = sentiment_counts.get(category, 0)
            percentage = (count / total_messages) * 100 if total_messages > 0 else 0
            color = get_category_color(category)
            print(f"{color}{category}: {count} ({percentage:.1f}%){RESET}")

    except Exception as e:
        print_error(f"Error generating sentiment report: {e}")

def get_category_color(category):
    color_map = {
        'High Alert': Fore.RED,
        'Potential Threat': Fore.YELLOW,
        'Neutral': Fore.WHITE,
        'Potentially Positive': Fore.LIGHTGREEN_EX,
        'Very Positive': Fore.GREEN
    }
    return color_map.get(category, '')

def interpret_overall_score(score):
    if score <= -50:
        return "Critical situation. Numerous severe threats detected. Immediate action required."
    elif -50 < score <= -10:
        return "Concerning situation. Multiple potential threats identified. Heightened vigilance needed."
    elif -10 < score < 10:
        return "Neutral situation. No significant threats or improvements detected. Maintain standard security measures."
    elif 10 <= score < 50:
        return "Positive situation. Some potential security improvements identified. Consider implementing suggested measures."
    else:
        return "Very positive situation. Strong security indicators present. Continue current security practices."

async def get_entity_name(entity):
    if isinstance(entity, User):
        return f"@{entity.username}" if entity.username else f"User({entity.id})"
    elif isinstance(entity, (Channel, Chat)):
        return entity.title or f"Channel({entity.id})"
    else:
        return f"Unknown({type(entity).__name__})"

def get_user_public_info(sender):
    if sender:
        info = []
        if sender.username:
            info.append(f"Username: @{sender.username}")
        if sender.first_name:
            info.append(f"First Name: {sender.first_name}")
        if sender.last_name:
            info.append(f"Last Name: {sender.last_name}")
        if sender.phone:
            info.append(f"Phone: {sender.phone}")
        return ', '.join(info) if info else "No public information available."
    return "No sender information."

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def create_default_config(config_path):
    default_config = {
        "api_id": "YOUR_API_ID",
        "api_hash": "YOUR_API_HASH",
        "phone_number": "YOUR_PHONE_NUMBER",
        "notification_target": "@yourusername_or_userid",
        "initial_channel_links": [
            "https://t.me/examplechannel1",
            "https://t.me/examplechannel2"
        ],
        "message_keywords": keywords,
        "batch_size": 1000,
        "channel_depth": 2
    }
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)
    print_success(f"Default config file created at {config_path}")
    print_info("Please edit this file with your API credentials, notification target, and channel links.")
    return default_config

async def run_scraper(client, config, message_depth, channel_depth):
    try:
        await client.start()
        
        channel_manager = ChannelManager()
        cybersecurity_sia = CybersecuritySentimentAnalyzer()
        batch_size = config.get('batch_size', 1000)
        batch_processor = BatchProcessor(batch_size=batch_size, cybersecurity_sia=cybersecurity_sia)
        notification_target = config.get('notification_target')

        if not notification_target:
            print_error("Notification target is not set in the config file.")
            return

        for link in config['initial_channel_links']:
            channel_manager.add_channel(link, depth=1)
        
        start_time = datetime.now()
        print_header(f"Scraping started at {start_time}")

        while not channel_queue.empty():
            link, current_depth = await channel_queue.get()
            affiliated_channel = channel_manager.channel_affiliations.get(link, None)
            if current_depth > channel_depth:
                print_info(f"Reached maximum channel depth for {link}. Skipping.")
                continue

            try:
                join_success = await join_channel(client, channel_manager, link)
                if join_success:
                    entity = await client.get_entity(link)
                    entity_messages, channel_name, match_found = await scrape_messages(
                        client, entity, message_depth, keywords, channel_manager, affiliated_channel, notification_target, current_depth
                    )
                    batch_processor.add_messages(entity_messages, channel_name, affiliated_channel)
                    
                    if not match_found:
                        await client(LeaveChannelRequest(entity))
                        print_success(f"Left channel: {channel_name} as no keywords were found.")
                        
                        notification_message = (
                            f"**No Keywords Found**\n\n"
                            f"**Channel**: {channel_name}\n"
                            f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
                            f"**Time**: {datetime.now().strftime('%H:%M:%S')}\n"
                            f"No messages containing the specified keywords were found in this channel."
                        )
                        await send_notification(client, notification_target, notification_message)
            except FloodWaitError as e:
                wait_time = min(e.seconds, 30)
                print_warning(f"FloodWaitError encountered in run_scraper. Waiting for {wait_time} seconds.")
                await asyncio.sleep(wait_time)
            except ChannelPrivateError:
                print_warning(f"Private channel {link} cannot be accessed.")
            except Exception as e:
                print_error(f"Failed to process entity {link}: {e}")
            finally:
                channel_manager.mark_as_processed(link)
            
            await asyncio.sleep(1)  # Rate limiting

        end_time = datetime.now()
        duration = end_time - start_time
        print_header(f"Scraping completed at {end_time}")
        print_info(f"Total duration: {duration}")
        print_info(f"Total messages scraped: {batch_processor.total_messages}")
        print_info(f"Total channels processed: {len(channel_manager.processed_channels)}")

        batch_processor.finalize()

    except FloodWaitError as e:
        wait_time = min(e.seconds, 30)
        print_warning(f"FloodWaitError in run_scraper: {e}. Waiting for {wait_time} seconds.")
        await asyncio.sleep(wait_time)
    except ChannelPrivateError:
        print_warning("Encountered a private channel during scraping.")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
    finally:
        await client.disconnect()

def ensure_nltk_data():
    nltk.download(['punkt', 'vader_lexicon'], quiet=True)

channel_queue = asyncio.Queue()

if __name__ == "__main__":
    banner()
    ensure_nltk_data()
    
    parser = argparse.ArgumentParser(description='Telegram Content Crawler')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--message-depth', type=int, default=1000, help='Number of messages to crawl per channel')
    parser.add_argument('--channel-depth', type=int, default=2, help='Depth of channel crawling')
    parser.add_argument('--api-id', type=str, help='API ID for Telegram client')
    parser.add_argument('--api-hash', type=str, help='API hash for Telegram client')
    parser.add_argument('--phone-number', type=str, help='Phone number for Telegram client')
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        user_input = input(f"Config file '{args.config}' not found. Create a default config? (y/n): ")
        if user_input.lower() == 'y':
            config = create_default_config(args.config)
        else:
            print_error("Please provide a valid config file. Exiting.")
            exit(1)

    API_ID = args.api_id or config.get('api_id')
    API_HASH = args.api_hash or config.get('api_hash')
    PHONE_NUMBER = args.phone_number or config.get('phone_number')
    channel_depth = args.channel_depth or config.get('channel_depth', 2)

    if not API_ID or not API_HASH or not PHONE_NUMBER:
        print_error("API credentials are missing. Please provide them either as command-line arguments or in the config file.")
        exit(1)

    print("\nSelect an option:")
    print("1. Crawl channels from initial list.")
    print("2. Generate and search for channels related to keywords.")
    choice = input("Enter choice (1/2): ")

    if choice == '1':
        for link in config['initial_channel_links']:
            cleaned = clean_link(link)
            if cleaned:
                channel_queue.put_nowait((cleaned, 1))
                print_info(f"Enqueued initial channel: {cleaned}")
    elif choice == '2':
        generated_links = generate_channel_names(config['message_keywords'])
        for link in generated_links:
            cleaned = clean_link(link)
            if cleaned:
                channel_queue.put_nowait((cleaned, 1))
                print_info(f"Enqueued AI-generated channel: {cleaned}")
    else:
        print_error("Invalid choice. Exiting.")
        exit(1)

    client = TelegramClient('session_name', API_ID, API_HASH)

    def shutdown(signal_received, frame):
        print_warning("\nReceived interrupt signal. Cleaning up...")
        asyncio.create_task(client.disconnect())
        exit(0)

    signal.signal(signal.SIGINT, shutdown)

    try:
        asyncio.run(run_scraper(client, config, args.message_depth, channel_depth))
    except KeyboardInterrupt:
        print_warning("\nReceived keyboard interrupt. Cleaning up...")
    except FloodWaitError as e:
        wait_time = min(e.seconds, 30)
        print_warning(f"FloodWaitError during execution: {e}. Waiting for {wait_time} seconds.")
        asyncio.run(asyncio.sleep(wait_time))
    except ChannelPrivateError:
        print_warning("Encountered a private channel during execution.")
    except Exception as e:
        print_error(f"An unexpected error occurred: {e}")
    finally:
        if client.is_connected():
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.run_until_complete(client.disconnect())

    print_success("Scraping completed successfully!")