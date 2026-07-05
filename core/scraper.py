import asyncio
import os
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional, Any
from colorama import Fore, Style

from telethon import TelegramClient
from telethon.errors import FloodWaitError, ChannelPrivateError
from telethon.tl.functions.channels import JoinChannelRequest, LeaveChannelRequest
from telethon.tl.types import Channel, User, Chat

from utils.logger import Logger
from utils.helpers import TextHelper
from core.channel_manager import ChannelManager
from core.batch_processor import BatchProcessor
from core.sentiment import CybersecuritySentimentAnalyzer

class TelegramScraper:
    def __init__(self, client: TelegramClient, config: dict, message_depth: int, channel_depth: int):
        self.client = client
        self.config = config
        self.message_depth = message_depth
        self.channel_depth = channel_depth
        self.notification_target = config.get('notification_target')
        self.keywords = config.get('message_keywords', [])
        
        # Dependencies injected or created
        self.queue = asyncio.Queue()
        self.channel_manager = ChannelManager(self.queue)
        self.cybersecurity_sia = CybersecuritySentimentAnalyzer()
        self.batch_processor = BatchProcessor(
            batch_size=config.get('batch_size', 1000), 
            cybersecurity_sia=self.cybersecurity_sia
        )

    @staticmethod
    async def get_entity_name(entity: Any) -> str:
        if isinstance(entity, User):
            return f"@{entity.username}" if entity.username else f"User({entity.id})"
        elif isinstance(entity, (Channel, Chat)):
            return entity.title or f"Channel({entity.id})"
        else:
            return f"Unknown({type(entity).__name__})"

    @staticmethod
    def get_user_public_info(sender: Any) -> str:
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

    async def send_notification(self, target: str, message: str) -> None:
        try:
            await self.client.send_message(target, message)
            Logger.success(f"Notification sent to {target}")
        except FloodWaitError as e:
            wait_time = min(e.seconds, 30)
            Logger.warning(f"FloodWaitError when sending notification. Waiting for {wait_time} seconds.")
            await asyncio.sleep(wait_time)
            await self.send_notification(target, message)
        except ChannelPrivateError:
            Logger.warning(f"Cannot send notification to private channel {target}.")
        except Exception as e:
            Logger.error(f"Failed to send notification: {e}")

    async def join_channel(self, link: str, max_retries: int = 3) -> bool:
        cleaned_link = TextHelper.clean_link(link)
        if not cleaned_link:
            Logger.warning(f"Invalid link format: {link}")
            return False

        retries = 0
        while retries < max_retries:
            try:
                entity = await self.client.get_entity(cleaned_link)
                entity_name = await self.get_entity_name(entity)
                
                if isinstance(entity, (Channel, Chat)):
                    if hasattr(entity, 'username') and entity.username:
                        await self.client(JoinChannelRequest(entity))
                    else:
                        Logger.warning(f"Cannot join private channel {entity_name} without an invite link")
                        return False
                elif isinstance(entity, User):
                    Logger.info(f"Entity {entity_name} is a user, no need to join")
                    return False
                else:
                    Logger.warning(f"Unknown entity type for {entity_name}")
                    return False
                
                Logger.success(f"Successfully joined and processed entity: {entity_name}")
                self.channel_manager.mark_as_joined(cleaned_link)
                
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
                Logger.success(f"Saved channel info for {entity.title} to 'channels_info.csv'")
                
                return True

            except FloodWaitError as e:
                wait_time = min(e.seconds, 30)
                Logger.warning(f"FloodWaitError encountered. Waiting for {wait_time} seconds. (Attempt {retries + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            except ChannelPrivateError:
                Logger.warning(f"Private channel {cleaned_link} cannot be accessed.")
                return False
            except Exception as e:
                Logger.error(f"Failed to process entity {cleaned_link}: {e}")
            
            retries += 1
            await asyncio.sleep(1)
        
        Logger.warning(f"Max retries exceeded. Failed to process entity: {cleaned_link}")
        return False

    async def scrape_messages(self, entity: Any, affiliated_channel: Optional[str] = None, current_depth: int = 1) -> Tuple[List[List], str, bool]:
        messages = []
        match_found = False
        entity_name = await self.get_entity_name(entity)
        
        try:
            async for message in self.client.iter_messages(entity, limit=self.message_depth):
                if message.text:
                    if affiliated_channel:
                        Logger.info(f"Message from {Fore.CYAN}{Style.BRIGHT}{entity_name}{Style.RESET_ALL}.{Fore.YELLOW}{Style.BRIGHT} <-- {affiliated_channel}{Style.RESET_ALL}: {message.text}")
                    else:
                        Logger.info(f"Message from {Fore.CYAN}{Style.BRIGHT}{entity_name}{Style.RESET_ALL}: {message.text}")
                    
                    messages.append([message.sender_id, message.date.strftime('%Y-%m-%d %H:%M:%S'), message.text, None, None])
                    
                    if any(keyword.lower() in message.text.lower() for keyword in self.keywords):
                        match_found = True
                        try:
                            sender = await message.get_sender()
                            user_info = self.get_user_public_info(sender)
                        except Exception as e:
                            user_info = "Could not retrieve user info."
                            Logger.warning(f"Error fetching sender info: {e}")
                        
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
                        await self.send_notification(self.notification_target, notification_message)
                    
                    links = TextHelper.extract_channel_links(message.text)
                    for link in links:
                        self.channel_manager.add_channel(link, depth=current_depth + 1, source_channel=entity_name)
                
                await asyncio.sleep(0.1)
        except FloodWaitError as e:
            Logger.warning(f"FloodWaitError in scrape_messages: {e}")
            await asyncio.sleep(min(e.seconds, 30))
        except ChannelPrivateError:
            Logger.warning(f"Cannot access messages from private channel {entity_name}.")
        except Exception as e:
            Logger.error(f"Error scraping entity {entity_name}: {e}")
        
        if messages:
            df_messages = pd.DataFrame(messages, columns=['Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment'])
            df_messages['Sentiment'] = df_messages['Message'].apply(lambda x: CybersecuritySentimentAnalyzer().polarity_scores(str(x)))
            df_messages['Compound_Sentiment'] = df_messages['Sentiment'].apply(lambda x: x['compound']).astype(float)
            df_messages['Channel Name'] = entity_name
            df_messages['Affiliated Channel'] = affiliated_channel if affiliated_channel else "Initial Config"
            
            df_messages.to_csv('messages.csv', mode='a', header=not os.path.exists('messages.csv'), index=False)
            Logger.success(f"Saved {len(messages)} messages from {entity_name} to 'messages.csv'")

        # Process user info for unique senders
        user_ids = [msg[0] for msg in messages if msg[0] is not None]
        unique_user_ids = set(user_ids)
        users_info = []
        for user_id in unique_user_ids:
            try:
                sender = await self.client.get_entity(user_id)
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
                Logger.warning(f"FloodWaitError when fetching user info. Waiting for {wait_time} seconds.")
                await asyncio.sleep(wait_time)
                try:
                    sender = await self.client.get_entity(user_id)
                    user_info = {
                        'User ID': sender.id,
                        'Username': sender.username or 'N/A',
                        'First Name': sender.first_name or '',
                        'Last Name': sender.last_name or '',
                        'Phone': sender.phone or 'N/A'
                    }
                    users_info.append(user_info)
                except Exception as ex:
                    Logger.warning(f"Could not fetch user info for user ID {user_id}: {ex}")
            except ChannelPrivateError:
                Logger.warning(f"Cannot access user ID {user_id} as they are in a private channel.")
            except Exception as e:
                Logger.warning(f"Could not fetch user info for user ID {user_id}: {e}")
        
        if users_info:
            df_users = pd.DataFrame(users_info)
            df_users.to_csv('users_info.csv', mode='a', header=not os.path.exists('users_info.csv'), index=False)
            Logger.success(f"Saved {len(users_info)} user infos to 'users_info.csv'")
        
        return messages, entity_name, match_found

    async def run(self, start_mode: str) -> None:
        try:
            await self.client.start()
            
            if not self.notification_target:
                Logger.error("Notification target is not set in the config file.")
                return

            if start_mode == '1':
                for link in self.config.get('initial_channel_links', []):
                    cleaned = TextHelper.clean_link(link)
                    if cleaned:
                        self.queue.put_nowait((cleaned, 1))
                        Logger.info(f"Enqueued initial channel: {cleaned}")
            elif start_mode == '2':
                generated_links = TextHelper.generate_channel_names(self.keywords)
                for link in generated_links:
                    cleaned = TextHelper.clean_link(link)
                    if cleaned:
                        self.queue.put_nowait((cleaned, 1))
                        Logger.info(f"Enqueued AI-generated channel: {cleaned}")
            else:
                Logger.error("Invalid choice. Exiting.")
                return
            
            start_time = datetime.now()
            Logger.header(f"Scraping started at {start_time}")

            while not self.queue.empty():
                link, current_depth = await self.queue.get()
                affiliated_channel = self.channel_manager.channel_affiliations.get(link, None)
                if current_depth > self.channel_depth:
                    Logger.info(f"Reached maximum channel depth for {link}. Skipping.")
                    continue

                try:
                    join_success = await self.join_channel(link)
                    if join_success:
                        entity = await self.client.get_entity(link)
                        entity_messages, channel_name, match_found = await self.scrape_messages(
                            entity, affiliated_channel, current_depth
                        )
                        self.batch_processor.add_messages(entity_messages, channel_name, affiliated_channel)
                        
                        if not match_found:
                            await self.client(LeaveChannelRequest(entity))
                            Logger.success(f"Left channel: {channel_name} as no keywords were found.")
                            
                            notification_message = (
                                f"**No Keywords Found**\n\n"
                                f"**Channel**: {channel_name}\n"
                                f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
                                f"**Time**: {datetime.now().strftime('%H:%M:%S')}\n"
                                f"No messages containing the specified keywords were found in this channel."
                            )
                            await self.send_notification(self.notification_target, notification_message)
                except FloodWaitError as e:
                    wait_time = min(e.seconds, 30)
                    Logger.warning(f"FloodWaitError encountered in run_scraper. Waiting for {wait_time} seconds.")
                    await asyncio.sleep(wait_time)
                except ChannelPrivateError:
                    Logger.warning(f"Private channel {link} cannot be accessed.")
                except Exception as e:
                    Logger.error(f"Failed to process entity {link}: {e}")
                finally:
                    self.channel_manager.mark_as_processed(link)
                
                await asyncio.sleep(1)  # Rate limiting

            end_time = datetime.now()
            duration = end_time - start_time
            Logger.header(f"Scraping completed at {end_time}")
            Logger.info(f"Total duration: {duration}")
            Logger.info(f"Total messages scraped: {self.batch_processor.total_messages}")
            Logger.info(f"Total channels processed: {len(self.channel_manager.processed_channels)}")

            self.batch_processor.finalize()

        except FloodWaitError as e:
            wait_time = min(e.seconds, 30)
            Logger.warning(f"FloodWaitError in run_scraper: {e}. Waiting for {wait_time} seconds.")
            await asyncio.sleep(wait_time)
        except ChannelPrivateError:
            Logger.warning("Encountered a private channel during scraping.")
        except Exception as e:
            Logger.error(f"An unexpected error occurred: {e}")
        finally:
            await self.client.disconnect()
