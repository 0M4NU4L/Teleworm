import asyncio
from typing import Set, Dict, Optional
from utils.logger import Logger
from utils.helpers import TextHelper

class ChannelManager:
    def __init__(self, queue: asyncio.Queue):
        self.joined_channels: Set[str] = set()
        self.processed_channels: Set[str] = set()
        self.channel_affiliations: Dict[str, str] = {}
        self.initial_channels: Set[str] = set()
        self.queue = queue

    def add_channel(self, link: str, depth: int, source_channel: Optional[str] = None) -> None:
        cleaned_link = TextHelper.clean_link(link)
        if cleaned_link and cleaned_link not in self.joined_channels and cleaned_link not in self.processed_channels:
            self.queue.put_nowait((cleaned_link, depth))
            if source_channel:
                self.channel_affiliations[cleaned_link] = source_channel
            else:
                self.initial_channels.add(cleaned_link)
            Logger.info(f"Discovered new channel: {cleaned_link} at depth {depth}")

    def mark_as_joined(self, link: str) -> None:
        cleaned_link = TextHelper.clean_link(link)
        if cleaned_link:
            self.joined_channels.add(cleaned_link)

    def mark_as_processed(self, link: str) -> None:
        cleaned_link = TextHelper.clean_link(link)
        if cleaned_link:
            self.processed_channels.add(cleaned_link)

    def has_unprocessed_channels(self) -> bool:
        return not self.queue.empty()

    def display_status(self) -> None:
        Logger.subheader("Channel Status")
        print(f"  Channels joined: {len(self.joined_channels)}")
        print(f"  Channels processed: {len(self.processed_channels)}")
