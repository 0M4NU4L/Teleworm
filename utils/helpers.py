import re
import nltk
from typing import List, Optional

class TextHelper:
    @staticmethod
    def extract_channel_links(text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        pattern = r'(?:https?://)?t\.me/(?:joinchat/)?[a-zA-Z0-9_-]+'
        return re.findall(pattern, text)

    @staticmethod
    def clean_link(link: str) -> Optional[str]:
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

    @staticmethod
    def generate_channel_names(keywords: List[str], num_channels: int = 5) -> List[str]:
        generated = []
        for keyword in keywords:
            for i in range(1, num_channels + 1):
                name = f"{keyword.capitalize()}Hub{i}"
                generated.append(f"https://t.me/{name}")
        return generated

class NLTKHelper:
    @staticmethod
    def ensure_nltk_data() -> None:
        nltk.download(['punkt', 'vader_lexicon'], quiet=True)
