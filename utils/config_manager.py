import os
import json
from typing import Dict, Any, Optional
from utils.logger import Logger

class ConfigManager:
    DEFAULT_KEYWORDS = ['hacking', 'scam', 'drugs', 'vulnerability', 'exploit', 'malware']

    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}

    def load_config(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config_data = json.load(f)
                return self.config_data
        return None

    def create_default_config(self) -> Dict[str, Any]:
        default_config = {
            "api_id": "YOUR_API_ID",
            "api_hash": "YOUR_API_HASH",
            "phone_number": "YOUR_PHONE_NUMBER",
            "notification_target": "@yourusername_or_userid",
            "initial_channel_links": [
                "https://t.me/examplechannel1",
                "https://t.me/examplechannel2"
            ],
            "message_keywords": self.DEFAULT_KEYWORDS,
            "batch_size": 1000,
            "channel_depth": 2
        }
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        Logger.success(f"Default config file created at {self.config_path}")
        Logger.info("Please edit this file with your API credentials, notification target, and channel links.")
        self.config_data = default_config
        return default_config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config_data.get(key, default)
