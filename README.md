# Teleworm

Teleworm is an automated threat intelligence tool and Telegram Content Crawler built on the Telethon library. It is designed to traverse Telegram channels, scrape messages containing specific cybersecurity keywords, and perform sentiment analysis on the findings.

## Features

- **Automated Channel Crawling:** Extracts and follows `t.me` links from messages to discover and crawl new channels up to a specified depth.
- **Keyword Monitoring:** Scrapes messages containing specific keywords (e.g., hacking, scam, malware, vulnerability).
- **Sentiment Analysis:** Utilizes NLTK's VADER with a custom cybersecurity lexicon to analyze the sentiment of scraped messages and identify potential threats.
- **Data Export:** Saves scraped messages, channel information, and user details into structured CSV files (`messages.csv`, `channels_info.csv`, `users_info.csv`).
- **Live Notifications:** Sends real-time notifications to a specified Telegram target (user or channel) when keywords are detected.
- **Comprehensive Reporting:** Generates a `sentiment_report.txt` summarizing the overall threat landscape and highlighting the most concerning messages.

## Prerequisites

- Python 3.x
- A Telegram API ID and Hash. You can obtain these by creating an application at [my.telegram.org](https://my.telegram.org).

## Installation

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

On the first run, the script will automatically create a `config.json` file. You need to populate it with your API credentials and target settings:

```json
{
    "api_id": "YOUR_API_ID",
    "api_hash": "YOUR_API_HASH",
    "phone_number": "YOUR_PHONE_NUMBER",
    "notification_target": "@yourusername_or_userid",
    "initial_channel_links": [
        "https://t.me/examplechannel1",
        "https://t.me/examplechannel2"
    ],
    "message_keywords": ["hacking", "scam", "drugs", "vulnerability", "exploit", "malware"],
    "batch_size": 1000,
    "channel_depth": 2
}
```

## Usage

Run the tool using Python:

```bash
python teleworm.py
```

### Command-line Arguments

You can also pass configuration options via command-line arguments:

- `--config`: Path to the configuration file (default: `config.json`).
- `--message-depth`: Number of messages to crawl per channel (default: `1000`).
- `--channel-depth`: Maximum depth for recursive channel crawling (default: `2`).
- `--api-id`: Your Telegram API ID.
- `--api-hash`: Your Telegram API hash.
- `--phone-number`: Your Telegram phone number.

**Example:**

```bash
python teleworm.py --message-depth 500 --channel-depth 3
```

## Disclaimer

This tool is intended for educational purposes and authorized threat intelligence gathering. Ensure you comply with Telegram's Terms of Service and applicable privacy laws when using Teleworm.