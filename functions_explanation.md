# Teleworm Functional Documentation

This document provides a comprehensive overview of all classes and functions in the refactored Teleworm codebase, which has been split into Object-Oriented modules.

## 1. Core Modules (`core/`)

### `core/scraper.py`
**Class: `TelegramScraper`**
The main orchestrator for interacting with the Telegram API.
- `__init__(self, client, config, message_depth, channel_depth)`: Initializes the scraper with the Telegram client, configuration, and dependencies (like `ChannelManager` and `BatchProcessor`).
- `get_entity_name(entity)` *(static)*: Extracts a human-readable name (username or title) from a Telegram `User`, `Chat`, or `Channel` entity.
- `get_user_public_info(sender)` *(static)*: Extracts available public information (username, first name, last name, phone) from a Telegram user entity.
- `send_notification(self, target, message)` *(async)*: Sends an alert message to a specified target via Telegram, handling flood waits.
- `join_channel(self, link, max_retries)` *(async)*: Attempts to join a Telegram channel using an invite link or username, managing retries and flood limits.
- `scrape_messages(self, entity, affiliated_channel, current_depth)` *(async)*: Iterates through messages in a channel up to the `message_depth`, logs keyword matches, sends notifications, extracts new channel links, and saves messages to a CSV file.
- `run(self, start_mode)` *(async)*: The main execution loop. It consumes channels from the queue, processes them, and stops when maximum depth or queue depletion is reached.

### `core/channel_manager.py`
**Class: `ChannelManager`**
Manages the discovery, queuing, and tracking of Telegram channels.
- `__init__(self, queue)`: Initializes the manager with an `asyncio.Queue` to track channels awaiting processing.
- `add_channel(self, link, depth, source_channel)`: Cleans a newly found link and adds it to the queue if it hasn't been processed or joined before.
- `mark_as_joined(self, link)`: Registers a channel as successfully joined.
- `mark_as_processed(self, link)`: Registers a channel as fully processed to avoid duplicate scraping.
- `has_unprocessed_channels(self)`: Checks if the queue is empty.
- `display_status(self)`: Logs the current count of joined and processed channels.

### `core/batch_processor.py`
**Class: `BatchProcessor`**
Handles the batching, sentiment evaluation, and saving of scraped messages.
- `__init__(self, batch_size, cybersecurity_sia)`: Initializes the processor with a specified batch limit and sentiment analyzer.
- `add_messages(self, messages, channel_name, affiliated_channel)`: Adds raw messages to the current batch. Triggers `save_batch` if the limit is reached.
- `save_batch(self)`: Converts the batch to a pandas DataFrame, evaluates sentiment, and saves it to a numbered CSV file (`telegram_scraped_messages_batch_X.csv`).
- `generate_final_report(self)`: Triggers the generation of the final sentiment summary using `ReportGenerator`.
- `finalize(self)`: A cleanup method called at the end of scraping to flush any remaining messages and generate the final report.

### `core/sentiment.py`
**Class: `CybersecuritySentimentAnalyzer`**
Evaluates text for sentiment, tailored specifically for cybersecurity contexts.
- `__init__(self)`: Initializes the NLTK `SentimentIntensityAnalyzer` and injects a custom cybersecurity lexicon (e.g., 'vulnerability' is positive, 'exploit' is negative).
- `polarity_scores(self, text)`: Calculates and returns the sentiment polarity scores (compound, positive, negative, neutral) for a given text.

### `core/report.py`
**Class: `ReportGenerator`**
Responsible for summarizing the results of a scraping session.
- `generate_sentiment_report(df)` *(static)*: Analyzes the complete DataFrame of scraped messages, calculates category breakdowns (e.g., 'High Alert', 'Potential Threat'), and saves a text report (`sentiment_report.txt`).
- `get_category_color(category)` *(static)*: Returns the appropriate CLI color code based on the threat severity category.
- `interpret_overall_score(score)` *(static)*: Provides a human-readable interpretation of the overall sentiment score.

---

## 2. Utility Modules (`utils/`)

### `utils/logger.py`
**Class: `Logger`**
Provides centralized, colored console output functionality.
- `info(message)` *(classmethod)*: Prints informational messages in purple/blue.
- `success(message)` *(classmethod)*: Prints success messages in light purple.
- `warning(message)` *(classmethod)*: Prints warning messages in yellow.
- `error(message)` *(classmethod)*: Prints error messages in red.
- `header(message)` *(classmethod)*: Prints a prominent, underlined header.
- `subheader(message)` *(classmethod)*: Prints a secondary header.
- `banner()` *(classmethod)*: Prints the Teleworm ASCII art logo.

### `utils/config_manager.py`
**Class: `ConfigManager`**
Handles loading, validating, and generating the JSON configuration.
- `__init__(self, config_path)`: Initializes with the path to `config.json`.
- `load_config(self)`: Reads and parses the JSON file if it exists.
- `create_default_config(self)`: Generates a template `config.json` file with placeholder API credentials.
- `get(self, key, default)`: Retrieves a specific configuration value safely.

### `utils/helpers.py`
**Class: `TextHelper`**
Provides text manipulation and sanitization utilities.
- `extract_channel_links(text)` *(static)*: Uses regex to find `t.me` links within a block of text.
- `clean_link(link)` *(static)*: Strips trailing punctuation and formatting to yield a standardized Telegram handle or join hash.
- `generate_channel_names(keywords, num_channels)` *(static)*: AI-generates a list of synthetic channel links based on keywords (e.g., `HackingHub1`).

**Class: `NLTKHelper`**
- `ensure_nltk_data()` *(static)*: Silently downloads required NLTK datasets (`punkt`, `vader_lexicon`) upon application startup.

---

## 3. Entry Point

### `teleworm.py`
The main execution script.
- `main()`: Parses command-line arguments, initializes the `ConfigManager` and `Logger`, prompts the user for the operating mode, instantiates the `TelegramClient` and `TelegramScraper`, and handles graceful shutdown via signal trapping (`SIGINT`).
