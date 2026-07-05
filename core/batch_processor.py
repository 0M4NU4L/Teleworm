import pandas as pd
from typing import List, Optional
from utils.logger import Logger
from core.sentiment import CybersecuritySentimentAnalyzer
from core.report import ReportGenerator

class BatchProcessor:
    def __init__(self, batch_size: int = 1000, cybersecurity_sia: Optional[CybersecuritySentimentAnalyzer] = None):
        self.batch: List[List] = []
        self.batch_size = batch_size
        self.batch_counter = 1
        self.total_messages = 0
        self.cybersecurity_sia = cybersecurity_sia or CybersecuritySentimentAnalyzer()
        self.all_messages_df = pd.DataFrame(columns=['Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment', 'Channel Name', 'Affiliated Channel'])

    def add_messages(self, messages: List[List], channel_name: str, affiliated_channel: Optional[str]) -> None:
        messages_with_info = [
            message + [channel_name, affiliated_channel if affiliated_channel else "Initial Config"]
            for message in messages
        ]
        self.batch.extend(messages_with_info)
        self.total_messages += len(messages)
        if len(self.batch) >= self.batch_size:
            self.save_batch()

    def save_batch(self) -> None:
        if self.batch:
            df = pd.DataFrame(self.batch, columns=['Sender ID', 'Date', 'Message', 'Sentiment', 'Compound_Sentiment', 'Channel Name', 'Affiliated Channel'])
            df['Sentiment'] = df['Message'].apply(self.cybersecurity_sia.polarity_scores)
            df['Compound_Sentiment'] = df['Sentiment'].apply(lambda x: x['compound']).astype(float)
            
            batch_filename = f"telegram_scraped_messages_batch_{self.batch_counter}.csv"
            df.to_csv(batch_filename, index=False)
            Logger.success(f"Saved batch {self.batch_counter} with {len(self.batch)} messages to {batch_filename}")
            
            self.all_messages_df = pd.concat([self.all_messages_df, df], ignore_index=True)
            
            self.batch = []
            self.batch_counter += 1

    def generate_final_report(self) -> None:
        if self.all_messages_df.empty:
            Logger.warning("No messages to generate report from.")
            return
        
        ReportGenerator.generate_sentiment_report(self.all_messages_df)

    def finalize(self) -> None:
        self.save_batch()
        self.generate_final_report()
