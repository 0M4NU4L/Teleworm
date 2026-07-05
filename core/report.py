import pandas as pd
from colorama import Fore, Style
from utils.logger import Logger

class ReportGenerator:
    @staticmethod
    def generate_sentiment_report(df: pd.DataFrame) -> None:
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
{ReportGenerator.interpret_overall_score(overall_score)}

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
                report += f"- {str(row['Message'])[:100]}... (Threat Level: {threat_level:.1f}/100)\n"

            report += f"\nTop 5 Most Positive Messages (Potential Security Improvements):\n"
            for _, row in df.nlargest(5, 'Compound_Sentiment').iterrows():
                positivity_level = row['Compound_Sentiment'] * 100
                report += f"- {str(row['Message'])[:100]}... (Positivity Level: {positivity_level:.1f}/100)\n"

            with open('sentiment_report.txt', 'w', encoding='utf-8') as f:
                f.write(report)

            Logger.success("Sentiment analysis report generated and saved to 'sentiment_report.txt'")
            
            Logger.info("Sentiment Category Counts:")
            for category, description in categories:
                count = sentiment_counts.get(category, 0)
                percentage = (count / total_messages) * 100 if total_messages > 0 else 0
                color = ReportGenerator.get_category_color(category)
                print(f"{color}{category}: {count} ({percentage:.1f}%){Style.RESET_ALL}")

        except Exception as e:
            Logger.error(f"Error generating sentiment report: {e}")

    @staticmethod
    def get_category_color(category: str) -> str:
        color_map = {
            'High Alert': Fore.RED,
            'Potential Threat': Fore.YELLOW,
            'Neutral': Fore.WHITE,
            'Potentially Positive': Fore.LIGHTGREEN_EX,
            'Very Positive': Fore.GREEN
        }
        return color_map.get(category, '')

    @staticmethod
    def interpret_overall_score(score: float) -> str:
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
