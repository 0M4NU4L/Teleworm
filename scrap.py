from telethon import TelegramClient
from telethon.tl.functions.channels import JoinChannelRequest, LeaveChannelRequest
import re
from colorama import Fore, Style

PURPLE_BLUE = '\033[38;2;100;100;255m'
LIGHT_PURPLE = '\033[38;2;200;180;255m'
BOLD_WHITE = '\033[1;37m'

api_id = 'TELEGRAM_API_ID'
api_hash = 'TELEGRAM_API_HASH'

client = TelegramClient('USERNAME', api_id, api_hash)

keywords = ['hacking', 'scam', 'drugs']

def print_info(message):
    print(f"{PURPLE_BLUE}ℹ {BOLD_WHITE}{message}{Style.RESET_ALL}")

def print_success(message):
    print(f"{LIGHT_PURPLE}✔ {BOLD_WHITE}{message}{Style.RESET_ALL}")

def print_warning(message):
    print(f"{Fore.YELLOW}{Style.BRIGHT}⚠ {BOLD_WHITE}{message}{Style.RESET_ALL}")

def print_error(message):
    print(f"{Fore.RED}✘ {message}{Style.RESET_ALL}")

def print_header(message):
    print(f"\n{PURPLE_BLUE}{Style.BRIGHT}{message}")
    print(f"{PURPLE_BLUE}{'-' * len(message)}{Style.RESET_ALL}")

def print_subheader(message):
    print(f"\n{LIGHT_PURPLE}{Style.BRIGHT}{message}")
    print(f"{LIGHT_PURPLE}{'-' * len(message)}{Style.RESET_ALL}")

def gear_banner():
    print(f"""
{Fore.LIGHTYELLOW_EX}{Style.BRIGHT}

                _______  _______
             _/       `--'       \\_
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

{Style.RESET_ALL}
    """)

def help_menu():
    print_header("Help Menu")
    print_info("1. Join Channel: Join a Telegram channel using its link.")
    print_info("2. Search Keywords: Check for keywords like 'hacking', 'scam', 'drugs'.")
    print_info("3. Save Message: Save the message ID and user details if a keyword is found.")
    print_info("4. Exit Channel: Exit the channel if no keyword is found.")
    print_info("5. Help: Display this help menu.")
    print("\n")
    gear_banner()

async def join_channel(client, channel_link):
    try:
        await client(JoinChannelRequest(channel_link))
        print_success(f"Successfully joined the channel {channel_link}")
    except Exception as e:
        print_error(f"Failed to join channel {channel_link}: {e}")

async def leave_channel(client, channel_link):
    try:
        await client(LeaveChannelRequest(channel_link))
        print_success(f"Successfully left the channel {channel_link}")
    except Exception as e:
        print_error(f"Failed to leave channel {channel_link}: {e}")

async def scrape_messages_for_keywords(client, channel, limit=50):
    keyword_found = False
    matched_message = None
    user_info = None

    try:
        async for message in client.iter_messages(channel, limit=limit):
            if message.text:
                if any(re.search(rf'\b{keyword}\b', message.text, re.IGNORECASE) for keyword in keywords):
                    keyword_found = True
                    matched_message = message
                    user_info = await client.get_entity(message.sender_id)
                    print_success(f"Keyword found in message: {message.text}")
                    print("-" * 40)
                    break
    except Exception as e:
        print_error(f"Failed to scrape messages from {channel}: {e}")
    
    return keyword_found, matched_message, user_info

async def main_scrapper():
    channel_link = input("Enter the Telegram channel link (e.g. @examplechannel): ")

    await join_channel(client, channel_link)

    keyword_found, message, user = await scrape_messages_for_keywords(client, channel_link)

    if keyword_found:
        print_info(f"Message ID: {message.id}")
        print_info(f"User ID: {user.id}")
        print_info(f"Username: {user.username}")
        print_info(f"Name: {user.first_name} {user.last_name}")
        await client.send_message('me', f"Keyword found in channel: {channel_link}, Message ID: {message.id}, User: {user.username}")
    else:
        await leave_channel(client, channel_link)
        await client.send_message('me', f"No keyword found in channel: {channel_link}. Left the channel.")

def main_menu():
    print_header("Welcome to Telegram Scraper!")
    gear_banner()
    print_subheader("Main Menu:")
    print_info("1. Run Scraper")
    print_info("2. Help")
    print_info("3. Exit")

    choice = input("\nChoose an option: ")
    
    if choice == '1':
        print_success("Running the scraper...")
        with client:
            client.loop.run_until_complete(main_scrapper())
    elif choice == '2':
        help_menu()
        main_menu()
    elif choice == '3':
        print_success("Exiting the application.")
        exit()
    else:
        print_warning("Invalid choice. Please select a valid option.")
        main_menu()

if __name__ == "__main__":
    main_menu()
