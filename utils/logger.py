from colorama import init, Fore, Style

init(autoreset=True)

class Logger:
    PURPLE_BLUE = '\033[38;2;100;100;255m'
    LIGHT_PURPLE = '\033[38;2;200;180;255m'
    BOLD_WHITE = '\033[1;37m'
    RESET = Style.RESET_ALL

    @classmethod
    def info(cls, message: str) -> None:
        print(f"{cls.PURPLE_BLUE}ℹ {cls.BOLD_WHITE}{message}{cls.RESET}")

    @classmethod
    def success(cls, message: str) -> None:
        print(f"{cls.LIGHT_PURPLE}✔ {cls.BOLD_WHITE}{message}{cls.RESET}")

    @classmethod
    def warning(cls, message: str) -> None:
        print(f"{Fore.YELLOW}{Style.BRIGHT}⚠ {cls.BOLD_WHITE}{message}{cls.RESET}")

    @classmethod
    def error(cls, message: str) -> None:
        print(f"{Fore.RED}✘ {message}{cls.RESET}")

    @classmethod
    def header(cls, message: str) -> None:
        print(f"\n{cls.PURPLE_BLUE}{Style.BRIGHT}{message}")
        print(f"{cls.PURPLE_BLUE}{'-' * len(message)}{cls.RESET}")

    @classmethod
    def subheader(cls, message: str) -> None:
        print(f"\n{cls.LIGHT_PURPLE}{Style.BRIGHT}{message}")
        print(f"{cls.LIGHT_PURPLE}{'-' * len(message)}{cls.RESET}")

    @classmethod
    def banner(cls) -> None:
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
{cls.RESET}
    """)
