from dotenv import find_dotenv, load_dotenv
def hello() -> str:
    return "Hello from email-assistant!"

_ = load_dotenv(find_dotenv())