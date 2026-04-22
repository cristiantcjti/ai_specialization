from dotenv import load_dotenv
from mem0 import MemoryClient

load_dotenv()

client = MemoryClient()

# To add a memory:
# 1 way:
# messages = [
#     {
#         "role": "user",
#         "content": "Hi, I'm Cris. I'm a vegetarian and I'm allergic to nuts.",
#     },
#     {
#         "role": "assistant",
#         "content": "Hello Cris. I see that you're a vegetarian with a nut allergy.",
#     },
# ]

# client.add(messages, user_id="cris")

# 2 way:
# client.add("I am a software engineer that is deepeing my knowledge in AI.", user_id="cris")

query = "What is your name?"
response = client.search(query=query, filters={"user_id": "cris"})
response["results"][0]["memory"]
