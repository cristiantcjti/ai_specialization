import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = client.responses.parse(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    input="John and Jass will be travelling next year.",
    instructions="Extraia informações do evento.",
    text_format=CalendarEvent,
)

event = response.output_parsed
if event is not None:
    print(event.model_dump_json(indent=2))
