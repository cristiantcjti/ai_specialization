import os
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

model = "meta-llama/llama-4-scout-17b-16e-instruct"


class EventExtraction(BaseModel):
    description: str = Field(description="Row event description")
    is_calendar_event: bool = Field(
        description="If this text is a calendar event"
    )
    confident_ponctuation: float = Field(
        description="Confident ponctuation between 0 and 1"
    )


class EventDetails(BaseModel):
    name: str = Field(description="Event name")
    date: str = Field(
        description="Event date and time. Standard format ISO 8601."
    )
    minutes_duration: int | None = Field(
        description="Expected duration in minutes"
    )
    participants: list[str] = Field(description="Participants list")


class EventConfimation(BaseModel):
    confirmation_message: str = Field(
        description="Confirmation message in natural language."
    )
    calendar_link: str = Field(
        description="Calendar link generated if applicable."
    )


def extract_event_information(user_input: str) -> EventExtraction:
    today = datetime.now()
    data_context = f"Today is {today.strftime('%A, %d of %B of %Y')}"

    response = client.responses.parse(
        model=model,
        input=f"{data_context} analyse if the text describes a calendar event.",
        instructions=f"Extract information about a possible event in this text: '{user_input}'",
        text_format=EventExtraction,
    )

    if response.output_parsed is None:
        raise ValueError(
            "Failed to extract event information from the input text."
        )
    return response.output_parsed


def analyse_event_details(description: str) -> EventDetails:
    today = datetime.now()
    data_context = f"Today is {today.strftime('%A, %d of %B of %Y')}"

    response = client.responses.parse(
        model=model,
        input=f"{data_context} Extract event's details information. When dates reference to next Tuesday or related similar dates, use the current date as reference.",
        instructions=f"Extract information about a possible event in this text: '{description}'",
        text_format=EventDetails,
    )

    if response.output_parsed is None:
        raise ValueError(
            "Failed to analyse event details from the description."
        )
    return response.output_parsed


def generate_confirmation(event_details: EventDetails) -> EventConfimation:
    response = client.responses.parse(
        model=model,
        input="Generate a natural event confirmation message. Sing the message with your name: Skynet",
        instructions=f"Create a confirmation for this event: {event_details.model_dump()}",
        text_format=EventConfimation,
    )

    if response.output_parsed is None:
        raise ValueError("Failed to generate event confirmation.")
    return response.output_parsed


def process_calendar_request(
    user_input: str,
) -> EventConfimation | None:
    initial_extraction = extract_event_information(user_input=user_input)

    if (
        not initial_extraction.is_calendar_event
        or initial_extraction.confident_ponctuation < 0.7
    ):
        return None

    event_details = analyse_event_details(initial_extraction.description)

    confirmation = generate_confirmation(event_details=event_details)

    return confirmation


# Valid calendar event example
user_input = """
    We will broadcast a live next monday at 08:00 pm with Luciano Ramalho and
    Cristian Silva to present Cris new website, it must be 2 hours long.
"""

# Invalid calendar event example
# user_input = """
#     Can you send an email to Cris and Luciano Ramalho to discuss a plan project?
# """


result = process_calendar_request(user_input=user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar link: {result.calendar_link}")
else:
    print("It seems not to be an event calendar request.")
