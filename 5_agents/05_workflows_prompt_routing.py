import os
from datetime import datetime
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

model = "meta-llama/llama-4-scout-17b-16e-instruct"


class CalendarTypeRequest(BaseModel):
    request_type: Literal["new_event", "update_event", "other"] = Field(
        description="Calendar type requested."
    )
    confident_ponctuation: float = Field(
        description="Confident ponctuation between 0 and 1"
    )
    description: str = Field(description="Row event description")


class CreateEventDetails(BaseModel):
    name: str = Field(description="Event name")
    date: str = Field(
        description="Event date and time. Standard format ISO 8601."
    )
    minutes_duration: int | None = Field(
        description="Expected duration in minutes"
    )
    participants: list[str] = Field(description="Participants list")


class Update(BaseModel):
    field_to_modify: str = Field(description="Field to be modified.")
    new_value: str = Field(description="New value to the field.")


class UpdateEventDetail(BaseModel):
    event_identifier: str = Field(
        description="Description to identify the existing event."
    )
    changes: list[Update] = Field(description="Change list to be made.")
    add_participants: list[str] = Field(
        description="New participants to be added."
    )
    remove_participants: list[str] = Field(
        description="Participants to be removed."
    )


class CalendarAnswer(BaseModel):
    is_success: bool = Field(
        description="Determine if the action was a success."
    )
    message: str = Field(description="Friendly answer to the user")
    calendar_link: str | None = Field(
        description="Calendar link if applicable."
    )


def route_calendar_request(user_input: str) -> CalendarTypeRequest:
    response = client.responses.parse(
        model=model,
        input="Define if this request is to create a new calendar event or to change an existing one.",
        instructions=f"Analyze this request: '{user_input}'",
        text_format=CalendarTypeRequest,
    )

    if response.output_parsed is None:
        raise ValueError("Failed to analyze the request.")
    return response.output_parsed


def process_new_event(description: str) -> CalendarAnswer:
    today = datetime.now()
    context_data = f"Today is {today.strftime('%A, %d of %B of %Y')}"

    response = client.responses.parse(
        model=model,
        input=f"{context_data} Extract details to create a new calendar event.",
        instructions=f"Extract structured information from this description: '{description}'",
        text_format=CreateEventDetails,
    )

    if response.output_parsed is None:
        raise ValueError(
            "Failed to analyse event details from the description."
        )

    details = response.output_parsed

    return CalendarAnswer(
        is_success=True,
        message=f"New event created '{details.name}' to {details.date} with {', '.join(details.participants)}",
        calendar_link=f"calendar://new?event={details.name}",
    )


def process_update_event(description: str) -> CalendarAnswer:
    today = datetime.now()
    context_data = f"Today is {today.strftime('%A, %d of %B of %Y')}"

    response = client.responses.parse(
        model=model,
        input=f"{context_data} Extract details to update a existing calendar event.",
        instructions=f"Extract the updated information from this description: '{description}'",
        text_format=UpdateEventDetail,
    )

    if response.output_parsed is None:
        raise ValueError(
            "Failed to analyse event details from the description."
        )

    details = response.output_parsed

    updated_text = ", ".join(
        [f"{m.field_to_modify} to {m.new_value}" for m in details.changes]
    )

    return CalendarAnswer(
        is_success=True,
        message=f"Updated event '{details.event_identifier}': {updated_text}",
        calendar_link=f"calendar://modify?event={details.event_identifier}",
    )


def process_calendar_request(
    user_input: str,
) -> CalendarAnswer | None:
    routing_result = route_calendar_request(user_input=user_input)

    if routing_result.confident_ponctuation < 0.7:
        return None

    if routing_result.request_type == "new_event":
        return process_new_event(description=routing_result.description)
    elif routing_result.request_type == "update_event":
        return process_update_event(description=routing_result.description)
    else:
        return None


# Valid new calendar event example
input_new_event = """
    We will schedule a new team meeting next friday at 02:00 pm with Luciano Ramalho and
    Cristian.
"""
result = process_calendar_request(user_input=input_new_event)
if result:
    print(f"Response: {result.message}")
else:
    print("It seems not to be an event calendar request.")


# Valid update calendar event example
update_new_event = """
    Can you re-schedule team meeting with Luciano Ramalho and Cristian to friday at 03:00 pm.
"""
result = process_calendar_request(user_input=update_new_event)
if result:
    print(f"Response: {result.message}")
else:
    print("It seems not to be an event calendar request.")


# invalid calendar event request
update_new_event = """
    How is the weather today?
"""
result = process_calendar_request(user_input=update_new_event)
if result:
    print(f"Response: {result.message}")
else:
    print("It seems not to be an calendar event request.")
