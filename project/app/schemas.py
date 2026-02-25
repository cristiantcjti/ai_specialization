from pydantic import BaseModel


class EventSchema(BaseModel):
    event_id: str
    event_type: str
    event_data: dict
