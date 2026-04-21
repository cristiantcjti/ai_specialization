import json
from http import HTTPStatus

from fastapi import APIRouter
from starlette.responses import Response

from app.schemas import EventSchema

router = APIRouter()


@router.post("/", dependencies=[])
def hadle_event(data: EventSchema) -> Response:
    print(data)

    return Response(
        content=json.dumps({"message": "Data received!"}),
        status_code=HTTPStatus.ACCEPTED,
    )
