from fastapi import APIRouter

from app import controller

router = APIRouter()
router.include_router(controller.router, prefix="/events", tags=["events"])
