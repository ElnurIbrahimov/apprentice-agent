"""Agent build mode API endpoints."""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_api_key
from api.services.build_service import build_service

router = APIRouter(prefix="/api/agent", tags=["agent"], dependencies=[Depends(require_api_key)])


class BuildFile(BaseModel):
    path: str = Field(..., min_length=1, max_length=300)
    content: str = Field(default="", max_length=500_000)


class BuildPlanItem(BaseModel):
    path: str = Field(..., min_length=1, max_length=300)
    purpose: str = Field(default="Generated file", max_length=300)
    priority: int = Field(default=1, ge=1, le=100)


class BuildPlanRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=20_000)
    framework: str = Field(default="static", max_length=32)
    files: List[BuildFile] = Field(default_factory=list, max_length=100)
    model: Optional[str] = Field(default=None, max_length=128)


class BuildStartRequest(BuildPlanRequest):
    plan: List[BuildPlanItem] = Field(default_factory=list, max_length=100)
    max_iterations: int = Field(default=24, ge=1, le=80)


@router.post("/build/plan")
async def create_build_plan(request: BuildPlanRequest):
    plan = build_service.create_plan(
        description=request.description,
        framework=request.framework,
        files=[item.model_dump() for item in request.files],
        model=request.model,
    )
    return {
        "plan": plan,
        "framework": request.framework,
        "description": request.description,
    }


@router.post("/build")
async def start_build(request: BuildStartRequest):
    task = build_service.start_build(
        description=request.description,
        framework=request.framework,
        files=[item.model_dump() for item in request.files],
        plan=[item.model_dump() for item in request.plan] if request.plan else None,
        model=request.model,
        max_iterations=request.max_iterations,
    )
    return {
        "task_id": task.task_id,
        "status": task.status,
        "plan": task.plan,
        "workspace": task.workspace,
    }


@router.get("/build/{task_id}")
async def get_build_status(task_id: str):
    task = build_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Build task not found")
    return task.to_dict()


@router.post("/build/{task_id}/cancel")
async def cancel_build(task_id: str):
    task = build_service.cancel_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Build task not found")
    return {
        "task_id": task.task_id,
        "status": "cancel_requested",
    }
