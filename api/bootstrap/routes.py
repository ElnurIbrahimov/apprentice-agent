"""Route loading and inclusion helpers for the AURA API."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from types import ModuleType

from fastapi import FastAPI


@dataclass(frozen=True)
class RouteModuleSpec:
    """Describe a route module and the router attributes it exports."""

    name: str
    router_attrs: tuple[str, ...] = ("router",)


ROUTE_MODULE_SPECS: tuple[RouteModuleSpec, ...] = (
    RouteModuleSpec("chat"),
    RouteModuleSpec("status", ("public_router", "router")),
    RouteModuleSpec("upload"),
    RouteModuleSpec("features"),
    RouteModuleSpec("multi_agent"),
    RouteModuleSpec("reasoning_tree"),
    RouteModuleSpec("proactive"),
    RouteModuleSpec("memory"),
    RouteModuleSpec("context"),
    RouteModuleSpec("conversation_starters"),
    RouteModuleSpec("thinking"),
    RouteModuleSpec("idle_behaviors"),
    RouteModuleSpec("self_improvement"),
    RouteModuleSpec("thinking_mode"),
    RouteModuleSpec("tools_new"),
    RouteModuleSpec("activity"),
    RouteModuleSpec("multi_model"),
    RouteModuleSpec("knowledge"),
    RouteModuleSpec("search"),
    RouteModuleSpec("pdf"),
    RouteModuleSpec("transcribe"),
    RouteModuleSpec("ocr"),
    RouteModuleSpec("image_gen"),
    RouteModuleSpec("agent_action"),
    RouteModuleSpec("build"),
    RouteModuleSpec("models"),
    RouteModuleSpec("summarize"),
    RouteModuleSpec("youtube"),
    RouteModuleSpec("math"),
    RouteModuleSpec("research"),
    RouteModuleSpec("evolution"),
    RouteModuleSpec("artifacts"),
    RouteModuleSpec("feed"),
    RouteModuleSpec("providers"),
    RouteModuleSpec("code"),
    RouteModuleSpec("webhooks"),
    RouteModuleSpec("generate"),
    RouteModuleSpec("share"),
    RouteModuleSpec("hands"),
    RouteModuleSpec("telegram_miniapp"),
    RouteModuleSpec("routing"),
    RouteModuleSpec("reliability"),
    RouteModuleSpec("auth"),
    RouteModuleSpec("web_auth"),
    RouteModuleSpec("bandit"),
    # SOTA Round 2 additions — deploy with `git pull` + restart aura-api
    RouteModuleSpec("ghost"),
    RouteModuleSpec("lifelog"),
    RouteModuleSpec("mcp_server_http"),
    RouteModuleSpec("mcp_manage"),
)

CRITICAL_ROUTE_MODULES = frozenset({"chat", "status", "auth", "web_auth"})


def _is_production() -> bool:
    return os.environ.get("AURA_ENV") == "production"


def load_route_modules(logger: logging.Logger) -> dict[str, ModuleType | None]:
    """Import route modules, failing closed for critical routes in production."""

    loaded_modules: dict[str, ModuleType | None] = {}
    for spec in ROUTE_MODULE_SPECS:
        try:
            loaded_modules[spec.name] = __import__(
                f"api.routes.{spec.name}",
                fromlist=["router"],
            )
        except Exception as exc:
            loaded_modules[spec.name] = None
            if _is_production() and spec.name in CRITICAL_ROUTE_MODULES:
                raise RuntimeError(
                    f"Critical route '{spec.name}' failed to import during production startup"
                ) from exc
            logger.warning("[API] Route '%s' unavailable: %s", spec.name, exc)
    return loaded_modules


def include_loaded_routers(
    app: FastAPI,
    loaded_modules: dict[str, ModuleType | None],
) -> None:
    """Register exported routers, failing closed for critical route wiring in production."""

    for spec in ROUTE_MODULE_SPECS:
        module = loaded_modules.get(spec.name)
        if module is None:
            continue

        included = 0
        for attr in spec.router_attrs:
            router = getattr(module, attr, None)
            if router is not None:
                app.include_router(router)
                included += 1

        if not included and _is_production() and spec.name in CRITICAL_ROUTE_MODULES:
            raise RuntimeError(
                f"Critical route '{spec.name}' loaded without any of the expected routers: "
                f"{', '.join(spec.router_attrs)}"
            )
