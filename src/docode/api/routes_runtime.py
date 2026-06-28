from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder

from docode.api.auth import UserContext, get_user_context
from docode.config import DocodeConfig
from docode.llm.credentials import APICredCredentialResolver
from docode.llm.model_policy import DocodeModelPolicy


def make_runtime_router(config: DocodeConfig, user_dependency=get_user_context) -> APIRouter:
    router = APIRouter(prefix="/v1/runtime", tags=["runtime"])

    @router.get("/providers")
    async def list_runtime_providers(user: UserContext = Depends(user_dependency)) -> dict[str, object]:
        resolver = APICredCredentialResolver(config.apicred_base_url, config.apicred_token, config.apicred_mode)
        resolver.use_access_token(user.apicred_access_token)
        policy = DocodeModelPolicy(config, resolver)
        options = await policy.list_options(user_id=user.user_id)
        defaults = {
            quality: asdict(await policy.resolve(provider=None, model=None, quality=quality, user_id=user.user_id))
            for quality in ("fast", "balanced", "strong")
        }
        return jsonable_encoder(
            {
                "default_provider": defaults["balanced"]["provider"],
                "default_model": defaults["balanced"]["model"],
                "default_quality": "balanced",
                "quality_defaults": defaults,
                "options": [asdict(option) for option in options],
            }
        )

    return router
