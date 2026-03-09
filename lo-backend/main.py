import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from routes import router as tasks_router


def _parse_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_allowed_origins() -> list[str]:
    app_env = os.getenv("APP_ENV", "development").strip().lower()
    raw = os.getenv("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    if origins:
        return origins
    if app_env == "production":
        # Force explicit origin configuration in production.
        return []
    return ["http://localhost:3000", "http://127.0.0.1:3000"]





app = FastAPI()

allow_origins = _parse_allowed_origins()
allow_credentials = _parse_bool(os.getenv("CORS_ALLOW_CREDENTIALS", "true"), True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    # Do not leak raw exceptions to clients in production responses.
    print(f"Unhandled server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(tasks_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
