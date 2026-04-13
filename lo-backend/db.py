from dotenv import load_dotenv
import os
import asyncpg
import ssl

try:
    import certifi
except Exception: 
    certifi = None

load_dotenv()


def _parse_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_ssl_context() -> ssl.SSLContext:
    verify_ssl = _parse_bool(os.getenv("DB_SSL_VERIFY"), True)

    if not verify_ssl:
        insecure_ctx = ssl.create_default_context()
        insecure_ctx.check_hostname = False
        insecure_ctx.verify_mode = ssl.CERT_NONE
        return insecure_ctx

    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())

    return ssl.create_default_context()

async def get_db():
    database_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL (or DB_URL) is not set")

    statement_cache_size = int(os.getenv("DB_STATEMENT_CACHE_SIZE", "0"))
    return await asyncpg.connect(
        database_url,
        ssl=_build_ssl_context(),
        statement_cache_size=statement_cache_size,
    )
    
    
    
    
