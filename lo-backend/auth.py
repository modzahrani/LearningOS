from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError
from fastapi import HTTPException, Header, Cookie
from typing import Optional, Any
import os
from pathlib import Path

from dotenv import load_dotenv
import requests
import time

load_dotenv(Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
SUPABASE_API_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL:
    raise RuntimeError(
        "SUPABASE_URL is not set. Add it to lo-backend/.env (see .env.example) "
        "or export it before starting the server."
    )

ACCESS_TOKEN_COOKIE = os.getenv("ACCESS_TOKEN_COOKIE", "access_token")
JWKS_CACHE_TTL_SECONDS = int(os.getenv("JWKS_CACHE_TTL_SECONDS", "300"))
JWKS_REQUEST_TIMEOUT_SECONDS = float(os.getenv("JWKS_REQUEST_TIMEOUT_SECONDS", "3"))
_jwks_cache: dict[str, Any] = {"expires_at": 0.0, "data": None}

def get_jwks():
    """Get public keys from Supabase with a short in-memory cache."""
    now = time.time()
    if _jwks_cache["data"] and now < _jwks_cache["expires_at"]:
        return _jwks_cache["data"]

    jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    response = requests.get(jwks_url, timeout=JWKS_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()

    _jwks_cache["data"] = data
    _jwks_cache["expires_at"] = now + JWKS_CACHE_TTL_SECONDS
    return data

def get_user_from_supabase(token: str) -> dict[str, Any]:
    """Ask Supabase Auth to validate token and return user payload."""
    if not SUPABASE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) is required for auth fallback",
        )

    response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_API_KEY,
        },
        timeout=JWKS_REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Invalid token")

    response.raise_for_status()
    return response.json()

def verify_token(
    authorization: Optional[str] = Header(None),
    access_token: Optional[str] = Cookie(None, alias=ACCESS_TOKEN_COOKIE),
):
    token: Optional[str] = None

    if authorization:
        token = authorization.replace("Bearer ", "").strip()
    elif access_token:
        token = access_token.strip()

    if not token:
        raise HTTPException(status_code=401, detail="No authorization token provided")
    
    try:
        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg")

        if algorithm == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise HTTPException(
                    status_code=500,
                    detail="SUPABASE_JWT_SECRET is required for HS256 token verification",
                )

            decoded = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=f"{SUPABASE_URL}/auth/v1",
            )
        else:
            jwks = get_jwks()
            signing_key = None
            for key in jwks["keys"]:
                if key.get("kid") == unverified_header.get("kid"):
                    signing_key = key
                    break

            if not signing_key:
                raise HTTPException(status_code=401, detail="Unable to find appropriate signing key")

            decoded = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256", "ES256"],
                audience="authenticated",
                issuer=f"{SUPABASE_URL}/auth/v1",
            )
        
        user_id = decoded.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token payload missing subject")
        return user_id
        # If we successfully decoded the token, return the user ID from the "sub" claim.
    except HTTPException:
        raise
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        print(f"JWT Error: {e}")
        # Fallback for environments where local JWT secret is not aligned.
        user_data = get_user_from_supabase(token)
        user_id = user_data.get("id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        return str(user_id)
    except Exception as e:
        print(f"Token verification error in verify_token: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")


