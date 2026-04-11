from jose import jwt, JWTError
from jose.exceptions import ExpiredSignatureError
from fastapi import HTTPException, Header, Cookie
from typing import Optional, Any
import os
from dotenv import load_dotenv
import requests
from supabase import create_client
import time

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET") 

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
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
    response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_KEY,
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
        jwks = get_jwks()
        unverified_header = jwt.get_unverified_header(token)

        rsa_key = None
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = key
                break
        
        if not rsa_key:
            user_data = get_user_from_supabase(token)
            user_id = user_data.get("id")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            return str(user_id)
        
        decoded = jwt.decode(
            token,
            rsa_key,
            algorithms=["ES256"],
            audience="authenticated"
        )
        
        user_id = decoded.get("sub")
        return user_id
        
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError as e:
        print(f"JWT Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Token verification failed")


