from fastapi import APIRouter, HTTPException, Depends, Header, Query, Response, Cookie
import os
import re
import asyncpg
from uuid import NAMESPACE_DNS, uuid5
from db import get_db
from auth import verify_token
from models.models import UserCreate
from models.models import UserLogin
from models.models import PathSelect
from supabase import create_client

# --- Load environment variables ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# --- converts strings to boolean values for control purposes ---
def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Allow bypassing supabase auth for testing or alternative auth setups, but default to secure behavior.
BYPASS_SUPABASE_AUTH = parse_bool(os.getenv("BYPASS_SUPABASE_AUTH"), False)

# Validate critical environment variables at startup
if not BYPASS_SUPABASE_AUTH and (not SUPABASE_URL or not SUPABASE_KEY):
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) must be set")

# Create Supabase client if not bypassing auth
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if not BYPASS_SUPABASE_AUTH else None

# start the router
router = APIRouter()

# ENV configuration for cookies and CORS
ACCESS_TOKEN_COOKIE = os.getenv("ACCESS_TOKEN_COOKIE", "access_token")
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true" if APP_ENV == "production" else "false").lower() in {"1", "true", "yes", "on"}
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "none" if APP_ENV == "production" else "lax")
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN")

# --- Helper functions for input validation ---
PASSWORD_PATTERN = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z\d]).{8,}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def normalize_email(value: str) -> str:
    email = value.strip().strip('"\'').lower()
    email = re.sub(r"[\u200B-\u200D\uFEFF]", "", email)
    if not email or not EMAIL_PATTERN.match(email):
        raise HTTPException(status_code=400, detail="Invalid email address")
    return email


def normalize_name(value: str, field_name: str = "Name") -> str:
    normalized = re.sub(r"\s+", " ", value.strip())
    if not normalized:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty")
    if len(normalized) > 120:
        raise HTTPException(status_code=400, detail=f"{field_name} is too long")
    return normalized

def validate_password_strength(password: str) -> None:
    if not PASSWORD_PATTERN.match(password):
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be at least 8 characters and include uppercase, "
                "lowercase, number, and special character."
            ),
        )
        
# --- API Endpoints ---
@router.post("/register")
async def create_user(user: UserCreate):
    email = normalize_email(user.email)
    first_name = normalize_name(user.first_name, "First name")
    last_name = normalize_name(user.last_name, "Last name")
    validate_password_strength(user.password)

    if not user.agree_terms:
        raise HTTPException(status_code=400, detail="You must agree to the terms and conditions")

    db = await get_db()
    try:
        if BYPASS_SUPABASE_AUTH:
            user_id = str(uuid5(NAMESPACE_DNS, email))
        else:
            auth_user = supabase.auth.sign_up({
                "email": email,
                "password": user.password,
                "options": {
                    "data": {
                        "first_name": first_name,
                        "last_name": last_name,
                        "agree_terms": user.agree_terms,
                    }
                }
            })

            if auth_user.user is None:
                raise HTTPException(status_code=400, detail="Registration failed")

            user_id = auth_user.user.id

        try:
            row = await db.fetchrow("""
                INSERT INTO users (id, email, first_name, last_name, agree_terms)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """, user_id, email, first_name, last_name, user.agree_terms)
        except asyncpg.UniqueViolationError:
            raise HTTPException(status_code=409, detail="Email already exists")
        finally:
            await db.close()

        return {
            "id": row["id"],
            "email": row["email"],
            "first_name": row["first_name"],
            "last_name": row["last_name"],
            "agree_terms": row["agree_terms"],
            "detail": "Registration successful. Check your email to confirm your account."
        }

    except HTTPException:
        raise
    except Exception as e:
        error_message = str(e).lower()
        if "already registered" in error_message or "user already registered" in error_message:
            raise HTTPException(status_code=409, detail="Email already exists")
        if "rate limit" in error_message:
            raise HTTPException(
                status_code=429,
                detail="Signup is temporarily rate-limited by Supabase. Please wait and try again.",
            )
        if "for security purposes" in error_message or "you can only request this after" in error_message:
            raise HTTPException(status_code=429, detail="Signup is rate-limited by Supabase. Please wait and retry.")
        if "invalid email" in error_message or ("email" in error_message and "is invalid" in error_message):
            raise HTTPException(status_code=400, detail="Invalid email address")
        if "invalid api key" in error_message or "apikey" in error_message:
            raise HTTPException(status_code=500, detail="Supabase API key is misconfigured on the server")
        if "column" in error_message and "does not exist" in error_message:
            raise HTTPException(status_code=500, detail="Database schema mismatch for users table")

        print(f"Register error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")
    
@router.get("/check-email")
async def check_email_exists(email: str = Query(..., min_length=3)):
    normalized_email = normalize_email(email)

    db = await get_db()
    exists = await db.fetchval("""
        SELECT EXISTS (
            SELECT 1
            FROM users
            WHERE LOWER(email) = $1
        )
    """, normalized_email)
    await db.close()

    return {"available": not exists}


# --- login route ---

@router.post("/login")
async def login_user(user: UserLogin, response: Response):
    email = normalize_email(user.email)

    try:
        if BYPASS_SUPABASE_AUTH:
            raise HTTPException(
                status_code=501,
                detail="Login is not available when BYPASS_SUPABASE_AUTH is enabled"
            )

        # Ask Supabase Auth to sign in
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": user.password
        })

        # If no user/session comes back, credentials are wrong
        if not auth_response.user or not auth_response.session:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        access_token = auth_response.session.access_token

        # Store token in an HTTP cookie
        response.set_cookie(
            key=ACCESS_TOKEN_COOKIE,
            value=access_token,
            httponly=True,
            secure=COOKIE_SECURE,
            samesite=COOKIE_SAMESITE,
            domain=COOKIE_DOMAIN,
            max_age=60 * 60 * 24 * 7,
            path="/"
        )

        
        db_user = None
        db = await get_db()
        try:
            db_user = await db.fetchrow("""
                SELECT id, email, first_name, last_name
                FROM users
                WHERE LOWER(email) = $1
            """, email)
        except Exception as db_error:
            # Supabase auth succeeded; treat local profile lookup as best-effort.
            print(f"Login profile lookup warning: {db_error}")
        finally:
            await db.close()

        # Return success message
        return {
            "detail": "Login successful",
            "user": {
                "id": str(db_user["id"]) if db_user else str(auth_response.user.id),
                "email": db_user["email"] if db_user else auth_response.user.email,
                "first_name": db_user["first_name"] if db_user else None,
                "last_name": db_user["last_name"] if db_user else None,
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        error_message = str(e).lower()

        if "invalid login credentials" in error_message:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if "email not confirmed" in error_message:
            raise HTTPException(
                status_code=401,
                detail="Please verify your email before logging in"
            )

        if "invalid api key" in error_message or "apikey" in error_message:
            raise HTTPException(
                status_code=500,
                detail="Supabase API key is misconfigured on the server"
            )

        #  unexpected server error
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")
    
    
# --- select-path  ---

@router.post("/select-path")
async def select_path(
    payload: PathSelect,
    user_id: str = Depends(verify_token)
):
  

    db = await get_db()
    try:
        # --- save path or update if
        #  the user has path ---

        row = await db.fetchrow("""
            INSERT INTO user_profiles (id, role)
            VALUES ($1, $2)
            ON CONFLICT (id)
            DO UPDATE SET role = EXCLUDED.role
            RETURNING id, role, difficulty
        """, user_id, payload.role)

        return {
            "detail": "Path saved successfully",
            "profile": {
                "id": str(row["id"]),
                "role": row["role"],
                "difficulty": row["difficulty"],
            }
        }

    except Exception as e:
        print(f"Select path error: {e}")
        raise HTTPException(status_code=500, detail="Failed to save selected path to database")
    finally:
        await db.close()



@router.get("/selected-path")
async def get_selected_path(user_id: str = Depends(verify_token)):
  
    db = await get_db()
    try:
        row = await db.fetchrow("""
            SELECT id, role, difficulty
            FROM user_profiles
            WHERE id = $1
        """, user_id)

        if not row:
            raise HTTPException(status_code=404, detail="User profile not found")

        return {
            "id": str(row["id"]),
            "role": row["role"],
            "difficulty": row["difficulty"],
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Get selected path error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch selected path")
    finally:
        await db.close()

