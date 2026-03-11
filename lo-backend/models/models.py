from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from enum import Enum
from typing import Optional

# User Models
class UserCreate(BaseModel):
    email: str
    first_name: str
    last_name: str
    agree_terms: bool
    password: str

class User(BaseModel):
    id: UUID
    email: str
    first_name: str
    last_name: str
    agree_terms: bool
    name: Optional[str] = None
    created_at: datetime

class UserLogin(BaseModel):
    email: str
    password: str