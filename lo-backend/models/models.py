from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from enum import Enum
from typing import Optional
from typing import Literal


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


class OAuthStartRequest(BaseModel):
    provider: Literal["google"]
    redirect_to: str


class OAuthSessionRequest(BaseModel):
    access_token: str


class ForgotPasswordRequest(BaseModel):
    email: str
    redirect_to: str


class ResetPasswordRequest(BaseModel):
    access_token: str
    new_password: str


class ChatMessageRequest(BaseModel):
    message: str
    source: Optional[str] = None


class ChatSource(BaseModel):
    source: str
    topic: str
    role: str
    level: int


class ChatMessageResponse(BaseModel):
    answer: str
    sources: list[ChatSource]

class PathSelect(BaseModel):
    role: Literal["student", "individual", "enterprise"]
    
class StartQuizRequest(BaseModel):
    user_profile: str

class AnswerRequest(BaseModel):
    session_id: str
    selected_index: int
    
class Question(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class QuizStartResponse(BaseModel):
    quiz_id: str
    question: dict
    question_number: int
    total_questions: int
    progress_percent: int


class QuizAnswerRequest(BaseModel):
    quiz_id: str
    answer_index: int


class QuizAnswerResponse(BaseModel):
    quiz_id: str
    correct: bool
    finished: bool
    score: int
    question_number: int
    total_questions: int
    progress_percent: int
    explanation: Optional[str] = None
    question: Optional[dict] = None
    assigned_level: Optional[int] = None
    assigned_lessons: Optional[list[dict]] = None


class QuizStartDevRequest(BaseModel):
    profile: dict


class QuizAnswerDevRequest(BaseModel):
    quiz_id: str
    answer_index: int


class LessonAssignmentResponse(BaseModel):
    role: str
    level: int
    lessons: list[dict]


class LessonCompletionRequest(BaseModel):
    source: str
    status: Literal["assigned", "in_progress", "completed"] = "completed"
    quiz_score: Optional[int] = None
    total_questions: Optional[int] = None


class DashboardStat(BaseModel):
    title: str
    value: str


class DashboardUser(BaseModel):
    name: str
    learned_minutes_today: int


class DashboardCurrentLesson(BaseModel):
    source: Optional[str] = None
    title: str
    description: str
    progress: int
    completed_label: str
    modules_label: str
    status: str


class DashboardDailyGoal(BaseModel):
    progress: int
    message: str


class DashboardRecommendation(BaseModel):
    title: str
    subtitle: str
    duration: str
    level: str


class DashboardActivityItem(BaseModel):
    type: Literal["quiz_completed", "module_started", "module_completed"]
    title: str
    subtitle: str


class DashboardResponse(BaseModel):
    user: DashboardUser
    stats: list[DashboardStat]
    current_lesson: DashboardCurrentLesson
    daily_goal: DashboardDailyGoal
    recommendations: list[DashboardRecommendation]
    recent_activity: list[DashboardActivityItem]


class ProfileLessonSummary(BaseModel):
    source: Optional[str] = None
    title: str
    status: str
    progress: int


class ProfileStats(BaseModel):
    lessons_completed: int
    lessons_in_progress: int
    lessons_assigned: int
    average_quiz_score: int
    last_quiz_score: int


class ProfileResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    full_name: str
    role: str
    difficulty: str
    current_level: int
    joined_at: Optional[datetime] = None
    stats: ProfileStats
    current_lesson: Optional[ProfileLessonSummary] = None


class LessonStoryboardSlide(BaseModel):
    title: str
    narrative: str
    bullets: list[str]
    scene_caption: Optional[str] = None
    dialogue_line: Optional[str] = None
    illustration_prompt: str
    speaker_note: Optional[str] = None
    checkpoint_question: Optional[str] = None


class LessonQuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str


class LessonNextRecommendation(BaseModel):
    source: str
    title: str
    reason: str
    level: int


class LessonStateRequest(BaseModel):
    source: str
    active_slide: int = 0
    phase: Literal["story", "quiz", "next"] = "story"
    quiz_answers: dict[str, int] = {}
    quiz_submitted: bool = False


class LessonStateResponse(BaseModel):
    source: str
    active_slide: int
    phase: Literal["story", "quiz", "next"]
    quiz_answers: dict[str, int]
    quiz_submitted: bool


class LessonDetailResponse(BaseModel):
    source: str
    topic: str
    role: str
    level: int
    status: str
    estimated_minutes: int
    progress: int
    overview: str
    learning_objectives: list[str]
    slides: list[LessonStoryboardSlide]
    end_quiz: list[LessonQuizQuestion]
    next_lesson: Optional[LessonNextRecommendation] = None
