import json
import os
import time
from typing import Any

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(MODEL_NAME)


def _fallback_selection(
    candidates: list[dict[str, Any]],
    target_level: int,
    max_lessons: int,
) -> list[str]:
    # Deterministic backup: prefer lessons whose stored level is closest to the
    # target level when the model cannot produce a valid recommendation.
    ordered = sorted(
        candidates,
        key=lambda item: (
            abs(int(item.get("level", 1)) - target_level),
            item.get("topic", ""),
        ),
    )
    return [item["source"] for item in ordered[:max_lessons] if item.get("source")]


async def recommend_lessons(
    profile: dict[str, Any],
    candidates: list[dict[str, Any]],
    target_level: int,
    max_lessons: int = 4,
    max_retries: int = 3,
) -> dict[str, Any]:
    # Recommendation happens at the lesson level. The chunk grouping work is
    # already done upstream before this function receives its candidates.
    if not candidates:
        return {
            "selected_sources": [],
            "reasoning": "No lesson candidates were available.",
        }

    compact_candidates = []
    for candidate in candidates:
        # Keep the prompt compact: enough metadata and sample text for ranking,
        # but not the full raw chunk payload for every lesson.
        compact_candidates.append(
            {
                "source": candidate.get("source"),
                "topic": candidate.get("topic"),
                "role": candidate.get("role"),
                "level": candidate.get("level"),
                "format": candidate.get("format"),
                "chunk_count": candidate.get("chunk_count"),
                "sample_content": candidate.get("sample_content", "")[:500],
            }
        )

    prompt = f"""
You are an AI curriculum planner for LearningOS.
Choose the best lessons for this learner after the questionnaire.

Learner profile:
{json.dumps(profile, ensure_ascii=True)}

Target level from questionnaire:
{target_level}

Available lesson candidates:
{json.dumps(compact_candidates, ensure_ascii=True)}

Rules:
- Select up to {max_lessons} lesson sources.
- Prioritize lessons that best fit the learner role and target level.
- Use the learner's final questionnaire score, weak areas, and answer history to pick lessons that close knowledge gaps.
- Avoid recommending lessons the learner has already completed unless they clearly need reinforcement.
- Use the sample chunk content to judge relevance.
- Prefer a coherent sequence from foundational to more advanced material.
- Return STRICT JSON only.

Format:
{{
  "selected_sources": ["source1", "source2"],
  "reasoning": "short explanation"
}}
"""

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()

            payload = json.loads(text)
            selected_sources = payload.get("selected_sources") or []
            if not isinstance(selected_sources, list):
                raise ValueError("selected_sources must be a list")

            # Never trust model output blindly; keep only sources that exist in
            # the known candidate set before assignment continues.
            valid_sources = {
                candidate["source"]
                for candidate in compact_candidates
                if candidate.get("source")
            }
            filtered = [
                source for source in selected_sources
                if isinstance(source, str) and source in valid_sources
            ]

            if not filtered:
                filtered = _fallback_selection(candidates, target_level, max_lessons)

            return {
                "selected_sources": filtered[:max_lessons],
                "reasoning": str(payload.get("reasoning") or ""),
            }
        except Exception as exc:
            last_error = exc
            print(f"Lesson_agent recommendation error: {exc}")
            time.sleep(0.5 * (attempt + 1))

    print(f"Lesson_agent fallback after {max_retries} failed attempts: {last_error}")
    return {
        "selected_sources": _fallback_selection(candidates, target_level, max_lessons),
        "reasoning": "Fallback selection based on target level proximity.",
    }


def _fallback_storyboard(
    lesson: dict[str, Any],
    lesson_text: str,
    next_lesson: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # This keeps lesson delivery working even if storyboard generation fails or
    # returns invalid JSON.
    clean_text = " ".join((lesson_text or "").split())
    paragraphs = [part.strip() for part in lesson_text.split("\n\n") if part.strip()]

    slides = []
    for index, paragraph in enumerate(paragraphs[:4], start=1):
        bullets = [
            sentence.strip()
            for sentence in paragraph.replace("\n", " ").split(".")
            if sentence.strip()
        ][:3]
        slides.append(
            {
                "title": f"{lesson.get('topic', 'Lesson')} Part {index}",
                "narrative": paragraph[:420],
                "bullets": bullets or [paragraph[:120]],
                "scene_caption": f"A cartoon scene showing {lesson.get('topic', 'the lesson topic')} in action.",
                "dialogue_line": "Let’s walk through this idea one step at a time.",
                "illustration_prompt": (
                    f"Friendly educational storyboard illustration for "
                    f"{lesson.get('topic', 'this lesson')}, slide {index}."
                ),
                "speaker_note": "Explain the idea in simple language and relate it to the learner's path.",
                "checkpoint_question": None,
            }
        )

    if not slides:
        slides = [
            {
                "title": lesson.get("topic", "Lesson Overview"),
                "narrative": clean_text[:420] or "This lesson introduces the main idea step by step.",
                "bullets": [
                    "Start with the core idea.",
                    "Connect it to the learner's current level.",
                    "End with one practical takeaway.",
                ],
                "scene_caption": "A playful classroom comic panel introducing the lesson.",
                "dialogue_line": "Here’s the big idea we want to understand today.",
                "illustration_prompt": (
                    f"Cartoon-style learning slide about {lesson.get('topic', 'AI learning')}."
                ),
                "speaker_note": "Keep the pace approachable and confidence-building.",
                "checkpoint_question": "What is the main idea you should remember from this lesson?",
            }
        ]

    return {
        "overview": clean_text[:500] or "A guided lesson tailored to the learner's current understanding.",
        "learning_objectives": [
            "Understand the core concept in simple terms.",
            "Connect the concept to practical examples.",
            "Leave with a clear mental model for the next lesson.",
        ],
        "slides": slides,
        "end_quiz": [
            {
                "question": f"What is the main goal of {lesson.get('topic', 'this lesson')}?",
                "options": [
                    "Understand the core idea and practical use",
                    "Memorize unrelated definitions",
                    "Skip directly to advanced theory",
                    "Ignore real-world examples",
                ],
                "correct_index": 0,
                "explanation": "The lesson is designed to build a practical understanding before moving on.",
            },
            {
                "question": "Which learning habit helps most after this lesson?",
                "options": [
                    "Review the key takeaways and connect them to examples",
                    "Avoid checking your understanding",
                    "Only memorize the hardest words",
                    "Skip the next lesson entirely",
                ],
                "correct_index": 0,
                "explanation": "Reviewing and applying the idea helps it stick and prepares you for the next topic.",
            },
        ],
        "next_lesson": (
            {
                "source": str(next_lesson.get("source") or ""),
                "title": str(next_lesson.get("topic") or "Next lesson"),
                "reason": "This is the natural next step based on the concept you just learned.",
                "level": int(next_lesson.get("level") or lesson.get("level") or 1),
            }
            if next_lesson and next_lesson.get("source")
            else None
        ),
    }


async def generate_lesson_storyboard(
    profile: dict[str, Any],
    lesson: dict[str, Any],
    lesson_text: str,
    next_lesson: dict[str, Any] | None = None,
    max_slides: int = 6,
    max_retries: int = 3,
) -> dict[str, Any]:
    # Use compact metadata for the learner and lesson, then pass the lesson text
    # separately as the source material for the generated slide-book.
    compact_profile = {
        "role": profile.get("role"),
        "difficulty": profile.get("difficulty"),
        "target_level": profile.get("target_level"),
        "final_questionnaire_score": profile.get("final_questionnaire_score"),
        "weak_areas": profile.get("weak_areas", [])[:5],
        "previously_completed_lessons": profile.get("previously_completed_lessons", [])[:5],
    }
    compact_lesson = {
        "source": lesson.get("source"),
        "topic": lesson.get("topic"),
        "role": lesson.get("role"),
        "level": lesson.get("level"),
        "status": lesson.get("status"),
    }
    compact_next_lesson = None
    if next_lesson:
        compact_next_lesson = {
            "source": next_lesson.get("source"),
            "topic": next_lesson.get("topic"),
            "level": next_lesson.get("level"),
            "status": next_lesson.get("status"),
        }
    compact_text = (lesson_text or "")[:8000]

    prompt = f"""
You are an AI lesson designer for LearningOS.
Create a personalized lesson storyboard as a slide-book, not a video.

Learner profile:
{json.dumps(compact_profile, ensure_ascii=True)}

Lesson metadata:
{json.dumps(compact_lesson, ensure_ascii=True)}

Suggested next lesson candidate:
{json.dumps(compact_next_lesson, ensure_ascii=True)}

Lesson source material:
{json.dumps(compact_text, ensure_ascii=True)}

Rules:
- Return between 4 and {max_slides} slides.
- Tailor the explanation to the learner's current knowledge and weak areas.
- Keep the tone encouraging, visual, and easy to follow.
- Each slide should feel like one page in an illustrated slide book.
- Make each slide feel like a cartoon/comic teaching panel.
- Provide a scene caption and a short dialogue line for each slide.
- Provide an internal illustration prompt for each slide describing a cartoon or educational scene.
- Use practical examples when possible.
- Generate a short end-of-lesson quiz with 2 or 3 multiple-choice questions.
- If a next lesson candidate exists, explain why it should come next.
- Return STRICT JSON only.

Format:
{{
  "overview": "short lesson overview",
  "learning_objectives": ["objective 1", "objective 2", "objective 3"],
  "slides": [
    {{
      "title": "Slide title",
      "narrative": "2-4 sentence explanation",
      "bullets": ["point 1", "point 2", "point 3"],
      "scene_caption": "what the cartoon panel shows",
      "dialogue_line": "short speech bubble line",
      "illustration_prompt": "cartoon or illustrated scene",
      "speaker_note": "short teaching guidance",
      "checkpoint_question": "short reflection or recall question"
    }}
  ],
  "end_quiz": [
    {{
      "question": "question text",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "why this is correct"
    }}
  ],
  "next_lesson": {{
    "source": "candidate source or empty string",
    "title": "candidate title",
    "reason": "why it should come next",
    "level": 1
  }}
}}
"""

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()

            payload = json.loads(text)
            slides = payload.get("slides") or []
            if not isinstance(slides, list) or not slides:
                raise ValueError("slides must be a non-empty list")

            # Normalize the LLM output into the exact shape expected by the API
            # before it reaches the frontend.
            normalized_slides = []
            for slide in slides[:max_slides]:
                if not isinstance(slide, dict):
                    continue
                bullets = slide.get("bullets") or []
                if not isinstance(bullets, list):
                    bullets = []
                normalized_slides.append(
                    {
                        "title": str(slide.get("title") or "Lesson Slide"),
                        "narrative": str(slide.get("narrative") or "").strip(),
                        "bullets": [str(item).strip() for item in bullets if str(item).strip()][:4],
                        "scene_caption": str(slide.get("scene_caption") or "").strip() or None,
                        "dialogue_line": str(slide.get("dialogue_line") or "").strip() or None,
                        "illustration_prompt": str(
                            slide.get("illustration_prompt") or "Friendly educational illustration."
                        ).strip(),
                        "speaker_note": str(slide.get("speaker_note") or "").strip() or None,
                        "checkpoint_question": str(slide.get("checkpoint_question") or "").strip() or None,
                    }
                )

            if not normalized_slides:
                raise ValueError("No valid slides generated")

            objectives = payload.get("learning_objectives") or []
            if not isinstance(objectives, list):
                objectives = []

            quiz_items = payload.get("end_quiz") or []
            normalized_quiz = []
            if isinstance(quiz_items, list):
                for item in quiz_items[:3]:
                    if not isinstance(item, dict):
                        continue
                    options = item.get("options") or []
                    if not isinstance(options, list) or len(options) != 4:
                        continue
                    try:
                        correct_index = int(item.get("correct_index"))
                    except Exception:
                        continue
                    if correct_index < 0 or correct_index > 3:
                        continue
                    normalized_quiz.append(
                        {
                            "question": str(item.get("question") or "").strip(),
                            "options": [str(option).strip() for option in options][:4],
                            "correct_index": correct_index,
                            "explanation": str(item.get("explanation") or "").strip(),
                        }
                    )

            next_payload = payload.get("next_lesson") or {}
            normalized_next_lesson = None
            if isinstance(next_payload, dict) and next_lesson and next_lesson.get("source"):
                normalized_next_lesson = {
                    "source": str(next_lesson.get("source") or ""),
                    "title": str(next_lesson.get("topic") or next_payload.get("title") or "Next lesson"),
                    "reason": str(next_payload.get("reason") or "").strip()
                    or "This is the best next step based on the lesson you just completed.",
                    "level": int(next_lesson.get("level") or lesson.get("level") or 1),
                }

            return {
                "overview": str(payload.get("overview") or "").strip()
                or _fallback_storyboard(lesson, lesson_text, next_lesson)["overview"],
                "learning_objectives": [
                    str(item).strip() for item in objectives if str(item).strip()
                ][:4]
                or _fallback_storyboard(lesson, lesson_text, next_lesson)["learning_objectives"],
                "slides": normalized_slides,
                "end_quiz": normalized_quiz
                or _fallback_storyboard(lesson, lesson_text, next_lesson)["end_quiz"],
                "next_lesson": normalized_next_lesson
                if normalized_next_lesson
                else _fallback_storyboard(lesson, lesson_text, next_lesson)["next_lesson"],
            }
        except Exception as exc:
            last_error = exc
            print(f"Lesson_agent storyboard error: {exc}")
            time.sleep(0.5 * (attempt + 1))

    print(f"Lesson_agent storyboard fallback after {max_retries} failed attempts: {last_error}")
    return _fallback_storyboard(lesson, lesson_text, next_lesson)
