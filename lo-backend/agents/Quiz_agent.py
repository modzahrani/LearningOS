import asyncio
import os 
import json
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(MODEL_NAME)

FALLBACK_QUESTION = {
    "question": "What is a common use case for classification models?",
    "options": [
        "A) Predicting a continuous house price value.",
        "B) Deciding whether an email is spam.",
        "C) Compressing images without losing quality.",
        "D) Finding the shortest path in a graph.",
    ],
    "correct_index": 1,
    "explanation": "Classification predicts discrete labels, such as spam vs. not spam.",
}

async def generate_question(profile,difficulty,history, max_retries: int = 3):
    role = str(profile.get("role", "student"))
    recent_history = history[-4:] if isinstance(history, list) else []

    prompt = f"""
You generate one AI learning quiz question.

Context:
- Learner role: {role}
- Difficulty: {difficulty}/10
- Recent questions to avoid repeating: {recent_history}

Requirements:
- Topic must be about AI concepts, history, applications, ethics, or machine learning basics.
- Tailor wording to the learner role.
- Do not repeat the recent questions.
- Return exactly 4 options.
- Match the difficulty level.
- Return strict JSON only.

JSON format:
{{
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "correct_index": 0,
  "explanation": "..."
}}
""".strip()
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            if "```" in text:
                text = text.split("```")[1].replace("json","").strip()
            print("Quiz_agent: generated question via Gemini model.")
            return json.loads(text)
        except Exception as e:
            last_error = e
            print("Error generating question:", e)
            await asyncio.sleep(0.5 * (attempt + 1))

    # Fallback to a safe static question if model is unavailable.
    print(f"Quiz_agent: falling back to static question after {max_retries} failed attempts: {last_error}")
    return FALLBACK_QUESTION
