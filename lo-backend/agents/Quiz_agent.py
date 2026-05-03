import os 
import json
import time
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
    prompt = f""" 
    You are an AI quiz generator for a learning platform. the platforn is about Learning whats an AI,
    so your questions should be focused on AI concepts, history, applications, and ethics.
    we have 3 roles (student, individual, enterprise) and the user can select one of them,
    so you should tailor the questions to the selected role.
    
    The user has the following profile: {profile}.
    diffuculty level: {difficulty}/10.
    
    Previous questions:
    {history}
    Generate a NEW multiple-choice questions.
    
    Rules:
    - do not repeat topics.
    - 4 options per question.
    - make it appropriate to the difficulty level.
    - return STRICT json only format (no text outside the json).
    
    Format:
    {{
        "question":"...",
        "options":["A","B","C","D"],
        "correct_index": 0,
        "explanation":"..."
    }} """
    
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
            time.sleep(0.5 * (attempt + 1))

    # Fallback to a safe static question if model is unavailable.
    print(f"Quiz_agent: falling back to static question after {max_retries} failed attempts: {last_error}")
    return FALLBACK_QUESTION
