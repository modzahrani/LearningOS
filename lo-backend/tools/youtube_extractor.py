# youtube_extractor.py

import os
import re
import sys
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "docs"))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "videos.csv")



# --- 1. Extract video ID from URL ---
def extract_video_id(url: str) -> str:
    """Supports standard, short, and embed YouTube URLs."""
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",        # https://www.youtube.com/watch?v=ID
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})", # https://youtu.be/ID
        r"(?:embed/)([a-zA-Z0-9_-]{11})",     # https://www.youtube.com/embed/ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"[ERROR] Could not extract video ID from URL: {url}")


# --- 2. Fetch transcript ---
def fetch_transcript(video_id: str, languages=["en"]) -> str:
    """Fetch and return transcript as plain text."""
    try:
        ytt = YouTubeTranscriptApi()
        transcript = ytt.fetch(video_id, languages=languages)
        full_text = " ".join(chunk.text for chunk in transcript)
        return full_text.strip()
    except TranscriptsDisabled:
        raise RuntimeError(f"[ERROR] Transcripts are disabled for video: {video_id}")
    except NoTranscriptFound:
        raise RuntimeError(f"[ERROR] No transcript found in {languages} for video: {video_id}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unexpected error fetching transcript: {e}")


# --- 3. Save to docs/ folder ---
def save_transcript(text: str, filename: str) -> str:
    os.makedirs(DOCS_FOLDER, exist_ok=True)

    if not filename.endswith(".txt"):
        filename += ".txt"

    # Validate naming convention
    parts = filename.replace(".txt", "").split("_")
    valid_roles = ["student", "individual", "enterprise"]

    role = parts[0].lower()
    if role not in valid_roles:
        print(f"[WARN] Filename should start with a role ({valid_roles}).")
        print(f"[WARN] Got '{role}' — defaulting to 'student' folder.")
        role = "student"

    if not any(f"level{i}" in filename.lower() for i in range(1, 4)):
        print(f"[WARN] No level detected in filename (level1/level2/level3).")
        print(f"[WARN] Will default to level1/easy when loaded.")

    # Save into the role subfolder
    role_folder = os.path.join(DOCS_FOLDER, role)
    os.makedirs(role_folder, exist_ok=True)

    filepath = os.path.join(role_folder, filename)

    if os.path.exists(filepath):
        print(f"[WARN] File already exists: {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[INFO] Transcript saved to: {filepath}")
    return filepath


# --- 4. Main runner ---
def extract_youtube_to_docs(url: str, filename: str):
    """Full pipeline: URL → transcript → .txt in docs/"""
    print(f"[INFO] Extracting video ID from URL...")
    video_id = extract_video_id(url)
    print(f"[INFO] Video ID: {video_id}")

    print(f"[INFO] Fetching transcript...")
    text = fetch_transcript(video_id)
    word_count = len(text.split())
    print(f"[INFO] Transcript fetched — {word_count} words")

    filepath = save_transcript(text, filename)
    return filepath


# --- 5. Batch from CSV ---
def batch_from_csv(csv_path: str, max_workers: int = 4):
    """
    Read a CSV of (url, filename) pairs and extract all transcripts in parallel.
    CSV format:
        url,filename
        https://youtube.com/...,enterprise_deepLearning_level2.txt
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("url") and row.get("filename")]

    if not rows:
        print("[ERROR] No valid rows found in CSV.")
        return

    print(f"[INFO] Found {len(rows)} videos to process\n")

    results = {"success": [], "failed": []}

    def process(row):
        url = row["url"].strip()
        filename = row["filename"].strip()
        try:
            extract_youtube_to_docs(url, filename)
            return ("success", filename)
        except (ValueError, RuntimeError) as e:
            print(e)
            return ("failed", filename)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, row): row for row in rows}
        for future in as_completed(futures):
            status, filename = future.result()
            results[status].append(filename)

    print(f"\n{'='*40}")
    print(f"[DONE] {len(results['success'])} succeeded, {len(results['failed'])} failed")
    if results["failed"]:
        print(f"[FAILED] {', '.join(results['failed'])}")
    print(f"{'='*40}")

# --- 6. CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Transcript Extractor")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--url", type=str, help="Single YouTube URL")
    parser.add_argument("--filename", type=str, help="Output filename")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to CSV file")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.url or not args.filename:
            print("[ERROR] --url and --filename are required for single mode.")
            sys.exit(1)
        try:
            extract_youtube_to_docs(args.url, args.filename)
            print("\n[DONE] File is ready in docs/")
        except (ValueError, RuntimeError) as e:
            print(e)
            sys.exit(1)

    else:  # batch mode (default)
        batch_from_csv(args.csv, max_workers=args.workers)
