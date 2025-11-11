import sys
import os
import datetime
import time
import random
import json
from functools import wraps
from typing import List

# --- FastAPI & Pydantic Imports ---
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn

# --- Vertex AI Imports ---
from vertexai import init
from vertexai.preview import caching
from vertexai.preview.generative_models import (
    GenerativeModel,
    Part,
    Content,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)

# --- Redis Import ---
import redis

# --- Configuration (from original script) ---
PROJECT_ID = "gen-lang-client-0537570594"
LOCATION = "asia-south1"
MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = """You are an expert legal assistant in India, specializing in the new criminal laws: the Bharatiya Nyaya Sanhita (BNS), the Bharatiya Nagarik Suraksha Sanhita (BNSS), and the Bharatiya Sakshya Adhiniyam (BSA).

Your task is to analyze the user's situation and provide guidance.
- **Always preface your advice with a clear disclaimer that you are an AI and not a substitute for a qualified lawyer.**
- Provide a clear, structured analysis for the user's legal question.
- **Crucially, you MUST cite the specific, relevant sections (e.g., "Section 101 of the BNS") from the provided legal documents (BNS, BNSS, BSA) to support your analysis, whenever possible.**
- Explain what the cited sections mean and how they apply to the user's situation.
- Maintain a professional, objective, and analytical tone.
- Ask the user for more details if the current information is insufficient.
"""

document1 = Part.from_uri(
    mime_type="application/pdf",
    uri="gs://my-indianlegal-docs-bucket/The Bharatiya Nyaya Sanhita, 2023.pdf",
)
document2 = Part.from_uri(
    mime_type="application/pdf",
    uri="gs://my-indianlegal-docs-bucket/The Bharatiya Nagarik Suraksha Sanhita, 2023.pdf",
)
document3 = Part.from_uri(
    mime_type="application/pdf",
    uri="gs://my-indianlegal-docs-bucket/The Bharatiya Sakshya Adhiniyam, 2023.pdf",
)


MODEL_NAME = "gemini-2.5-flash"
generation_config = {"max_output_tokens": 8192, "temperature": 0.5, "top_p": 0.95}

safety_settings = [
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT, 
        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, 
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE 
    ),
    SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, 
        threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    ),
]

# --- Global Variables for FastAPI ---
app = FastAPI(title="Legal-Bot API")
model = None  # Will be loaded on startup
redis_client = None  # Will be loaded on startup

# --- Redis Configuration ---
# You MUST set these environment variables in Cloud Run
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
SESSION_EXPIRATION_SECONDS = 3600  # 1 hour

def exponential_backoff_with_jitter(
    max_retries=5, base_delay=1, max_backoff=32, jitter=True
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {retries + 1} failed: {e}")
                    time.sleep(
                        min(
                            delay + random.uniform(0, delay) if jitter else delay,
                            max_backoff,
                        )
                    )
                    retries += 1
                    delay = min(delay * 2, max_backoff)
            raise Exception(f"All {max_retries} retries failed.")

        return wrapper

    return decorator


# --- Caching Function (Unchanged) ---
@exponential_backoff_with_jitter()
def get_or_create_cache():
    try:
        print("Checking for existing cached content...")
        for cached_content in caching.CachedContent.list():
            if cached_content.display_name == "legal-chatbot-cache-v1":
                print(f"Found existing cache: {cached_content.name}")
                return cached_content.name
        print("No existing cache found.")
    except Exception as e:
        print(f"Error while listing cached contents: {e}")

    print("Creating a new cached content object for chatbot...")
    cached_content = caching.CachedContent.create(
        model_name=MODEL_NAME,  # Use the cost-effective model for caching
        system_instruction=SYSTEM_INSTRUCTION,
        contents=[document1, document2, document3],
        ttl=datetime.timedelta(minutes=60),
        display_name="legal-chatbot-cache-v1",
    )
    print(f"Cache created. Name: {cached_content.name}")
    return cached_content.name


# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes Vertex AI, gets/creates the cache, loads the
    GenerativeModel, and connects to Redis.
    """
    global model, redis_client
    try:
        print("Initializing Vertex AI...")
        init(project=PROJECT_ID, location=LOCATION)

        cached_content_name = get_or_create_cache()

        print(f"Loading model from cached content: {cached_content_name}...")
        model = GenerativeModel.from_cached_content(
            cached_content_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )
        print("✅ Model loaded successfully.")

        # --- Initialize Redis Connection ---
        print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connected successfully.")

    except Exception as e:
        print(f"❌ Failed to initialize the chatbot model or Redis: {e}")
        model = None
        redis_client = None

# --- API Request/Response Models ---
# We no longer need ApiPart or ApiContent for the request
class ChatRequest(BaseModel):
    """The JSON request body the frontend must send."""
    session_id: str  # The client MUST send a unique session ID
    message: str


# --- Helper functions for Redis History ---
def get_history_from_redis(session_id: str) -> List[Content]:
    """Retrieves and deserializes chat history from Redis."""
    try:
        history_json = redis_client.get(session_id)
        if not history_json:
            return []  # No history, new session

        # Deserialize JSON string to a list of dicts
        history_data = json.loads(history_json)
        
        # Convert list of dicts back into Vertex AI Content objects
        history = [
            Content(
                role=c.get("role"),
                parts=[Part.from_text(p.get("text")) for p in c.get("parts", [])],
            )
            for c in history_data
        ]
        return history
    except Exception as e:
        print(f"Error getting history from Redis: {e}")
        return [] # Start fresh if history is corrupt

def save_history_to_redis(session_id: str, history: List[Content]):
    """Serializes and saves chat history to Redis."""
    try:
        # Convert Content objects to a list of simple dicts for JSON
        history_data = [
            {"role": c.role, "parts": [{"text": p.text} for p in c.parts]}
            for c in history
        ]
        
        history_json = json.dumps(history_data)
        
        # Save to Redis with an expiration time
        redis_client.set(session_id, history_json, ex=SESSION_EXPIRATION_SECONDS)
    except Exception as e:
        print(f"Error saving history to Redis: {e}")


# --- API Endpoints ---
@app.get("/")
async def root():
    if not model or not redis_client:
        return {"status": "API is running, but Model or Redis is not initialized."}
    return {"status": "Legal-Bot API is running and connected."}


async def stream_generator(chat_session, history: List[Content], user_message: str, session_id: str):
    """
    A generator that yields text chunks and saves the full
    history to Redis when complete.
    """
    full_response = ""
    try:
        response_stream = chat_session.send_message(user_message, stream=True)
        for chunk in response_stream:
            if chunk.text:
                full_response += chunk.text
                yield chunk.text
    except Exception as e:
        print(f"Error during model streaming: {e}")
        yield f"Error: Could not process request. {e}"
    finally:
        # This block executes after the stream is finished or breaks
        # We now update the history with the new user message and full model response
        if full_response:
            history.append(Content(role="user", parts=[Part.from_text(user_message)]))
            history.append(Content(role="model", parts=[Part.from_text(full_response)]))
            save_history_to_redis(session_id, history)


@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    The main streaming endpoint. Takes a session_id and message,
    loads history from Redis, and returns a streaming response.
    """
    if not model or not redis_client:
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Model or Redis is not initialized. Please check server logs.",
        )

    try:
        # 1. Get existing history from Redis
        history = get_history_from_redis(request.session_id)

        # 2. Start a new chat session with the retrieved history
        chat = model.start_chat(history=history)

        # 3. Return a streaming response, passing all necessary info to the generator
        return StreamingResponse(
            stream_generator(chat, history, request.message, request.session_id),
            media_type="text/plain",
        )

    except Exception as e:
        print(f"Error in /chat-stream endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Main execution (for local testing) ---
if __name__ == "__main__":
    """
    This allows you to run the API locally for testing:
    `python main.py`
    You MUST have a Redis server running locally on localhost:6379 for this to work.
    """
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)