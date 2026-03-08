import os
import json
import uvicorn
import traceback
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()
app = FastAPI()

# Initialize Client
client = genai.Client(
    vertexai=True, 
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

class BrainRequest(BaseModel):
    country: str
    date: str
    context: str
    native_language: str
    english_language: str

@app.post("/verify")
async def verify_claim(request: BrainRequest):
    # --- 1. LANGUAGE CHECK ---
    # Determine if we actually need a translation
    native_input = request.native_language.strip().lower()
    is_english_only = not native_input or native_input == "english"
    
    # We still give the AI a target for the prompt to avoid errors, 
    # but we will empty the result later.
    target_lang = "English" if is_english_only else request.native_language

    try:
        prompt = f"""
        TASK: Research and verify: "{request.context}"
        Location: {request.country}
        Date: {request.date}

        STRICT TRUTH-SCORING RULES:
        1. If confirmed by news/official reports: set truth_score to 0.9.
        2. If NOT FOUND or debunked (no news for a major event): set truth_score to 0.1.
        3. If location is incorrect: set truth_score to 0.1.
        4. If conflicting/vague: set truth_score to 0.5.

        OUTPUT FORMAT: Respond ONLY with raw JSON.
        SCHEMA:
        {{
          "truth_score": <float>,
          "explanation_en": "Summarize findings clearly in English.",
          "explanation_native": "Translate the summary into {target_lang}.",
          "sources": []
        }}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.0
            )
        )

        # --- 2. EXTRACT URLS ---
        actual_urls = []
        try:
            metadata = response.candidates[0].grounding_metadata
            if metadata and metadata.grounding_chunks:
                actual_urls = list(set([chunk.web.uri for chunk in metadata.grounding_chunks if chunk.web]))
        except:
            actual_urls = []

        # --- 3. CLEAN & PARSE JSON ---
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

        # --- 4. APPLY EMPTY STRING LOGIC ---
        if is_english_only:
            data["explanation_native"] = ""

        # --- 5. VERDICT LOGIC ---
        score = data.get("truth_score", 0.0)
        if score >= 0.7:
            data["classification"] = "TRUE"
        elif 0.4 <= score <= 0.6:
            data["classification"] = "UNVERIFIED"
        else:
            data["classification"] = "FALSE"

        data["sources"] = actual_urls
        return data

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)