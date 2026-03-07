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
    try:
        # Prompt optimized for research and raw data
        prompt = f"""
        TASK: Research and verify the following claim:
        Claim: "{request.context}"
        Location: {request.country}
        Date: {request.date}

        INSTRUCTIONS:
        1. PREMISE CHECK: Ensure the entity (e.g. mall/place) actually exists in {request.country}.
        2. SEARCH: Use Google Search to find real-time news for {request.date}.
        3. OUTPUT: Respond ONLY with a raw JSON object. Do not use markdown tags.

        SCHEMA:
        {{
          "raw_confidence": 0.0-1.0,
          "explanation_en": "Start with location confirmation, then the fact check.",
          "explanation_native": "Same as above in {request.native_language}",
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

        # --- 1. EXTRACT ACTUAL URLS FROM METADATA ---
        actual_urls = []
        try:
            metadata = response.candidates[0].grounding_metadata
            if metadata and metadata.grounding_chunks:
                # This grabs the actual web links, not the citation numbers
                actual_urls = list(set([chunk.web.uri for chunk in metadata.grounding_chunks if chunk.web]))
        except Exception:
            actual_urls = []

        # --- 2. CLEAN & PARSE JSON TEXT ---
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

        # --- 3. APPLY CONFIDENCE THRESHOLD LOGIC ---
        # 0.7 - 1.0 -> TRUE
        # 0.4 - 0.6 -> UNCERTAIN
        # < 0.4     -> FALSE
        conf = data.get("raw_confidence", 0.0)
        
        if conf >= 0.7:
            data["classification"] = "TRUE"
            # data["is_true"] = True
        elif 0.4 <= conf <= 0.6:
            data["classification"] = "UNCERTAIN"
            # data["is_true"] = False
        else:
            data["classification"] = "FALSE"
            # data["is_true"] = False

        # --- 4. OVERWRITE SOURCES WITH REAL LINKS ---
        data["sources"] = actual_urls

        return data

    except Exception as e:
        print("\n--- HAND ERROR LOG ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)