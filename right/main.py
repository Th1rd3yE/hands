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
        # UPDATED PROMPT: Forces AI to separate "Research Certainty" from "Truth Value"
        prompt = f"""
        TASK: Research and verify the claim: "{request.context}"
        Location: {request.country}
        Date: {request.date}

        STRICT TRUTH-SCORING RULES:
        1. If the event is CONFIRMED by news: set truth_score to 0.9 - 1.0.
        2. If the event is NOT FOUND or debunked (No news exists for a major event): set truth_score to 0.1 - 0.2.
        3. If there are conflicting reports or vague rumors: set truth_score to 0.5.
        4. If the location is incorrect (e.g. Bedok in Japan): set truth_score to 0.1.

        OUTPUT FORMAT: Respond ONLY with a raw JSON object.
        SCHEMA:
        {{
          "truth_score": <float based on rules>,
          "explanation_en": "Start with 'Fact:' or 'False:'",
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
        
        # 1. Extract Real URLs from Metadata
        actual_urls = []
        try:
            metadata = response.candidates[0].grounding_metadata
            if metadata and metadata.grounding_chunks:
                actual_urls = list(set([chunk.web.uri for chunk in metadata.grounding_chunks if chunk.web]))
        except:
            actual_urls = []

        # 2. Cleanup & Parse JSON
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

        # # Background Logging
        # with open("research_history.jsonl", "a", encoding="utf-8") as f:
        #     log_entry = {"time": datetime.now().isoformat(), "query": request.context, "res": data}
        #     f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return data

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)