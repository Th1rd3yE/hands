import os
import json
import uvicorn
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. SETUP & CONFIGURATION
load_dotenv()
app = FastAPI(title="Fact-Check Researcher Agent v2.0")

# Initialize Gemini Client (using Vertex AI mode)
client = genai.Client(
    vertexai=True, 
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION")
)

# 2. DATA MODELS
class BrainRequest(BaseModel):
    country: str
    date: str
    context: str
    native_language: str
    english_language: str

class ResearchResult(BaseModel):
    claim: str
    classification: str
    confidence: float
    explanation_en: str
    explanation_native: str
    sources: list[str]

# 3. THE RESEARCH & LOGGING LOGIC
@app.post("/verify", response_model=ResearchResult)
async def verify_claim(request: BrainRequest):
    try:
        # --- PASS 1: DEEP RESEARCH (Google Search) ---
        search_prompt = f"""
        Research the following claim for {request.country} as of {request.date}:
        CLAIM: {request.context}
        
        Instructions:
        1. Use Google Search to find current facts from reliable news or gov sites.
        2. Provide a detailed summary in {request.english_language}.
        3. Provide a professional translation in {request.native_language}.
        4. Create a list of the ACTUAL source URLs you used. Do NOT invent URLs.
        """

        research_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=search_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=1.0
            )
        )
        
        raw_notes = research_response.text

        # --- PASS 2: JSON STRUCTURING ---
        format_prompt = f"""
        Convert these research notes into a valid JSON object.
        
        NOTES:
        {raw_notes}

        OUTPUT ONLY THIS JSON STRUCTURE:
        {{
            "classification": "TRUE/FALSE/UNCERTAIN",
            "confidence": 0.0 to 1.0,
            "explanation_en": "string",
            "explanation_native": "string",
            "sources": ["actual_url1", "actual_url2"]
        }}
        """

        json_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=format_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        # Parse the JSON
        data = json.loads(json_response.text)

        # # --- NEW: AUTO-SAVE TO HISTORY FILE ---
        # # We save this as a JSONL (JSON Lines) file for easy database import later.
        # with open("research_history.jsonl", "a", encoding="utf-8") as f:
        #     log_entry = {
        #         "timestamp": datetime.now().isoformat(),
        #         "request": request.model_dump(),
        #         "result": data
        #     }
        #     f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return ResearchResult(
            claim=request.context,
            classification=data.get("classification", "UNCERTAIN"),
            confidence=data.get("confidence", 0.0),
            explanation_en=data.get("explanation_en", ""),
            explanation_native=data.get("explanation_native", ""),
            sources=data.get("sources", [])
        )

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail="The Hand failed to process research.")

@app.get("/")
def health_check():
    return {"status": "Hand is active", "project": os.getenv("GOOGLE_CLOUD_PROJECT")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)