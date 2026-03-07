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
        # Optimized prompt to get raw data for the threshold logic
        prompt = f"""
        TASK: Research and verify the following claim:
        Claim: {request.context}
        Location: {request.country}
        Date: {request.date}

        OUTPUT FORMAT: Respond ONLY with a raw JSON object. Do not include markdown tags.
        
        SCHEMA:
        {{
          "raw_confidence": 0.0-1.0,
          "explanation_en": "Summary of findings in English",
          "explanation_native": "Summary of findings in {request.native_language}",
          "sources": ["url1", "url2"]
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
        
        # Cleanup: Remove any accidental markdown backticks
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
        if raw_text.startswith("json"):
            raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)

        # --- CUSTOM THRESHOLD LOGIC ---
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
        # ------------------------------

        # Background Logging (Optional - uncomment if needed)
        # with open("research_history.jsonl", "a", encoding="utf-8") as f:
        #     log_entry = {"time": datetime.now().isoformat(), "query": request.context, "res": data}
        #     f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return data

    except Exception as e:
        print("\n--- HAND ERROR LOG ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Works for Render (PORT env) and Local (8080)
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)