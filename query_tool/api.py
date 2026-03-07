import json
import os
import re
import uvicorn
from typing import List, Optional, Set
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from query_tool.fetcher import fetch_data

load_dotenv()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SQLRequest(BaseModel):
    question: str
    keywords: List[str]
    country: Optional[str] = None
    languages: Optional[List[str]] = None
    date: Optional[str] = None


class ResearchResult(BaseModel):
    claim: str
    classification: str
    confidence: float
    explanation_en: str
    explanation_native: str
    sources: List[str]


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=2048,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# -- Chain 0: Keyword expansion ----------------------------------------------

KEYWORD_EXPANSION_SYSTEM = """
You are a search query expert. Given a list of keywords, expand each one into a list of
normalized search terms that cover:
- Root/lemma forms (e.g. "allocations" -> "allocation")
- Singular and plural variants (e.g. "rumor", "rumors", "rumour", "rumours")
- Common abbreviations and acronyms (e.g. "HDB" -> also "Housing Development Board")
- Common slangs or informal forms (e.g. "BTO" -> "Build-To-Order")
- Common misspellings (e.g. "clarifies" -> also "clarify", "clarification")
- Synonyms that would appear in news articles

Return ONLY a JSON array of unique, flat strings — all variants combined into one list.
No markdown, no backticks, no explanations, no nested arrays.

Example input: ["HDB", "BTO allocations"]
Example output: ["HDB", "Housing Development Board", "BTO", "Build-To-Order", "BTO allocation", "BTO allocations", "ballot", "balloting", "flat allocation", "flat ballot"]
"""

keyword_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", KEYWORD_EXPANSION_SYSTEM),
    ("human", "Keywords: {keywords}\n\nReturn only the JSON array."),
])

keyword_expansion_chain = keyword_expansion_prompt | llm | StrOutputParser()

# -- Chain 1: SQL generation --------------------------------------------------

SQL_SYSTEM = """
You are a SQL expert. Generate a single, valid PostgreSQL SELECT query based on the user's question and expanded keywords.

Rules:
- Return ONLY the raw SQL query — no markdown, no backticks, no explanations.
- Never include a trailing semicolon.
- Use only these tables and columns:

  countries     : id (int), name (varchar)
  media_outlets : id (int), name (varchar), country_id (int FK -> countries.id)
  articles      : id (int), media_outlet_id (int FK -> media_outlets.id),
                  title (text), content (text), original_url (text),
                  language (varchar), published_at (timestamp), collected_at (timestamp)

- Always JOIN media_outlets and countries so outlet name and country name are available.
- Always SELECT: a.id, a.title, a.content, a.original_url, a.language,
                 a.published_at, mo.name AS outlet_name, c.name AS country_name
- Filter by country name (case-insensitive ILIKE) when country is provided.
- Filter by language (case-insensitive) when languages is provided (use ILIKE ANY or IN).
- Filter published_at <= date when date is provided.
- For keywords: use OR across ALL expanded keyword variants on both title AND content.
  Each keyword variant: (a.title ILIKE '%variant%' OR a.content ILIKE '%variant%')
  Connect all variants with OR so any single match returns the article.
- Limit results to 20 rows.
"""

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    (
        "human",
        "Question: {question}\n"
        "Expanded keywords: {keywords}\n"
        "Country: {country}\n"
        "Languages: {languages}\n"
        "Date: {date}\n\n"
        "Return only the SQL query.",
    ),
])

sql_chain = sql_prompt | llm | StrOutputParser()

# -- Chain 2: Fact-check analysis ---------------------------------------------

ANALYSIS_SYSTEM = """
You are a fact-checking analyst. You will be given a claim/question and a list of articles retrieved from a database.
Analyze the articles and return a JSON object with exactly these fields:

{{
  "claim": "<restate the original question as a claim>",
  "classification": "<one of: TRUE, FALSE, MISLEADING, UNVERIFIED>",
  "confidence": <float between 0.0 and 1.0>,
  "explanation_en": "<clear English explanation based on the articles>",
  "explanation_native": "<same explanation translated to the primary language of the articles>",
  "sources": ["<original_url_1>", "<original_url_2>", ...]
}}

Rules:
- Base your analysis ONLY on the provided articles.
- If articles are insufficient to verify, set classification to UNVERIFIED and confidence below 0.5.
- Return ONLY the JSON object — no markdown, no backticks, no extra text.
- Include all relevant source URLs from the articles.
"""

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_SYSTEM),
    (
        "human",
        "Question: {question}\n\n"
        "Articles:\n{articles}\n\n"
        "Return only the JSON object.",
    ),
])

analysis_chain = analysis_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_sql(raw: str) -> str:
    """Strip markdown fences and stray semicolons the LLM may add."""
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("`", "").strip()
    raw = raw.split(";")[0].strip()
    return raw


def _clean_json(raw: str) -> str:
    """Strip markdown fences around JSON if present."""
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("`", "").strip()
    return raw


def _expand_keywords(keywords: List[str]) -> List[str]:
    """
    Use LLM to expand keywords into root forms, synonyms, abbreviations,
    slangs, and common misspellings. Falls back to original keywords on error.
    """
    try:
        raw = keyword_expansion_chain.invoke({"keywords": keywords})
        cleaned = _clean_json(raw)
        expanded: List[str] = json.loads(cleaned)
        # Deduplicate while preserving order, always include originals
        seen: Set[str] = set()
        result: List[str] = []
        for kw in (keywords + expanded):
            lower = kw.lower().strip()
            if lower and lower not in seen:
                seen.add(lower)
                result.append(kw.strip())
        print(f"[api] expanded keywords: {result}")
        return result
    except Exception as e:
        print(f"[api] keyword expansion failed, using originals: {e}")
        return keywords


def _format_articles_for_prompt(rows: List[dict]) -> str:
    """Serialize DB rows into a compact string for the analysis prompt."""
    articles = []
    for i, row in enumerate(rows, 1):
        articles.append(
            f"[{i}] Title: {row.get('title', '')}\n"
            f"    Outlet: {row.get('outlet_name', '')} ({row.get('country_name', '')})\n"
            f"    Language: {row.get('language', '')}\n"
            f"    Published: {row.get('published_at', '')}\n"
            f"    URL: {row.get('original_url', '')}\n"
            f"    Content: {str(row.get('content', ''))[:800]}"
        )
    return "\n\n".join(articles)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="SQL Query Generator API")


@app.post("/generate-sql")
async def generate_sql(request: SQLRequest):
    """Return expanded keywords and generated SQL — useful for debugging."""
    try:
        expanded = _expand_keywords(request.keywords)
        payload = request.dict()
        payload["keywords"] = expanded
        raw = sql_chain.invoke(payload)
        sql = _clean_sql(raw)
        return {"expanded_keywords": expanded, "sql": sql}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate-and-fetch", response_model=ResearchResult)
async def generate_and_fetch(request: SQLRequest):
    """
    1. Expand keywords (root forms, synonyms, abbreviations, slangs, misspellings).
    2. Generate SQL using expanded keywords.
    3. Fetch matching articles from the database.
    4. Pass articles to a second LLM call for fact-check analysis.
    5. Return a single structured ResearchResult.
    """
    try:
        # Step 1 — expand keywords
        expanded_keywords = _expand_keywords(request.keywords)

        # Step 2 — generate and clean SQL
        payload = request.dict()
        payload["keywords"] = expanded_keywords
        raw_sql = sql_chain.invoke(payload)
        sql = _clean_sql(raw_sql)
        print("[api] generated SQL:", sql)

        # Step 3 — fetch articles from DB
        rows = fetch_data(sql)
        print(f"[api] fetched {len(rows)} row(s)")

        if not rows:
            return ResearchResult(
                claim=request.question,
                classification="UNVERIFIED",
                confidence=0.0,
                explanation_en="No relevant articles were found in the database for this query.",
                explanation_native="No relevant articles were found in the database for this query.",
                sources=[],
            )

        # Step 4 — analyse articles with second LLM call
        articles_text = _format_articles_for_prompt(rows)
        raw_analysis = analysis_chain.invoke({
            "question": request.question,
            "articles": articles_text,
        })

        # Step 5 — parse and return
        cleaned = _clean_json(raw_analysis)
        data = json.loads(cleaned)

        return ResearchResult(
            claim=data.get("claim", request.question),
            classification=data.get("classification", "UNVERIFIED"),
            confidence=float(data.get("confidence", 0.0)),
            explanation_en=data.get("explanation_en", ""),
            explanation_native=data.get("explanation_native", ""),
            sources=data.get("sources", []),
        )

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM analysis response: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
