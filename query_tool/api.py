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

from query_tool.fetcher import fetch_data, SQLExecutionError

load_dotenv()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SQLRequest(BaseModel):
    context: str                              # the claim or question to fact-check
    country: Optional[str] = None
    date: Optional[str] = None
    native_language: Optional[str] = None    # e.g. "Burmese", "Mandarin" — for native explanation
    english_language: Optional[str] = "English"  # always English by default


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
    max_tokens=4096,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Separate LLM for keyword expansion — higher token limit to fit all multilingual terms
llm_expand = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=8192,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# -- Chain 0: Keyword expansion ----------------------------------------------

KEYWORD_EXPANSION_SYSTEM = """
You are a multilingual search query expert. Given a claim or question and a native language,
first extract the key topics/entities from the claim, then expand each into ALL of the following:

MANDATORY: Always translate into ALL of these languages regardless of the native language provided:
English, Mandarin (Simplified + Traditional), Cantonese, Malay, Bahasa Indonesia, Hindi (Devanagari), Bengali (Bengali script).
PLUS translate into the native_language if it is different from the above.
Keep the total list under 60 terms — prioritize the most searchable/common variants per language.


- Root/lemma forms (e.g. "allocations" -> "allocation")
- Singular and plural variants (e.g. "rumor", "rumors", "rumour", "rumours")
- Common abbreviations and acronyms — expand both ways:
  * Short form -> full form (e.g. "HDB" -> "Housing Development Board")
  * Full form -> short form (e.g. "Build-To-Order" -> "BTO")
- IATA / ICAO airport codes -> city and country names:
  * e.g. "MDY" -> "Mandalay", "Mandalay International Airport", "Myanmar"
  * e.g. "SIN" -> "Singapore", "Changi Airport"
- Country codes, currency codes, stock tickers -> their full names:
  * e.g. "SGD" -> "Singapore dollar", "SG" -> "Singapore"
- Place abbreviations -> full place names:
  * e.g. "NYC" -> "New York City", "New York"
  * e.g. "MDY" -> "Mandalay" (city in Myanmar)
- Slangs and informal forms — expand BOTH ways:
  * Slang -> formal equivalent (e.g. "shiok" -> "excellent", "delicious", "enjoyable")
  * Formal -> slang equivalent (e.g. "expensive" -> "ex", "pricey", "costly")
  * Regional/local slangs -> standard English (e.g. "lah", "lor", "leh", "meh", "sia", "walao", "kiasu", "kopitiam", "makan")
  * Internet/youth slangs -> formal forms (e.g. "gonna" -> "going to", "wanna" -> "want to", "tbh" -> "to be honest")
  * News/political slangs -> formal terms (e.g. "pork barrel" -> "political funding", "lame duck" -> "outgoing", "red tape" -> "bureaucracy")
- Common misspellings and alternate spellings (e.g. "colour" / "color", "centre" / "center")
- Synonyms that would appear in news articles

MULTILINGUAL TRANSLATION (most important):
- Translate EVERY keyword and ALL its variants into EACH of the target languages provided.
- Include the translated terms alongside the English terms in the same flat array.
- Always expand into ALL of these languages by default regardless of what is requested:
  * English
  * Mandarin (Simplified 简体 AND Traditional 繁體 where they differ)
  * Cantonese (written Cantonese, e.g. 係、唔係、點解)
  * Malay / Bahasa Melayu
  * Bahasa Indonesia
  * Hindi (Devanagari script, e.g. अफवाह, आवंटन)
  * Bengali / Bangla (Bengali script, e.g. গুজব, বরাদ্দ)
- For each language include: the direct translation, formal news terms, and common informal/slang variants used in that language's media.
- e.g. "HDB" -> also "建屋发展局" (Mandarin), "建屋局" (Mandarin short), "组屋" (Mandarin), "HDB" (Malay), "rumah HDB" (Malay informal)
- e.g. "BTO allocation" -> "预购组屋" (Mandarin), "组屋分配" (Mandarin), "peruntukan BTO" (Malay), "आवंटन" (Hindi)
- e.g. "rumor" -> "谣言" (Mandarin), "传言" (Mandarin), "khabar angin" (Malay), "अफवाह" (Hindi), "গুজব" (Bengali)
- e.g. "clarify" -> "澄清" (Mandarin), "menjelaskan" (Malay), "स्पष्ट करना" (Hindi), "স্পষ্ট করা" (Bengali)

Return ONLY a JSON array of unique, flat strings — English and translated terms all in one list.
No markdown, no backticks, no explanations, no nested arrays.

Example input: keywords=["HDB", "BTO allocations"], languages=["English", "Chinese"]
Example output: [
  "HDB", "Housing Development Board", "BTO", "Build-To-Order", "BTO allocation", "BTO allocations", "ballot", "balloting",
  "建屋发展局", "建屋局", "预购组屋", "组屋分配", "建屋局抽签", "组屋抽签", "谣言", "澄清", "传言",
  "HDB rumah", "peruntukan BTO", "undi BTO", "khabar angin", "menjelaskan",
  "आवंटन", "अफवाह", "स्पष्ट करना",
  "বরাদ্দ", "গুজব", "স্পষ্ট করা"
]
"""

keyword_expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", KEYWORD_EXPANSION_SYSTEM),
    ("human", "Extract and expand keywords from this claim/question: {context}\nNative language: {native_language}\n\nReturn only the JSON array."),
])

keyword_expansion_chain = keyword_expansion_prompt | llm_expand | StrOutputParser()

# -- Chain 1: SQL generation --------------------------------------------------

SQL_SYSTEM = """
You are a PostgreSQL expert. Generate a single valid SELECT query.

Schema:
  countries     : id (int), name (varchar)
  media_outlets : id (int), name (varchar), country_id (int)
  articles      : id (int), media_outlet_id (int), title (text), content (text),
                  original_url (text), language (varchar), published_at (timestamp)

STRICT RULES — follow exactly:
1. Return ONLY raw SQL. No markdown, no backticks, no comments, no explanations.
2. No trailing semicolon.
3. Always use these exact JOINs:
   FROM articles a
   JOIN media_outlets mo ON mo.id = a.media_outlet_id
   JOIN countries c ON c.id = mo.country_id
4. Always SELECT exactly:
   a.id, a.title, a.content, a.original_url, a.language, a.published_at,
   mo.name AS outlet_name, c.name AS country_name
5. Country filter (when provided):
   AND c.name ILIKE '%{country}%'
6. DO NOT filter by language — articles in ANY language must be returned if keywords match.
7. Date filter (when provided):
   AND a.published_at <= '{date}'
8. Use the placeholder __KEYWORD_FILTER__ exactly where the keyword filter should go — do NOT write any ILIKE conditions yourself:
   AND __KEYWORD_FILTER__
9. NEVER filter or order by id — all matching rows must be returned regardless of their id value.
10. End with: ORDER BY a.published_at DESC LIMIT 50
11. No dangling parentheses.

Example output:
SELECT a.id, a.title, a.content, a.original_url, a.language, a.published_at, mo.name AS outlet_name, c.name AS country_name
FROM articles a
JOIN media_outlets mo ON mo.id = a.media_outlet_id
JOIN countries c ON c.id = mo.country_id
WHERE c.name ILIKE '%Singapore%'
AND a.published_at <= '2026-03-10'
AND __KEYWORD_FILTER__
ORDER BY a.published_at DESC LIMIT 50
"""

sql_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_SYSTEM),
    (
        "human",
        "Context/Question: {context}\n"
        "Expanded keywords: {keywords}\n"
        "Country: {country}\n"
        "Date: {date}\n\n"
        "Return only the SQL query.",
    ),
])

sql_chain = sql_prompt | llm | StrOutputParser()

# -- Chain 1b: SQL fix -------------------------------------------------------

SQL_FIX_SYSTEM = """
You are a PostgreSQL expert. Fix the broken SQL query below.

STRICT RULES:
1. Return ONLY the fixed raw SQL — no markdown, no backticks, no explanations.
2. No trailing semicolon.
3. ONLY use these three tables: articles, media_outlets, countries. NEVER reference any other table.
4. Always use these aliases: articles AS a, media_outlets AS mo, countries AS c.
5. Always include these JOINs:
   FROM articles a
   JOIN media_outlets mo ON mo.id = a.media_outlet_id
   JOIN countries c ON c.id = mo.country_id
6. NEVER use ANY() for ILIKE — always expand to OR conditions instead:
   WRONG:  a.language ILIKE ANY(ARRAY['English', 'Chinese'])
   CORRECT: (a.language ILIKE '%English%' OR a.language ILIKE '%Chinese%')
7. Make sure all parentheses are balanced.
8. Make sure all string literals are properly quoted with single quotes.
"""

sql_fix_prompt = ChatPromptTemplate.from_messages([
    ("system", SQL_FIX_SYSTEM),
    (
        "human",
        "Broken SQL:\n{sql}\n\n"
        "Error message:\n{error}\n\n"
        "Return only the fixed SQL query.",
    ),
])

sql_fix_chain = sql_fix_prompt | llm | StrOutputParser()

# -- Chain 2: Fact-check analysis ---------------------------------------------

ANALYSIS_SYSTEM = """
You are an expert multilingual fact-checking analyst. Your job is to assess claims fairly,
surface useful context, work across languages, and reduce real-world harm from misinformation.

You will be given a claim and a list of articles. Return a JSON object with EXACTLY these fields:

{{
  "claim": "<restate the original question as a clear, neutral claim>",
  "classification": "<one of: TRUE, FALSE, MISLEADING, UNVERIFIED>",
  "confidence": <float 0.0–1.0>,
  "explanation_en": "<rich English explanation — see requirements below>",
  "explanation_native": "<same explanation in the native language requested — see requirements below>",
  "sources": ["<original_url_1>", "<original_url_2>", ...]
}}

━━━ CLASSIFICATION RULES ━━━
- TRUE         : Multiple credible sources consistently confirm the claim.
- FALSE        : Multiple credible sources consistently contradict the claim.
- MISLEADING   : Claim contains partial truth but omits key context, or sources contradict each other.
- UNVERIFIED   : Insufficient or no articles to draw a conclusion.
Always make uncertainty VISIBLE — never present a weak conclusion as strong.

━━━ CONFIDENCE SCORING ━━━
Base score by article count:
  0 → 0.0 | 1 → 0.40 | 2 → 0.55 | 3 → 0.65 | 4-5 → 0.75 | 6-9 → 0.85 | 10+ → 0.90
Adjustments:
  + 0.02 per article with a valid original_url (max +0.10)
  + 0.05 if articles come from 2 or more different outlets
  - 0.10 if articles contradict each other (→ set MISLEADING)
  - 0.15 if articles are only tangentially related to the claim
Hard limits: max 0.98 | UNVERIFIED never exceeds 0.49

━━━ EXPLANATION REQUIREMENTS ━━━
Both explanation_en and explanation_native MUST include ALL of the following:

1. CREDIBILITY ASSESSMENT
   - State clearly why the claim is classified as TRUE / FALSE / MISLEADING / UNVERIFIED
   - Name the outlet(s) and how many sources support the conclusion
   - Explicitly state if evidence is weak, strong, or conflicting

2. USEFUL CONTEXT
   - Highlight any credibility signals (e.g. official source, government statement, eyewitness)
   - Point out any MISSING context that would change the interpretation
   - Note if the claim is only partially true or cherry-picks facts

3. LANGUAGE & CULTURAL SENSITIVITY
   - Use plain language suitable for general public literacy
   - If the claim uses slang, abbreviations, or cultural references — explain them
   - For explanation_native: write naturally in that language, not a word-for-word translation
   - If the claim mixes languages, acknowledge both versions

4. HARM REDUCTION
   - If the claim could cause panic, add a calm, factual reassurance
   - If the claim relates to a policy change, briefly clarify the actual policy
   - Avoid over-censoring: if the claim is true, say so clearly and without hedging

━━━ OTHER RULES ━━━
- Base analysis ONLY on the provided articles — do not use outside knowledge.
- Include ALL source URLs from articles in the sources array.
- Return ONLY the JSON object — no markdown, no backticks, no extra text.
"""

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", ANALYSIS_SYSTEM),
    (
        "human",
        "Question: {context}\n"
        "Native language for explanation_native: {native_language}\n\n"
        "Articles:\n{articles}\n\n"
        "Return only the JSON object.",
    ),
])

analysis_chain = analysis_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_sql(raw: str) -> str:
    """Extract SQL from markdown fences if present, then sanitize."""
    raw = raw.strip()

    # If fenced, extract content INSIDE the fence rather than deleting everything
    fenced = re.search(r"```(?:sql)?\s*\n?(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()

    # Remove any remaining backticks
    raw = raw.replace("`", "")
    # Strip inline -- comments
    raw = re.sub(r"--[^\n]*", "", raw)
    # Collapse whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    # Take only the first statement
    raw = raw.split(";")[0].strip()
    return raw


def _clean_json(raw: str) -> str:
    """Strip markdown fences around JSON if present."""
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("`", "").strip()
    return raw


def _expand_keywords(context: str, native_language: Optional[str] = None) -> List[str]:
    """
    Extract keywords from the context/question, then expand into root forms,
    synonyms, abbreviations, slangs, misspellings, and multilingual translations
    across all default languages plus the user's native language.
    Falls back to splitting the context into words on error.
    """
    try:
        raw = keyword_expansion_chain.invoke({
            "context": context,
            "native_language": native_language or "none",
        })
        cleaned = _clean_json(raw)
        # Recover from truncated JSON — trim to last valid entry
        if not cleaned.endswith("]"):
            last_quote_comma = cleaned.rfind('",')
            if last_quote_comma != -1:
                cleaned = cleaned[:last_quote_comma + 1] + "]"
            else:
                cleaned = cleaned.rstrip(",") + "]"
        expanded: List[str] = json.loads(cleaned)
        seen: Set[str] = set()
        result: List[str] = []
        for kw in expanded:
            stripped = kw.strip()
            lower = stripped.lower()
            if lower and lower not in seen:
                seen.add(lower)
                result.append(stripped)
        # Cap at 50 terms to prevent SQL from becoming too long and truncating mid-string
        result = result[:50]
        print(f"[api] expanded keywords ({len(result)} terms): {result}")
        return result
    except Exception as e:
        print(f"[api] keyword expansion failed, falling back: {e}")
        # Fallback: use meaningful words from context (length > 3)
        return [w.strip("?.,!") for w in context.split() if len(w) > 3]


# Allowed tables — any SQL referencing other tables is rejected immediately
ALLOWED_TABLES = {"articles", "media_outlets", "countries"}


def _build_keyword_filter(keywords: List[str]) -> str:
    """
    Build the SQL keyword filter block in Python to avoid LLM truncation.
    Returns a safe WHERE clause fragment using ILIKE OR across title and content.
    """
    if not keywords:
        return "TRUE"
    conditions = []
    for kw in keywords:
        # Escape single quotes in keyword
        safe_kw = kw.replace("'", "''")
        conditions.append(f"a.title ILIKE '%{safe_kw}%'")
        conditions.append(f"a.content ILIKE '%{safe_kw}%'")
    joined = "\n    OR ".join(conditions)
    return "(\n    " + joined + "\n  )"


def _validate_tables(sql: str) -> None:
    """Raise ValueError if the SQL references any table not in ALLOWED_TABLES."""
    referenced = set(re.findall(r"(?:FROM|JOIN)\s+(\w+)", sql, flags=re.IGNORECASE))
    unknown = referenced - ALLOWED_TABLES
    if unknown:
        raise SQLExecutionError(
            f"SQL references unknown table(s): {unknown}. "
            f"Only allowed: {ALLOWED_TABLES}"
        )


def _strip_id_filters(sql: str) -> str:
    """
    Remove any id-based WHERE conditions the LLM may have added
    (e.g. a.id < 20, id <= 50, articles.id > 10) which would silently
    exclude relevant rows with higher IDs.
    Also ensure LIMIT is at least 50.
    """
    # Remove id comparison conditions like: a.id < 20, id <= 50, a.id BETWEEN 1 AND 30
    sql = re.sub(r"AND\s+\w*\.?id\s*(<=|>=|<|>|=|!=|BETWEEN)\s*[\d\s]+(?:AND\s+\d+)?", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"WHERE\s+\w*\.?id\s*(<=|>=|<|>|=|!=)\s*\d+", "WHERE", sql, flags=re.IGNORECASE)
    # Ensure LIMIT is at least 50
    sql = re.sub(r"LIMIT\s+(\d+)", lambda m: f"LIMIT {max(int(m.group(1)), 50)}", sql, flags=re.IGNORECASE)
    # Collapse any double spaces or dangling ANDs left behind
    sql = re.sub(r"AND\s+AND", "AND", sql, flags=re.IGNORECASE)
    sql = re.sub(r"WHERE\s+AND", "WHERE", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s{2,}", " ", sql).strip()
    return sql


def _fetch_with_retry(sql: str, max_retries: int = 2) -> List[dict]:
    """
    Validate tables, execute SQL via fetch_data.
    On any error, ask the LLM to fix and retry up to max_retries times.
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if not sql:
                raise SQLExecutionError("Generated SQL is empty after cleaning.")

            # Guard against hallucinated table names before hitting the DB
            _validate_tables(sql)

            print(f"[api] attempt {attempt + 1} SQL: {sql}")
            return fetch_data(sql)

        except (SQLExecutionError, ValueError) as e:
            last_error = str(e)
            if attempt < max_retries:
                print(f"[api] SQL error on attempt {attempt + 1}, asking LLM to fix...")
                print(f"[api] error: {last_error}")
                raw_fixed = sql_fix_chain.invoke({"sql": sql, "error": last_error})
                sql = _clean_sql(raw_fixed)
                print(f"[api] fixed SQL: {sql}")
            else:
                raise SQLExecutionError(
                    f"SQL failed after {max_retries + 1} attempts. "
                    f"Last error: {last_error}\nLast SQL: {sql}"
                )

    raise SQLExecutionError(f"SQL failed: {last_error}")


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
        expanded = _expand_keywords(request.context, request.native_language)
        payload = request.dict()
        payload["keywords"] = expanded
        raw = sql_chain.invoke(payload)
        sql = _clean_sql(raw)
        keyword_filter = _build_keyword_filter(expanded)
        sql = sql.replace("__KEYWORD_FILTER__", keyword_filter)
        sql = _strip_id_filters(sql)
        return {"expanded_keywords": expanded, "sql": sql}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/generate-and-fetch", response_model=ResearchResult)
async def generate_and_fetch(request: SQLRequest):
    """
    1. Expand keywords (root forms, synonyms, abbreviations, slangs, misspellings).
    2. Generate SQL using expanded keywords.
    3. Fetch matching articles — auto-retry with LLM fix on SQL errors.
    4. Pass articles to a second LLM call for fact-check analysis.
    5. Return a single structured ResearchResult.
    """
    try:
        # Step 1 — expand keywords from context
        expanded_keywords = _expand_keywords(request.context, request.native_language)

        # Step 2 — generate SQL structure (LLM), inject keyword filter (Python)
        payload = request.dict()
        payload["keywords"] = expanded_keywords
        raw_sql = sql_chain.invoke(payload)
        sql = _clean_sql(raw_sql)

        # Inject the keyword filter built in Python — avoids LLM truncation entirely
        keyword_filter = _build_keyword_filter(expanded_keywords)
        sql = sql.replace("__KEYWORD_FILTER__", keyword_filter)
        # Strip any id-based filters and enforce minimum LIMIT 50
        sql = _strip_id_filters(sql)
        print("[api] final SQL:", sql)

        # Step 3 — fetch with auto-retry on syntax errors
        rows = _fetch_with_retry(sql)
        print(f"[api] fetched {len(rows)} row(s)")

        if not rows:
            return ResearchResult(
                claim=request.context,
                classification="UNVERIFIED",
                confidence=0.0,
                explanation_en="No relevant articles were found in the database for this query.",
                explanation_native="No relevant articles were found in the database for this query.",
                sources=[],
            )

        # Step 4 — analyse articles with second LLM call
        articles_text = _format_articles_for_prompt(rows)
        raw_analysis = analysis_chain.invoke({
            "context": request.context,
            "native_language": request.native_language or "English",
            "articles": articles_text,
        })

        # Step 5 — parse and return
        cleaned = _clean_json(raw_analysis)
        data = json.loads(cleaned)

        classification = data.get("classification", "UNVERIFIED")
        raw_confidence = float(data.get("confidence", 0.0))

        # Python-side confidence sanity check based on source count
        source_urls = [s for s in data.get("sources", []) if s and s.strip()]
        n_articles = len(rows)
        n_sources  = len(source_urls)

        # Floor: confidence can't be higher than what the source count justifies
        count_ceiling = {0: 0.0, 1: 0.50, 2: 0.65, 3: 0.75, 4: 0.82, 5: 0.87}
        ceiling = count_ceiling.get(n_articles, 0.90 if n_articles < 10 else 0.98)

        # Boost for having verifiable URLs
        url_boost = min(n_sources * 0.02, 0.10)
        ceiling = min(ceiling + url_boost, 0.98)

        # UNVERIFIED can never exceed 0.5
        if classification == "UNVERIFIED":
            ceiling = min(ceiling, 0.49)

        confidence = min(raw_confidence, ceiling)

        return ResearchResult(
            claim=data.get("claim", request.context),
            classification=classification,
            confidence=round(confidence, 2),
            explanation_en=data.get("explanation_en", ""),
            explanation_native=data.get("explanation_native", ""),
            sources=source_urls,
        )

    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse LLM analysis response: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
