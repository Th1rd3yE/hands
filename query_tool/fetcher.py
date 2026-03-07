import logging
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

logger = logging.getLogger(__name__)

__all__ = [
    "fetch_data",
    "SupabaseConfigError",
    "SupabaseRPCError",
    "SQLExecutionError",
]


class SupabaseConfigError(Exception):
    """Raised when Supabase credentials are missing or invalid."""


class SupabaseRPCError(Exception):
    """Raised when the execute_sql RPC function is missing or misconfigured."""


class SQLExecutionError(Exception):
    """Raised when SQL execution fails."""


@lru_cache(maxsize=1)
def _get_client() -> Client:
    """Return a cached Supabase client, created once per process."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_KEY environment variables are required."
        )
    return create_client(url, key)


def _sanitize_sql(sql: str) -> str:
    """
    Sanitize SQL before sending to Supabase RPC:
    - Strip whitespace
    - Remove trailing semicolons
    - Collapse multiple statements to the first one only
    - Block obviously destructive statements
    """
    sql = sql.strip()

    # Block destructive operations
    forbidden = re.compile(
        r"^\s*(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )
    if forbidden.match(sql):
        raise ValueError(
            "Forbidden SQL operation detected. Only SELECT statements are allowed."
        )

    # Take only the first statement if multiple are present
    sql = sql.split(";")[0].strip()

    return sql


def _unwrap(data: Any) -> list[dict[str, Any]]:
    """
    Supabase RPC can return results in several shapes depending on how the
    execute_sql function is defined. This normalises all of them to a flat
    list of row dicts.

    Handles:
      - None / empty                  → []
      - Already a flat list of dicts  → returned as-is
      - Jsonb wrapped: [{"json_agg": [...rows...]}]
      - Double-wrapped list           → unwrapped recursively (one level)
    """
    if not data:
        return []

    # Already a flat list of row dicts
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        first_values = list(data[0].values())
        # If the first value is itself a list, the rows are nested under a key
        # e.g. [{"json_agg": [{...}, {...}]}]
        if len(data) == 1 and len(first_values) == 1 and isinstance(first_values[0], list):
            logger.debug("[fetcher] unwrapping nested jsonb response")
            return first_values[0]

        # Flat list of row dicts — the happy path
        return data

    # Scalar or unexpected shape — log and return empty
    logger.warning("[fetcher] unexpected response shape: %s", type(data))
    return []


def fetch_data(sql: str) -> list[dict[str, Any]]:
    """
    Execute a SELECT query via Supabase RPC and return results as a list of dicts.

    Args:
        sql: A single SELECT SQL statement.

    Returns:
        List of row dicts, or an empty list if no rows matched.

    Raises:
        SupabaseConfigError: Missing environment credentials.
        SupabaseRPCError:    The execute_sql function is absent in the DB.
        SQLExecutionError:   The query failed for any other reason.
        ValueError:          Forbidden or malformed SQL was provided.
    """
    sql = _sanitize_sql(sql)
    logger.debug("[fetcher] executing SQL via RPC: %s", sql)

    client = _get_client()

    try:
        response = client.rpc("execute_sql", {"query": sql}).execute()

        # Log raw response to help debug shape issues
        logger.debug("[fetcher] raw response data: %s", response.data)
        logger.debug("[fetcher] raw response type: %s", type(response.data))

        data = _unwrap(response.data)
        logger.debug("[fetcher] returned %d row(s)", len(data))
        return data

    except (SupabaseConfigError, ValueError):
        raise

    except Exception as e:
        msg = str(e)

        if "PGRST202" in msg or "Could not find the function" in msg:
            raise SupabaseRPCError(
                "Supabase RPC 'public.execute_sql' is missing. "
                "Please create it in your database — see README for the definition."
            ) from e

        if "42601" in msg:
            raise SQLExecutionError(
                f"SQL syntax error. Check the generated query.\nSQL: {sql}\nDetail: {msg}"
            ) from e

        if "42804" in msg:
            raise SQLExecutionError(
                "Return type mismatch in execute_sql function. "
                "Re-create it using the definition in the README."
            ) from e

        raise SQLExecutionError(f"Unexpected error during SQL execution: {msg}") from e
