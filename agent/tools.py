"""
External API tool integrations.

search_protocol — queries NCBI PubMed E-utilities (free, no auth required).
  Step 1: esearch.fcgi  → find PMIDs matching the clinical query
  Step 2: efetch.fcgi   → retrieve plain-text abstracts for those PMIDs

Truncated to 2000 chars to stay within GPT-4o's 300-token response budget.
PUBMED_API_KEY is optional but raises the rate limit from 3 to 10 req/s.
"""
import structlog
import httpx

logger = structlog.get_logger()


async def search_protocol(query: str, settings) -> str:
    """
    Search PubMed for clinical protocol abstracts matching the query.
    Returns up to 2000 characters of abstract text, or an informative fallback.
    """
    base = settings.pubmed_base_url
    common: dict = {}
    if settings.pubmed_api_key:
        common["api_key"] = settings.pubmed_api_key

    logger.info("pubmed_search_start", query=query[:100])

    # ── Step 1: esearch — find matching PMIDs ─────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{base}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": 3,
                    "retmode": "json",
                    **common,
                },
            )
            r.raise_for_status()
            pmids: list[str] = r.json().get("esearchresult", {}).get("idlist", [])
    except httpx.TimeoutException:
        logger.error("pubmed_esearch_timeout", query=query[:50])
        return "PubMed search timed out. Please try a more specific clinical query."
    except httpx.HTTPStatusError as exc:
        logger.error("pubmed_esearch_http_error", status=exc.response.status_code)
        return f"PubMed search returned HTTP {exc.response.status_code}."
    except Exception as exc:
        logger.error("pubmed_esearch_unexpected", error=str(exc))
        return "An error occurred while searching PubMed."

    logger.info("pubmed_esearch_done", pmids_found=len(pmids))

    if not pmids:
        return (
            "No relevant clinical protocols found in PubMed for this query. "
            "Please try different search terms or consult your institutional guidelines."
        )

    # ── Step 2: efetch — retrieve abstracts ───────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(
                f"{base}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "rettype": "abstract",
                    "retmode": "text",
                    **common,
                },
            )
            r.raise_for_status()
            abstracts = r.text
    except httpx.TimeoutException:
        logger.error("pubmed_efetch_timeout", pmids=pmids)
        return "PubMed abstract retrieval timed out."
    except httpx.HTTPStatusError as exc:
        logger.error("pubmed_efetch_http_error", status=exc.response.status_code)
        return f"PubMed abstract fetch returned HTTP {exc.response.status_code}."
    except Exception as exc:
        logger.error("pubmed_efetch_unexpected", error=str(exc))
        return "An error occurred while fetching protocol abstracts."

    truncated = abstracts[:2000]
    logger.info(
        "pubmed_search_complete",
        pmids=len(pmids),
        raw_chars=len(abstracts),
        returned_chars=len(truncated),
    )
    return truncated
