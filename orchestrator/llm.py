"""
llm.py — the one place that creates the LLM used by the LLM policies.

One fixed model for every decision in every experiment, so differences between policies
come from the policy and not from which model happened to answer. The model is reached
through NRP's OpenAI-compatible endpoint with LangChain's ChatOpenAI (the same client the
AoI project uses), which gives `with_structured_output(...)` for the verdict types.

Configuration comes from environment variables (or the repo's .env, which is gitignored):
    NRP_LLM_API_KEY     required — the NRP LLM key
    ORCH_LLM_MODEL      optional — defaults to "gpt-oss"
    ORCH_LLM_BASE_URL   optional — defaults to NRP's endpoint

Usage:
    from orchestrator.llm import get_llm, llm_config, STRUCTURED_OUTPUT_METHOD
    llm = get_llm()
    chain = prompt | llm.with_structured_output(AssimilationDecision,
                                                method=STRUCTURED_OUTPUT_METHOD,
                                                include_raw=True)
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_openai import ChatOpenAI

# Both locations are read, because the repo tells you to use both: the key is
# listed in user_info_template.env, so user_info.env is where a reader naturally
# puts it, while this module originally looked only at .env -- and a key in the
# wrong one of the two fails silently later, at the first LLM decision. .env is
# loaded first and wins, since load_dotenv does not overwrite what is already set.
# Both are gitignored.
try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parent.parent
    for _candidate in (".env", "user_info.env"):
        _p = _root / _candidate
        if _p.exists():
            load_dotenv(_p)
except ImportError:
    pass

DEFAULT_MODEL = "gpt-oss"               # NRP alias; was openai/gpt-oss-120b (120B MoE, ~5B active)
DEFAULT_BASE_URL = "https://ellm.nrp-nautilus.io/v1"

# Tested 2026-11-23 on NRP: gpt-oss returned valid AssimilationDecisions with both
# "json_schema" and "function_calling"; json_schema was consistent and used fewer tokens.
STRUCTURED_OUTPUT_METHOD = "json_schema"

TEMPERATURE = 0.0                       # same situation -> same answer, as far as possible
TIMEOUT_S = 120
MAX_RETRIES = 2


def get_llm(model: str | None = None, temperature: float = TEMPERATURE) -> ChatOpenAI:
    """Create the chat model. Call once per run and share it between the LLM policies."""
    api_key = os.environ.get("NRP_LLM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NRP_LLM_API_KEY is not set. Add it to the repo's .env or "
            "user_info.env (both gitignored), or export it.")
    return ChatOpenAI(
        model=model or os.environ.get("ORCH_LLM_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("ORCH_LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        temperature=temperature,
        timeout=TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )


def llm_config(llm: ChatOpenAI) -> dict:
    """What to record in a run's config.json. The model name is an NRP alias that NRP can
    repoint, so the date matters: run all LLM experiments within a short window."""
    return {
        "model": llm.model_name,
        "base_url": str(llm.openai_api_base),
        "temperature": llm.temperature,
        "structured_output_method": STRUCTURED_OUTPUT_METHOD,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def total_tokens(raw_message) -> int:
    """Total tokens of one call, from the raw AIMessage that include_raw=True returns."""
    usage = getattr(raw_message, "usage_metadata", None) or {}
    return int(usage.get("total_tokens", 0) or 0)