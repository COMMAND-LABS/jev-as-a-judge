"""Command Labs API client: stream an agent's answer and fetch its config."""

import json
from typing import Any
from urllib.parse import quote

import requests


def _agent_url(api_url: str, agent_id: str) -> str:
    return f"{api_url.rstrip('/')}/api/agents/{quote(agent_id, safe='')}"


def _headers(api_key: str | None, accept: str) -> dict[str, str]:
    headers = {"Accept": accept}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _read_answer(response: requests.Response) -> str:
    """Read the agent's answer from the stream.

    Each event is a line like `data: {"event": "...", "data": ...}`. The full answer
    arrives in `on_chain_end`; `on_chat_model_stream` carries it in pieces, as a list
    of {"type": "text", "text": ...} blocks, which are used if `on_chain_end` is missing.
    """
    final_answer: str | None = None
    pieces: list[str] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        event = json.loads(line[len("data:"):])
        name, data = event.get("event"), event.get("data")
        if name == "error":
            message = data.get("message") if isinstance(data, dict) else data
            raise RuntimeError(f"Command Labs stream error: {message or 'unknown error'}")
        if name == "on_chain_end" and isinstance(data, str):
            final_answer = data
        elif name == "on_chat_model_stream" and isinstance(data, list):
            pieces.extend(block.get("text", "") for block in data if block.get("type") == "text")
    return (final_answer if final_answer is not None else "".join(pieces)).strip()


def call_cmdlabs_completion(
    *,
    api_url: str,
    api_key: str | None,
    agent_id: str,
    session_id: str,
    prompt: str,
    cmdlabs_api_timeout_seconds: int = 120,
    cmdlabs_api_retries: int = 2,
) -> str:
    """Ask an agent a question and return its answer, retrying on timeouts and connection errors."""
    url = f"{_agent_url(api_url, agent_id)}/stream"
    attempts = max(1, cmdlabs_api_retries + 1)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            with requests.post(
                url,
                headers=_headers(api_key, "text/event-stream"),
                json={"sessionId": session_id, "prompt": prompt},
                stream=True,
                timeout=cmdlabs_api_timeout_seconds,
            ) as response:
                if not response.ok:
                    raise RuntimeError(f"Command Labs API error {response.status_code}: {response.text}")
                return _read_answer(response)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc

    raise RuntimeError(f"Command Labs request failed after {attempts} attempt(s) at {url}: {last_error}")


def fetch_cmdlabs_agent_config(
    *,
    api_url: str,
    api_key: str | None,
    agent_id: str,
    cmdlabs_api_timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Fetch an agent's name and config (model, tools, system prompt)."""
    response = requests.get(
        _agent_url(api_url, agent_id),
        headers=_headers(api_key, "application/json"),
        timeout=cmdlabs_api_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
