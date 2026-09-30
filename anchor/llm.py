import json
import os
import random
import re
import threading
import time

import httpx
from groq import BadRequestError, Groq, RateLimitError
from pydantic import BaseModel, ValidationError

from anchor.config import GEMINI_API_KEY, GROQ_API_KEYS, WORKER_MODEL, WRITER_MODEL

# An explicit timeout, because the default is generous enough that a request which never
# comes back sits there indefinitely. One did, during calibration, and took half an hour
# of a run with it. One retry rather than none, so a dropped connection still recovers,
# but the worst case stays bounded at roughly three minutes instead of forever.
clients = [Groq(api_key=key, timeout=90.0, max_retries=1) for key in GROQ_API_KEYS]

# A daily budget is per account AND per model. Losing gpt-oss-120b on one account says
# nothing about gpt-oss-20b on that same account, so exhaustion is recorded as a
# (model, account) pair. The earlier version tracked a single current account for
# everything, which meant the first model to run out dragged every other model off an
# account that still had budget left.
_spent: set[tuple[str, int]] = set()
_turn: dict[str, int] = {}
_keys = threading.Lock()


def _pick(model: str) -> int:
    """Round robin over the accounts that still have budget for this model.

    Spreading calls rather than draining one account at a time is what makes several
    keys worth having. Each account gets its own tokens-per-minute allowance, so three
    accounts in rotation is three times the throughput, not merely three times the total
    budget. Draining them in order would have given the second thing without the first.
    """
    with _keys:
        live = [i for i in range(len(clients)) if (model, i) not in _spent]
        if not live:
            raise RuntimeError(f"every account has spent its daily budget for {model}")
        turn = _turn.get(model, 0)
        _turn[model] = turn + 1
        return live[turn % len(live)]


def _retire(model: str, index: int) -> bool:
    """Take one account out of rotation for one model. True if others remain."""
    with _keys:
        if (model, index) not in _spent:
            _spent.add((model, index))
            usage["retired"].append(f"{model} on key {index + 1}")
        return any((model, i) not in _spent for i in range(len(clients)))


usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "waited": 0.0,
         "retired": [], "per_key": [0] * max(len(GROQ_API_KEYS), 1)}

MAX_TRIES = 6

# Groq enforces a per minute budget and a rolling daily one. A minute of waiting is
# normal during an eval and worth sitting through; anything longer means the daily
# budget is gone and sleeping on it would just stall the run for an hour, so the
# ceiling is deliberately short unless a caller says otherwise.
PATIENCE = float(os.getenv("ANCHOR_MAX_WAIT", "90"))


def _wait_for(error: RateLimitError, attempt: int) -> float:
    """Groq says how long to wait in the error body, so use that rather than guessing.

    Its per minute token budget is shared across every call the pipeline makes, so
    during an eval run the limit is hit constantly. Falling back to plain doubling
    without reading the message either sleeps far longer than needed or not long
    enough, and a run of 150 questions turns into an afternoon either way.
    """
    match = re.search(r"try again in ([0-9.]+)(ms|s)", str(error))
    if match:
        seconds = float(match.group(1))
        if match.group(2) == "ms":
            seconds /= 1000
        return seconds + 0.25
    return min(2 ** attempt, 30) + random.random()


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _gemini(prompt: str, system: str, model: str, temperature: float,
            max_tokens: int, json_mode: bool) -> str:
    """Same contract as chat(), different wire format.

    Worth having for one reason: the judges grading this pipeline were all open-weight
    models served by the same provider as the pipeline itself, so their agreeing with
    each other proved less than it looked like it did. This one is a different lineage.

    Two things about Gemini 3 that cost me an hour. It thinks before answering and the
    thinking comes out of the same output budget, so a tight maxOutputTokens returns an
    empty answer with a MAX_TOKENS finish reason rather than a short one — which is why
    there is a floor under the budget regardless of what the caller asked for. And
    thinkingBudget, which is what the older models take, is quietly ignored here;
    thinkingLevel is the one that works.
    """
    floor = max(max_tokens, 1024)

    for attempt in range(MAX_TRIES):
        config = {
            "temperature": temperature,
            "maxOutputTokens": floor * (attempt + 1),
            "thinkingConfig": {"thinkingLevel": "low"},
        }
        if json_mode:
            config["responseMimeType"] = "application/json"

        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        response = httpx.post(GEMINI_URL.format(model=model),
                              headers={"x-goog-api-key": GEMINI_API_KEY},
                              json=body, timeout=120)

        # 503 here means the shared model is busy rather than anything being wrong with
        # the request, and it clears on its own within a few seconds.
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == MAX_TRIES - 1:
                response.raise_for_status()
            time.sleep(min(2 ** attempt, 30) + random.random())
            continue

        response.raise_for_status()
        payload = response.json()

        usage["calls"] += 1
        meta = payload.get("usageMetadata", {})
        usage["prompt_tokens"] += meta.get("promptTokenCount", 0)
        usage["completion_tokens"] += meta.get("candidatesTokenCount", 0)

        candidates = payload.get("candidates", [])
        if not candidates:
            return ""

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()

        # Thinking ate the budget. Same request, more room.
        if not text and candidates[0].get("finishReason") == "MAX_TOKENS":
            continue

        return text

    return ""


def chat(prompt: str, system: str = "", model: str = WORKER_MODEL,
         temperature: float = 0.0, max_tokens: int = 900, json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if model.startswith("gemini"):
        return _gemini(prompt, system, model, temperature, max_tokens, json_mode)

    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}

    for attempt in range(MAX_TRIES):
        index = _pick(model)
        try:
            response = clients[index].chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, **kwargs,
            )
            usage["per_key"][index] += 1
            break
        except RateLimitError as error:
            pause = _wait_for(error, attempt)

            if "per day" in str(error) or pause > PATIENCE:
                if _retire(model, index):
                    continue
                raise

            if attempt == MAX_TRIES - 1:
                raise

            # A per minute limit belongs to one account. With others in rotation the
            # next attempt lands on a different one, so there is no reason to sit out
            # the full delay the server suggested; a short pause is only there to stop
            # a hot loop.
            with _keys:
                spare = sum(1 for i in range(len(clients)) if (model, i) not in _spent) > 1
            pause = min(pause, 1.0) if spare else pause
            usage["waited"] += pause
            time.sleep(pause)

    usage["calls"] += 1
    if response.usage:
        usage["prompt_tokens"] += response.usage.prompt_tokens
        usage["completion_tokens"] += response.usage.completion_tokens

    return (response.choices[0].message.content or "").strip()


def write(prompt: str, system: str = "", max_tokens: int = 900) -> str:
    return chat(prompt, system, model=WRITER_MODEL, max_tokens=max_tokens)


def _salvage(text: str) -> str:
    """Models like wrapping JSON in prose or a code fence. Take the outermost braces."""
    start, end = text.find("{"), text.rfind("}")
    return text[start: end + 1] if start != -1 and end > start else text


def structured(prompt: str, schema: type[BaseModel], system: str = "",
               max_tokens: int = 900, attempts: int = 2, model: str = WORKER_MODEL):
    """Ask for JSON and keep asking until it parses into the model we wanted.

    The retry hands the model its own broken output and the validation error, which
    fixes far more cases than simply asking again, because the second attempt is a
    correction task rather than a fresh guess. Structured output is where an LLM
    pipeline quietly breaks in production, so this path is worth the extra call.
    """
    hint = json.dumps(schema.model_json_schema())
    last_error = ""
    reply = ""

    for attempt in range(attempts + 1):
        if attempt == 0:
            ask = f"{prompt}\n\nReply with JSON matching this schema:\n{hint}"
        else:
            ask = (f"{prompt}\n\nYour previous reply was rejected.\n"
                   f"Reply was:\n{reply}\n\nThe problem:\n{last_error}\n\n"
                   f"Return corrected JSON matching:\n{hint}")

        # Groq validates JSON mode output on its side and returns a 400 when the model
        # fails to produce any, so the first retry drops JSON mode entirely and lets
        # _salvage pull the object out of whatever comes back. Reusing the mode that
        # just failed tends to fail the same way.
        try:
            reply = chat(ask, system, model=model, json_mode=attempt == 0,
                         max_tokens=max_tokens)
        except BadRequestError as error:
            last_error = f"the api rejected the generation: {str(error)[:200]}"
            reply = ""
            continue

        try:
            return schema.model_validate_json(_salvage(reply))
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            last_error = str(error)[:400]

    raise ValueError(f"model never produced valid {schema.__name__}: {last_error}")
