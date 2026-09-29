import json
import random
import re
import time

from groq import BadRequestError, Groq, RateLimitError
from pydantic import BaseModel, ValidationError

from anchor.config import GROQ_API_KEY, WORKER_MODEL, WRITER_MODEL

client = Groq(api_key=GROQ_API_KEY)

usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "waited": 0.0}

MAX_TRIES = 6


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


def chat(prompt: str, system: str = "", model: str = WORKER_MODEL,
         temperature: float = 0.0, max_tokens: int = 900, json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}

    for attempt in range(MAX_TRIES):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, **kwargs,
            )
            break
        except RateLimitError as error:
            if attempt == MAX_TRIES - 1:
                raise
            pause = _wait_for(error, attempt)
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
