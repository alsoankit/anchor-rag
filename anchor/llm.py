import json

from groq import Groq
from pydantic import BaseModel, ValidationError

from anchor.config import GROQ_API_KEY, WORKER_MODEL, WRITER_MODEL

client = Groq(api_key=GROQ_API_KEY)

usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def chat(prompt: str, system: str = "", model: str = WORKER_MODEL,
         temperature: float = 0.0, max_tokens: int = 900, json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
        max_tokens=max_tokens, **kwargs,
    )

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
               max_tokens: int = 900, attempts: int = 2):
    """Ask for JSON and keep asking until it parses into the model we wanted.

    The retry hands the model its own broken output and the validation error, which
    fixes far more cases than simply asking again, because the second attempt is a
    correction task rather than a fresh guess. Structured output is where an LLM
    pipeline quietly breaks in production, so this path is worth the extra call.
    """
    hint = json.dumps(schema.model_json_schema())
    last_error = ""
    reply = ""

    for attempt in range(attempts):
        if attempt == 0:
            ask = f"{prompt}\n\nReply with JSON matching this schema:\n{hint}"
        else:
            ask = (f"{prompt}\n\nYour previous reply was rejected.\n"
                   f"Reply was:\n{reply}\n\nThe problem:\n{last_error}\n\n"
                   f"Return corrected JSON matching:\n{hint}")

        reply = chat(ask, system, json_mode=True, max_tokens=max_tokens)
        try:
            return schema.model_validate_json(_salvage(reply))
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            last_error = str(error)[:400]

    raise ValueError(f"model never produced valid {schema.__name__}: {last_error}")
