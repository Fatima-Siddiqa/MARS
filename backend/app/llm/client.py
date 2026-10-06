"""Single LLM wrapper. Provider chosen by LLM_PROVIDER env var (ollama | groq).

All agents call `complete()`; never call a provider SDK/API directly elsewhere.
"""
import httpx

from app.core.config import get_settings


async def complete(prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
    s = get_settings()
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    async with httpx.AsyncClient(timeout=120) as client:
        if s.llm_provider == "groq":
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {s.groq_api_key}"},
                json={"model": s.groq_model, "messages": messages, "temperature": temperature},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        r = await client.post(
            f"{s.ollama_base_url}/api/chat",
            json={
                "model": s.ollama_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        r.raise_for_status()
        return r.json()["message"]["content"]
