import httpx

from app.application.errors import LLMUnavailableError
from app.application.ports.llm_client import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client or httpx.Client()

    def generate(self, *, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        try:
            response = self._http_client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as exc:  # includes timeout and connection errors
            raise LLMUnavailableError(details={"reason": str(exc)}) from exc

        if response.status_code >= 400:
            raise LLMUnavailableError(
                details={"status_code": response.status_code, "body": response.text},
            )

        data = response.json()
        generated = data.get("message", {}).get("content")
        if not isinstance(generated, str):
            raise LLMUnavailableError(
                message="Respuesta inválida del modelo local",
                details={"payload": data},
            )
        return generated
