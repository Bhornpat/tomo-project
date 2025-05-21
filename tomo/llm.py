from langchain.llms.base import LLM
from typing import Optional, List
import requests

class LlamaCppServerLLM(LLM):
    endpoint: str = "http://192.168.137.1:8080/completion"
    n_predict: int = 200

    @property
    def _llm_type(self) -> str:
        return "llama_cpp_server"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "prompt": prompt,
            "n_predict": self.n_predict,
            "temperature": 0.7,
            "stop": stop if stop else []
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} - {response.text}")

        result = response.json()
        return result["content"]  # llama.cpp returns 'content' field
