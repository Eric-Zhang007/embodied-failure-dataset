"""
VLM 调用客户端。

支持：本地 Ollama / 远程 OpenAI 兼容 API（硅基流动、OpenRouter 等）。
"""

import json
import os
import base64
import io
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# 日志
# ------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FULL_API = os.environ.get("EFD_LOG_FULL_API", "1") != "0"

# Per-instance api_calls.jsonl directory. Set via VLMClient.set_api_log_dir().
_api_log_dir: Path | None = None


def _get_api_log_path() -> Path:
    """Return the api_calls.jsonl path. Uses per-test dir if set, else global."""
    base = _api_log_dir or LOG_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / "api_calls.jsonl"


logger = logging.getLogger("vlm_client")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(LOG_DIR / "api_calls.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)


def _summarize_messages(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            texts = [p["text"] for p in content if p.get("type") == "text"]
            img_count = sum(1 for p in content if p.get("type") == "image_url")
            parts.append(f"{role}: {len(texts)} text parts, {img_count} images")
        elif isinstance(content, str):
            parts.append(f"{role}: {len(content)} chars")
    return " | ".join(parts)


# ------------------------------------------------------------------
# 客户端
# ------------------------------------------------------------------

class VLMClient:
    @staticmethod
    def set_api_log_dir(log_dir: str | Path | None):
        """Set per-test directory for api_calls.jsonl. Call before creating clients."""
        global _api_log_dir
        _api_log_dir = Path(log_dir) if log_dir else None

    def __init__(self, backend: str, model: str, base_url: str = None, api_key: str = None,
                 reasoning_effort: str | None = None):
        self.backend = backend
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort  # "low" | "medium" | "high" | "xhigh" | "max"

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def ollama(cls, model: str = "qwen2.5-vl:7b", host: str = "http://localhost:11434") -> "VLMClient":
        return cls(backend="ollama", model=model, base_url=host)

    @classmethod
    def openrouter(cls, model: str = "anthropic/claude-opus-4", api_key: str = None,
                   reasoning_effort: str | None = None) -> "VLMClient":
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        return cls(backend="openai", model=model,
                   base_url="https://openrouter.ai/api/v1", api_key=key,
                   reasoning_effort=reasoning_effort)

    @classmethod
    def openai(cls, model: str = "gpt-5", api_key: str = None, base_url: str = None,
               reasoning_effort: str | None = None) -> "VLMClient":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        url = base_url or "https://api.openai.com/v1"
        return cls(backend="openai", model=model, base_url=url, api_key=key,
                   reasoning_effort=reasoning_effort)

    @classmethod
    def siliconflow(cls, model: str, api_key: str,
                    reasoning_effort: str | None = None) -> "VLMClient":
        return cls(backend="openai", model=model,
                   base_url="https://api.siliconflow.cn/v1", api_key=api_key,
                   reasoning_effort=reasoning_effort)

    # ------------------------------------------------------------------
    # 核心调用
    # ------------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if self.backend == "openai":
            return self._chat_openai(messages, json_mode, temperature, max_tokens)
        elif self.backend == "ollama":
            return self._chat_ollama(messages, json_mode, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def chat_text(
        self,
        system_prompt: str,
        user_text: str,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return self.chat(messages, json_mode=json_mode, temperature=temperature,
                         max_tokens=max_tokens)

    def chat_with_images(
        self,
        system_prompt: str,
        user_text: str,
        images: list[tuple[str, 'np.ndarray']],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """多图对话。images: [(label, image_array), ...]"""
        import numpy as np
        from PIL import Image
        content_parts = [{"type": "text", "text": user_text}]
        for idx, (label, img) in enumerate(images, start=1):
            if isinstance(img, str):
                img = np.array(Image.open(img))
            b64 = _encode_image(img)
            content_parts.append({"type": "text", "text": f"VIEW {idx}: {label}"})
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]
        return self.chat(messages, json_mode=False, temperature=temperature, max_tokens=max_tokens)

    def chat_with_images_json(
        self,
        system_prompt: str,
        user_text: str,
        images: list[tuple[str, 'np.ndarray']],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 2,
        required_fields: tuple[str, ...] | None = None,
    ) -> dict:
        return self._chat_json_with_retry(
            call_fn=lambda prompt: self.chat_with_images(
                system_prompt=system_prompt,
                user_text=prompt,
                images=images,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            user_text=user_text,
            max_retries=max_retries,
            required_fields=required_fields,
            retry_name="JSON_RETRY_IMAGES",
        )

    def chat_with_image_json(
        self,
        system_prompt: str,
        user_text: str,
        image: np.ndarray | str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 2,
        required_fields: tuple[str, ...] | None = None,
    ) -> dict:
        """
        带图像的对话 + 自动 JSON 解析 + 解析失败重试。
        返回解析后的 dict。失败后直接抛错，不伪造动作。
        """
        return self._chat_json_with_retry(
            call_fn=lambda prompt: self.chat_with_image(
                system_prompt=system_prompt,
                user_text=prompt,
                image=image,
                json_mode=False,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            user_text=user_text,
            max_retries=max_retries,
            required_fields=required_fields,
            retry_name="JSON_RETRY",
        )

    def chat_text_json(
        self,
        system_prompt: str,
        user_text: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 2,
        required_fields: tuple[str, ...] | None = None,
    ) -> dict:
        """
        纯文本对话 + 自动 JSON 解析 + 解析失败重试。
        """
        return self._chat_json_with_retry(
            call_fn=lambda prompt: self.chat_text(
                system_prompt=system_prompt,
                user_text=prompt,
                json_mode=False,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            user_text=user_text,
            max_retries=max_retries,
            required_fields=required_fields,
            retry_name="JSON_RETRY_TEXT",
        )

    def _chat_json_with_retry(
        self,
        call_fn,
        user_text: str,
        max_retries: int,
        required_fields: tuple[str, ...] | None,
        retry_name: str,
    ) -> dict:
        last_raw = ""
        last_error = ""
        for attempt in range(max_retries):
            response = call_fn(user_text)
            last_raw = response
            parsed = parse_json_response(response)
            last_error = _json_response_error(parsed, required_fields)
            if last_error is None:
                return parsed
            if attempt < max_retries - 1:
                user_text = user_text + "\n\n" + _json_retry_instruction(last_error)
                logger.warning("%s attempt=%d/%d error=%s", retry_name, attempt + 1, max_retries, last_error)

        raise ValueError(
            f"VLM JSON response invalid after {max_retries} attempts: "
            f"{last_error}. Last response: {last_raw[:1000]}"
        )

    def chat_with_image(
        self,
        system_prompt: str,
        user_text: str,
        image: np.ndarray | str,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if isinstance(image, str):
            image = np.array(Image.open(image))
        image_b64 = _encode_image(image)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        return self.chat(messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # OpenAI 兼容后端（含日志）
    # ------------------------------------------------------------------
    def _chat_openai(self, messages, json_mode, temperature, max_tokens):
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        msg_summary = _summarize_messages(messages)
        body_size = len(json.dumps(body, ensure_ascii=False))
        t_start = time.time()

        logger.debug("REQ model=%s mode=%s body_size=%d summary=[%s]",
                     self.model, "json" if json_mode else "text", body_size, msg_summary)

        last_conn_error = None
        timeout = 300
        for attempt in range(3):  # 最多 3 次尝试（1 次正常 + 2 次重试）
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=timeout,
                )
                elapsed = time.time() - t_start
                if resp.status_code == 200:
                    msg = resp.json()["choices"][0]["message"]
                    content = msg["content"]
                    reasoning = msg.get("reasoning_content", "")
                    logger.debug("RESP model=%s status=200 elapsed=%.1fs content_len=%d reasoning_len=%d preview=%s",
                                 self.model, elapsed, len(content), len(reasoning or ""), content[:500])
                    self._write_api_exchange(body, 200, content, elapsed, attempt + 1,
                                             reasoning_content=reasoning or None)
                    return content
                # 429: rate limited - wait and retry
                if resp.status_code == 429:
                    self._write_api_exchange(body, 429, resp.text, elapsed, attempt + 1)
                    wait = 5 * (attempt + 1)
                    logger.warning("RATE_LIMIT retry %d/3 wait %ds", attempt + 1, wait)
                    time.sleep(wait)
                    continue
                # 5xx: server errors (502/503/504/524) — transient, retry with backoff
                if resp.status_code >= 500:
                    self._write_api_exchange(body, resp.status_code, resp.text, elapsed, attempt + 1)
                    if attempt < 2:
                        wait = 2 ** attempt  # 1s, 2s
                        logger.warning("SERVER_ERR retry %d/3 status=%d wait=%ds",
                                       attempt + 1, resp.status_code, wait)
                        time.sleep(wait)
                        continue
                    # All retries exhausted — dump and raise
                    self._dump_failure(body, resp.status_code, resp.text, elapsed)
                    resp.raise_for_status()
                # Other errors (4xx client errors): dump and raise immediately
                self._write_api_exchange(body, resp.status_code, resp.text, elapsed, attempt + 1)
                self._dump_failure(body, resp.status_code, resp.text, elapsed)
                resp.raise_for_status()

            except requests.exceptions.ConnectionError as e:
                last_conn_error = e
                elapsed = time.time() - t_start
                self._write_api_exchange(body, "CONNECTION_ERROR", str(e), elapsed, attempt + 1)
                logger.warning("CONN_ERR model=%s attempt=%d/3 error=%s", self.model, attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)

            except requests.exceptions.Timeout:
                elapsed = time.time() - t_start
                self._write_api_exchange(body, "TIMEOUT", f"timeout={timeout}s", elapsed, attempt + 1)
                logger.warning("TIMEOUT model=%s attempt=%d/3 timeout=%ds elapsed=%.1fs",
                              self.model, attempt + 1, timeout, elapsed)
                if attempt < 2:
                    timeout += 100  # 300 → 400 → 500
                    continue
                self._dump_failure(body, "TIMEOUT_EXHAUSTED",
                                   f"All retries exhausted, last timeout={timeout}s", elapsed)
                raise

        # ConnectionError 重试全部失败
        elapsed = time.time() - t_start
        self._dump_failure(body, "CONNECTION_EXHAUSTED", str(last_conn_error), elapsed)
        raise RuntimeError(f"API unreachable after 3 attempts: {last_conn_error}")

    def _dump_failure(self, body: dict, status, response_text: str, elapsed: float):
        """失败时把完整请求体写到日志文件。"""
        dump_path = LOG_DIR / f"failure_{int(time.time())}_{self.model.replace('/', '_')}.json"
        text_sample = response_text[:500] if isinstance(response_text, str) else str(response_text)[:500]
        dump = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "base_url": self.base_url,
            "status": str(status),
            "elapsed_s": round(elapsed, 1),
            "response_sample": text_sample,
            "request_body": body,
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump, f, ensure_ascii=False, indent=2)
        logger.error("FAILURE_DUMP model=%s status=%s elapsed=%.1fs dump=%s",
                     self.model, status, elapsed, dump_path)

    def _write_api_exchange(self, body: dict, status, response_text: str, elapsed: float, attempt: int,
                            reasoning_content: str | None = None):
        if not LOG_FULL_API:
            return
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "base_url": self.base_url,
            "attempt": attempt,
            "status": str(status),
            "elapsed_s": round(elapsed, 1),
            "request_body": _sanitize_for_log(body),
            "response": response_text,
        }
        if reasoning_content:
            entry["reasoning_content"] = reasoning_content
        log_path = _get_api_log_path()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Ollama 后端
    # ------------------------------------------------------------------
    def _chat_ollama(self, messages, json_mode, temperature, max_tokens):
        import requests

        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"

        ollama_msgs = []
        for m in messages:
            if isinstance(m.get("content"), list):
                text_parts = []
                images = []
                for part in m["content"]:
                    if part["type"] == "text":
                        text_parts.append(part["text"])
                    elif part["type"] == "image_url":
                        b64 = part["image_url"]["url"].split(",", 1)[-1]
                        images.append(b64)
                ollama_msgs.append({
                    "role": m["role"],
                    "content": " ".join(text_parts),
                    "images": images,
                })
            else:
                ollama_msgs.append(m)

        body["messages"] = ollama_msgs

        for attempt in range(3):
            resp = requests.post(f"{self.base_url}/api/chat", json=body, timeout=40)
            if resp.status_code == 200:
                content = resp.json()["message"]["content"]
                self._write_api_exchange(body, 200, content, 0.0, attempt + 1)
                return content
            if attempt < 2:
                time.sleep(2 ** attempt)
            self._write_api_exchange(body, resp.status_code, resp.text, 0.0, attempt + 1)
        resp.raise_for_status()


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------
def _encode_image(image: np.ndarray) -> str:
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _repair_json(text: str) -> str:
    """Multi-stage JSON repair: fix common 8B model output issues.
    Adapted from Voyager's fix_and_parse_json pipeline.
    """
    # Strip leading/trailing whitespace and tabs
    repaired = text.strip().replace("\t", " ")

    # Fix invalid escape sequences (e.g. \s, \d, \w that aren't valid JSON escapes)
    import re
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', repaired)

    # Fix unquoted property names: {key: "val"} → {"key": "val"}
    # Matches word chars before a colon that aren't inside quotes
    repaired = re.sub(r'(?<=[{,])\s*(\w+)\s*:', r'"\1":', repaired)

    # Balance braces: if missing closing braces, add them
    open_count = repaired.count("{")
    close_count = repaired.count("}")
    if open_count > close_count:
        repaired += "}" * (open_count - close_count)
    elif close_count > open_count:
        # Too many closing braces — truncate from last valid position
        # Find the position where braces balance
        balance = 0
        cut_idx = 0
        for i, ch in enumerate(repaired):
            if ch == "{":
                balance += 1
            elif ch == "}":
                balance -= 1
            if balance == 0 and i > 0:
                cut_idx = i + 1
        if cut_idx > 0:
            repaired = repaired[:cut_idx]

    return repaired


def parse_json_response(response: str, default=None) -> dict | None:
    text = response.strip()
    if not text:
        return default

    # Stage 1: direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Stage 2: extract from {...} (handles CoT before/after)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                logger.debug("JSON_EXTRACTED_FROM_THINKING start=%d end=%d", start, end)
                return parsed
        except json.JSONDecodeError:
            pass

    # Stage 3: repair common issues and retry
    repaired = _repair_json(text)
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict):
            logger.debug("JSON_REPAIRED")
            return parsed
    except json.JSONDecodeError:
        pass

    # Stage 4: extract from repaired text
    start = repaired.find("{")
    end = repaired.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(repaired[start:end + 1])
            if isinstance(parsed, dict):
                logger.debug("JSON_REPAIRED_AND_EXTRACTED")
                return parsed
        except json.JSONDecodeError:
            pass

    logger.warning("JSON_PARSE_FAIL raw=%s", response[:300])
    return default


def _json_response_error(parsed, required_fields: tuple[str, ...] | None) -> str | None:
    if parsed is None:
        return "previous response was not valid JSON"
    missing = [field for field in (required_fields or ()) if field not in parsed]
    if missing:
        return "missing required JSON fields: " + ", ".join(missing)
    return None


def _json_retry_instruction(error: str) -> str:
    return (
        f"Your previous output was INVALID: {error}. "
        "Output ONLY valid JSON — start with {, end with }. "
        "No markdown fences, no text outside braces. "
        "Keep to at most 12 actions. Use 'repeat' to batch same-direction movement."
    )


def _sanitize_for_log(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, val in value.items():
            if key == "image_url" and isinstance(val, dict) and isinstance(val.get("url"), str):
                url = val["url"]
                sanitized[key] = {"url": f"<image_base64 len={len(url)}>"}
            else:
                sanitized[key] = _sanitize_for_log(val)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_for_log(item) for item in value]
    return value
