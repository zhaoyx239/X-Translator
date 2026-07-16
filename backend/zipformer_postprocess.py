from __future__ import annotations

import re
import threading

from .logging_utils import setup_logger

logger = setup_logger()

_CJK_CHAR = r"\u4e00-\u9fff"
_TOKEN_RE = re.compile(rf"[{_CJK_CHAR}]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]")


def _contains_chinese(text: str) -> bool:
    return bool(re.search(rf"[{_CJK_CHAR}]", text or ""))


def _normalize_spacing(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)
    value = re.sub(rf"(?<=[{_CJK_CHAR}])\s+", "", value)
    value = re.sub(rf"\s+(?=[{_CJK_CHAR}])", "", value)
    value = re.sub(r"\s+([,.;:!?，。！？；：])", r"\1", value)
    value = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", value)
    return value.strip()


def _capitalize_sentence_head(text: str) -> str:
    if not text:
        return ""
    match = re.match(r'^([\s"\'“‘(\[]*)([a-z])', text)
    if match is None:
        return text
    prefix, first_char = match.groups()
    return f"{prefix}{first_char.upper()}{text[len(prefix) + 1:]}"


def normalize_zipformer_text(text: str) -> str:
    value = _normalize_spacing((text or "").lower())
    if not value:
        return ""

    pieces: list[str] = []
    prev_ascii = False
    for token in _TOKEN_RE.findall(value):
        is_ascii = bool(re.fullmatch(r"[a-z0-9]+(?:'[a-z]+)?", token))
        if pieces and prev_ascii and is_ascii:
            pieces.append(" ")
        pieces.append(token)
        prev_ascii = is_ascii
    return _capitalize_sentence_head("".join(pieces).strip())


class ZipformerTextPostProcessor:
    """Zipformer 专用文本后处理。"""

    DEFAULT_MODEL = "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    DEFAULT_MODEL_REVISION = "v2.0.4"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        model_revision: str = DEFAULT_MODEL_REVISION,
    ):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        self._model = model
        self._model_revision = model_revision
        self._lock = threading.Lock()
        self._pipeline = pipeline(
            task=Tasks.punctuation,
            model=model,
            model_revision=model_revision,
        )

    def process(self, text: str, *, apply_punctuation: bool = True) -> str:
        normalized = normalize_zipformer_text(text)
        if not normalized:
            return ""
        if not apply_punctuation or not _contains_chinese(normalized):
            return normalized

        with self._lock:
            cache: dict = {}
            result = self._pipeline(normalized, cache=cache)
        if result and isinstance(result, list):
            punctuated = str(result[0].get("text", "") or "").strip()
        else:
            punctuated = normalized
        return normalize_zipformer_text(punctuated)

    @property
    def model(self) -> str:
        return self._model

    @property
    def model_revision(self) -> str:
        return self._model_revision
