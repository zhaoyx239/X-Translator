from __future__ import annotations

import asyncio
import base64
import audioop
from datetime import datetime
import io
import os
import re
import shutil
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import websockets

from .logging_utils import setup_logger
from .zipformer_postprocess import normalize_zipformer_text

logger = setup_logger()

SESSION_AUDIO_RETENTION_COUNT = 10


TARGET_LANGUAGE_NAMES = {
    "af": "Afrikaans",
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "pt": "Portuguese",
    "es": "Spanish",
    "ja": "Japanese",
    "ru": "Russian",
    "ko": "Korean",
    "th": "Thai",
    "it": "Italian",
    "de": "German",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
}

DEFAULT_TRANSLATION_MODELS = {
    "hunyuan": "hunyuan_mt",
    "lmt": "lmt",
}


@dataclass(slots=True)
class SessionState:
    """会话状态。"""

    session_id: str
    source_lang: str
    target_lang: str
    speech_active: bool = False
    stable_text: str = ""
    last_partial_text: str = ""
    pending_translation_text: str = ""
    last_sent_segment: str = ""
    last_emit_at: float = field(default_factory=time.time)


class SessionAudioStore:
    """会话音频存储。"""

    def __init__(
        self,
        session_id: str,
        output_dir: Path,
        sample_rate: int,
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.created_at_label = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.output_root = output_dir
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_root / self.created_at_label
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.session_dir / "input.wav"
        self.clone_path = self.session_dir / "clone_recent.wav"
        self.tts_path = self.session_dir / "output.wav"
        self._buffer = bytearray()
        self._tts_buffer = bytearray()
        self._lock = asyncio.Lock()
        self._input_file_write_lock = asyncio.Lock()
        self._started_at: float | None = None
        self._last_input_write_at = 0.0
        self._input_write_interval_seconds = 1.0

    def mark_started(self) -> None:
        """标记会话开始时间，对齐输入输出时间线。"""
        if self._started_at is None:
            self._started_at = time.perf_counter()

    async def append(self, pcm_bytes: bytes) -> None:
        """追加音频。"""
        if not pcm_bytes:
            return
        async with self._lock:
            self._buffer.extend(pcm_bytes)
            now = time.perf_counter()
            should_write = (
                not self.session_path.exists()
                or now - self._last_input_write_at >= self._input_write_interval_seconds
            )
            if not should_write:
                return
            self._last_input_write_at = now
            pcm_snapshot = bytes(self._buffer)
        loop = asyncio.get_running_loop()
        async with self._input_file_write_lock:
            await loop.run_in_executor(None, self._write_all, pcm_snapshot)

    async def export_recent_clone(self, duration_seconds: float) -> Optional[str]:
        """导出最近片段作为 clone 参考。"""
        async with self._lock:
            if not self._buffer:
                return None
            pcm_bytes = bytes(self._buffer)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._write_recent_clip,
            pcm_bytes,
            duration_seconds,
        )

    def _write_recent_clip(self, pcm_bytes: bytes, duration_seconds: float) -> Optional[str]:
        """写出最近片段的 clone 音频。"""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        clip_samples = int(self.sample_rate * duration_seconds)
        if samples.size == 0 or clip_samples <= 0:
            return None
        clip = samples[-clip_samples:]
        with wave.open(str(self.clone_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(clip.tobytes())
        return os.path.abspath(self.clone_path)

    async def flush(self) -> str:
        """写出整段音频。"""
        async with self._lock:
            pcm_bytes = bytes(self._buffer)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._write_all, pcm_bytes)

    def _write_all(self, pcm_bytes: bytes) -> str:
        """同步写出整段音频。"""
        with wave.open(str(self.session_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        return os.path.abspath(self.session_path)

    async def save_tts_audio(self, pcm_bytes: bytes, sample_rate: int) -> Optional[str]:
        """按会话时间线缓存 TTS 输出音频，保留中间静音段。"""
        if not pcm_bytes:
            return None
        async with self._lock:
            if self._started_at is None:
                self._started_at = time.perf_counter()
            elapsed_seconds = max(0.0, time.perf_counter() - self._started_at)
            current_samples = len(self._tts_buffer) // 2
            target_samples = max(0, round(elapsed_seconds * sample_rate))
            write_start_samples = max(current_samples, target_samples)
            gap_samples = max(0, write_start_samples - current_samples)
            if gap_samples:
                self._tts_buffer.extend(b"\x00\x00" * gap_samples)
            self._tts_buffer.extend(pcm_bytes)
            tts_snapshot = bytes(self._tts_buffer)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_tts_all, tts_snapshot, sample_rate)
        return os.path.abspath(self.tts_path)

    async def flush_tts(self, sample_rate: int) -> Optional[str]:
        """写出会话级完整 TTS 音频。"""
        async with self._lock:
            if not self._tts_buffer:
                return None
            pcm_bytes = bytes(self._tts_buffer)
            started_at = self._started_at
        if started_at is not None:
            elapsed_samples = max(0, round((time.perf_counter() - started_at) * sample_rate))
            current_samples = len(pcm_bytes) // 2
            if elapsed_samples > current_samples:
                pcm_bytes += b"\x00\x00" * (elapsed_samples - current_samples)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._write_tts_all, pcm_bytes, sample_rate)

    async def prune_old_sessions(self, keep: int = SESSION_AUDIO_RETENTION_COUNT) -> None:
        """只保留最近的会话音频目录。"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._prune_old_sessions, keep)

    def _write_tts_all(self, pcm_bytes: bytes, sample_rate: int) -> str:
        with wave.open(str(self.tts_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return os.path.abspath(self.tts_path)

    def _prune_old_sessions(self, keep: int) -> None:
        keep = max(1, int(keep))
        root = self.output_root.resolve()
        if not root.exists():
            return
        session_dirs = [
            path
            for path in root.iterdir()
            if path.is_dir()
            and re.fullmatch(r"\d{8}_\d{6}_\d{3}", path.name)
            and (path / "input.wav").exists()
        ]
        stale_dirs = sorted(session_dirs, key=lambda path: path.name, reverse=True)[keep:]
        for path in stale_dirs:
            resolved = path.resolve()
            if resolved.parent != root:
                continue
            try:
                shutil.rmtree(resolved)
                logger.info("old session audio pruned path=%s", resolved)
            except OSError as exc:
                logger.warning("old session audio prune failed path=%s err=%s", resolved, exc)


class Qwen3ASRService:
    """Qwen3ASR HTTP 客户端。"""

    def __init__(self, base_url: str, sample_rate: int, chunk_seconds: float):
        self.base_url = base_url.rstrip("/")
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.session = requests.Session()

    def recognize_stream(
        self,
        audio: np.ndarray,
        cache: dict,
        is_final: bool,
        sample_rate: int | None = None,
        committed_text: str = "",
    ) -> str:
        """按 session_id 执行流式识别。"""
        wav_bytes = self._pcm_to_wav_bytes(audio)
        audio_data_url = self._to_data_url(wav_bytes)
        if "session_id" not in cache:
            cache["session_id"] = os.urandom(16).hex()
        payload = {
            "audio": audio_data_url,
            "session_id": cache["session_id"],
            "is_final": is_final,
            "sample_rate": sample_rate or self.sample_rate,
            "committed_text": committed_text,
        }
        start_time = time.perf_counter()
        response = self.session.post(self.base_url, json=payload, timeout=30)
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        text = response.json().get("text", "")
        logger.info(
            "qwen3 request final=%s elapsed_ms=%.1f text_len=%s",
            is_final,
            elapsed_ms,
            len((text or "").strip()),
        )
        return text

    def commit_stream(self, cache: dict, committed_text: str, sample_rate: int | None = None) -> str:
        """通知远端提交已翻译前缀并重建流式 state。"""
        if "session_id" not in cache or not committed_text.strip():
            return ""
        silence = np.zeros((0,), dtype=np.float32)
        return self.recognize_stream(
            silence,
            cache,
            is_final=False,
            sample_rate=sample_rate or self.sample_rate,
            committed_text=committed_text,
        )

    def reset_stream(self, cache: dict) -> None:
        """清理本地缓存的流式会话。"""
        cache.clear()

    def _pcm_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """float32 转 wav bytes。"""
        pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def _to_data_url(self, wav_bytes: bytes) -> str:
        """转 data url。"""
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return f"data:audio/wav;base64,{b64}"


class ZipformerASRService:
    """Zipformer websocket 客户端。"""

    def __init__(self, server_uri: str, sample_rate: int):
        self.server_uri = server_uri
        self.sample_rate = sample_rate
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()
        self._started = False
        self._ws = None
        self._request_lock = threading.Lock()

    def recognize_stream(self, audio: np.ndarray) -> str:
        """发送当前 chunk，返回 partial。"""
        pcm = self._float_to_pcm_bytes(audio)
        response = self._run_coro(self._send_audio(pcm))
        return str(response.get("text", "") or "").strip()

    def finalize_stream(self) -> str:
        """结束当前流式轮次，返回 final。"""
        if self._ws is None or not self._started:
            return ""
        response = self._run_coro(self._send_json({"type": "end"}))
        self._started = False
        return str(response.get("text", "") or "").strip()

    def reset_stream(self) -> None:
        """重置远端流式会话。"""
        if self._ws is None:
            self._started = False
            return
        try:
            self._run_coro(self._send_json({"type": "reset"}))
        except Exception as exc:
            logger.warning("zipformer reset failed: %s", exc)
        self._started = False

    def close(self) -> None:
        """关闭后台连接。"""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            self._run_coro(self._close_ws())
        except Exception as exc:
            logger.warning("zipformer close failed: %s", exc)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            return
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread = None
        self._loop = None
        self._loop_ready.clear()
        self._started = False

    def _float_to_pcm_bytes(self, audio: np.ndarray) -> bytes:
        if audio.size == 0:
            return b""
        pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
        return pcm.tobytes()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is not None and loop.is_running():
            return loop

        self._loop_ready.clear()

        def runner() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            self._loop = new_loop
            self._loop_ready.set()
            new_loop.run_forever()
            pending = asyncio.all_tasks(new_loop)
            for task in pending:
                task.cancel()
            if pending:
                new_loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            new_loop.close()

        self._thread = threading.Thread(target=runner, name="zipformer-asr", daemon=True)
        self._thread.start()
        self._loop_ready.wait(timeout=5)
        if self._loop is None:
            raise RuntimeError("zipformer websocket loop failed to start")
        return self._loop

    def _run_coro(self, coro):
        with self._request_lock:
            loop = self._ensure_loop()
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)

    async def _ensure_ws(self):
        if self._ws is None or getattr(self._ws, "closed", False):
            self._ws = await websockets.connect(self.server_uri, max_size=None)
        if not self._started:
            response = await self._exchange_json({"type": "start"}, ws=self._ws)
            if response.get("type") != "started":
                raise RuntimeError(f"zipformer start failed: {response}")
            self._started = True
        return self._ws

    async def _send_json(self, payload: dict) -> dict:
        ws = await self._ensure_ws()
        return await self._exchange_json(payload, ws=ws)

    async def _exchange_json(self, payload: dict, ws=None) -> dict:
        if ws is None:
            ws = await self._ensure_ws()
        await ws.send(_json_dumps(payload))
        message = await ws.recv()
        return _json_loads(message)

    async def _send_audio(self, audio_bytes: bytes) -> dict:
        if not audio_bytes:
            return {"type": "partial", "text": ""}
        ws = await self._ensure_ws()
        await ws.send(audio_bytes)
        message = await ws.recv()
        return _json_loads(message)

    async def _close_ws(self) -> None:
        ws = self._ws
        if ws is not None and not getattr(ws, "closed", False):
            await ws.close()
        self._ws = None


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _json_loads(message: str | bytes) -> dict:
    import json

    if isinstance(message, bytes):
        message = message.decode("utf-8")
    return json.loads(message)


class OpenAIChatTranslateService:
    """OpenAI chat-completions 兼容的 MT HTTP 客户端。"""

    def __init__(
        self,
        api_url: str,
        provider: str = "hunyuan",
        model: str = "",
        max_tokens: int = 256,
    ):
        self.api_url = api_url
        self.provider = (provider or "hunyuan").strip().lower().replace("_", "-")
        self.model = model.strip() if model else DEFAULT_TRANSLATION_MODELS.get(
            self.provider,
            self.provider,
        )
        self.max_tokens = max_tokens
        self.session = requests.Session()

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """执行翻译。"""
        normalized_text = text.strip()
        if not normalized_text:
            return ""

        source_language = TARGET_LANGUAGE_NAMES.get(source_lang, source_lang)
        target_language = TARGET_LANGUAGE_NAMES.get(target_lang, target_lang)
        messages = self._build_messages(
            normalized_text,
            source_language=source_language,
            target_language=target_language,
        )
        # prompt = (
        #     "You are a real-time speech translation assistant for everyday conversation. "
        # f"Translate into concise, natural, reliable {target_language}. "
        # "Before translating, silently refine the ASR text with minimal intervention. "
        # "Keep the speaker's final intended meaning only. "
        # "Apply these rules internally: "
        # "remove filler words, stutters, obvious repetitions, and noise; "
        # "resolve simple ASR breakage or malformed phrases when the intent is clear; "
        # "apply corrections or spelling clarifications to recover the intended wording, then omit the meta correction language; "
        # "preserve tone, politeness, and interpersonal style; "
        # "prefer short spoken phrasing over formal or written phrasing; "
        # "do not add unsupported information; "
        # "if part of the input is uncertain, translate conservatively rather than inventing details. "
        # f"If the input is already in {target_language}, output it unchanged. "
        # "Output translation only, with no explanation or notes."
        # )
        payload = {
            "model": self.model,
            "messages": messages,
            "top_k": 20,
            "top_p": 0.6,
            "temperature": 0.0 if self.provider == "lmt" else 0.7,
            "max_tokens": self.max_tokens,
        }
        start_time = time.perf_counter()
        response = self.session.post(self.api_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        text_out = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        text_out = self._normalize_output(text_out, target_language=target_language)
        logger.info(
            "mt request provider=%s model=%s elapsed_ms=%.1f input_len=%s output_len=%s",
            self.provider,
            self.model,
            (time.perf_counter() - start_time) * 1000,
            len(normalized_text),
            len(text_out),
        )
        return text_out

    def _build_messages(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str,
    ) -> list[dict[str, str]]:
        if self.provider == "lmt":
            prompt = (
                f"Translate the following text from {source_language} into {target_language}.\n"
                f"{source_language}: {text}\n"
                f"{target_language}:"
            )
            return [{"role": "user", "content": prompt}]

        prompt = (
            "You are a real-time speech translator. "
            f"If the input language is already {target_language}, output the input text unchanged. "
            f"Translate the input into natural, fluent {target_language}. "
            "Output translation only. Do not explain."
        )
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

    def _normalize_output(self, text: str, *, target_language: str) -> str:
        if self.provider != "lmt":
            return text
        label_pattern = rf"^\s*{re.escape(target_language)}\s*[:：]\s*"
        text = re.sub(label_pattern, "", text, count=1, flags=re.IGNORECASE).strip()
        return text.strip("\"' \n\t")


class HunyuanTranslateService(OpenAIChatTranslateService):
    """兼容旧名字的 Hunyuan MT 客户端。"""

    def __init__(
        self,
        api_url: str,
        provider: str = "hunyuan",
        model: str = "",
        max_tokens: int = 256,
    ):
        super().__init__(
            api_url=api_url,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
        )


class IndexTTSService:
    """IndexTTS HTTP 客户端。"""

    def __init__(self, api_url: str, sample_rate: int = 48000):
        self.api_url = api_url
        self.sample_rate = sample_rate
        self.session = requests.Session()
        self.default_audio_paths: list[str] = []

    def synthesize(self, text: str, clone_audio_path: Optional[str]) -> bytes:
        """执行合成。"""
        payload = {
            "text": text,
            "audio_paths": [clone_audio_path] if clone_audio_path else self.default_audio_paths,
        }
        start_time = time.perf_counter()
        response = self.session.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        pcm = self._wav_bytes_to_pcm(response.content)
        logger.info(
            "tts request elapsed_ms=%.1f text_len=%s audio_bytes=%s clone=%s",
            (time.perf_counter() - start_time) * 1000,
            len(text.strip()),
            len(pcm),
            clone_audio_path,
        )
        return pcm

    def _wav_bytes_to_pcm(self, wav_bytes: bytes) -> bytes:
        """将 TTS 返回的 WAV 转为 16bit PCM 48k 单声道。"""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if channels > 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            channels = 1

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2

        if sample_rate != self.sample_rate:
            frames, _ = audioop.ratecv(
                frames,
                sample_width,
                channels,
                sample_rate,
                self.sample_rate,
                None,
            )

        return frames


class OmniTTSService:
    """OmniVoice TTS HTTP 客户端。"""

    def __init__(
        self,
        api_url: str,
        asr_service: Optional[Qwen3ASRService | ZipformerASRService] = None,
        asr_sample_rate: int = 16000,
        output_sample_rate: int = 24000,
        target_lang: Optional[str] = None,
    ):
        self.api_url = api_url
        self.asr_service = asr_service
        self.asr_sample_rate = asr_sample_rate
        self.sample_rate = output_sample_rate
        self.target_lang = target_lang
        self.session = requests.Session()

    def synthesize(
        self,
        text: str,
        clone_audio_path: Optional[str],
        ref_text: Optional[str] = None,
    ) -> bytes:
        """执行合成，返回 16-bit PCM bytes。"""
        if clone_audio_path and ref_text is None and self.asr_service is not None:
            try:
                ref_text = self._transcribe_ref_audio(clone_audio_path)
                logger.info(
                    "omni tts asr fallback path=%s ref_text_len=%s",
                    clone_audio_path,
                    len((ref_text or "").strip()),
                )
            except Exception as exc:
                logger.warning("omni tts asr fallback failed: %s", exc)
                ref_text = None

        payload = {
            "text": text,
            "audio_paths": [clone_audio_path] if clone_audio_path else [],
            "ref_text": ref_text,
            "language": self.target_lang,
        }
        start_time = time.perf_counter()
        response = self.session.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        pcm = self._wav_bytes_to_pcm(response.content)
        logger.info(
            "omni tts request elapsed_ms=%.1f text_len=%s audio_bytes=%s clone=%s",
            (time.perf_counter() - start_time) * 1000,
            len(text.strip()),
            len(pcm),
            clone_audio_path,
        )
        return pcm

    def close(self) -> None:
        """关闭 TTS 专用的辅助资源。"""
        if hasattr(self.asr_service, "close"):
            self.asr_service.close()

    def _transcribe_ref_audio(self, audio_path: str) -> str:
        """读取参考音频文件并调用 ASR 得到转写文本。"""
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            file_sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if channels > 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            channels = 1

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2

        if file_sample_rate != self.asr_sample_rate:
            frames, _ = audioop.ratecv(
                frames,
                sample_width,
                channels,
                file_sample_rate,
                self.asr_sample_rate,
                None,
            )

        pcm = np.frombuffer(frames, dtype=np.int16)
        audio_float = pcm.astype(np.float32) / 32768.0

        if isinstance(self.asr_service, Qwen3ASRService):
            text = self.asr_service.recognize_stream(
                audio_float,
                cache={},
                is_final=True,
                sample_rate=self.asr_sample_rate,
            )
        elif isinstance(self.asr_service, ZipformerASRService):
            self.asr_service.recognize_stream(audio_float)
            text = self.asr_service.finalize_stream()
        else:
            text = ""

        if isinstance(self.asr_service, ZipformerASRService):
            return normalize_zipformer_text(text)
        return (text or "").strip()

    def _wav_bytes_to_pcm(self, wav_bytes: bytes) -> bytes:
        """将 OmniVoice 返回的 WAV 转为 16-bit PCM，并重采样到配置采样率。"""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            src_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if channels > 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            channels = 1

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2

        if src_rate != self.sample_rate:
            frames, _ = audioop.ratecv(
                frames,
                sample_width,
                channels,
                src_rate,
                self.sample_rate,
                None,
            )

        return frames


class XVoiceTTSService:
    """X-Voice TTS HTTP 客户端。"""

    def __init__(
        self,
        api_url: str,
        output_sample_rate: int = 48000,
        target_lang: Optional[str] = None,
    ):
        self.api_url = api_url
        self.sample_rate = output_sample_rate
        self.target_lang = target_lang
        self.session = requests.Session()

    def synthesize(self, text: str, clone_audio_path: Optional[str]) -> bytes:
        """执行合成，返回 16-bit PCM bytes。"""
        payload = {
            "text": text,
            "audio_paths": [clone_audio_path] if clone_audio_path else [],
            "language": self.target_lang,
        }
        start_time = time.perf_counter()
        response = self.session.post(self.api_url, json=payload, timeout=90)
        response.raise_for_status()
        pcm = self._wav_bytes_to_pcm(response.content)
        logger.info(
            "xvoice tts request elapsed_ms=%.1f text_len=%s audio_bytes=%s clone=%s",
            (time.perf_counter() - start_time) * 1000,
            len(text.strip()),
            len(pcm),
            clone_audio_path,
        )
        return pcm

    def _wav_bytes_to_pcm(self, wav_bytes: bytes) -> bytes:
        """将 X-Voice 返回的 WAV 转为 16-bit PCM，重采样到配置采样率。"""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            src_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        if channels > 1:
            frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
            channels = 1

        if sample_width != 2:
            frames = audioop.lin2lin(frames, sample_width, 2)
            sample_width = 2

        if src_rate != self.sample_rate:
            frames, _ = audioop.ratecv(
                frames,
                sample_width,
                channels,
                src_rate,
                self.sample_rate,
                None,
            )

        return frames


def pcm_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """PCM16 转 float32。"""
    if not audio_bytes:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def split_text_with_punctuation(text: str) -> list[str]:
    """按中英、数字和标点切分，并保留标点。"""
    import re

    return re.findall(
        r"[\u4e00-\u9fff]|[A-Za-z]+|[0-9]+|[，。！？；：、,.!?;:…]",
        text,
    )


def check_en(text: str) -> bool:
    """检查 token 是否为英文。"""
    import re

    if not text:
        return False
    symbol_pattern = re.compile(
        r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
        r"\u2000-\u206F\u3000-\u303F\uFF00-\uFFEF]"
    )
    for char in reversed(text):
        if char.isdigit() or symbol_pattern.match(char):
            continue
        if "\u4e00" <= char <= "\u9fff":
            return False
        return True
    return True


def join_tokens(tokens: list[str]) -> str:
    """将 token 拼回文本。"""
    pieces: list[str] = []
    for token in tokens:
        if not pieces:
            pieces.append(token)
            continue
        if check_en(token) and check_en(pieces[-1][-1:]):
            pieces.append(f" {token}")
        elif check_en(token) and pieces[-1] and pieces[-1][-1].isalnum():
            pieces.append(f" {token}")
        else:
            pieces.append(token)
    return "".join(pieces).strip()


def normalize_token(token: str) -> str:
    """token 级归一化，仅用于比较。"""
    if check_en(token):
        return token.lower().strip()
    return token.strip()


def longest_common_prefix_len(left: list[str], right: list[str]) -> int:
    """获取 token 级最长公共前缀长度。"""
    size = min(len(left), len(right))
    idx = 0
    while idx < size and normalize_token(left[idx]) == normalize_token(right[idx]):
        idx += 1
    return idx


def stabilize_partial_text(previous_stable: str, current_partial: str) -> tuple[str, str]:
    """根据稳定前缀计算稳定文本和增量。"""
    stable_tokens = split_text_with_punctuation(previous_stable)
    partial_tokens = split_text_with_punctuation(current_partial)
    prefix_len = longest_common_prefix_len(stable_tokens, partial_tokens)

    if prefix_len < len(stable_tokens):
        stable_tokens = stable_tokens[:prefix_len]

    if len(partial_tokens) > len(stable_tokens):
        appended_tokens = partial_tokens[len(stable_tokens) :]
        next_stable_tokens = stable_tokens + appended_tokens
        delta_tokens = appended_tokens
    else:
        next_stable_tokens = stable_tokens
        delta_tokens = []

    next_stable = join_tokens(next_stable_tokens)
    delta = join_tokens(delta_tokens)
    return next_stable, delta


def should_emit_segment(text: str, min_chars: int) -> bool:
    """按简单规则判断是否应当翻译。"""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    return stripped.endswith(("。", "，", ",", ".", "!", "！", "?", "？", ";", "；", ":"))
