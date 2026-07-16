from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from .asr_backends import (
    ASRStreamingConfig,
    LocalASRStreamingRecognizer,
    ParaformerASR,
    Qwen3ASR,
    Qwen3StreamingRecognizer,
    SensevoiceASR,
    ZipformerASR,
    ZipformerStreamingRecognizer,
)
from .config import Settings
from .logging_utils import setup_logger
from .services import (
    IndexTTSService,
    OmniTTSService,
    OpenAIChatTranslateService,
    Qwen3ASRService,
    SessionAudioStore,
    SessionState,
    XVoiceTTSService,
    ZipformerASRService,
    pcm_bytes_to_float32,
    should_emit_segment,
)
from .speaker_prompt import SpeakerChangeEvent, SpeakerPromptTracker
from .zipformer_postprocess import ZipformerTextPostProcessor, normalize_zipformer_text

logger = setup_logger()

FILLER_WORDS = {
    "嗯",
    "啊",
    "呃",
    "额",
    "哦",
    "噢",
    "唉",
    "哎",
}
FILLER_PUNCTUATION = "，。！？；：、,.!?;:~… "
FILLER_WORD_PATTERN = "|".join(
    re.escape(word) for word in sorted(FILLER_WORDS, key=len, reverse=True)
)
FILLER_PREFIX_RE = re.compile(
    rf"^(?:[{re.escape(FILLER_PUNCTUATION)}]*(?:{FILLER_WORD_PATTERN}))+"
    rf"[{re.escape(FILLER_PUNCTUATION)}]*"
)
FILLER_SUFFIX_RE = re.compile(
    rf"[{re.escape(FILLER_PUNCTUATION)}]*(?:(?:{FILLER_WORD_PATTERN})"
    rf"[{re.escape(FILLER_PUNCTUATION)}]*)+$"
)


def build_latency_metrics(
    asr_elapsed_ms: float,
    mt_elapsed_ms: float,
    tts_elapsed_ms: float,
) -> dict[str, int]:
    """构建面向前端展示的延时指标。"""
    asr_ms = max(0, round(asr_elapsed_ms))
    return {
        "asr_ms": int(asr_ms),
        "mt_ms": int(max(0, round(mt_elapsed_ms))),
        "tts_ms": int(max(0, round(tts_elapsed_ms))),
    }


class TranslationSession:
    """同传会话控制器。"""

    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        shared_local_asr: Optional[ParaformerASR | SensevoiceASR] = None,
        zipformer_postprocessor: ZipformerTextPostProcessor | None = None,
    ):
        self.websocket = websocket
        self.settings = settings
        self.zipformer_postprocessor = zipformer_postprocessor
        self.state = SessionState(
            session_id=str(uuid.uuid4()),
            source_lang=settings.source_lang,
            target_lang=settings.target_lang,
        )
        self.sample_rate = settings.sample_rate
        self.translation_queue: asyncio.Queue[dict | str] = asyncio.Queue()
        self.audio_store = SessionAudioStore(
            session_id=self.state.session_id,
            output_dir=settings.session_audio_dir,
            sample_rate=settings.sample_rate,
        )
        self.speaker_prompt_tracker: SpeakerPromptTracker | None = None
        if settings.speaker_prompt_enabled:
            self.speaker_prompt_tracker = SpeakerPromptTracker(
                session_id=self.state.session_id,
                output_dir=settings.session_audio_dir,
                sample_rate=settings.sample_rate,
                session_label=self.audio_store.created_at_label,
                target_seconds=settings.clone_window_seconds,
                min_seconds=settings.speaker_prompt_min_seconds,
                max_seconds=settings.speaker_prompt_max_seconds,
                frame_ms=settings.speaker_vad_frame_ms,
                silence_dbfs=settings.speaker_silence_dbfs,
                vad_margin_db=settings.speaker_vad_margin_db,
                embedding_provider=settings.speaker_embedding_provider,
                embedding_model=settings.speaker_embedding_model,
                embedding_window_seconds=settings.speaker_embedding_window_seconds,
                embedding_hop_seconds=settings.speaker_embedding_hop_seconds,
                speaker_change_threshold=settings.speaker_change_threshold,
            )

        self._audio_store_tasks: set[asyncio.Task] = set()
        self._asr_task: Optional[asyncio.Task] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._speaker_task: Optional[asyncio.Task] = None
        self._speaker_audio_queue: asyncio.Queue[bytes | None] | None = (
            asyncio.Queue() if self.speaker_prompt_tracker is not None else None
        )
        self._closed = False
        self._asr_flush_requested = False
        self._asr_boundary_flush_requested = False
        self._asr_boundary_flush_future: Optional[asyncio.Future[None]] = None
        self._asr_boundary_flush_bytes: int | None = None
        self._asr_stop_received = False
        self._forced_prompt_speaker_id: str | None = None
        self._last_asr_emit = ""
        self._last_asr_type = ""
        self._asr_audio_event: Optional[asyncio.Event] = None
        self._asr_audio_lock: Optional[asyncio.Lock] = None
        self._asr_pending_audio = bytearray()
        self._asr_bytes_per_sample = 2
        selected_chunk_size = (
            settings.zipformer_chunk_size
            if settings.asr_provider == "zipformer"
            else settings.paraformer_chunk_size
        )
        self._asr_chunk_bytes = selected_chunk_size * self._asr_bytes_per_sample
        self._current_audio_started_at: float | None = None
        self._current_audio_bytes = 0
        self._session_audio_bytes_total = 0
        self._current_turn_started_audio_ms = 0.0
        self._current_turn_committed_audio_ms = 0
        self._current_turn_id = ""
        self._zipformer_prev_raw_text = ""
        self._zipformer_last_chunk_text = ""
        self._zipformer_no_change_count = 0
        self._zipformer_unpunctuated_buffer = ""
        self._zipformer_translation_buffer = ""
        self._zipformer_use_ctpunc = (
            settings.asr_provider == "zipformer"
            and settings.zipformer_use_ctpunc
            and self.zipformer_postprocessor is not None
        )

        self.asr_recognizer: Optional[
            LocalASRStreamingRecognizer | Qwen3StreamingRecognizer | ZipformerStreamingRecognizer
        ] = None
        asr_config = ASRStreamingConfig(
            sample_rate=settings.sample_rate,
            chunk_size=selected_chunk_size,
            asr_window_sec=(
                settings.zipformer_window_seconds
                if settings.asr_provider == "zipformer"
                else settings.paraformer_window_seconds
            ),
            init_turn_sec=(
                settings.zipformer_init_turn_seconds
                if settings.asr_provider == "zipformer"
                else settings.paraformer_init_turn_seconds
            ),
            device=settings.paraformer_device,
            model=settings.paraformer_model,
            hub=settings.paraformer_hub,
        )
        if settings.asr_provider == "qwen3":
            self.asr_recognizer = Qwen3StreamingRecognizer(
                asr_config,
                Qwen3ASR(
                    service=Qwen3ASRService(
                        base_url=settings.qwen3_asr_url,
                        sample_rate=settings.sample_rate,
                        chunk_seconds=settings.asr_chunk_seconds,
                    ),
                    sample_rate=settings.sample_rate,
                ),
            )
        elif settings.asr_provider == "zipformer":
            self.asr_recognizer = ZipformerStreamingRecognizer(
                asr_config,
                ZipformerASR(
                    service=ZipformerASRService(
                        server_uri=settings.zipformer_server_uri,
                        sample_rate=settings.sample_rate,
                    ),
                    sample_rate=settings.sample_rate,
                ),
            )
        else:
            self.asr_recognizer = LocalASRStreamingRecognizer(asr_config, asr=shared_local_asr)
        self._asr_audio_event = asyncio.Event()
        self._asr_audio_lock = asyncio.Lock()

        translation_provider = (
            settings.translation_provider or "hunyuan"
        ).strip().lower().replace("_", "-")
        translation_url = (
            settings.lmt_url if translation_provider == "lmt" else settings.hunyuan_url
        )
        self.mt_service = OpenAIChatTranslateService(
            translation_url,
            provider=translation_provider,
            model=settings.translation_model,
            max_tokens=settings.translation_max_tokens,
        )
        tts_provider = (settings.tts_provider or "index").strip().lower().replace("_", "-")
        if tts_provider == "omni":
            tts_asr_service: Qwen3ASRService | ZipformerASRService | None = None
            if settings.asr_provider == "qwen3":
                tts_asr_service = Qwen3ASRService(
                    base_url=settings.qwen3_asr_url,
                    sample_rate=settings.sample_rate,
                    chunk_seconds=settings.asr_chunk_seconds,
                )
            elif settings.asr_provider == "zipformer":
                tts_asr_service = ZipformerASRService(
                    server_uri=settings.zipformer_server_uri,
                    sample_rate=settings.sample_rate,
                )
            self.tts_service: IndexTTSService | OmniTTSService | XVoiceTTSService = OmniTTSService(
                api_url=settings.omni_tts_url,
                asr_service=tts_asr_service,
                asr_sample_rate=settings.sample_rate,
                output_sample_rate=settings.tts_sample_rate,
                target_lang=settings.target_lang,
            )
        elif tts_provider in {"xvoice", "x-voice"}:
            self.tts_service = XVoiceTTSService(
                api_url=settings.xvoice_tts_url,
                output_sample_rate=settings.tts_sample_rate,
                target_lang=settings.target_lang,
            )
        else:
            if tts_provider != "index":
                logger.warning("unknown tts provider=%s, fallback to index", settings.tts_provider)
            self.tts_service = IndexTTSService(settings.index_tts_url)
            self.tts_service.sample_rate = settings.tts_sample_rate
    async def run(self) -> None:
        """运行会话。"""
        await self.websocket.accept()
        self.audio_store.mark_started()
        logger.info(
            "session start id=%s asr=%s src=%s dst=%s",
            self.state.session_id,
            self.settings.asr_provider,
            self.state.source_lang,
            self.state.target_lang,
        )
        if not await self._send_json(
            {
                "action": "session_ready",
                "data": {
                    "session_id": self.state.session_id,
                    "source_lang": self.state.source_lang,
                    "target_lang": self.state.target_lang,
                },
            }
        ):
            return

        if self.asr_recognizer is not None:
            self._asr_task = asyncio.create_task(self._asr_loop())
        if self.speaker_prompt_tracker is not None and self._speaker_audio_queue is not None:
            self._speaker_task = asyncio.create_task(self._speaker_loop())
        self._tts_task = asyncio.create_task(self._tts_loop())

        try:
            while True:
                message = await self.websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    break
                if message_type != "websocket.receive":
                    continue
                if message.get("text"):
                    await self._handle_text(message["text"])
                elif message.get("bytes") is not None:
                    await self._handle_audio(message["bytes"])
        except WebSocketDisconnect:
            logger.info("websocket disconnected: %s", self.state.session_id)
        except RuntimeError as exc:
            if "disconnect message has been received" in str(exc):
                logger.info("websocket disconnected: %s", self.state.session_id)
            else:
                raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """关闭会话。"""
        if self._closed:
            return
        self._closed = True

        if self._speaker_audio_queue is not None:
            await self._speaker_audio_queue.put(None)
        if self._speaker_task and not self._speaker_task.done():
            self._speaker_task.cancel()
            try:
                await self._speaker_task
            except asyncio.CancelledError:
                pass

        if self.asr_recognizer is not None:
            self._asr_flush_requested = True
            if self._asr_audio_event is not None:
                self._asr_audio_event.set()

        if self._asr_task and not self._asr_task.done():
            try:
                await self._asr_task
            except asyncio.CancelledError:
                pass

        await self.translation_queue.put("")
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try:
                await self._tts_task
            except asyncio.CancelledError:
                pass

        if self._audio_store_tasks:
            results = await asyncio.gather(*self._audio_store_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("audio append task failed: %s", result)

        zipformer_asr = getattr(self.asr_recognizer, "asr", None)
        if zipformer_asr is not None and hasattr(zipformer_asr, "close"):
            zipformer_asr.close()
        if hasattr(self.tts_service, "close"):
            self.tts_service.close()

        session_audio_path = await self.audio_store.flush()
        tts_audio_path = await self.audio_store.flush_tts(self.settings.tts_sample_rate)
        await self.audio_store.prune_old_sessions()
        logger.info("session audio saved: %s", session_audio_path)
        if tts_audio_path:
            logger.info("session output audio saved: %s", tts_audio_path)

    async def _handle_text(self, raw_text: str) -> None:
        """处理控制消息。"""
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return

        action = payload.get("action") or payload.get("type")
        event = payload.get("event")
        if action == "session_config":
            self.state.source_lang = payload.get("source_lang", self.state.source_lang)
            self.state.target_lang = payload.get("target_lang", self.state.target_lang)
            if hasattr(self.tts_service, "target_lang"):
                self.tts_service.target_lang = self.state.target_lang
            sample_rate = payload.get("sample_rate")
            if sample_rate:
                self.sample_rate = int(sample_rate)
            logger.info(
                "session config id=%s src=%s dst=%s sample_rate=%s",
                self.state.session_id,
                self.state.source_lang,
                self.state.target_lang,
                self.sample_rate,
            )
            return

        if event == "config_audio":
            sample_rate = payload.get("sample_rate")
            if sample_rate:
                self.sample_rate = int(sample_rate)
            logger.info(
                "asr audio config id=%s sample_rate=%s",
                self.state.session_id,
                self.sample_rate,
            )
        elif event == "stop":
            logger.info("session stop requested id=%s", self.state.session_id)
            self._asr_stop_received = True
            self._asr_flush_requested = True
            if self._asr_audio_event is not None:
                self._asr_audio_event.set()
        elif event == "turn_started":
            self._current_turn_id = str(payload.get("turn_id") or "").strip()

    async def _handle_audio(self, audio_bytes: bytes) -> None:
        """处理音频帧。"""
        if not audio_bytes:
            return

        audio_start_ms = self._session_audio_duration_ms()
        self._session_audio_bytes_total += len(audio_bytes)
        self._schedule_audio_store(audio_bytes)

        if self.asr_recognizer is not None:
            await self._buffer_asr_audio_for_turn(audio_bytes, audio_start_ms)

        if self._speaker_audio_queue is not None:
            self._speaker_audio_queue.put_nowait(audio_bytes)

        if self.asr_recognizer is not None:
            return

    async def _speaker_loop(self) -> None:
        if self.speaker_prompt_tracker is None or self._speaker_audio_queue is None:
            return
        try:
            while True:
                audio_bytes = await self._speaker_audio_queue.get()
                if audio_bytes is None:
                    break
                speaker_change_events = await self.speaker_prompt_tracker.feed(audio_bytes)
                for event in speaker_change_events:
                    await self._handle_speaker_change_event(event)
        except asyncio.CancelledError:
            pass

    async def _buffer_asr_audio_for_turn(self, audio_bytes: bytes, audio_start_ms: float) -> None:
        if not audio_bytes:
            return
        if self._current_audio_started_at is None:
            self._current_audio_started_at = time.perf_counter()
            self._current_turn_started_audio_ms = audio_start_ms
        self._current_audio_bytes += len(audio_bytes)
        await self._buffer_asr_audio(audio_bytes)

    async def _handle_speaker_change_event(self, event: SpeakerChangeEvent) -> None:
        logger.info(
            "speaker boundary id=%s previous=%s current=%s boundary_ms=%.0f detected_at_ms=%.0f",
            self.state.session_id,
            event.previous_speaker_id,
            event.current_speaker_id,
            event.boundary_ms,
            event.detected_at_ms,
        )
        self._forced_prompt_speaker_id = event.previous_speaker_id
        await self._request_asr_boundary_flush()

    async def _request_asr_boundary_flush(self) -> None:
        if (
            self.asr_recognizer is None
            or self._asr_audio_event is None
            or self._asr_task is None
            or self._asr_task.done()
        ):
            self._reset_asr_turn()
            return
        pending_bytes = await self._pending_asr_audio_bytes()
        if self._current_audio_bytes <= 0 and pending_bytes <= 0:
            self._reset_asr_turn()
            return

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._asr_boundary_flush_future = future
        self._asr_boundary_flush_bytes = pending_bytes
        self._asr_boundary_flush_requested = True
        self._asr_audio_event.set()
        await future

    async def _pending_asr_audio_bytes(self) -> int:
        if self._asr_audio_lock is None:
            return 0
        async with self._asr_audio_lock:
            return len(self._asr_pending_audio)

    def _byte_offset_for_audio_ms(self, elapsed_ms: float, total_bytes: int) -> int:
        if self.sample_rate <= 0:
            return 0
        samples = round(max(0.0, elapsed_ms) / 1000 * self.sample_rate)
        offset = max(0, min(total_bytes, samples * self._asr_bytes_per_sample))
        return offset - (offset % self._asr_bytes_per_sample)

    def _audio_ms_for_byte_offset(self, byte_offset: int) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return byte_offset / (self.sample_rate * self._asr_bytes_per_sample) * 1000

    def _schedule_audio_store(self, audio_bytes: bytes) -> None:
        """异步保存会话音频，避免阻塞热路径。"""
        task = asyncio.create_task(self.audio_store.append(audio_bytes))
        self._audio_store_tasks.add(task)
        task.add_done_callback(self._audio_store_tasks.discard)

    async def _buffer_asr_audio(self, audio_bytes: bytes) -> None:
        """缓存最新音频，交给 ASR worker 异步消费。"""
        if self._asr_stop_received:
            return
        if self._asr_audio_lock is None or self._asr_audio_event is None:
            return
        async with self._asr_audio_lock:
            self._asr_pending_audio.extend(audio_bytes)
        self._asr_audio_event.set()

    async def _asr_loop(self) -> None:
        """ASR worker，始终处理当前累计到的最新音频。"""
        if self.asr_recognizer is None:
            return

        try:
            while True:
                merged_audio = await self._next_asr_chunk()
                if merged_audio is None:
                    if self._asr_flush_requested:
                        break
                    if self._asr_boundary_flush_requested:
                        await self._flush_asr_finalize()
                        self._asr_boundary_flush_requested = False
                        self._asr_boundary_flush_bytes = None
                        future = self._asr_boundary_flush_future
                        self._asr_boundary_flush_future = None
                        if future is not None and not future.done():
                            future.set_result(None)
                    continue
                await self._process_asr_chunk(merged_audio)

            if self._asr_flush_requested:
                await self._flush_asr_finalize()
        except asyncio.CancelledError:
            pass

    async def _next_asr_chunk(self) -> bytes | None:
        """读取当前累计音频块。"""
        if self._asr_audio_event is None or self._asr_audio_lock is None:
            return None
        while True:
            await self._asr_audio_event.wait()
            async with self._asr_audio_lock:
                if not self._asr_pending_audio:
                    self._asr_audio_event.clear()
                    if self._asr_flush_requested:
                        return None
                    if self._asr_boundary_flush_requested:
                        return None
                    continue

                if (
                    self._asr_boundary_flush_requested
                    and self._asr_boundary_flush_bytes is not None
                    and self._asr_boundary_flush_bytes <= 0
                ):
                    self._asr_audio_event.set()
                    return None

                dropped_bytes = 0
                if self.settings.asr_provider == "zipformer":
                    max_step_bytes = self._asr_chunk_bytes
                    max_backlog_bytes = self._asr_chunk_bytes * 64
                else:
                    max_step_bytes = self._asr_chunk_bytes
                    max_backlog_bytes = self._asr_chunk_bytes * 2

                if len(self._asr_pending_audio) > max_backlog_bytes:
                    dropped_bytes = len(self._asr_pending_audio) - max_backlog_bytes
                    del self._asr_pending_audio[:dropped_bytes]

                if (
                    len(self._asr_pending_audio) < self._asr_chunk_bytes
                    and not self._asr_flush_requested
                    and not self._asr_boundary_flush_requested
                ):
                    self._asr_audio_event.clear()
                    continue

                take_bytes = min(len(self._asr_pending_audio), max_step_bytes)
                if (
                    self._asr_boundary_flush_requested
                    and self._asr_boundary_flush_bytes is not None
                ):
                    take_bytes = min(take_bytes, self._asr_boundary_flush_bytes)
                if self.settings.asr_provider == "zipformer":
                    take_bytes -= take_bytes % self._asr_chunk_bytes
                    if take_bytes <= 0 and (
                        self._asr_flush_requested or self._asr_boundary_flush_requested
                    ):
                        if (
                            self._asr_boundary_flush_requested
                            and self._asr_boundary_flush_bytes is not None
                        ):
                            take_bytes = min(
                                len(self._asr_pending_audio),
                                self._asr_boundary_flush_bytes,
                            )
                        else:
                            take_bytes = len(self._asr_pending_audio)
                if take_bytes <= 0:
                    return None
                merged_audio = bytes(self._asr_pending_audio[:take_bytes])
                del self._asr_pending_audio[:take_bytes]
                if (
                    self._asr_boundary_flush_requested
                    and self._asr_boundary_flush_bytes is not None
                ):
                    self._asr_boundary_flush_bytes = max(
                        0,
                        self._asr_boundary_flush_bytes - take_bytes,
                    )
                remaining_bytes = len(self._asr_pending_audio)
                if remaining_bytes:
                    self._asr_audio_event.set()
                else:
                    self._asr_audio_event.clear()
            break
        if dropped_bytes:
            logger.warning(
                "asr backlog trimmed id=%s dropped_bytes=%s remaining_bytes=%s",
                self.state.session_id,
                dropped_bytes,
                remaining_bytes,
            )
        logger.info(
            "asr latest chunk id=%s bytes=%s remaining_bytes=%s",
            self.state.session_id,
            len(merged_audio),
            remaining_bytes,
        )
        return merged_audio

    async def _process_asr_chunk(self, audio_bytes: bytes) -> None:
        """执行一次 ASR 推理。"""
        if self.asr_recognizer is None:
            return

        audio_float = pcm_bytes_to_float32(audio_bytes)
        start_time = time.perf_counter()
        outputs = await self._run_asr_feed(audio_float)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        output_types = ",".join(str(item.get("type", "")) for item in outputs)
        logger.info(
            "asr feed id=%s samples=%s outputs=%s elapsed_ms=%.1f",
            self.state.session_id,
            audio_float.size,
            output_types or "none",
            elapsed_ms,
        )
        for result in outputs:
            await self._handle_asr_output(result, processing_elapsed_ms=elapsed_ms)

    async def _flush_asr_finalize(self) -> None:
        """结束当前 ASR 轮次。"""
        if self.asr_recognizer is None:
            return

        start_time = time.perf_counter()
        result = await self._run_asr_finalize()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "asr finalize id=%s text_len=%s elapsed_ms=%.1f",
            self.state.session_id,
            len((result.get("text") or "").strip()),
            elapsed_ms,
        )
        is_boundary_flush = self._asr_boundary_flush_requested
        await self._handle_asr_output(result, processing_elapsed_ms=elapsed_ms)
        if result.get("type") == "final" and (
            is_boundary_flush or not (result.get("text") or "").strip()
        ):
            self._reset_asr_turn()
        self._asr_flush_requested = False
        self._asr_stop_received = False

    async def _run_asr_feed(self, audio_float):
        if self.settings.asr_provider == "zipformer":
            return await asyncio.to_thread(self.asr_recognizer.feed_audio, audio_float)
        return self.asr_recognizer.feed_audio(audio_float)

    async def _run_asr_finalize(self):
        if self.settings.asr_provider == "zipformer":
            return await asyncio.to_thread(self.asr_recognizer.finalize)
        return self.asr_recognizer.finalize()

    async def _handle_asr_output(self, result: dict, processing_elapsed_ms: float = 0.0) -> None:
        """转发 ASR 结果。"""
        result_type = result.get("type", "partial")
        text = await self._postprocess_asr_text(result_type, (result.get("text") or "").strip())
        delta = result.get("delta", "")

        if self.settings.asr_provider == "zipformer":
            await self._handle_zipformer_asr_text(
                result_type=result_type,
                text=text,
                processing_elapsed_ms=processing_elapsed_ms,
                raw_result=result,
            )
            return

        await self._emit_asr_output(
            {
                "type": result_type,
                "text": text,
                "delta": delta,
                "ts": result.get("ts"),
            },
            processing_elapsed_ms=processing_elapsed_ms,
        )

    async def _handle_zipformer_asr_text(
        self,
        result_type: str,
        text: str,
        processing_elapsed_ms: float,
        raw_result: dict | None = None,
    ) -> None:
        chunk_text = (text or "").strip()
        current_raw_text = normalize_zipformer_text(chunk_text)

        if (
            result_type == "final"
            and current_raw_text
            and current_raw_text == self._zipformer_last_chunk_text
        ):
            logger.info(
                "zipformer skip duplicate final id=%s text=%s",
                self.state.session_id,
                _preview_text(current_raw_text),
            )
            await self._finalize_zipformer_pending_text(
                processing_elapsed_ms=processing_elapsed_ms
            )
            self._reset_zipformer_stream_state()
            return

        if not current_raw_text:
            self._zipformer_no_change_count += 1
            logger.info(
                "zipformer no-text id=%s type=%s streak=%s",
                self.state.session_id,
                result_type,
                self._zipformer_no_change_count,
            )
            await self._emit_zipformer_live_text(self._zipformer_unpunctuated_buffer)
            if (
                result_type == "final"
                or self._asr_flush_requested
                or self._zipformer_no_change_count >= 2
            ):
                await self._finalize_zipformer_pending_text(
                    processing_elapsed_ms=processing_elapsed_ms,
                    force_terminal_punctuation=(
                        self._zipformer_use_ctpunc and self._zipformer_no_change_count >= 2
                    ),
                )
            if result_type == "final" or self._asr_flush_requested:
                self._reset_zipformer_stream_state()
            return

        if current_raw_text == self._zipformer_prev_raw_text:
            self._zipformer_no_change_count += 1
            logger.info(
                "zipformer no-new-text id=%s type=%s streak=%s text=%s",
                self.state.session_id,
                result_type,
                self._zipformer_no_change_count,
                _preview_text(current_raw_text),
            )
            await self._emit_zipformer_live_text(self._zipformer_unpunctuated_buffer)
            if (
                result_type == "final"
                or self._asr_flush_requested
                or self._zipformer_no_change_count >= 2
            ):
                await self._finalize_zipformer_pending_text(
                    processing_elapsed_ms=processing_elapsed_ms,
                    force_terminal_punctuation=(
                        self._zipformer_use_ctpunc and self._zipformer_no_change_count >= 2
                    ),
                )
                self._zipformer_no_change_count = 0
            if result_type == "final" or self._asr_flush_requested:
                self._reset_zipformer_stream_state()
            return

        self._zipformer_no_change_count = 0
        self._zipformer_last_chunk_text = current_raw_text
        current_increment = _extract_zipformer_increment(
            previous=self._zipformer_prev_raw_text,
            current=current_raw_text,
        )
        if not current_increment:
            logger.info(
                "zipformer empty-increment flush id=%s type=%s raw=%s prev=%s",
                self.state.session_id,
                result_type,
                _preview_text(current_raw_text),
                _preview_text(self._zipformer_prev_raw_text),
            )
            self._zipformer_prev_raw_text = current_raw_text
            await self._emit_zipformer_live_text(self._zipformer_unpunctuated_buffer)
            if result_type == "final" or self._asr_flush_requested:
                await self._finalize_zipformer_pending_text(
                    processing_elapsed_ms=processing_elapsed_ms
                )
                self._reset_zipformer_stream_state()
            return

        if (
            _content_token_count(current_increment) == 0
            and not self._zipformer_unpunctuated_buffer.strip()
        ):
            logger.info(
                "zipformer skip trailing punctuation-only increment id=%s type=%s raw=%s increment=%s",
                self.state.session_id,
                result_type,
                _preview_text(current_raw_text),
                _preview_text(current_increment),
            )
            self._zipformer_prev_raw_text = current_raw_text
            await self._emit_zipformer_live_text("")
            if result_type == "final":
                self._reset_zipformer_stream_state()
            return

        self._zipformer_unpunctuated_buffer = _concat_zipformer_text(
            self._zipformer_unpunctuated_buffer,
            current_increment,
        )
        punctuated_buffer = self._zipformer_unpunctuated_buffer
        if self._zipformer_use_ctpunc:
            punctuated_buffer = await asyncio.to_thread(
                self.zipformer_postprocessor.process,
                self._zipformer_unpunctuated_buffer,
                apply_punctuation=True,
            )
        punctuated_buffer = punctuated_buffer.strip()
        committed_prefix, punctuated_tail = _split_text_at_last_punctuation(punctuated_buffer)
        logger.info(
            "zipformer aggregate id=%s type=%s raw=%s increment=%s buffer=%s punct=%s",
            self.state.session_id,
            result_type,
            _preview_text(current_raw_text),
            _preview_text(current_increment),
            _preview_text(self._zipformer_unpunctuated_buffer),
            _preview_text(punctuated_buffer),
        )
        self._zipformer_prev_raw_text = current_raw_text

        if committed_prefix:
            raw_committed, raw_tail = _split_raw_text_by_content_count(
                self._zipformer_unpunctuated_buffer,
                _content_token_count(committed_prefix),
            )
            logger.info(
                "zipformer commit id=%s committed=%s raw_committed=%s tail=%s",
                self.state.session_id,
                _preview_text(committed_prefix),
                _preview_text(raw_committed),
                _preview_text(raw_tail),
            )
            if _content_token_count(committed_prefix) > 0:
                await self._emit_zipformer_stable_text(committed_prefix)
                await self._enqueue_translation_text(
                    committed_prefix,
                    processing_elapsed_ms=processing_elapsed_ms,
                )
            else:
                logger.info(
                    "zipformer skip punctuation-only commit id=%s text=%s",
                    self.state.session_id,
                    _preview_text(committed_prefix),
                )
            self._zipformer_unpunctuated_buffer = raw_tail
            await self._emit_zipformer_live_text(self._zipformer_unpunctuated_buffer)
        else:
            await self._emit_zipformer_live_text(self._zipformer_unpunctuated_buffer)

        if result_type == "final":
            await self._finalize_zipformer_pending_text(
                processing_elapsed_ms=processing_elapsed_ms
            )
            self._reset_zipformer_stream_state()

    async def _emit_asr_output(self, result: dict, processing_elapsed_ms: float = 0.0) -> None:
        result_type = result.get("type", "partial")
        text = (result.get("text") or "").strip()
        delta = result.get("delta", "")

        if result_type == "blank":
            return
        if result_type == self._last_asr_type and text == self._last_asr_emit:
            return
        if result_type == "final" and text == self.state.stable_text:
            return

        self._last_asr_type = result_type
        self._last_asr_emit = text
        if result_type == "partial":
            self.state.last_partial_text = text
        elif result_type == "final":
            self.state.last_partial_text = ""
            self.state.stable_text = text

        sent = await self._send_json(
            {
                "action": "asr_result",
                "data": {
                    "type": result_type,
                    "text": text,
                    "delta": delta,
                    "ts": result.get("ts"),
                },
            }
        )
        if not sent:
            return

        logger.info(
            "asr output id=%s type=%s text=%s",
            self.state.session_id,
            result_type,
            _preview_text(text),
        )
        if result_type == "final" and text:
            forced_speaker_id = self._forced_prompt_speaker_id
            segment_pairs = translation_segment_pairs_for_text(text)
            if not segment_pairs:
                if result.get("turn_complete", True):
                    self._reset_asr_turn()
                return
            segments = [pair[0] for pair in segment_pairs]
            logger.info(
                "translation enqueue id=%s segments=%s text=%s",
                self.state.session_id,
                len(segments),
                _preview_text(text),
            )
            source_audio_duration_ms = (
                self._current_audio_bytes
                / (self.sample_rate * self._asr_bytes_per_sample)
                * 1000
                if self.sample_rate > 0
                else 0
            )
            segment_count = len(segments)
            segment_source_audio_durations = estimate_segment_source_audio_durations(
                segments,
                source_audio_duration_ms,
            )
            previous_segment_audio_ms = self._current_turn_committed_audio_ms
            for index, (segment, source_segment) in enumerate(segment_pairs, start=1):
                segment_audio_end_ms = segment_source_audio_durations[index - 1]
                prompt_fields = await self._prompt_fields_for_source_span(
                    self._current_turn_started_audio_ms + previous_segment_audio_ms,
                    self._current_turn_started_audio_ms + segment_audio_end_ms,
                    speaker_id_override=forced_speaker_id,
                )
                await self._enqueue_translation_item(
                    {
                        "text": segment,
                        "source_text": source_segment,
                        "turn_id": self._current_turn_id,
                        "segment_index": index,
                        "segment_count": segment_count,
                        "source_audio_duration_ms": segment_source_audio_durations[index - 1],
                        "asr_elapsed_ms": processing_elapsed_ms,
                        **prompt_fields,
                    }
                )
                previous_segment_audio_ms = segment_audio_end_ms
            self._current_turn_committed_audio_ms = max(
                self._current_turn_committed_audio_ms,
                previous_segment_audio_ms,
            )
            if result.get("turn_complete", True):
                self._reset_asr_turn()

    async def _emit_zipformer_stable_text(self, text: str) -> None:
        value = (text or "").strip()
        if not value:
            return
        self.state.stable_text = join_session_text(self.state.stable_text, value)
        self.state.last_partial_text = ""
        self._last_asr_type = "stable"
        self._last_asr_emit = value
        await self._send_json(
            {
                "action": "asr_result",
                "data": {
                    "type": "stable",
                    "text": value,
                    "delta": "",
                    "ts": time.time(),
                },
            }
        )
        await self._send_json(
            {
                "action": "asr_result",
                "data": {
                    "type": "blank",
                    "text": "",
                    "delta": "",
                    "ts": time.time(),
                },
            }
        )
        logger.info(
            "zipformer stable id=%s text=%s",
            self.state.session_id,
            _preview_text(value),
        )

    async def _emit_zipformer_live_text(self, text: str) -> None:
        value = (text or "").strip()
        self.state.last_partial_text = value
        self._last_asr_type = "partial" if value else "blank"
        self._last_asr_emit = value
        await self._send_json(
            {
                "action": "asr_result",
                "data": {
                    "type": "partial" if value else "blank",
                    "text": value,
                    "delta": "",
                    "ts": time.time(),
                },
            }
        )

    async def _enqueue_translation_text(
        self,
        text: str,
        processing_elapsed_ms: float,
    ) -> None:
        value = (text or "").strip()
        if not value:
            return
        segment_pairs = translation_segment_pairs_for_text(value)
        if not segment_pairs:
            return
        segments = [pair[0] for pair in segment_pairs]
        logger.info(
            "zipformer translation enqueue id=%s segments=%s text=%s",
            self.state.session_id,
            len(segments),
            _preview_text(value),
        )
        source_audio_duration_ms = (
            self._current_audio_bytes
            / (self.sample_rate * self._asr_bytes_per_sample)
            * 1000
            if self.sample_rate > 0
            else 0
        )
        segment_count = len(segments)
        segment_source_audio_durations = estimate_segment_source_audio_durations(
            segments,
            source_audio_duration_ms,
        )
        previous_segment_audio_ms = self._current_turn_committed_audio_ms
        for index, (segment, source_segment) in enumerate(segment_pairs, start=1):
            segment_audio_end_ms = segment_source_audio_durations[index - 1]
            prompt_fields = await self._prompt_fields_for_source_span(
                self._current_turn_started_audio_ms + previous_segment_audio_ms,
                self._current_turn_started_audio_ms + segment_audio_end_ms,
                speaker_id_override=self._forced_prompt_speaker_id,
            )
            await self._enqueue_translation_item(
                {
                    "text": segment,
                    "source_text": source_segment,
                    "turn_id": self._current_turn_id,
                    "segment_index": index,
                    "segment_count": segment_count,
                    "source_audio_duration_ms": segment_source_audio_durations[index - 1],
                    "asr_elapsed_ms": processing_elapsed_ms,
                    **prompt_fields,
                }
            )
            previous_segment_audio_ms = segment_audio_end_ms
        self._current_turn_committed_audio_ms = max(
            self._current_turn_committed_audio_ms,
            previous_segment_audio_ms,
        )

    async def _enqueue_zipformer_translations(
        self,
        stable_increment: str,
        processing_elapsed_ms: float,
    ) -> None:
        self._zipformer_translation_buffer = join_session_text(
            self._zipformer_translation_buffer,
            stable_increment,
        )
        committable_prefix, tail = _split_text_at_last_punctuation(self._zipformer_translation_buffer)
        if not committable_prefix:
            return
        new_text = committable_prefix.strip()
        if not new_text:
            self._zipformer_translation_buffer = tail
            return

        segments = translation_segments_for_text(new_text)
        if not segments:
            self._zipformer_translation_buffer = tail
            return

        logger.info(
            "zipformer translation enqueue id=%s segments=%s text=%s",
            self.state.session_id,
            len(segments),
            _preview_text(new_text),
        )
        source_audio_duration_ms = (
            self._current_audio_bytes
            / (self.sample_rate * self._asr_bytes_per_sample)
            * 1000
            if self.sample_rate > 0
            else 0
        )
        segment_count = len(segments)
        segment_source_audio_durations = estimate_segment_source_audio_durations(
            segments,
            source_audio_duration_ms,
        )
        previous_segment_audio_ms = self._current_turn_committed_audio_ms
        for index, segment in enumerate(segments, start=1):
            segment_audio_end_ms = segment_source_audio_durations[index - 1]
            prompt_fields = await self._prompt_fields_for_source_span(
                self._current_turn_started_audio_ms + previous_segment_audio_ms,
                self._current_turn_started_audio_ms + segment_audio_end_ms,
                speaker_id_override=self._forced_prompt_speaker_id,
            )
            await self._enqueue_translation_item(
                {
                    "text": segment,
                    "turn_id": self._current_turn_id,
                    "segment_index": index,
                    "segment_count": segment_count,
                    "source_audio_duration_ms": segment_source_audio_durations[index - 1],
                    "asr_elapsed_ms": processing_elapsed_ms,
                    **prompt_fields,
                }
            )
            previous_segment_audio_ms = segment_audio_end_ms
        self._current_turn_committed_audio_ms = max(
            self._current_turn_committed_audio_ms,
            previous_segment_audio_ms,
        )
        self._zipformer_translation_buffer = tail

    async def _flush_zipformer_translation_buffer(
        self,
        processing_elapsed_ms: float,
    ) -> None:
        pending = self._zipformer_translation_buffer.strip()
        if not pending:
            return
        segments = translation_segments_for_text(pending)
        if not segments:
            self._zipformer_translation_buffer = ""
            return
        logger.info(
            "zipformer translation flush id=%s segments=%s text=%s",
            self.state.session_id,
            len(segments),
            _preview_text(pending),
        )
        source_audio_duration_ms = (
            self._current_audio_bytes
            / (self.sample_rate * self._asr_bytes_per_sample)
            * 1000
            if self.sample_rate > 0
            else 0
        )
        segment_count = len(segments)
        segment_source_audio_durations = estimate_segment_source_audio_durations(
            segments,
            source_audio_duration_ms,
        )
        previous_segment_audio_ms = self._current_turn_committed_audio_ms
        for index, segment in enumerate(segments, start=1):
            segment_audio_end_ms = segment_source_audio_durations[index - 1]
            prompt_fields = await self._prompt_fields_for_source_span(
                self._current_turn_started_audio_ms + previous_segment_audio_ms,
                self._current_turn_started_audio_ms + segment_audio_end_ms,
                speaker_id_override=self._forced_prompt_speaker_id,
            )
            await self._enqueue_translation_item(
                {
                    "text": segment,
                    "turn_id": self._current_turn_id,
                    "segment_index": index,
                    "segment_count": segment_count,
                    "source_audio_duration_ms": segment_source_audio_durations[index - 1],
                    "asr_elapsed_ms": processing_elapsed_ms,
                    **prompt_fields,
                }
            )
            previous_segment_audio_ms = segment_audio_end_ms
        self._current_turn_committed_audio_ms = max(
            self._current_turn_committed_audio_ms,
            previous_segment_audio_ms,
        )
        self._zipformer_translation_buffer = ""

    async def _finalize_zipformer_pending_text(
        self,
        processing_elapsed_ms: float,
        force_terminal_punctuation: bool = False,
    ) -> None:
        pending = self._zipformer_unpunctuated_buffer.strip()
        if not pending:
            await self._emit_zipformer_live_text("")
            return
        punctuated = pending
        if self._zipformer_use_ctpunc:
            punctuated = await asyncio.to_thread(
                self.zipformer_postprocessor.process,
                pending,
                apply_punctuation=True,
            )
        punctuated = punctuated.strip() or pending
        if force_terminal_punctuation:
            punctuated = _ensure_terminal_punctuation(punctuated)
        logger.info(
            "zipformer finalize pending id=%s forced=%s raw=%s punct=%s",
            self.state.session_id,
            force_terminal_punctuation,
            _preview_text(pending),
            _preview_text(punctuated),
        )
        if _content_token_count(punctuated) > 0:
            await self._emit_zipformer_stable_text(punctuated)
            await self._enqueue_translation_text(
                punctuated,
                processing_elapsed_ms=processing_elapsed_ms,
            )
        else:
            logger.info(
                "zipformer skip punctuation-only finalize id=%s text=%s",
                self.state.session_id,
                _preview_text(punctuated),
            )
        self._zipformer_unpunctuated_buffer = ""
        await self._emit_zipformer_live_text("")

    async def _postprocess_asr_text(self, result_type: str, text: str) -> str:
        if (
            self.settings.asr_provider != "zipformer"
            or not text
            or not self._zipformer_use_ctpunc
        ):
            return text
        try:
            return await asyncio.to_thread(
                self.zipformer_postprocessor.process,
                text,
                apply_punctuation=False,
            )
        except Exception as exc:
            logger.warning(
                "zipformer text postprocess failed id=%s type=%s err=%s",
                self.state.session_id,
                result_type,
                exc,
            )
            return text

    async def _current_prompt_fields(self) -> dict[str, str]:
        if self.speaker_prompt_tracker is None:
            return {}
        clone_audio_path = await self.speaker_prompt_tracker.export_prompt()
        return {
            "speaker_id": self.speaker_prompt_tracker.current_speaker_id,
            "clone_audio_path": clone_audio_path or "",
        }

    async def _prompt_fields_for_source_span(
        self,
        start_ms: float,
        end_ms: float,
        speaker_id_override: str | None = None,
    ) -> dict[str, str | float]:
        if self.speaker_prompt_tracker is None:
            return {
                "source_start_ms": start_ms,
                "source_end_ms": end_ms,
            }
        speaker_id = speaker_id_override or await self.speaker_prompt_tracker.speaker_for_span(
            start_ms,
            end_ms,
        )
        clone_audio_path = await self.speaker_prompt_tracker.export_prompt_for_speaker(speaker_id)
        logger.info(
            "speaker prompt bound id=%s speaker=%s forced=%s source_start_ms=%.0f source_end_ms=%.0f path=%s",
            self.state.session_id,
            speaker_id,
            bool(speaker_id_override),
            start_ms,
            end_ms,
            clone_audio_path,
        )
        return {
            "speaker_id": speaker_id or "",
            "clone_audio_path": clone_audio_path or "",
            "source_start_ms": start_ms,
            "source_end_ms": end_ms,
        }

    async def _enqueue_translation_item(self, item: dict) -> None:
        """先向前端发送已绑定 speaker 的源文本片段，再进入翻译队列。"""
        await self._send_json(
            {
                "action": "source_segment_ready",
                "data": {
                    "source_text": item.get("source_text") or item.get("text", ""),
                    "turn_id": item.get("turn_id", ""),
                    "segment_index": item.get("segment_index", 1),
                    "segment_count": item.get("segment_count", 1),
                    "speaker_id": item.get("speaker_id", ""),
                    "source_start_ms": item.get("source_start_ms", 0),
                    "source_end_ms": item.get("source_end_ms", 0),
                },
            }
        )
        await self.translation_queue.put(item)

    def _session_audio_duration_ms(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self._session_audio_bytes_total / (self.sample_rate * self._asr_bytes_per_sample) * 1000

    async def _tts_loop(self) -> None:
        """串行执行翻译和 TTS。"""
        try:
            while True:
                item = await self.translation_queue.get()
                if item == "":
                    break
                if not item:
                    continue
                segment = item["text"]
                turn_id = item.get("turn_id", "")
                segment_index = item.get("segment_index", 1)
                segment_count = item.get("segment_count", 1)
                asr_elapsed_ms = item.get("asr_elapsed_ms", 0)
                speaker_id = item.get("speaker_id", "")

                loop = asyncio.get_running_loop()
                logger.info(
                    "translation start id=%s text=%s",
                    self.state.session_id,
                    _preview_text(segment),
                )
                translate_start = time.perf_counter()
                try:
                    translated = await loop.run_in_executor(
                        None,
                        self.mt_service.translate,
                        segment,
                        self.state.source_lang,
                        self.state.target_lang,
                    )
                except Exception as exc:
                    logger.error("translation failed: %s", exc)
                    await self._send_json({"action": "error", "data": {"message": str(exc)}})
                    continue
                mt_elapsed_ms = (time.perf_counter() - translate_start) * 1000
                logger.info(
                    "translation done id=%s elapsed_ms=%.1f text=%s output=%s",
                    self.state.session_id,
                    mt_elapsed_ms,
                    _preview_text(segment),
                    _preview_text(translated),
                )

                clone_audio_path = item.get("clone_audio_path") or None
                if self.speaker_prompt_tracker is None:
                    clone_start = time.perf_counter()
                    clone_audio_path = await self.audio_store.export_recent_clone(
                        self.settings.clone_window_seconds
                    )
                    logger.info(
                        "clone export id=%s elapsed_ms=%.1f path=%s",
                        self.state.session_id,
                        (time.perf_counter() - clone_start) * 1000,
                        clone_audio_path,
                    )
                else:
                    logger.info(
                        "speaker prompt selected id=%s speaker=%s path=%s",
                        self.state.session_id,
                        speaker_id,
                        clone_audio_path,
                    )

                tts_start = time.perf_counter()
                try:
                    audio_bytes = await loop.run_in_executor(
                        None,
                        self.tts_service.synthesize,
                        translated,
                        clone_audio_path,
                    )
                except Exception as exc:
                    logger.error("tts failed: %s", exc)
                    await self._send_json({"action": "error", "data": {"message": str(exc)}})
                    continue
                tts_elapsed_ms = (time.perf_counter() - tts_start) * 1000
                logger.info(
                    "tts done id=%s elapsed_ms=%.1f audio_bytes=%s text=%s",
                    self.state.session_id,
                    tts_elapsed_ms,
                    len(audio_bytes),
                    _preview_text(translated),
                )

                save_start = time.perf_counter()
                tts_audio_path = await self.audio_store.save_tts_audio(
                    audio_bytes,
                    self.settings.tts_sample_rate,
                )
                logger.info(
                    "tts saved id=%s elapsed_ms=%.1f path=%s",
                    self.state.session_id,
                    (time.perf_counter() - save_start) * 1000,
                    tts_audio_path,
                )

                sent = await self._send_json(
                    {
                        "action": "translation_ready",
                        "data": {
                            "source_text": segment,
                            "translated_text": translated,
                            "turn_id": turn_id,
                            "segment_index": segment_index,
                            "segment_count": segment_count,
                            "speaker_id": speaker_id,
                            "clone_audio_path": clone_audio_path,
                            "tts_audio_path": tts_audio_path,
                            "metrics": build_latency_metrics(
                                asr_elapsed_ms=asr_elapsed_ms,
                                mt_elapsed_ms=mt_elapsed_ms,
                                tts_elapsed_ms=tts_elapsed_ms,
                            ),
                        },
                    }
                )
                if not sent or not audio_bytes:
                    continue
                if not await self._send_bytes(audio_bytes):
                    continue
                await self._send_json({"action": "tts_finished", "data": {"text": translated}})
        except asyncio.CancelledError:
            pass

    async def _send_json(self, payload: dict) -> bool:
        """发送 JSON 消息。"""
        try:
            await self.websocket.send_text(json.dumps(payload, ensure_ascii=False))
            return True
        except WebSocketDisconnect:
            logger.info("websocket disconnected while sending: %s", self.state.session_id)
            return False
        except RuntimeError as exc:
            message = str(exc)
            if (
                "disconnect message has been received" in message
                or "close message has been sent" in message
                or "Unexpected ASGI message 'websocket.send'" in message
            ):
                logger.info("websocket disconnected while sending: %s", self.state.session_id)
                return False
            raise

    async def _send_bytes(self, payload: bytes) -> bool:
        """发送二进制消息。"""
        try:
            await self.websocket.send_bytes(payload)
            return True
        except WebSocketDisconnect:
            logger.info("websocket disconnected while sending audio: %s", self.state.session_id)
            return False
        except RuntimeError as exc:
            message = str(exc)
            if (
                "disconnect message has been received" in message
                or "close message has been sent" in message
                or "Unexpected ASGI message 'websocket.send'" in message
            ):
                logger.info("websocket disconnected while sending audio: %s", self.state.session_id)
                return False
            raise

    def _reset_asr_turn(self) -> None:
        """重置当前 ASR 轮次状态。"""
        self.state.stable_text = ""
        self.state.last_partial_text = ""
        self._last_asr_emit = ""
        self._last_asr_type = ""
        self._current_audio_started_at = None
        self._current_audio_bytes = 0
        self._current_turn_started_audio_ms = self._session_audio_duration_ms()
        self._current_turn_committed_audio_ms = 0
        self._current_turn_id = ""
        self._forced_prompt_speaker_id = None
        self._reset_zipformer_stream_state()

    def _reset_zipformer_stream_state(self) -> None:
        self._zipformer_prev_raw_text = ""
        self._zipformer_last_chunk_text = ""
        self._zipformer_no_change_count = 0
        self._zipformer_unpunctuated_buffer = ""
        self._zipformer_translation_buffer = ""

def split_translation_segments(text: str) -> list[str]:
    """按标点切分翻译片段。"""
    parts = re.findall(r"[^，。！？；：,.!?;:]+[，。！？；：,.!?;:]?|[^，。！？；：,.!?;:]+$", text)
    return [
        part.strip()
        for part in parts
        if part and part.strip() and should_emit_segment(part.strip(), 1)
    ] or [text.strip()]


def translation_segments_for_text(text: str) -> list[str]:
    """切分并清理不会进入翻译/TTS 的语气词片段。"""
    return [segment for segment, _source_segment in translation_segment_pairs_for_text(text)]


def translation_segment_pairs_for_text(text: str) -> list[tuple[str, str]]:
    """返回翻译用清理文本和前端展示用原始源文。"""
    pairs: list[tuple[str, str]] = []
    for source_segment in split_translation_segments(text):
        cleaned = strip_filler_edges(source_segment)
        if cleaned and _content_token_count(cleaned) > 0:
            pairs.append((cleaned, source_segment.strip()))
    return pairs


def _concat_zipformer_text(previous: str, current: str) -> str:
    left = normalize_zipformer_text(previous)
    raw_right = current or ""
    right = normalize_zipformer_text(raw_right)
    if not left:
        return right
    if not right:
        return left
    if left.endswith(right):
        return left
    if right.startswith(left):
        return right

    # Keep the whitespace boundary that already exists in the cumulative
    # zipformer text instead of guessing based on token shape.
    if raw_right[:1].isspace():
        return f"{left} {right}"
    if left[-1:].isalnum() and right[:1].isalnum():
        return f"{left}{right}"
    return f"{left}{right}"


def _extract_zipformer_increment(previous: str, current: str) -> str:
    left = normalize_zipformer_text(previous)
    right = normalize_zipformer_text(current)
    if not right:
        return ""
    if not left:
        return right
    if right.startswith(left):
        return right[len(left) :]
    if left.startswith(right):
        return ""

    logger.warning(
        "zipformer prefix mismatch previous=%s current=%s",
        _preview_text(left),
        _preview_text(right),
    )
    return right


def _split_text_at_last_punctuation(text: str) -> tuple[str, str]:
    value = (text or "").strip()
    if not value:
        return "", ""
    punctuation_indexes = [
        index
        for index, char in enumerate(value)
        if char in "，。！？；：,.!?;:"
    ]
    if not punctuation_indexes:
        return "", value
    cutoff = punctuation_indexes[-1] + 1
    return value[:cutoff].strip(), value[cutoff:].strip()


def _ensure_terminal_punctuation(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    if value[-1] in "，。！？；：,.!?;:":
        return value
    return f"{value}。"


def _content_token_count(text: str) -> int:
    return len(_content_tokens(text))


def _content_tokens(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", text or "")


def _split_raw_text_by_content_count(text: str, left_content_count: int) -> tuple[str, str]:
    tokens = re.findall(
        r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s]",
        text or "",
    )
    left_tokens: list[str] = []
    right_tokens: list[str] = []
    content_seen = 0
    in_right = left_content_count == 0

    for token in tokens:
        is_content = bool(re.fullmatch(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", token))
        if is_content and content_seen >= left_content_count:
            in_right = True
        if in_right:
            right_tokens.append(token)
        else:
            left_tokens.append(token)
        if is_content:
            content_seen += 1

    return "".join(left_tokens).strip(), "".join(right_tokens).strip()


def _split_punctuated_by_content_count(text: str, left_content_count: int) -> tuple[str, str]:
    tokens = re.findall(
        r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[，。！？；：、,.!?;:…]",
        text or "",
    )
    left_tokens: list[str] = []
    right_tokens: list[str] = []
    content_seen = 0
    in_right = left_content_count == 0

    for token in tokens:
        is_content = bool(re.fullmatch(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", token))
        if is_content and content_seen >= left_content_count:
            in_right = True
        if in_right:
            right_tokens.append(token)
        else:
            left_tokens.append(token)
        if is_content:
            content_seen += 1

    while right_tokens and right_tokens[0] in "，。！？；：、,.!?;:…":
        break
    return "".join(left_tokens).strip(), "".join(right_tokens).strip()


def join_session_text(previous: str, current: str) -> str:
    left = (previous or "").strip()
    right = (current or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith(right):
        return left
    return left + right


def estimate_segment_source_audio_durations(
    segments: list[str],
    total_source_audio_duration_ms: float,
) -> list[int]:
    """按累计文本占比估算每个翻译片段对应的累计源语音时长。"""
    total_ms = max(0, round(total_source_audio_duration_ms))
    if not segments:
        return []
    if len(segments) == 1 or total_ms <= 0:
        return [total_ms for _ in segments]

    weights = [max(1, len(_segment_weight_text(segment))) for segment in segments]
    total_weight = sum(weights)
    cumulative_weight = 0
    durations: list[int] = []

    for weight in weights:
        cumulative_weight += weight
        durations.append(max(0, round(total_ms * cumulative_weight / total_weight)))

    durations[-1] = total_ms
    return durations


def _segment_weight_text(text: str) -> str:
    return re.sub(r"[，。！？；：,.!?;:\s]+", "", text or "")


def strip_filler_edges(text: str) -> str:
    value = (text or "").strip()
    previous = None
    while value and value != previous:
        previous = value
        value = FILLER_PREFIX_RE.sub("", value).strip()
        value = FILLER_SUFFIX_RE.sub("", value).strip()
    return value


def is_filler_segment(text: str) -> bool:
    return not strip_filler_edges(text)


def filter_translation_segments(segments: list[str]) -> list[str]:
    filtered: list[str] = []
    for segment in segments:
        cleaned = strip_filler_edges(segment)
        if cleaned:
            filtered.append(cleaned)
    return filtered


def _preview_text(text: str, limit: int = 80) -> str:
    """压缩日志文本长度。"""
    value = text.strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."
