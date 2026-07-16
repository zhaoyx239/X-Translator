from __future__ import annotations

import asyncio
import os
import threading
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from .logging_utils import setup_logger

logger = setup_logger()

AudioTagKind = Literal["speech", "silence", "speaker_change"]


@dataclass(slots=True)
class AudioTag:
    """时间线上的音频标签。"""

    kind: AudioTagKind
    start_ms: float
    end_ms: float
    speaker_id: str | None = None
    confidence: float = 1.0


@dataclass(slots=True)
class SpeakerChangeEvent:
    """窗口级说话人变化事件。"""

    previous_speaker_id: str
    current_speaker_id: str
    boundary_ms: float
    detected_at_ms: float
    confidence: float
    similarity: float


class SpeakerEmbeddingBackend(Protocol):
    """说话人 embedding 后端。"""

    name: str

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """返回归一化后的 speaker embedding。"""


class SpectralSpeakerEmbeddingBackend:
    """无模型时的轻量 fallback。"""

    name = "spectral"

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        return _voice_signature(samples)


class ModelScopeSpeakerEmbeddingBackend:
    """ModelScope CAM++ speaker verification embedding 后端。"""

    name = "modelscope_campplus"

    def __init__(self, model: str) -> None:
        self.model = model
        self._pipeline = None
        self._lock = threading.Lock()

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        pipeline = self._ensure_pipeline()
        audio = np.asarray(samples, dtype=np.float32)
        if sample_rate != 16000:
            audio = _resample_linear(audio, sample_rate, 16000)
        result = pipeline([audio], output_emb=True)
        embedding = np.asarray(result["embs"], dtype=np.float32)
        if embedding.ndim > 1:
            embedding = embedding[0]
        return _normalize_vector(embedding)

    def _ensure_pipeline(self):
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model=self.model,
            )
            logger.info("speaker embedding backend loaded model=%s", self.model)
            return self._pipeline


class SpeakerPromptTracker:
    """维护当前说话人的干净 TTS prompt 候选音频。

    帧级能量 VAD 负责过滤静音；说话人切换用较长语音窗口上的
    speaker embedding 判断。默认后端是 ModelScope CAM++。
    """

    def __init__(
        self,
        session_id: str,
        output_dir: Path,
        sample_rate: int,
        *,
        session_label: str | None = None,
        target_seconds: float,
        min_seconds: float,
        max_seconds: float,
        frame_ms: int,
        silence_dbfs: float,
        vad_margin_db: float,
        embedding_provider: str,
        embedding_model: str,
        embedding_window_seconds: float,
        embedding_hop_seconds: float,
        speaker_change_threshold: float,
    ) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.output_label = _safe_path_label(session_label) or _timestamp_label()
        self.output_dir = output_dir / self.output_label
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_output_dir = self.output_dir / "prompts"
        self.rolling_output_dir = self.prompt_output_dir / "rolling"
        self.fixed_output_dir = self.prompt_output_dir / "fixed"
        self.rolling_output_dir.mkdir(parents=True, exist_ok=True)
        self.fixed_output_dir.mkdir(parents=True, exist_ok=True)
        self.target_seconds = max(0.5, target_seconds)
        self.min_seconds = max(0.2, min_seconds)
        self.max_seconds = max(0.5, max_seconds)
        self.fixed_prompt_seconds = min(self.target_seconds, self.max_seconds)
        self.fixed_prompt_samples = round(self.fixed_prompt_seconds * sample_rate)
        self.frame_samples = max(160, round(sample_rate * frame_ms / 1000))
        self.silence_dbfs = silence_dbfs
        self.vad_margin_db = vad_margin_db
        self.embedding_backend = _create_embedding_backend(
            embedding_provider,
            embedding_model,
        )
        self.embedding_window_samples = max(
            self.frame_samples,
            round(sample_rate * embedding_window_seconds),
        )
        self.embedding_hop_samples = max(
            self.frame_samples,
            round(sample_rate * embedding_hop_seconds),
        )
        self.speaker_change_threshold = speaker_change_threshold
        self.speaker_reidentify_threshold = speaker_change_threshold + 0.1

        self._pending = bytearray()
        self._processed_samples = 0
        self._prompt_frames: list[bytes] = []
        self._prompt_samples = 0
        self._prompt_frames_by_speaker: dict[str, list[bytes]] = {}
        self._prompt_samples_by_speaker: dict[str, int] = {}
        self._prompt_export_indexes: dict[str, int] = {}
        self._prompt_paths_by_speaker: dict[str, str] = {}
        self._prompt_revisions: dict[str, int] = {}
        self._prompt_exported_revisions: dict[str, int] = {}
        self._fixed_prompt_paths_by_speaker: dict[str, str] = {}
        self._prompt_active_by_speaker: dict[str, bool] = {}
        self._prompt_trailing_silence_by_speaker: dict[str, int] = {}
        self._prompt_min_reset_silence_samples = round(1.2 * sample_rate)
        self._embedding_frames: list[bytes] = []
        self._embedding_samples = 0
        self._last_embedding_sample = 0
        self._last_embedding_wait_log_sample = 0
        self._ignore_embedding_windows_before_ms = 0.0
        self._tags: list[AudioTag] = []
        self._lock = asyncio.Lock()
        self._noise_floor_db = -60.0
        self._current_speaker_index = 0
        self._current_speaker_id = "speaker_00"
        self._centroid: np.ndarray | None = None
        self._centroid_weight = 0
        self._speaker_centroids: dict[str, np.ndarray] = {}
        self._speaker_centroid_weights: dict[str, int] = {}
        self._last_prompt_path: str | None = None
        logger.info(
            "speaker tracker init id=%s dir=%s backend=%s model=%s sample_rate=%s frame_ms=%s "
            "embedding_window_sec=%.2f embedding_hop_sec=%.2f prompt_min_sec=%.2f "
            "prompt_max_sec=%.2f threshold=%.3f",
            self.session_id,
            self.output_label,
            self.embedding_backend.name,
            embedding_model,
            self.sample_rate,
            frame_ms,
            self.embedding_window_samples / self.sample_rate,
            self.embedding_hop_samples / self.sample_rate,
            self.min_seconds,
            self.max_seconds,
            self.speaker_change_threshold,
        )

    @property
    def current_speaker_id(self) -> str:
        return self._current_speaker_id

    async def feed(self, pcm_bytes: bytes) -> list[SpeakerChangeEvent]:
        """追加音频并在线打 speech/silence/speaker_change 标签。"""
        if not pcm_bytes:
            return []
        embedding_windows: list[tuple[bytes, float, float]] = []
        events: list[SpeakerChangeEvent] = []
        async with self._lock:
            self._pending.extend(pcm_bytes)
            frame_bytes = self.frame_samples * 2
            while len(self._pending) >= frame_bytes:
                frame = bytes(self._pending[:frame_bytes])
                del self._pending[:frame_bytes]
                embedding_window = self._process_frame(frame)
                if embedding_window is not None:
                    embedding_windows.append(embedding_window)

        for window_bytes, start_ms, end_ms in embedding_windows:
            try:
                samples = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                logger.info(
                    "speaker embedding start id=%s backend=%s speaker=%s window_sec=%.2f "
                    "end_ms=%.0f prompt_sec=%.2f",
                    self.session_id,
                    self.embedding_backend.name,
                    self._current_speaker_id,
                    samples.size / self.sample_rate,
                    end_ms,
                    self._prompt_samples / self.sample_rate,
                )
                embedding = await asyncio.to_thread(
                    self.embedding_backend.embed,
                    samples,
                    self.sample_rate,
                )
                logger.info(
                    "speaker embedding done id=%s backend=%s speaker=%s dim=%s end_ms=%.0f",
                    self.session_id,
                    self.embedding_backend.name,
                    self._current_speaker_id,
                    int(np.asarray(embedding).size),
                    end_ms,
                )
            except Exception as exc:
                logger.warning(
                    "speaker embedding failed id=%s backend=%s err=%s; fallback=spectral",
                    self.session_id,
                    self.embedding_backend.name,
                    exc,
                )
                self.embedding_backend = SpectralSpeakerEmbeddingBackend()
                samples = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                embedding = self.embedding_backend.embed(samples, self.sample_rate)
            async with self._lock:
                event = self._handle_embedding(embedding, window_bytes, start_ms, end_ms)
                if event is not None:
                    events.append(event)
        return events

    async def export_prompt(self) -> str | None:
        """导出当前说话人的候选 prompt，不包含静音和已切走的上一说话人。"""
        return await self.export_prompt_for_speaker(self._current_speaker_id)

    async def export_prompt_for_speaker(self, speaker_id: str | None) -> str | None:
        """导出指定 speaker 的 prompt。

        rolling prompt 继续随最新音频更新；fixed prompt 在达到目标长度后冻结，
        并优先作为 TTS clone prompt 返回。
        """
        target_speaker = speaker_id or self._current_speaker_id
        fixed_frames: bytes | None = None
        rolling_frames: bytes | None = None
        rolling_export_index = 0
        async with self._lock:
            fixed_path = self._fixed_prompt_paths_by_speaker.get(target_speaker)
            revision = self._prompt_revisions.get(target_speaker, 0)
            last_exported_revision = self._prompt_exported_revisions.get(target_speaker, -1)
            last_path = self._prompt_paths_by_speaker.get(target_speaker)
            frames = b"".join(self._prompt_frames_by_speaker.get(target_speaker, []))
            if not frames:
                return fixed_path or last_path
            current_samples = self._prompt_samples_by_speaker.get(
                target_speaker,
                len(frames) // 2,
            )
            if fixed_path:
                logger.info(
                    "speaker fixed prompt selected id=%s speaker=%s path=%s",
                    self.session_id,
                    target_speaker,
                    fixed_path,
                )
            elif current_samples >= self.fixed_prompt_samples:
                fixed_frames = frames

            if last_path and revision == last_exported_revision:
                logger.info(
                    "speaker rolling prompt reuse id=%s speaker=%s revision=%s path=%s",
                    self.session_id,
                    target_speaker,
                    revision,
                    last_path,
                )
            else:
                rolling_frames = frames
                rolling_export_index = self._prompt_export_indexes.get(target_speaker, 0) + 1
                self._prompt_export_indexes[target_speaker] = rolling_export_index

        loop = asyncio.get_running_loop()
        fixed_written_path: str | None = None
        if fixed_frames is not None:
            fixed_written_path = await loop.run_in_executor(
                None,
                self._write_prompt,
                fixed_frames,
                target_speaker,
                1,
                "fixed",
            )
        rolling_written_path: str | None = None
        if rolling_frames is not None:
            rolling_written_path = await loop.run_in_executor(
                None,
                self._write_prompt,
                rolling_frames,
                target_speaker,
                rolling_export_index,
                "rolling",
            )
        async with self._lock:
            if fixed_written_path:
                self._fixed_prompt_paths_by_speaker[target_speaker] = fixed_written_path
                logger.info(
                    "speaker fixed prompt frozen id=%s speaker=%s target_sec=%.2f path=%s",
                    self.session_id,
                    target_speaker,
                    self.fixed_prompt_seconds,
                    fixed_written_path,
                )
            if rolling_written_path:
                self._prompt_paths_by_speaker[target_speaker] = rolling_written_path
                self._prompt_exported_revisions[target_speaker] = self._prompt_revisions.get(
                    target_speaker,
                    0,
                )
                if self._current_speaker_id == target_speaker:
                    self._last_prompt_path = rolling_written_path
            fixed_path = self._fixed_prompt_paths_by_speaker.get(target_speaker)
            rolling_path = self._prompt_paths_by_speaker.get(target_speaker)
        return fixed_path or rolling_path

    async def tags_snapshot(self) -> list[AudioTag]:
        async with self._lock:
            return list(self._tags)

    async def speaker_for_span(self, start_ms: float, end_ms: float) -> str | None:
        """返回与源音频时间段重叠最长的 speaker。"""
        async with self._lock:
            return self._speaker_for_span_unlocked(start_ms, end_ms)

    def _process_frame(self, frame: bytes) -> tuple[bytes, float, float] | None:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        start_ms = self._processed_samples / self.sample_rate * 1000
        self._processed_samples += samples.size
        end_ms = self._processed_samples / self.sample_rate * 1000

        dbfs = _rms_dbfs(samples)
        is_speech = self._is_speech(dbfs)
        if not is_speech:
            self._update_noise_floor(dbfs)
            self._append_tag("silence", start_ms, end_ms)
            self._append_prompt_silence_frame(frame)
            return None

        self._append_tag("speech", start_ms, end_ms, self._current_speaker_id)
        self._append_prompt_speech_frame(frame)
        return self._append_embedding_frame(frame, end_ms)

    def _is_speech(self, dbfs: float) -> bool:
        adaptive_threshold = max(self.silence_dbfs, self._noise_floor_db + self.vad_margin_db)
        return dbfs >= adaptive_threshold

    def _update_noise_floor(self, dbfs: float) -> None:
        if dbfs <= self.silence_dbfs + 6:
            self._noise_floor_db = 0.98 * self._noise_floor_db + 0.02 * dbfs

    def _handle_embedding(
        self,
        embedding: np.ndarray,
        window_bytes: bytes,
        start_ms: float,
        end_ms: float,
    ) -> SpeakerChangeEvent | None:
        if start_ms < self._ignore_embedding_windows_before_ms:
            logger.info(
                "speaker embedding skip-stale id=%s speaker=%s window_start_ms=%.0f "
                "window_end_ms=%.0f ignore_before_ms=%.0f",
                self.session_id,
                self._current_speaker_id,
                start_ms,
                end_ms,
                self._ignore_embedding_windows_before_ms,
            )
            return None

        embedding = _normalize_vector(embedding)
        if self._centroid is None:
            self._update_centroid(embedding)
            logger.info(
                "speaker centroid init id=%s speaker=%s backend=%s dim=%s end_ms=%.0f",
                self.session_id,
                self._current_speaker_id,
                self.embedding_backend.name,
                int(embedding.size),
                end_ms,
            )
            return None

        similarity = _cosine_similarity(embedding, self._centroid)
        logger.info(
            "speaker embedding compare id=%s speaker=%s similarity=%.3f threshold=%.3f "
            "centroid_weight=%s end_ms=%.0f",
            self.session_id,
            self._current_speaker_id,
            similarity,
            self.speaker_change_threshold,
            self._centroid_weight,
            end_ms,
        )
        if similarity < self.speaker_change_threshold:
            return self._start_new_speaker(
                boundary_ms=start_ms,
                detected_at_ms=end_ms,
                similarity=similarity,
                embedding=embedding,
                quarantine_samples=len(window_bytes) // 2,
            )

        self._update_centroid(embedding)
        logger.info(
            "speaker embedding same id=%s speaker=%s similarity=%.3f threshold=%.3f",
            self.session_id,
            self._current_speaker_id,
            similarity,
            self.speaker_change_threshold,
        )
        return None

    def _start_new_speaker(
        self,
        boundary_ms: float,
        detected_at_ms: float,
        similarity: float,
        embedding: np.ndarray,
        quarantine_samples: int = 0,
    ) -> SpeakerChangeEvent:
        previous = self._current_speaker_id
        if quarantine_samples > 0:
            self._remove_recent_prompt_samples(previous, quarantine_samples)
        matched_speaker, matched_similarity = self._match_existing_speaker(
            embedding,
            exclude=previous,
        )
        if matched_speaker is not None:
            self._current_speaker_id = matched_speaker
            logger.info(
                "speaker reidentified id=%s previous=%s current=%s similarity=%.3f threshold=%.3f",
                self.session_id,
                previous,
                matched_speaker,
                matched_similarity,
                self.speaker_reidentify_threshold,
            )
        else:
            self._current_speaker_index += 1
            self._current_speaker_id = f"speaker_{self._current_speaker_index:02d}"
        self._prompt_frames = self._prompt_frames_by_speaker.setdefault(self._current_speaker_id, [])
        self._prompt_samples = self._prompt_samples_by_speaker.get(self._current_speaker_id, 0)
        self._prompt_active_by_speaker[self._current_speaker_id] = False
        self._prompt_trailing_silence_by_speaker[self._current_speaker_id] = 0
        self._embedding_frames.clear()
        self._embedding_samples = 0
        self._last_embedding_sample = self._processed_samples
        self._ignore_embedding_windows_before_ms = detected_at_ms
        self._centroid = self._speaker_centroids.get(self._current_speaker_id)
        self._centroid_weight = self._speaker_centroid_weights.get(self._current_speaker_id, 0)
        self._last_prompt_path = None
        self._append_tag(
            "speaker_change",
            boundary_ms,
            boundary_ms,
            self._current_speaker_id,
            confidence=max(0.0, min(1.0, 1.0 - similarity)),
        )
        logger.info(
            "speaker change id=%s previous=%s current=%s boundary_ms=%.0f detected_at_ms=%.0f "
            "similarity=%.3f quarantine_sec=%.2f",
            self.session_id,
            previous,
            self._current_speaker_id,
            boundary_ms,
            detected_at_ms,
            similarity,
            quarantine_samples / self.sample_rate,
        )
        return SpeakerChangeEvent(
            previous_speaker_id=previous,
            current_speaker_id=self._current_speaker_id,
            boundary_ms=boundary_ms,
            detected_at_ms=detected_at_ms,
            confidence=max(0.0, min(1.0, 1.0 - similarity)),
            similarity=similarity,
        )

    def _append_prompt_speech_frame(self, frame: bytes) -> None:
        self._append_prompt_frame_for_speaker(self._current_speaker_id, frame, is_speech=True)

    def _append_prompt_silence_frame(self, frame: bytes) -> None:
        self._append_prompt_frame_for_speaker(self._current_speaker_id, frame, is_speech=False)

    def _append_prompt_frame_for_speaker(
        self,
        speaker_id: str,
        frame: bytes,
        *,
        is_speech: bool,
    ) -> None:
        if not is_speech and not self._prompt_active_by_speaker.get(speaker_id, False):
            return

        if is_speech and not self._prompt_active_by_speaker.get(speaker_id, False):
            self._prompt_active_by_speaker[speaker_id] = True

        frames = self._prompt_frames_by_speaker.setdefault(speaker_id, [])
        frames.append(frame)
        samples = self._prompt_samples_by_speaker.get(speaker_id, 0) + len(frame) // 2
        if is_speech:
            self._prompt_trailing_silence_by_speaker[speaker_id] = 0
        else:
            trailing_silence = (
                self._prompt_trailing_silence_by_speaker.get(speaker_id, 0)
                + len(frame) // 2
            )
            self._prompt_trailing_silence_by_speaker[speaker_id] = trailing_silence
            if trailing_silence > self._prompt_min_reset_silence_samples:
                self._remove_recent_prompt_samples(
                    speaker_id,
                    trailing_silence,
                    reason="trailing_silence",
                )
                self._prompt_active_by_speaker[speaker_id] = False
                self._prompt_trailing_silence_by_speaker[speaker_id] = 0
                return

        max_samples = round(self.max_seconds * self.sample_rate)
        while samples > max_samples and frames:
            dropped = frames.pop(0)
            samples -= len(dropped) // 2
        trailing_silence = self._prompt_trailing_silence_by_speaker.get(speaker_id, 0)
        if trailing_silence > samples:
            self._prompt_trailing_silence_by_speaker[speaker_id] = samples
        self._prompt_samples_by_speaker[speaker_id] = samples
        self._prompt_revisions[speaker_id] = self._prompt_revisions.get(speaker_id, 0) + 1
        if speaker_id == self._current_speaker_id:
            self._prompt_frames = frames
            self._prompt_samples = samples

    def _reset_prompt_for_speaker(self, speaker_id: str) -> None:
        self._prompt_frames_by_speaker[speaker_id] = []
        self._prompt_samples_by_speaker[speaker_id] = 0
        self._prompt_trailing_silence_by_speaker[speaker_id] = 0
        self._prompt_revisions[speaker_id] = self._prompt_revisions.get(speaker_id, 0) + 1
        if speaker_id == self._current_speaker_id:
            self._prompt_frames = self._prompt_frames_by_speaker[speaker_id]
            self._prompt_samples = 0
        logger.info(
            "speaker prompt run reset id=%s speaker=%s",
            self.session_id,
            speaker_id,
        )

    def _remove_recent_prompt_samples(
        self,
        speaker_id: str,
        samples_to_remove: int,
        *,
        reason: str = "speaker_change",
    ) -> None:
        frames = self._prompt_frames_by_speaker.get(speaker_id)
        if not frames or samples_to_remove <= 0:
            return
        remaining = samples_to_remove
        while frames and remaining > 0:
            frame_samples = len(frames[-1]) // 2
            if frame_samples <= remaining:
                frames.pop()
                remaining -= frame_samples
            else:
                break
        samples = sum(len(frame) // 2 for frame in frames)
        self._prompt_samples_by_speaker[speaker_id] = samples
        self._prompt_revisions[speaker_id] = self._prompt_revisions.get(speaker_id, 0) + 1
        if speaker_id == self._current_speaker_id:
            self._prompt_frames = frames
            self._prompt_samples = samples
        logger.info(
            "speaker prompt trim id=%s speaker=%s reason=%s removed_sec=%.2f remain_sec=%.2f",
            self.session_id,
            speaker_id,
            reason,
            samples_to_remove / self.sample_rate,
            samples / self.sample_rate,
        )

    def _speaker_for_span_unlocked(self, start_ms: float, end_ms: float) -> str | None:
        left = max(0.0, min(start_ms, end_ms))
        right = max(left, max(start_ms, end_ms))
        change_points = [
            tag
            for tag in self._tags
            if tag.kind == "speaker_change" and tag.speaker_id is not None
        ]
        if change_points:
            boundaries: list[tuple[float, float, str]] = []
            current_start = 0.0
            current_speaker = "speaker_00"
            for tag in change_points:
                change_at = max(0.0, tag.start_ms)
                if change_at > current_start:
                    boundaries.append((current_start, change_at, current_speaker))
                current_start = change_at
                current_speaker = tag.speaker_id or current_speaker
            boundaries.append((current_start, max(right, current_start), current_speaker))
            weights: dict[str, float] = {}
            for boundary_start, boundary_end, speaker_id in boundaries:
                overlap = min(right, boundary_end) - max(left, boundary_start)
                if overlap > 0:
                    weights[speaker_id] = weights.get(speaker_id, 0.0) + overlap
            if weights:
                return max(weights.items(), key=lambda item: item[1])[0]

        weights: dict[str, float] = {}
        for tag in self._tags:
            if tag.kind != "speech" or not tag.speaker_id:
                continue
            overlap = min(right, tag.end_ms) - max(left, tag.start_ms)
            if overlap > 0:
                weights[tag.speaker_id] = weights.get(tag.speaker_id, 0.0) + overlap
        if weights:
            return max(weights.items(), key=lambda item: item[1])[0]
        for tag in reversed(self._tags):
            if tag.kind == "speech" and tag.speaker_id and tag.start_ms <= right:
                return tag.speaker_id
        return self._current_speaker_id

    def _append_embedding_frame(self, frame: bytes, end_ms: float) -> tuple[bytes, float, float] | None:
        self._embedding_frames.append(frame)
        self._embedding_samples += len(frame) // 2
        while self._embedding_frames:
            oldest_samples = len(self._embedding_frames[0]) // 2
            if self._embedding_samples - oldest_samples < self.embedding_window_samples:
                break
            dropped = self._embedding_frames.pop(0)
            self._embedding_samples -= len(dropped) // 2
        if self._embedding_samples < self.embedding_window_samples:
            self._log_embedding_wait(
                "window",
                self._embedding_samples,
                self.embedding_window_samples,
            )
            return None
        if self._processed_samples - self._last_embedding_sample < self.embedding_hop_samples:
            self._log_embedding_wait(
                "hop",
                self._processed_samples - self._last_embedding_sample,
                self.embedding_hop_samples,
            )
            return None
        self._last_embedding_sample = self._processed_samples
        logger.info(
            "speaker embedding window ready id=%s speaker=%s window_sec=%.2f end_ms=%.0f",
            self.session_id,
            self._current_speaker_id,
            self._embedding_samples / self.sample_rate,
            end_ms,
        )
        window_bytes = b"".join(self._embedding_frames)
        start_ms = max(0.0, end_ms - self._embedding_samples / self.sample_rate * 1000)
        return window_bytes, start_ms, end_ms

    def _log_embedding_wait(self, reason: str, current_samples: int, needed_samples: int) -> None:
        if self._processed_samples - self._last_embedding_wait_log_sample < self.sample_rate:
            return
        self._last_embedding_wait_log_sample = self._processed_samples
        logger.info(
            "speaker embedding wait-%s id=%s speaker=%s current_sec=%.2f need_sec=%.2f "
            "prompt_sec=%.2f",
            reason,
            self.session_id,
            self._current_speaker_id,
            current_samples / self.sample_rate,
            needed_samples / self.sample_rate,
            self._prompt_samples / self.sample_rate,
        )

    def _update_centroid(self, signature: np.ndarray) -> None:
        signature = _normalize_vector(signature)
        if self._centroid is None:
            self._centroid = signature
            self._centroid_weight = 1
            self._speaker_centroids[self._current_speaker_id] = self._centroid
            self._speaker_centroid_weights[self._current_speaker_id] = self._centroid_weight
            return
        weight = min(self._centroid_weight, 30)
        self._centroid = (self._centroid * weight + signature) / (weight + 1)
        norm = np.linalg.norm(self._centroid)
        if norm > 0:
            self._centroid = self._centroid / norm
        self._centroid_weight = weight + 1
        self._speaker_centroids[self._current_speaker_id] = self._centroid
        self._speaker_centroid_weights[self._current_speaker_id] = self._centroid_weight

    def _match_existing_speaker(
        self,
        embedding: np.ndarray,
        *,
        exclude: str,
    ) -> tuple[str | None, float]:
        best_speaker: str | None = None
        best_similarity = -1.0
        for speaker_id, centroid in self._speaker_centroids.items():
            if speaker_id == exclude:
                continue
            candidate_similarity = _cosine_similarity(embedding, centroid)
            if candidate_similarity > best_similarity:
                best_speaker = speaker_id
                best_similarity = candidate_similarity
        if best_speaker is None or best_similarity < self.speaker_reidentify_threshold:
            return None, best_similarity
        return best_speaker, best_similarity

    def _append_tag(
        self,
        kind: AudioTagKind,
        start_ms: float,
        end_ms: float,
        speaker_id: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        if self._tags:
            last = self._tags[-1]
            if (
                last.kind == kind
                and last.speaker_id == speaker_id
                and kind != "speaker_change"
                and abs(last.end_ms - start_ms) < 1.0
            ):
                last.end_ms = end_ms
                last.confidence = min(last.confidence, confidence)
                return
        self._tags.append(AudioTag(kind, start_ms, end_ms, speaker_id, confidence))
        if len(self._tags) > 1000:
            del self._tags[:200]

    def _write_prompt(
        self,
        pcm_bytes: bytes,
        speaker_id: str,
        export_index: int,
        prompt_kind: str,
    ) -> str | None:
        if not pcm_bytes:
            return None
        if prompt_kind == "fixed":
            path = self.fixed_output_dir / f"{speaker_id}_fixed_prompt.wav"
        else:
            path = self.rolling_output_dir / f"{speaker_id}_rolling_prompt_{export_index:04d}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        prompt_path = os.path.abspath(path)
        logger.info(
            "speaker %s prompt exported id=%s speaker=%s seconds=%.2f path=%s",
            prompt_kind,
            self.session_id,
            speaker_id,
            len(pcm_bytes) / 2 / self.sample_rate,
            prompt_path,
        )
        return prompt_path


def _rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-8)
    return 20.0 * np.log10(rms)


def _voice_signature(samples: np.ndarray, bands: int = 16) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((bands,), dtype=np.float32)
    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed)) + 1e-6
    spectrum = spectrum[1:]
    if spectrum.size < bands:
        padded = np.zeros((bands,), dtype=np.float32)
        padded[: spectrum.size] = spectrum
        spectrum = padded
    splits = np.array_split(np.log(spectrum), bands)
    features = np.array([float(np.mean(part)) for part in splits], dtype=np.float32)
    features -= float(np.mean(features))
    return _normalize_vector(features)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm > 0:
        value = value / norm
    return value


def _create_embedding_backend(provider: str, model: str) -> SpeakerEmbeddingBackend:
    normalized = (provider or "").strip().lower()
    if normalized in {"modelscope_campplus", "campplus", "modelscope"}:
        return ModelScopeSpeakerEmbeddingBackend(model)
    if normalized in {"spectral", "fallback", "none"}:
        return SpectralSpeakerEmbeddingBackend()
    logger.warning("unknown speaker embedding provider=%s; fallback=spectral", provider)
    return SpectralSpeakerEmbeddingBackend()


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _safe_path_label(label: str | None) -> str:
    value = (label or "").strip()
    if not value:
        return ""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32)
    duration = samples.size / source_rate
    target_size = max(1, round(duration * target_rate))
    source_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)
