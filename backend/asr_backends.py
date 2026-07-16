from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
import soxr

from .logging_utils import setup_logger
from .services import Qwen3ASRService, ZipformerASRService

logger = setup_logger()


def _preview_text(text: str, limit: int = 80) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def split_cn_en(text: str) -> list[str]:
    """按中英数 token 切分。"""
    return re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|[0-9]+", text)


def check_en(text: str) -> bool:
    """判断 token 是否英文。"""
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


def get_lcs_substrings(s1: list[str], s2: list[str]) -> tuple[list[str], list[str]]:
    """获取最长公共子序列后的切片。"""
    if not s1 or not s2:
        return s1, s2
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if s1[i] == s2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    max_len = dp[0][0]
    if max_len == 0:
        return s1, s2
    start_i, start_j = -1, -1
    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j] and dp[i][j] == max_len:
                start_i, start_j = i, j
                break
        if start_i != -1:
            break
    return s1[start_i:], s2[start_j:]


def remove_leading_backchannel(text: str) -> str:
    """去掉句首语气词。"""
    backchannel_chars = {"嗯", "啊", "哦", "噢", "呃", "哎", "哼", "嘿"}
    punctuation_chars = {
        " ",
        ",",
        ".",
        "?",
        "!",
        "，",
        "。",
        "？",
        "！",
        "、",
        "；",
        ";",
        "…",
        ":",
        "：",
    }
    skip_chars = backchannel_chars.union(punctuation_chars)
    for index, char in enumerate(text):
        if char not in skip_chars:
            return text[index:]
    return ""


def normalize_text(text: str) -> str:
    """归一化比较文本。"""
    text = remove_leading_backchannel(text).strip().lower()
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~，。！？；：、…"""
    trans = str.maketrans("", "", punctuation)
    text = text.translate(trans)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_text_prefix(prefix: str, current: str) -> str:
    """将已有前缀与当前文本拼接，并尽量消除边界重复。"""
    left = prefix.strip()
    right = current.strip()
    if not left:
        return right
    if not right:
        return left
    max_overlap = min(len(left), len(right))
    for size in range(max_overlap, 0, -1):
        if left[-size:] == right[:size]:
            return left + right[size:]
    return left + right


def strip_prefix_text(text: str, prefix: str) -> str:
    """从完整文本中去掉已提交前缀。"""
    value = text.strip()
    base = prefix.strip()
    if not base:
        return value
    if value.startswith(base):
        return value[len(base) :].strip()
    if base.endswith(value):
        return ""
    return value


def split_text_by_committable_punctuation(text: str) -> tuple[str, str]:
    """切分可提前提交的前缀和仍在波动的尾部。"""
    value = text.strip()
    if not value:
        return "", ""
    punctuation_indexes = [
        index
        for index, char in enumerate(value)
        if char in "，。！？；：,.!?;:"
    ]
    if len(punctuation_indexes) < 2:
        return "", value
    cutoff = punctuation_indexes[-2] + 1
    return value[:cutoff].strip(), value[cutoff:].strip()


class ParaformerASR:
    """完全按 ASR_demo 路线工作的 Paraformer backend。"""

    def __init__(
        self,
        device: str = "cuda",
        model: str = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        hub: str = "ms",
    ):
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        try:
            self.asr_pipeline = pipeline(
                task=Tasks.auto_speech_recognition,
                model=model,
                model_revision="v2.0.4",
                device=device,
                disable_pbar=True,
                disable_update=True,
            )
        except Exception:
            self.asr_pipeline = pipeline(
                task=Tasks.auto_speech_recognition,
                model=model,
                device=device,
                disable_pbar=True,
                disable_update=True,
            )

    def recognize(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """执行识别。"""
        total_start = time.perf_counter()
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        mean_elapsed_ms = (time.perf_counter() - total_start) * 1000

        resample_start = time.perf_counter()
        if sample_rate != 16000:
            audio_chunk = soxr.resample(audio_chunk, sample_rate, 16000)
        resample_elapsed_ms = (time.perf_counter() - resample_start) * 1000

        pipeline_start = time.perf_counter()
        try:
            text = self.asr_pipeline(audio_chunk)[0]["text"].strip()
        except Exception as exc:
            logger.warning("Paraformer recognize failed: %s", exc)
            return ""
        pipeline_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        total_elapsed_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "asr paraformer recognize samples=%s mean_ms=%.1f resample_ms=%.1f pipeline_ms=%.1f total_ms=%.1f",
            len(audio_chunk),
            mean_elapsed_ms,
            resample_elapsed_ms,
            pipeline_elapsed_ms,
            total_elapsed_ms,
        )
        return text


class SensevoiceASR:
    """保留 SenseVoice backend。"""

    def __init__(self, language: str = "auto", device: str = "cuda"):
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self.sensevoice_model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=False,
            device=device,
            disable_pbar=True,
            disable_update=True,
        )
        self.language = language
        self.rich_transcription_postprocess = rich_transcription_postprocess
        self.pattern = "[" + "".join(
            {"😊", "😔", "😡", "😰", "🤢", "😮", "🎼", "👏", "😀", "😭", "🤧", "😷"}
        ) + "]"

    def clean_text(self, text: str) -> str:
        """清理情绪符号。"""
        if not re.search(r"[\u4e00-\u9fff]|[a-zA-Z]", text):
            return ""
        return re.sub(self.pattern, "", text)

    def recognize(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> str:
        """执行识别。"""
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=1)
        if sample_rate != 16000:
            audio_chunk = soxr.resample(audio_chunk, sample_rate, 16000)
        current_language = self.language if language is None else language
        try:
            result = self.rich_transcription_postprocess(
                self.sensevoice_model.generate(
                    input=audio_chunk,
                    cache={},
                    language=current_language,
                    use_itn=True,
                    batch_size=16,
                )[0]["text"]
            ).strip()
            result = self.clean_text(result)
        except Exception as exc:
            logger.warning("SenseVoice recognize failed: %s", exc)
            result = ""
        return result


class Qwen3ASR:
    """对接外部 Qwen3ASR service 的 backend。"""

    def __init__(self, service: Qwen3ASRService, sample_rate: int = 16000):
        self.service = service
        self.sample_rate = sample_rate
        self.cache: dict[str, str] = {}

    def recognize(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        is_final: bool = False,
        committed_text: str = "",
    ) -> str:
        """执行流式识别。"""
        return self.service.recognize_stream(
            audio_chunk,
            self.cache,
            is_final=is_final,
            sample_rate=sample_rate or self.sample_rate,
            committed_text=committed_text,
        )

    def finalize(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """结束当前流式轮次。"""
        return self.recognize(audio_chunk, sample_rate=sample_rate, is_final=True)

    def reset(self) -> None:
        """重置远端流式会话。"""
        self.service.reset_stream(self.cache)

    def commit(self, committed_text: str, sample_rate: int = 16000) -> str:
        """提交已消费前缀，让远端重建流式状态。"""
        return self.service.commit_stream(
            self.cache,
            committed_text=committed_text,
            sample_rate=sample_rate or self.sample_rate,
        )


class ZipformerASR:
    """对接 streaming Zipformer websocket service 的 backend。"""

    def __init__(self, service: ZipformerASRService, sample_rate: int = 16000):
        self.service = service
        self.sample_rate = sample_rate

    def recognize(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """执行流式识别。"""
        if sample_rate != self.sample_rate:
            audio_chunk = soxr.resample(audio_chunk, sample_rate, self.sample_rate)
        return self.service.recognize_stream(audio_chunk)

    def finalize(self) -> str:
        """结束当前流式轮次。"""
        return self.service.finalize_stream()

    def reset(self) -> None:
        """重置远端流式会话。"""
        self.service.reset_stream()

    def close(self) -> None:
        """关闭远端连接。"""
        self.service.close()


@dataclass
class ASRStreamingConfig:
    sample_rate: int = 16000
    chunk_size: int = 2560
    asr_window_sec: float = 3.2
    init_turn_sec: float = 1.6
    silence_threshold: float = 0.01
    max_wait_chunks: int = 8
    device: str = "cuda"
    model: str = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    hub: str = "ms"


class LocalASRStreamingRecognizer:
    """本地 ASR 的伪流式识别器。"""

    def __init__(
        self,
        config: ASRStreamingConfig,
        asr: ParaformerASR | SensevoiceASR | None = None,
    ):
        self.cfg = config
        self.asr = asr or ParaformerASR(
            device=config.device,
            model=config.model,
            hub=config.hub,
        )
        self.reset()

    def reset(self) -> None:
        """重置状态。"""
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.cascade_buffer = np.zeros(
            int(self.cfg.asr_window_sec * self.cfg.sample_rate), dtype=np.float32
        )
        self.turn_audio_buffer = np.zeros(
            int(self.cfg.init_turn_sec * self.cfg.sample_rate), dtype=np.float32
        )
        self.cascade_text = ""
        self.delta_text: list[str] = []
        self.speech_detected = False
        self.wait_silence_cnt = 0

    def _normalize_tokens(self, text: str) -> list[str]:
        return split_cn_en(normalize_text(text))

    def _join_tokens(self, tokens: list[str]) -> str:
        return "".join([(token + " ") if check_en(token) else token for token in tokens]).strip()

    def _diff_text(self, full_text: str) -> tuple[str, str]:
        norm_full = self._normalize_tokens(full_text)
        norm_hist = self._normalize_tokens(self.cascade_text)

        if len(norm_full) >= 5 and len(norm_hist) >= 5:
            backup_norm_full = norm_full.copy()
            backup_norm_hist = norm_hist.copy()
            norm_full, norm_hist = get_lcs_substrings(norm_full, norm_hist)
        else:
            backup_norm_full = norm_full
            backup_norm_hist = norm_hist

        prev_delta = self.delta_text[-1] if self.delta_text else ""
        prev_delta_split = split_cn_en(prev_delta)
        len_prev = len(prev_delta_split)

        if len_prev > len(norm_hist):
            norm_full = backup_norm_full
            norm_hist = backup_norm_hist

        history_base = norm_hist[:-len_prev] if len_prev > 0 else norm_hist

        need_correction = False
        corrected_prev_delta = ""
        delta = ""

        if len(norm_full) > len(norm_hist):
            current_segment = norm_full[len(history_base) : len(history_base) + len_prev]
            if current_segment == prev_delta_split:
                delta = self._join_tokens(norm_full[len(norm_hist) :])
            else:
                need_correction = True
                corrected_prev_delta = self._join_tokens(current_segment)
                delta = self._join_tokens(norm_full[len(history_base) + len_prev :])
        elif len(norm_full) == len(norm_hist):
            current_segment = norm_full[len(history_base) :]
            if current_segment != prev_delta_split:
                need_correction = True
                corrected_prev_delta = self._join_tokens(current_segment)
        else:
            need_correction = True
            remainder = norm_full[len(history_base) :]
            corrected_prev_delta = ""
            delta = self._join_tokens(remainder)

        if need_correction and self.delta_text:
            self.delta_text[-1] = corrected_prev_delta

        self.cascade_text = self._join_tokens(norm_full)
        self.delta_text.append(delta)
        self.delta_text = self.delta_text[-20:]
        return delta, self.cascade_text

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))

    def _recognize(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        return self.asr.recognize(audio, sample_rate=self.cfg.sample_rate)

    def feed_audio(self, audio: np.ndarray) -> list[dict]:
        """喂入音频。"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        outputs: list[dict] = []

        while len(self.audio_buffer) >= self.cfg.chunk_size:
            chunk_start = time.perf_counter()
            chunk = self.audio_buffer[: self.cfg.chunk_size]
            self.audio_buffer = self.audio_buffer[self.cfg.chunk_size :]

            cascade_start = time.perf_counter()
            self.cascade_buffer = np.concatenate([self.cascade_buffer, chunk])
            max_len = int(self.cfg.asr_window_sec * self.cfg.sample_rate)
            if len(self.cascade_buffer) > max_len:
                self.cascade_buffer = self.cascade_buffer[-max_len:]
            cascade_elapsed_ms = (time.perf_counter() - cascade_start) * 1000

            rms_start = time.perf_counter()
            rms = self._rms(chunk)
            if rms >= self.cfg.silence_threshold:
                self.speech_detected = True
                self.wait_silence_cnt = 0
                self.turn_audio_buffer = np.concatenate([self.turn_audio_buffer, chunk])
            elif self.speech_detected:
                self.wait_silence_cnt += 1
            rms_elapsed_ms = (time.perf_counter() - rms_start) * 1000

            recognize_start = time.perf_counter()
            full = self._recognize(self.cascade_buffer) if self.speech_detected else ""
            recognize_elapsed_ms = (time.perf_counter() - recognize_start) * 1000

            diff_start = time.perf_counter()
            delta, stable = self._diff_text(full) if full else ("", self.cascade_text)
            diff_elapsed_ms = (time.perf_counter() - diff_start) * 1000
            outputs.append(
                {
                    "type": "partial",
                    "text": stable,
                    "delta": delta,
                    "ts": time.time(),
                }
            )

            if (
                self.speech_detected
                and self.wait_silence_cnt >= self.cfg.max_wait_chunks
                and self.turn_audio_buffer.size > 0
            ):
                final_start = time.perf_counter()
                final_text = self._recognize(self.turn_audio_buffer)
                final_tokens = self._normalize_tokens(final_text)
                outputs.append(
                    {
                        "type": "final",
                        "text": self._join_tokens(final_tokens),
                        "delta": "",
                        "ts": time.time(),
                    }
                )
                self.turn_audio_buffer = np.zeros(
                    int(self.cfg.init_turn_sec * self.cfg.sample_rate), dtype=np.float32
                )
                self.speech_detected = False
                self.wait_silence_cnt = 0
                final_elapsed_ms = (time.perf_counter() - final_start) * 1000
            else:
                final_elapsed_ms = 0.0

            chunk_elapsed_ms = (time.perf_counter() - chunk_start) * 1000
            logger.info(
                "asr feed chunk chunk_samples=%s cascade_samples=%s speech=%s rms=%.5f "
                "cascade_ms=%.1f rms_ms=%.1f recognize_ms=%.1f diff_ms=%.1f final_ms=%.1f chunk_ms=%.1f",
                len(chunk),
                len(self.cascade_buffer),
                self.speech_detected,
                rms,
                cascade_elapsed_ms,
                rms_elapsed_ms,
                recognize_elapsed_ms,
                diff_elapsed_ms,
                final_elapsed_ms,
                chunk_elapsed_ms,
            )

        if not outputs:
            outputs.append(
                {
                    "type": "blank",
                    "text": self.cascade_text,
                    "delta": "",
                    "ts": time.time(),
                }
            )
        return outputs

    def finalize(self) -> dict:
        """结束一轮。"""
        if self.audio_buffer.size > 0:
            self.turn_audio_buffer = np.concatenate([self.turn_audio_buffer, self.audio_buffer])
            self.audio_buffer = np.zeros(0, dtype=np.float32)
        final_text = self._recognize(self.turn_audio_buffer)
        result = {
            "type": "final",
            "text": final_text.strip(),
            "delta": "",
            "ts": time.time(),
        }
        self.reset()
        return result


class Qwen3StreamingRecognizer:
    """远端 Qwen3ASR 的增量流式识别器。"""

    def __init__(self, config: ASRStreamingConfig, asr: Qwen3ASR):
        self.cfg = config
        self.asr = asr
        self.reset()

    def reset(self) -> None:
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.speech_detected = False
        self.wait_silence_cnt = 0
        self.last_text = ""
        self.last_partial = ""
        self.committed_text = ""
        self.pending_tail_text = ""
        self.asr.reset()

    @staticmethod
    def _preview(text: str, limit: int = 80) -> str:
        value = (text or "").strip()
        if len(value) <= limit:
            return value
        return f"{value[:limit]}..."

    def _commit_prefix(self, committed_text: str) -> str:
        if not committed_text.strip():
            return self.pending_tail_text
        refreshed_text = self.asr.commit(
            committed_text,
            sample_rate=self.cfg.sample_rate,
        ).strip()
        logger.info(
            "qwen3 commit committed=%s refreshed=%s",
            self._preview(committed_text),
            self._preview(refreshed_text),
        )
        self.last_text = refreshed_text
        self.pending_tail_text = refreshed_text
        self.last_partial = refreshed_text
        return refreshed_text

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))

    def _recognize_chunk(self, audio: np.ndarray, is_final: bool = False) -> str:
        if audio.size == 0 and not is_final:
            return self.pending_tail_text
        text = self.asr.recognize(
            audio,
            sample_rate=self.cfg.sample_rate,
            is_final=is_final,
        ).strip()
        logger.info(
            "qwen3 recognize is_final=%s text=%s",
            is_final,
            self._preview(text),
        )
        self.last_text = text
        self.pending_tail_text = text
        return text

    def _extract_outputs_from_text(self, full_text: str) -> list[dict]:
        outputs: list[dict] = []
        committable, _tail = split_text_by_committable_punctuation(full_text)

        if committable:
            merged_committed = merge_text_prefix(self.committed_text, committable)
            new_final = strip_prefix_text(merged_committed, self.committed_text)
            if new_final:
                self.committed_text = merged_committed
                full_text = self._commit_prefix(self.committed_text)
                outputs.append(
                    {
                        "type": "final",
                        "text": new_final,
                        "delta": "",
                        "ts": time.time(),
                    }
                )
            else:
                self.committed_text = merged_committed

        partial_text = strip_prefix_text(full_text, self.committed_text)
        logger.info(
            "qwen3 extract full=%s committed=%s partial=%s",
            self._preview(full_text),
            self._preview(self.committed_text),
            self._preview(partial_text),
        )
        if partial_text and partial_text != self.last_partial:
            self.pending_tail_text = partial_text
            self.last_partial = partial_text
            outputs.append(
                {
                    "type": "partial",
                    "text": partial_text,
                    "delta": "",
                    "ts": time.time(),
                }
            )

        return outputs

    def feed_audio(self, audio: np.ndarray) -> list[dict]:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        outputs: list[dict] = []

        while len(self.audio_buffer) >= self.cfg.chunk_size:
            chunk_start = time.perf_counter()
            chunk = self.audio_buffer[: self.cfg.chunk_size]
            self.audio_buffer = self.audio_buffer[self.cfg.chunk_size :]

            rms = self._rms(chunk)
            should_recognize = False
            if rms >= self.cfg.silence_threshold:
                self.speech_detected = True
                self.wait_silence_cnt = 0
                should_recognize = True
            elif self.speech_detected:
                self.wait_silence_cnt += 1
                should_recognize = True

            text = self._recognize_chunk(chunk, is_final=False) if should_recognize else ""
            if text:
                outputs.extend(self._extract_outputs_from_text(text))

            if self.speech_detected and self.wait_silence_cnt >= self.cfg.max_wait_chunks:
                final_text = self._recognize_chunk(
                    np.zeros(0, dtype=np.float32),
                    is_final=True,
                )
                remaining_text = strip_prefix_text(final_text, self.committed_text)
                outputs.append(
                    {
                        "type": "final",
                        "text": remaining_text or final_text,
                        "delta": "",
                        "ts": time.time(),
                    }
                )
                self.speech_detected = False
                self.wait_silence_cnt = 0
                self.last_text = ""
                self.pending_tail_text = ""
                self.last_partial = ""
                self.committed_text = ""

            logger.info(
                "asr qwen3 feed chunk chunk_samples=%s speech=%s rms=%.5f chunk_ms=%.1f",
                len(chunk),
                self.speech_detected,
                rms,
                (time.perf_counter() - chunk_start) * 1000,
            )

        if not outputs:
            outputs.append(
                {
                    "type": "blank",
                    "text": self.pending_tail_text if self.speech_detected else "",
                    "delta": "",
                    "ts": time.time(),
                }
            )
        return outputs

    def finalize(self) -> dict:
        pending = self.audio_buffer
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        if pending.size > 0:
            self._recognize_chunk(pending, is_final=False)
        final_text = self._recognize_chunk(np.zeros(0, dtype=np.float32), is_final=True)
        remaining_text = strip_prefix_text(final_text, self.committed_text)
        logger.info(
            "qwen3 finalize final=%s committed=%s remaining=%s",
            self._preview(final_text),
            self._preview(self.committed_text),
            self._preview(remaining_text),
        )
        result = {
            "type": "final",
            "text": remaining_text.strip(),
            "delta": "",
            "ts": time.time(),
        }
        self.reset()
        return result


class ZipformerStreamingRecognizer:
    """远端 Zipformer websocket 的流式识别器。"""

    def __init__(self, config: ASRStreamingConfig, asr: ZipformerASR):
        self.cfg = config
        self.asr = asr
        self.reset()

    def reset(self) -> None:
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.speech_detected = False
        self.wait_silence_cnt = 0
        self.last_partial = ""
        self.asr.reset()

    @staticmethod
    def _rms(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32))))

    def feed_audio(self, audio: np.ndarray) -> list[dict]:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])
        outputs: list[dict] = []

        while len(self.audio_buffer) >= self.cfg.chunk_size:
            chunk_start = time.perf_counter()
            chunk = self.audio_buffer[: self.cfg.chunk_size]
            self.audio_buffer = self.audio_buffer[self.cfg.chunk_size :]

            rms = self._rms(chunk)
            partial_text = self.asr.recognize(chunk, sample_rate=self.cfg.sample_rate).strip()

            if rms >= self.cfg.silence_threshold or partial_text:
                self.speech_detected = True
                self.wait_silence_cnt = 0
            elif self.speech_detected:
                self.wait_silence_cnt += 1

            if partial_text and partial_text != self.last_partial:
                self.last_partial = partial_text
                outputs.append(
                    {
                        "type": "partial",
                        "text": partial_text,
                        "delta": "",
                        "ts": time.time(),
                    }
                )

            if self.speech_detected and self.wait_silence_cnt >= self.cfg.max_wait_chunks:
                final_text = self.asr.finalize().strip()
                outputs.append(
                    {
                        "type": "final",
                        "text": final_text or self.last_partial,
                        "delta": "",
                        "ts": time.time(),
                    }
                )
                self.audio_buffer = np.zeros(0, dtype=np.float32)
                self.speech_detected = False
                self.wait_silence_cnt = 0
                self.last_partial = ""
                self.asr.reset()

            logger.info(
                "asr zipformer feed chunk chunk_samples=%s speech=%s rms=%.5f chunk_ms=%.1f",
                len(chunk),
                self.speech_detected,
                rms,
                (time.perf_counter() - chunk_start) * 1000,
            )

        if not outputs:
            outputs.append(
                {
                    "type": "blank",
                    "text": self.last_partial if self.speech_detected else "",
                    "delta": "",
                    "ts": time.time(),
                }
            )
        return outputs

    def finalize(self) -> dict:
        pending = self.audio_buffer
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        if pending.size > 0:
            text = self.asr.recognize(pending, sample_rate=self.cfg.sample_rate).strip()
            if text:
                self.last_partial = text
        final_text = self.asr.finalize().strip()
        result = {
            "type": "final",
            "text": final_text or self.last_partial,
            "delta": "",
            "ts": time.time(),
        }
        self.reset()
        return result
