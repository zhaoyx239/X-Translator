from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent


def _resolve_runtime_path(value: str | Path, *, config_path: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if str(path).startswith("xtranslate/"):
        return (PROJECT_ROOT / path).resolve()
    if config_path is not None:
        return (config_path.resolve().parent / path).resolve()
    return (APP_DIR / path).resolve()


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


@dataclass(slots=True)
class Settings:
    """运行配置。"""

    host: str = "0.0.0.0"
    port: int = 7654
    asr_provider: str = "sensevoice"
    qwen3_asr_url: str = "http://localhost:8001/v1/recognize"
    zipformer_server_uri: str = "ws://127.0.0.1:8765"
    zipformer_chunk_size: int = 16000
    zipformer_window_seconds: float = 3.2
    zipformer_init_turn_seconds: float = 1.6
    zipformer_use_ctpunc: bool = True
    zipformer_punc_model: str = (
        "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727"
    )
    zipformer_punc_model_revision: str = "v2.0.4"
    paraformer_model: str = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    paraformer_hub: str = "ms"
    sensevoice_language: str = "zh"
    sensevoice_device: str = "cuda"
    translation_provider: str = "hunyuan"
    hunyuan_url: str = "http://localhost:8000/v1/chat/completions"
    lmt_url: str = "http://localhost:8002/v1/chat/completions"
    translation_model: str = ""
    translation_max_tokens: int = 256
    index_tts_url: str = "http://localhost:11996/tts_url"
    omni_tts_url: str = "http://localhost:11997/synthesize"
    xvoice_tts_url: str = "http://localhost:11998/synthesize"
    tts_provider: str = "index"
    source_lang: str = "zh"
    target_lang: str = "en"
    sample_rate: int = 16000
    tts_sample_rate: int = 48000
    asr_chunk_seconds: float = 1.6
    pre_buffer_frames: int = 20
    paraformer_chunk_size: int = 2560
    paraformer_window_seconds: float = 3.2
    paraformer_init_turn_seconds: float = 1.6
    paraformer_device: str = "cuda"
    clone_window_seconds: float = 3.0
    speaker_prompt_enabled: bool = True
    speaker_embedding_provider: str = "modelscope_campplus"
    speaker_embedding_model: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    speaker_embedding_window_seconds: float = 1.0
    speaker_embedding_hop_seconds: float = 0.5
    speaker_prompt_min_seconds: float = 1.0
    speaker_prompt_max_seconds: float = 6.0
    speaker_vad_frame_ms: int = 40
    speaker_silence_dbfs: float = -42.0
    speaker_vad_margin_db: float = 10.0
    speaker_change_threshold: float = 0.55
    emit_min_chars: int = 6
    emit_interval_seconds: float = 0.8
    show_speaker: bool = True
    session_audio_dir: Path = APP_DIR / "runtime/session_audio"
    log_dir: Path = APP_DIR / "runtime/logs"
    visit_counter_path: Path = APP_DIR / "runtime/visit_stats.json"
    config_path: Path = APP_DIR / "config.json"

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量加载配置。"""
        settings = cls()
        config_path = Path(os.getenv("XTRANSLATE_CONFIG", str(settings.config_path)))
        if config_path.exists():
            settings = cls.from_config_file(config_path)
        else:
            settings.config_path = config_path.resolve()

        settings.host = os.getenv("XTRANSLATE_HOST", settings.host)
        settings.port = int(os.getenv("XTRANSLATE_PORT", str(settings.port)))
        settings.asr_provider = os.getenv("XTRANSLATE_ASR_PROVIDER", settings.asr_provider)
        settings.qwen3_asr_url = os.getenv("XTRANSLATE_QWEN3_ASR_URL", settings.qwen3_asr_url)
        settings.zipformer_server_uri = os.getenv(
            "XTRANSLATE_ZIPFORMER_SERVER_URI",
            settings.zipformer_server_uri,
        )
        zipformer_use_ctpunc_env = os.getenv("XTRANSLATE_ZIPFORMER_USE_CTPUNC")
        if zipformer_use_ctpunc_env is not None:
            settings.zipformer_use_ctpunc = _parse_bool(
                zipformer_use_ctpunc_env,
                settings.zipformer_use_ctpunc,
            )
        settings.zipformer_punc_model = os.getenv(
            "XTRANSLATE_ZIPFORMER_PUNC_MODEL",
            settings.zipformer_punc_model,
        )
        settings.zipformer_punc_model_revision = os.getenv(
            "XTRANSLATE_ZIPFORMER_PUNC_MODEL_REVISION",
            settings.zipformer_punc_model_revision,
        )
        settings.sensevoice_language = os.getenv(
            "XTRANSLATE_SENSEVOICE_LANGUAGE",
            settings.sensevoice_language,
        )
        settings.sensevoice_device = os.getenv(
            "XTRANSLATE_SENSEVOICE_DEVICE",
            settings.sensevoice_device,
        )
        settings.translation_provider = os.getenv(
            "XTRANSLATE_TRANSLATION_PROVIDER",
            settings.translation_provider,
        )
        settings.hunyuan_url = os.getenv("XTRANSLATE_HUNYUAN_URL", settings.hunyuan_url)
        settings.lmt_url = os.getenv("XTRANSLATE_LMT_URL", settings.lmt_url)
        settings.translation_model = os.getenv(
            "XTRANSLATE_TRANSLATION_MODEL",
            settings.translation_model,
        )
        settings.translation_max_tokens = int(
            os.getenv(
                "XTRANSLATE_TRANSLATION_MAX_TOKENS",
                str(settings.translation_max_tokens),
            )
        )
        settings.index_tts_url = os.getenv("XTRANSLATE_INDEX_TTS_URL", settings.index_tts_url)
        settings.omni_tts_url = os.getenv("XTRANSLATE_OMNI_TTS_URL", settings.omni_tts_url)
        settings.xvoice_tts_url = os.getenv("XTRANSLATE_XVOICE_TTS_URL", settings.xvoice_tts_url)
        settings.tts_provider = os.getenv("XTRANSLATE_TTS_PROVIDER", settings.tts_provider)
        settings.source_lang = os.getenv("XTRANSLATE_SOURCE_LANG", settings.source_lang)
        settings.target_lang = os.getenv("XTRANSLATE_TARGET_LANG", settings.target_lang)
        show_speaker_env = os.getenv("XTRANSLATE_SHOW_SPEAKER")
        if show_speaker_env is not None:
            settings.show_speaker = show_speaker_env.strip().lower() not in ("0", "false", "no", "off", "")
        settings.session_audio_dir = _resolve_runtime_path(
            os.getenv("XTRANSLATE_SESSION_AUDIO_DIR", str(settings.session_audio_dir)),
            config_path=settings.config_path,
        )
        settings.log_dir = _resolve_runtime_path(
            os.getenv("XTRANSLATE_LOG_DIR", str(settings.log_dir)),
            config_path=settings.config_path,
        )
        settings.visit_counter_path = _resolve_runtime_path(
            os.getenv("XTRANSLATE_VISIT_COUNTER_PATH", str(settings.visit_counter_path)),
            config_path=settings.config_path,
        )
        return settings

    @classmethod
    def from_config_file(cls, config_path: Path) -> "Settings":
        """从 JSON 配置文件加载。"""
        with config_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)

        server = raw.get("server", {})
        audio = raw.get("audio", {})
        asr = raw.get("asr", {})
        translation = raw.get("translation", {})
        tts = raw.get("tts", {})
        storage = raw.get("storage", {})
        frontend = raw.get("frontend", {})

        resolved_config_path = config_path.resolve()
        return cls(
            host=server.get("host", "0.0.0.0"),
            port=int(server.get("port", 7654)),
            asr_provider=asr.get("provider", "sensevoice"),
            qwen3_asr_url=asr.get("qwen3_asr_url", "http://localhost:8001/v1/recognize"),
            zipformer_server_uri=asr.get("zipformer_server_uri", "ws://127.0.0.1:8765"),
            zipformer_chunk_size=int(asr.get("zipformer_chunk_size", 16000)),
            zipformer_window_seconds=float(asr.get("zipformer_window_seconds", 3.2)),
            zipformer_init_turn_seconds=float(asr.get("zipformer_init_turn_seconds", 1.6)),
            zipformer_use_ctpunc=_parse_bool(asr.get("zipformer_use_ctpunc"), True),
            zipformer_punc_model=asr.get(
                "zipformer_punc_model",
                "iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
            ),
            zipformer_punc_model_revision=asr.get(
                "zipformer_punc_model_revision",
                "v2.0.4",
            ),
            paraformer_model=asr.get(
                "paraformer_model",
                "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            ),
            paraformer_hub=asr.get("paraformer_hub", "ms"),
            sensevoice_language=asr.get("sensevoice_language", "zh"),
            sensevoice_device=asr.get("sensevoice_device", "cuda"),
            translation_provider=translation.get("provider", "hunyuan"),
            hunyuan_url=translation.get("hunyuan_url", "http://localhost:8000/v1/chat/completions"),
            lmt_url=translation.get("lmt_url", "http://localhost:8002/v1/chat/completions"),
            translation_model=translation.get("model", ""),
            translation_max_tokens=int(translation.get("max_tokens", 256)),
            index_tts_url=tts.get("index_tts_url", "http://localhost:11996/tts_url"),
            omni_tts_url=tts.get("omni_tts_url", "http://localhost:11997/synthesize"),
            xvoice_tts_url=tts.get("xvoice_tts_url", "http://localhost:11998/synthesize"),
            tts_provider=tts.get("provider", "index"),
            source_lang=translation.get("source_lang", "zh"),
            target_lang=translation.get("target_lang", "en"),
            sample_rate=int(audio.get("sample_rate", 16000)),
            tts_sample_rate=int(audio.get("tts_sample_rate", 48000)),
            asr_chunk_seconds=float(asr.get("chunk_seconds", 1.6)),
            pre_buffer_frames=int(audio.get("pre_buffer_frames", 20)),
            paraformer_chunk_size=int(asr.get("paraformer_chunk_size", 2560)),
            paraformer_window_seconds=float(asr.get("paraformer_window_seconds", 3.2)),
            paraformer_init_turn_seconds=float(asr.get("paraformer_init_turn_seconds", 1.6)),
            paraformer_device=asr.get("paraformer_device", "cuda"),
            clone_window_seconds=float(tts.get("clone_window_seconds", 3.0)),
            speaker_prompt_enabled=bool(tts.get("speaker_prompt_enabled", True)),
            speaker_embedding_provider=tts.get("speaker_embedding_provider", "modelscope_campplus"),
            speaker_embedding_model=tts.get(
                "speaker_embedding_model",
                "iic/speech_campplus_sv_zh-cn_16k-common",
            ),
            speaker_embedding_window_seconds=float(
                tts.get("speaker_embedding_window_seconds", 1.0)
            ),
            speaker_embedding_hop_seconds=float(tts.get("speaker_embedding_hop_seconds", 0.5)),
            speaker_prompt_min_seconds=float(tts.get("speaker_prompt_min_seconds", 1.0)),
            speaker_prompt_max_seconds=float(tts.get("speaker_prompt_max_seconds", 6.0)),
            speaker_vad_frame_ms=int(tts.get("speaker_vad_frame_ms", 40)),
            speaker_silence_dbfs=float(tts.get("speaker_silence_dbfs", -42.0)),
            speaker_vad_margin_db=float(tts.get("speaker_vad_margin_db", 10.0)),
            speaker_change_threshold=float(tts.get("speaker_change_threshold", 0.55)),
            emit_min_chars=int(translation.get("emit_min_chars", 6)),
            emit_interval_seconds=float(translation.get("emit_interval_seconds", 0.8)),
            show_speaker=bool(frontend.get("show_speaker", True)),
            session_audio_dir=_resolve_runtime_path(
                storage.get("session_audio_dir", "xtranslate/runtime/session_audio"),
                config_path=resolved_config_path,
            ),
            log_dir=_resolve_runtime_path(
                storage.get("log_dir", "xtranslate/runtime/logs"),
                config_path=resolved_config_path,
            ),
            visit_counter_path=_resolve_runtime_path(
                storage.get("visit_counter_path", "xtranslate/runtime/visit_stats.json"),
                config_path=resolved_config_path,
            ),
            config_path=resolved_config_path,
        )
