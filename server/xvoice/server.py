"""X-Voice TTS HTTP server.

The API intentionally matches the OmniVoice server used by xtranslate:

    POST /synthesize
        JSON: {"text": str, "audio_paths": [str], "ref_text": str | null, "language": str | null}
        Response: WAV bytes (audio/wav)

Stage2 drop-text mode is the default because it only needs reference audio.
Stage1 can be enabled with --stage 1 when a reference transcript is available.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

REPO_ROOT = Path(os.environ.get("XVOICE_ROOT", Path(__file__).resolve().parent)).expanduser().resolve()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

logger = logging.getLogger("xvoice_tts_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class SynthesizeRequest(BaseModel):
    text: str
    audio_paths: List[str] = []
    ref_text: Optional[str] = None
    language: Optional[str] = None
    gen_lang: Optional[str] = None
    ref_lang: Optional[str] = None
    speed: Optional[float] = None
    nfe_step: Optional[int] = None


@dataclass(slots=True)
class ServerConfig:
    stage: int
    model: str
    model_cfg: str
    ckpt_file: str
    ckpt_step: int | None
    vocab_file: str
    srp_ckpt_file: str
    srp_model_cfg: str
    device: str
    ref_lang: str | None
    gen_lang: str | None
    auto_detect_lang: bool
    normalize_text: bool
    vocoder_name: str
    load_vocoder_from_local: bool
    target_rms: float
    cross_fade_duration: float
    nfe_step: int
    cfg_strength: float
    layered: bool
    cfg_strength2: float
    cfg_schedule: str | None
    cfg_decay_time: float
    sway_sampling_coef: float
    speed: float
    denoise_ref: bool
    loudness_norm: bool
    post_processing: bool
    reverse: bool
    sp_type: str


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_repo_path(value: str | None) -> str:
    if not value:
        return ""
    if value.startswith("hf://"):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _parse_config() -> ServerConfig:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", type=int, default=int(os.getenv("XVOICE_STAGE", "2")))
    parser.add_argument("--model", type=str, default=os.getenv("XVOICE_MODEL"))
    parser.add_argument("--model-cfg", type=str, default=os.getenv("XVOICE_MODEL_CFG"))
    parser.add_argument("--ckpt-file", type=str, default=os.getenv("XVOICE_CKPT_FILE", ""))
    parser.add_argument("--ckpt-step", type=int, default=None)
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=os.getenv("XVOICE_VOCAB_FILE", "src/x_voice/infer/examples/vocab.txt"),
    )
    parser.add_argument("--srp-ckpt-file", type=str, default=os.getenv("XVOICE_SRP_CKPT_FILE", ""))
    parser.add_argument(
        "--srp-model-cfg",
        type=str,
        default=os.getenv("XVOICE_SRP_MODEL_CFG", "src/rate_pred/configs/SpeedPredict_Multilingual.yaml"),
    )
    parser.add_argument("--device", type=str, default=os.getenv("XVOICE_DEVICE", ""))
    parser.add_argument("--ref-lang", type=str, default=os.getenv("XVOICE_REF_LANG"))
    parser.add_argument("--gen-lang", type=str, default=os.getenv("XVOICE_GEN_LANG"))
    parser.add_argument("--auto-detect-lang", action="store_true", default=_env_bool("XVOICE_AUTO_DETECT_LANG", True))
    parser.add_argument("--no-auto-detect-lang", dest="auto_detect_lang", action="store_false")
    parser.add_argument("--normalize-text", action="store_true", default=_env_bool("XVOICE_NORMALIZE_TEXT", True))
    parser.add_argument("--no-normalize-text", dest="normalize_text", action="store_false")
    parser.add_argument("--vocoder-name", type=str, default=os.getenv("XVOICE_VOCODER_NAME", ""))
    parser.add_argument(
        "--load-vocoder-from-local",
        action="store_true",
        default=_env_bool("XVOICE_LOAD_VOCODER_FROM_LOCAL", True),
    )
    parser.add_argument("--target-rms", type=float, default=float(os.getenv("XVOICE_TARGET_RMS", "0.1")))
    parser.add_argument(
        "--cross-fade-duration",
        type=float,
        default=float(os.getenv("XVOICE_CROSS_FADE_DURATION", "0.15")),
    )
    parser.add_argument("--nfe-step", type=int, default=int(os.getenv("XVOICE_NFE_STEP", "32")))
    parser.add_argument("--cfg-strength", type=float, default=float(os.getenv("XVOICE_CFG_STRENGTH", "2.5")))
    parser.add_argument("--layered", action="store_true", default=_env_bool("XVOICE_LAYERED", True))
    parser.add_argument("--no-layered", dest="layered", action="store_false")
    parser.add_argument("--cfg-strength2", type=float, default=float(os.getenv("XVOICE_CFG_STRENGTH2", "4.0")))
    parser.add_argument("--cfg-schedule", type=str, default=os.getenv("XVOICE_CFG_SCHEDULE", "square"))
    parser.add_argument("--cfg-decay-time", type=float, default=float(os.getenv("XVOICE_CFG_DECAY_TIME", "0.6")))
    parser.add_argument("--sway-sampling-coef", type=float, default=float(os.getenv("XVOICE_SWAY_SAMPLING_COEF", "-1.0")))
    parser.add_argument("--speed", type=float, default=float(os.getenv("XVOICE_SPEED", "1.0")))
    parser.add_argument("--denoise-ref", action="store_true", default=_env_bool("XVOICE_DENOISE_REF", True))
    parser.add_argument("--no-denoise-ref", dest="denoise_ref", action="store_false")
    parser.add_argument("--loudness-norm", action="store_true", default=_env_bool("XVOICE_LOUDNESS_NORM", True))
    parser.add_argument("--no-loudness-norm", dest="loudness_norm", action="store_false")
    parser.add_argument("--post-processing", action="store_true", default=_env_bool("XVOICE_POST_PROCESSING", True))
    parser.add_argument("--no-post-processing", dest="post_processing", action="store_false")
    parser.add_argument("--reverse", action="store_true", default=_env_bool("XVOICE_REVERSE", False))
    parser.add_argument("--sp-type", type=str, default=os.getenv("XVOICE_SP_TYPE", "syllable"))
    args, _ = parser.parse_known_args()

    stage = 2 if args.stage not in {1, 2} else args.stage
    model = args.model or ("XVoice_Base_Stage2" if stage == 2 else "XVoice_Base_Stage1")
    model_cfg = args.model_cfg or f"src/x_voice/configs/{model}.yaml"
    cfg_schedule = args.cfg_schedule
    if cfg_schedule == "none":
        cfg_schedule = None
    device_name = args.device
    if not device_name:
        from x_voice.infer.utils_infer import device as default_device

        device_name = default_device

    return ServerConfig(
        stage=stage,
        model=model,
        model_cfg=_resolve_repo_path(model_cfg),
        ckpt_file=_resolve_repo_path(args.ckpt_file),
        ckpt_step=args.ckpt_step,
        vocab_file=_resolve_repo_path(args.vocab_file),
        srp_ckpt_file=_resolve_repo_path(args.srp_ckpt_file),
        srp_model_cfg=_resolve_repo_path(args.srp_model_cfg),
        device=device_name,
        ref_lang=args.ref_lang,
        gen_lang=args.gen_lang,
        auto_detect_lang=args.auto_detect_lang,
        normalize_text=args.normalize_text,
        vocoder_name=args.vocoder_name,
        load_vocoder_from_local=args.load_vocoder_from_local,
        target_rms=args.target_rms,
        cross_fade_duration=args.cross_fade_duration,
        nfe_step=args.nfe_step,
        cfg_strength=args.cfg_strength,
        layered=args.layered,
        cfg_strength2=args.cfg_strength2,
        cfg_schedule=cfg_schedule,
        cfg_decay_time=args.cfg_decay_time,
        sway_sampling_coef=args.sway_sampling_coef,
        speed=args.speed,
        denoise_ref=args.denoise_ref,
        loudness_norm=args.loudness_norm,
        post_processing=args.post_processing,
        reverse=args.reverse,
        sp_type=args.sp_type,
    )


class XVoiceRuntime:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.lock = threading.Lock()
        self.normalizer_cache: dict[str, object] = {}
        self._load_models()

    def _load_models(self) -> None:
        from hydra.utils import get_class
        from omegaconf import OmegaConf
        from x_voice.infer.utils_infer import (
            get_ipa_tokenizer_cache,
            load_model,
            load_model_sft,
            load_srp_model,
            load_vocoder,
            resolve_cached_path,
            resolve_ckpt_path,
        )

        cfg = self.config
        model_cfg = OmegaConf.load(cfg.model_cfg)
        self.model_cfg = model_cfg
        model_cls = get_class(f"x_voice.model.{model_cfg.model.backbone}")
        model_arc = OmegaConf.to_container(model_cfg.model.arch, resolve=True)
        tokenizer = model_cfg.model.tokenizer
        tokenizer_path = model_cfg.model.get("tokenizer_path", None)
        dataset_name = model_cfg.datasets.name
        self.tokenizer = tokenizer
        self.ipa_tokenizer_getter = get_ipa_tokenizer_cache(
            tokenizer,
            bool(model_cfg.model.get("stress", True)),
        )

        vocoder_name = cfg.vocoder_name or model_cfg.model.mel_spec.mel_spec_type
        vocoder_cfg = model_cfg.model.get("vocoder", {})
        vocoder_local_path = _resolve_repo_path(vocoder_cfg.get("local_path", "my_vocoder/vocos-mel-24khz"))
        self.vocoder_name = vocoder_name
        self.vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            is_local=cfg.load_vocoder_from_local or bool(vocoder_cfg.get("is_local", False)),
            local_path=vocoder_local_path,
            device=cfg.device,
        )

        ckpt_file = cfg.ckpt_file
        if not ckpt_file and cfg.ckpt_step is None:
            raise ValueError("XVOICE_CKPT_FILE or --ckpt-file is required.")
        ckpt_file = resolve_ckpt_path(ckpt_file, model_cfg, cfg.model, cfg.ckpt_step)
        vocab_file = resolve_cached_path(cfg.vocab_file) if cfg.vocab_file else ""
        mel_spec_kwargs = OmegaConf.to_container(model_cfg.model.mel_spec, resolve=True)

        if cfg.stage == 2:
            use_total_text = bool(model_cfg.model.get("use_total_text", False))
            self.model = load_model_sft(
                model_cls,
                model_arc,
                ckpt_file,
                mel_spec_type=vocoder_name,
                vocab_file=vocab_file,
                device=cfg.device,
                use_total_text=use_total_text,
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                dataset_name=dataset_name,
                mel_spec_kwargs=mel_spec_kwargs,
            )
            if not cfg.srp_ckpt_file:
                raise ValueError("Stage2 requires XVOICE_SRP_CKPT_FILE or --srp-ckpt-file.")
            self.srp_model = load_srp_model(cfg.srp_model_cfg, cfg.srp_ckpt_file, cfg.device)
        else:
            self.model = load_model(
                model_cls,
                model_arc,
                ckpt_file,
                mel_spec_type=vocoder_name,
                vocab_file=vocab_file,
                device=cfg.device,
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                dataset_name=dataset_name,
                mel_spec_kwargs=mel_spec_kwargs,
            )
            self.srp_model = None

        self.lang_to_id_map = getattr(self.model.transformer, "lang_to_id", {})
        logger.info(
            "X-Voice loaded stage=%s model=%s ckpt=%s device=%s sample_rate=%s",
            cfg.stage,
            cfg.model,
            ckpt_file,
            cfg.device,
            24000,
        )

    def synthesize(
        self,
        *,
        text: str,
        ref_audio: str,
        ref_text: str | None,
        language: str | None,
        ref_lang: str | None,
        speed: float | None,
        nfe_step: int | None,
    ) -> tuple[np.ndarray, int]:
        with self.lock:
            if self.config.stage == 2:
                return self._synthesize_stage2(
                    text=text,
                    ref_audio=ref_audio,
                    language=language,
                    speed=speed,
                    nfe_step=nfe_step,
                )
            return self._synthesize_stage1(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
                ref_lang=ref_lang,
                speed=speed,
                nfe_step=nfe_step,
            )

    def _language_spans(self, text: str, language: str | None) -> list[tuple[str, str]]:
        from x_voice.infer.utils_infer import (
            auto_split_mixed_text,
            detect_segment_lang,
            normalize_lang_code,
            normalize_text_for_lang,
        )

        fallback_lang = normalize_lang_code(language or self.config.gen_lang)
        if self.config.auto_detect_lang:
            fallback_lang = fallback_lang or detect_segment_lang(text, "en") or "en"
            spans = auto_split_mixed_text(text, fallback_lang)
        else:
            if not fallback_lang:
                raise ValueError("gen language is required when auto_detect_lang is disabled.")
            spans = [(fallback_lang, text)]

        if self.config.normalize_text:
            spans = [
                (lang, normalize_text_for_lang(span_text, lang, self.normalizer_cache))
                for lang, span_text in spans
            ]
        return spans

    def _synthesize_stage2(
        self,
        *,
        text: str,
        ref_audio: str,
        language: str | None,
        speed: float | None,
        nfe_step: int | None,
    ) -> tuple[np.ndarray, int]:
        from x_voice.infer.utils_infer import (
            infer_xvoice_droptext_process,
            preprocess_ref_audio_text,
        )

        cfg = self.config
        preprocess_start = time.perf_counter()
        ref_audio, _ = preprocess_ref_audio_text(ref_audio, "Drop-text mode ignores reference text.")
        spans = self._language_spans(text, language)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
        infer_start = time.perf_counter()
        audio, sample_rate, _ = infer_xvoice_droptext_process(
            ref_audio,
            [text],
            [spans],
            self.tokenizer,
            self.ipa_tokenizer_getter,
            self.model,
            self.vocoder,
            self.lang_to_id_map,
            self.srp_model,
            mel_spec_type_value=self.vocoder_name,
            target_rms_value=cfg.target_rms,
            cross_fade_duration_value=cfg.cross_fade_duration,
            nfe_step_value=nfe_step or cfg.nfe_step,
            cfg_strength_value=cfg.cfg_strength,
            layered=cfg.layered,
            cfg_strength2_value=cfg.cfg_strength2,
            cfg_schedule_value=cfg.cfg_schedule,
            cfg_decay_time_value=cfg.cfg_decay_time,
            sway_sampling_coef_value=cfg.sway_sampling_coef,
            local_speed=speed or cfg.speed,
            reverse=cfg.reverse,
            denoise_ref=cfg.denoise_ref,
            loudness_norm=cfg.loudness_norm,
            post_processing=cfg.post_processing,
            device_name=cfg.device,
            remove_silence_chunk=True,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000
        logger.info(
            "xvoice stage2 synthesized preprocess_ms=%.1f infer_ms=%.1f text_len=%s spans=%s samples=%s",
            preprocess_ms,
            infer_ms,
            len(text.strip()),
            len(spans),
            int(np.asarray(audio).size) if audio is not None else 0,
        )
        audio_array = audio if audio is not None else np.zeros((0,), dtype=np.float32)
        return np.asarray(audio_array, dtype=np.float32), sample_rate

    def _synthesize_stage1(
        self,
        *,
        text: str,
        ref_audio: str,
        ref_text: str | None,
        language: str | None,
        ref_lang: str | None,
        speed: float | None,
        nfe_step: int | None,
    ) -> tuple[np.ndarray, int]:
        from x_voice.infer.utils_infer import (
            detect_segment_lang,
            ensure_ref_text_punctuation,
            infer_xvoice_process,
            normalize_lang_code,
            normalize_text_for_lang,
            preprocess_ref_audio_text,
        )

        cfg = self.config
        preprocess_start = time.perf_counter()
        ref_audio, ref_text_out = preprocess_ref_audio_text(ref_audio, ref_text or "")
        ref_lang_out = normalize_lang_code(ref_lang or cfg.ref_lang)
        if cfg.auto_detect_lang and not ref_lang_out:
            ref_lang_out = detect_segment_lang(ref_text_out, None)
        if not ref_lang_out:
            raise ValueError("ref language is required for Stage1.")
        if cfg.normalize_text:
            ref_text_out = normalize_text_for_lang(ref_text_out, ref_lang_out, self.normalizer_cache)
            ref_text_out = ensure_ref_text_punctuation(ref_text_out)
        spans = self._language_spans(text, language)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000
        infer_start = time.perf_counter()
        audio, sample_rate, _ = infer_xvoice_process(
            ref_audio,
            ref_text_out,
            [text],
            ref_lang_out,
            [spans],
            self.tokenizer,
            self.ipa_tokenizer_getter,
            self.model,
            self.vocoder,
            self.lang_to_id_map,
            mel_spec_type_value=self.vocoder_name,
            target_rms_value=cfg.target_rms,
            cross_fade_duration_value=cfg.cross_fade_duration,
            nfe_step_value=nfe_step or cfg.nfe_step,
            cfg_strength_value=cfg.cfg_strength,
            layered=cfg.layered,
            cfg_strength2_value=cfg.cfg_strength2,
            cfg_schedule_value=cfg.cfg_schedule,
            cfg_decay_time_value=cfg.cfg_decay_time,
            sway_sampling_coef_value=cfg.sway_sampling_coef,
            local_speed=speed or cfg.speed,
            sp_type=cfg.sp_type,
            reverse=cfg.reverse,
            denoise_ref=cfg.denoise_ref,
            loudness_norm=cfg.loudness_norm,
            post_processing=cfg.post_processing,
            device_name=cfg.device,
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000
        logger.info(
            "xvoice stage1 synthesized preprocess_ms=%.1f infer_ms=%.1f ref_text_len=%s "
            "text_len=%s spans=%s samples=%s",
            preprocess_ms,
            infer_ms,
            len(ref_text_out.strip()),
            len(text.strip()),
            len(spans),
            int(np.asarray(audio).size) if audio is not None else 0,
        )
        audio_array = audio if audio is not None else np.zeros((0,), dtype=np.float32)
        return np.asarray(audio_array, dtype=np.float32), sample_rate


def _ndarray_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _parse_config()
    logger.info("loading X-Voice stage=%s model=%s", config.stage, config.model)
    runtime = XVoiceRuntime(config)
    app.state.runtime = runtime
    yield


app = FastAPI(title="X-Voice TTS Server", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        return {"status": "loading"}
    return {"status": "ok", "stage": runtime.config.stage, "sample_rate": 24000}


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    ref_audio = req.audio_paths[0] if req.audio_paths else None
    if not ref_audio:
        raise HTTPException(status_code=400, detail="audio_paths[0] is required")

    start_time = time.perf_counter()
    try:
        audio, sample_rate = await _run_sync(
            app.state.runtime.synthesize,
            text=req.text,
            ref_audio=ref_audio,
            ref_text=req.ref_text,
            language=req.gen_lang or req.language,
            ref_lang=req.ref_lang,
            speed=req.speed,
            nfe_step=req.nfe_step,
        )
    except Exception as exc:
        logger.exception("X-Voice synthesis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wav_start = time.perf_counter()
    wav_bytes = _ndarray_to_wav_bytes(audio, sample_rate)
    wav_ms = (time.perf_counter() - wav_start) * 1000
    total_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "xvoice response total_ms=%.1f wav_ms=%.1f audio_samples=%s wav_bytes=%s",
        total_ms,
        wav_ms,
        audio.shape[0],
        len(wav_bytes),
    )
    return Response(content=wav_bytes, media_type="audio/wav")


async def _run_sync(func, **kwargs):
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(**kwargs))


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="X-Voice TTS HTTP server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11998)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model-cfg", type=str, default=None)
    parser.add_argument("--ckpt-file", type=str, default=None)
    parser.add_argument("--srp-ckpt-file", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.stage is not None:
        os.environ["XVOICE_STAGE"] = str(args.stage)
    if args.model is not None:
        os.environ["XVOICE_MODEL"] = args.model
    if args.model_cfg is not None:
        os.environ["XVOICE_MODEL_CFG"] = args.model_cfg
    if args.ckpt_file is not None:
        os.environ["XVOICE_CKPT_FILE"] = args.ckpt_file
    if args.srp_ckpt_file is not None:
        os.environ["XVOICE_SRP_CKPT_FILE"] = args.srp_ckpt_file
    if args.device is not None:
        os.environ["XVOICE_DEVICE"] = args.device

    uvicorn.run(app, host=args.host, port=args.port)
