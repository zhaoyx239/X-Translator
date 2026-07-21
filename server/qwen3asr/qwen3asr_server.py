#!/usr/bin/env python3
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import base64
import io
import wave
import asyncio
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from qwen_asr import Qwen3ASRModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Qwen3ASR")

app = FastAPI(title="Qwen3 ASR Service")

model = None
session_cache = {}
gpu_lock = Lock()
executor = ThreadPoolExecutor(max_workers=10)

MAX_SESSION_STEPS = 60
OVERLAP_CHUNKS = 4
VAD_THRESHOLD = 0.001


def preview_text(text: str, limit: int = 120) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def merge_text_prefix(prefix: str, current: str) -> str:
    left = (prefix or "").strip()
    right = (current or "").strip()
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
    value = (text or "").strip()
    base = (prefix or "").strip()
    if not base:
        return value
    if value.startswith(base):
        return value[len(base):].strip()
    if base.endswith(value):
        return ""
    return value

class RecognitionRequest(BaseModel):
    audio: str
    session_id: str = None
    is_final: bool = False
    sample_rate: int = 16000
    committed_text: str = ""


def rebuild_streaming_state(ctx: dict) -> None:
    logger.info(
        "rebuild state history_chunks=%s prefix=%s",
        len(ctx["history_chunks"]),
        preview_text(ctx.get("prefix_text", "")),
    )
    new_state = model.init_streaming_state(
        unfixed_chunk_num=2,
        unfixed_token_num=5,
        chunk_size_sec=0.6
    )
    for chunk in ctx["history_chunks"]:
        if chunk.size == 0:
            continue
        model.streaming_transcribe(chunk, new_state)
    logger.info("rebuild state done text=%s", preview_text(getattr(new_state, "text", "")))
    ctx["state"] = new_state
    ctx["step_count"] = len(ctx["history_chunks"])


def apply_committed_prefix(ctx: dict, committed_text: str) -> None:
    committed = (committed_text or "").strip()
    if not committed:
        return
    if committed == ctx["prefix_text"]:
        return
    logger.info(
        "apply commit old_prefix=%s new_prefix=%s state_text=%s",
        preview_text(ctx["prefix_text"]),
        preview_text(committed),
        preview_text(getattr(ctx["state"], "text", "")),
    )
    ctx["prefix_text"] = committed
    ctx["last_valid_text"] = ""
    model.finish_streaming_transcribe(ctx["state"])
    rebuild_streaming_state(ctx)

def run_inference(request: RecognitionRequest, audio_data: np.ndarray):
    global model, session_cache
    
    # 如果是纯静音包，跳过推理以防止模型“胡思乱想”产生重复字
    is_speech = np.abs(audio_data).mean() > VAD_THRESHOLD
    
    with gpu_lock:
        try:
            if request.session_id:
                # 初始化 Session
                if request.session_id not in session_cache:
                    session_cache[request.session_id] = {
                        "state": model.init_streaming_state(
                            unfixed_chunk_num=2,
                            unfixed_token_num=5,
                            chunk_size_sec=0.6
                        ),
                        "step_count": 0,
                        "history_chunks": [], 
                        "last_valid_text": "",
                        "prefix_text": "",
                    }
                
                ctx = session_cache[request.session_id]
                apply_committed_prefix(ctx, request.committed_text)
                
                # 记录音频到滑动窗口
                if audio_data.size > 0:
                    ctx["history_chunks"].append(audio_data)
                    if len(ctx["history_chunks"]) > OVERLAP_CHUNKS:
                        ctx["history_chunks"].pop(0)

                # 滑动窗口重置逻辑：达到上限时，平滑衔接
                if ctx["step_count"] >= MAX_SESSION_STEPS and not request.is_final:
                    logger.info(f"Session {request.session_id} 滑动重置：保留上下文重新注入")
                    ctx["prefix_text"] = merge_text_prefix(
                        ctx["prefix_text"],
                        ctx["state"].text,
                    )
                    model.finish_streaming_transcribe(ctx["state"])
                    rebuild_streaming_state(ctx)
                
                # 执行识别 (仅在有声音或强制结束时执行)
                if is_speech or request.is_final:
                    model.streaming_transcribe(audio_data, ctx["state"])
                    ctx["step_count"] += 1
                
                if request.is_final:
                    model.finish_streaming_transcribe(ctx["state"])
                    final_text = merge_text_prefix(
                        ctx["prefix_text"],
                        ctx["state"].text,
                    ).strip()
                    logger.info(
                        "return final prefix=%s state_text=%s merged=%s",
                        preview_text(ctx["prefix_text"]),
                        preview_text(getattr(ctx["state"], "text", "")),
                        preview_text(final_text),
                    )
                    del session_cache[request.session_id]
                    return final_text
                else:
                    # 如果当前识别结果为空（由于静音过滤等），返回上一帧文本避免闪烁
                    full_text = merge_text_prefix(
                        ctx["prefix_text"],
                        ctx["state"].text,
                    ).strip()
                    current_text = strip_prefix_text(full_text, ctx["prefix_text"])
                    logger.info(
                        "return partial prefix=%s state_text=%s full=%s current=%s",
                        preview_text(ctx["prefix_text"]),
                        preview_text(getattr(ctx["state"], "text", "")),
                        preview_text(full_text),
                        preview_text(current_text),
                    )
                    if not current_text:
                        ctx["last_valid_text"] = ""
                        return ""
                    if ctx["last_valid_text"] and current_text == ctx["last_valid_text"]:
                        return ctx["last_valid_text"]
                    ctx["last_valid_text"] = current_text
                    return current_text
            else:
                # 非流式处理
                state = model.init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)
                model.streaming_transcribe(audio_data, state)
                model.finish_streaming_transcribe(state)
                return state.text
                
        except Exception as e:
            logger.error(f"Inference internal error: {e}")
            if request.session_id in session_cache:
                del session_cache[request.session_id]
            raise e

@app.post("/v1/recognize")
async def recognize(request: RecognitionRequest):
    try:
        if not request.audio.startswith("data:audio/wav;base64,"):
            raise ValueError("不支持的音频格式")
        
        base64_data = request.audio.split("data:audio/wav;base64,")[-1]
        wav_bytes = base64.b64decode(base64_data)
        
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
            num_channels = wf.getnchannels()
            wav_sample_rate = wf.getframerate()
        
        if sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio = np.frombuffer(frames, dtype=np.float32)
        
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1)

        if wav_sample_rate != request.sample_rate:
            logger.warning(
                "sample rate mismatch request=%s wav=%s session_id=%s",
                request.sample_rate,
                wav_sample_rate,
                request.session_id,
            )
        
        loop = asyncio.get_running_loop()
        # 增加超时保护
        text = await asyncio.wait_for(
            loop.run_in_executor(executor, run_inference, request, audio),
            timeout=15.0
        )
        return {"text": text, "session_id": request.session_id, "is_final": request.is_final}
        
    except Exception as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def main():
    global model
    parser = argparse.ArgumentParser(description="Qwen3 ASR服务器")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()
    
    model = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=32,
        trust_remote_code=True
    )
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)

if __name__ == "__main__":
    main()
