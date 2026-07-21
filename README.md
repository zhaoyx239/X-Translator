# X-Translator

English | [Chinese version](README_zh.md)

X-Translator is a real-time speaker-aware speech-to-speech translation system. It connects streaming ASR, machine translation, and prompt-conditioned TTS through a lightweight runtime controller, so the browser can display source text, translated text, and synthesized target speech during a live session. Try the online demo at [https://translate.sjtuxlance.com/](https://translate.sjtuxlance.com/).

The current release includes the local demo and backend service adapters. Evaluation code will be released later.

## Architecture

![X-Translator system architecture](assets/overview.png)

## Runtime Design

![ASR pipeline and segment commitment](assets/asr_pipeline.png)

![Speaker prompt manager](assets/speaker_prompt_manager.png)

## Repository Layout

- `backend/`: FastAPI backend, runtime controller, ASR/MT/TTS clients, and session logic.
- `frontend/`: Static browser demo UI.
- `server/`: ASR, MT, and TTS service adapters, setup notes, and tmux launchers.
- `main.py`: Local application entry point.
- `config.json`: Default runtime configuration.
- `start.sh`: Convenience script for launching the demo.

## Environment Setup

```bash
cd xtranslate
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you use CUDA 12.4, install the matching PyTorch build:

```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torch-complex==0.4.4 --extra-index-url https://download.pytorch.org/whl/cu124
```

## Basic Configuration

Edit `config.json` before running the demo. In most cases, only these fields need to be changed:

- `server.host` and `server.port`: local web server address.
- `asr.provider`: ASR backend, such as `qwen3`, `sensevoice`, `paraformer`, or `zipformer`.
- `translation.provider`: MT backend, such as `lmt` or `hunyuan`.
- `tts.provider`: TTS backend, such as `xvoice` or `index`.
- Backend service URLs, for example `asr.qwen3_asr_url`, `translation.lmt_url`, and `tts.xvoice_tts_url`.
- `translation.source_lang` and `translation.target_lang`: source and target language codes.

The default configuration assumes local backend services. Start the ASR, MT, and TTS services you select in `config.json` before launching the browser demo.

See [`server/README.md`](server/README.md) for backend installation and multi-service startup instructions.

## Run the Demo

```bash
bash start.sh
```

The default local demo URL is:

```text
http://0.0.0.0:7654
```

## TODO

- [x] Release demo code.
- [x] Release server adapters and launchers.
- [ ] Release evaluation code.
- [x] Release paper.

## Citation


```bibtex
@misc{zhao2026xtranslatorrealtimemultilingualspeakeraware,
      title={X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System}, 
      author={Yuxiang Zhao and Yichi Zhang and Yanjie An and Yanqiao Zhu and Zhanxun Liu and Yushen Chen and Qixi Zheng and Haina Zhu and Yunchong Xiao and Keqi Deng and Shuai Fan and Kai Yu and Xie Chen},
      year={2026},
      eprint={2607.17544},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2607.17544}, 
}
```

## Acknowledgements

We thank [XTalk](https://github.com/xcc-zach/xtalk), [X-ASR](https://github.com/Gilgamesh-J/X-ASR), [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [Paraformer](https://github.com/modelscope/FunASR), [SenseVoice](https://github.com/FunAudioLLM/SenseVoice), [NiuTrans LMT](https://github.com/NiuTrans/LMT), [Hunyuan-MT](https://github.com/Tencent-Hunyuan/Hunyuan-MT), [X-Voice](https://github.com/sunnyxrxrx/X-Voice), [IndexTTS](https://github.com/index-tts/index-tts), and [OpenSTBench](https://github.com/sjtuayj/OpenSTBench) for their contributions to the broader speech translation ecosystem.

X-Translator code is released under the MIT License. Third-party modules, models, and services used with this project remain governed by their original licenses.
