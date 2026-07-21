# X-Translator

[English](README.md) | 中文

X-Translator 是一个实时、说话人感知的语音到语音翻译系统。它通过轻量级运行时控制器连接流式 ASR、机器翻译和基于提示音频的 TTS，让浏览器在实时会话中展示源语音识别文本、翻译文本和合成后的目标语音。在线 demo 地址：[https://translate.sjtuxlance.com/](https://translate.sjtuxlance.com/)。

当前版本主要发布本地 demo 代码。评测代码、server 代码和论文将在后续发布。

## 系统架构

![X-Translator system architecture](assets/overview.png)

## 运行时设计

![ASR pipeline and segment commitment](assets/asr_pipeline.png)

![Speaker prompt manager](assets/speaker_prompt_manager.png)

## 目录结构

- `backend/`：FastAPI 后端、运行时控制器、ASR/MT/TTS 客户端和会话逻辑。
- `frontend/`：静态浏览器 demo 界面。
- `main.py`：本地应用入口。
- `config.json`：默认运行配置。
- `start.sh`：demo 启动脚本。

## 环境配置

```bash
cd xtranslate
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果使用 CUDA 12.4，可以安装对应的 PyTorch 版本：

```bash
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torch-complex==0.4.4 --extra-index-url https://download.pytorch.org/whl/cu124
```

## 基本配置

运行前修改 `config.json`。通常只需要关注以下字段：

- `server.host` 和 `server.port`：本地 Web 服务地址。
- `asr.provider`：ASR 后端，例如 `qwen3`、`sensevoice`、`paraformer` 或 `zipformer`。
- `translation.provider`：MT 后端，例如 `lmt` 或 `hunyuan`。
- `tts.provider`：TTS 后端，例如 `xvoice` 或 `index`。
- 各后端服务 URL，例如 `asr.qwen3_asr_url`、`translation.lmt_url` 和 `tts.xvoice_tts_url`。
- `translation.source_lang` 和 `translation.target_lang`：源语言和目标语言代码。

默认配置假设后端服务运行在本地。启动浏览器 demo 前，请先启动 `config.json` 中选择的 ASR、MT 和 TTS 服务。

## 运行 Demo

```bash
bash start.sh
```

默认本地 demo 地址为：

```text
http://0.0.0.0:7654
```

## TODO

- [x] Release demo code.
- [ ] Release full server code.
- [ ] Release evaluation code.
- [ ] Release paper.

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

感谢 [XTalk](https://github.com/xcc-zach/xtalk)、[X-ASR](https://github.com/Gilgamesh-J/X-ASR)、[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[Paraformer](https://github.com/modelscope/FunASR)、[SenseVoice](https://github.com/FunAudioLLM/SenseVoice)、[NiuTrans LMT](https://github.com/NiuTrans/LMT)、[Hunyuan-MT](https://github.com/Tencent-Hunyuan/Hunyuan-MT)、[X-Voice](https://github.com/sunnyxrxrx/X-Voice)、[IndexTTS](https://github.com/index-tts/index-tts) 和 [OpenSTBench](https://github.com/sjtuayj/OpenSTBench) 对语音翻译生态的贡献。

X-Translator 代码使用 MIT License 发布。本项目使用到的第三方模块、模型和服务遵循其原始协议。
