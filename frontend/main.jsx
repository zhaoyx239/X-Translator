const { useEffect, useRef, useState } = React;

const LANGUAGE_OPTIONS = [
  ["zh", "Chinese"],
  ["en", "English"],
  ["fr", "French"],
  ["pt", "Portuguese"],
  ["es", "Spanish"],
  ["ja", "Japanese"],
  ["ru", "Russian"],
  ["ko", "Korean"],
  ["th", "Thai"],
  ["it", "Italian"],
  ["de", "German"],
  ["vi", "Vietnamese"],
  ["id", "Indonesian"],
  ["pl", "Polish"],
  ["cs", "Czech"],
  ["nl", "Dutch"],
];

function App() {
  const [status, setStatus] = useState("idle");
  const [visitCount, setVisitCount] = useState(null);
  const [showSpeaker, setShowSpeaker] = useState(true);
  const [targetLang, setTargetLang] = useState("en");
  const [asrPartial, setAsrPartial] = useState("...");
  const [translations, setTranslations] = useState([]);
  const [sourceSpeakerSegments, setSourceSpeakerSegments] = useState([]);
  const [translationSpeakerSegments, setTranslationSpeakerSegments] = useState([]);
  const [streamStartedAt, setStreamStartedAt] = useState(null);
  const [durationNow, setDurationNow] = useState(Date.now());
  const [recentSessions, setRecentSessions] = useState([]);
  const [latencyMetrics, setLatencyMetrics] = useState({
    e2e_ms: 0,
    asr_ms: 0,
    mt_ms: 0,
    tts_ms: 0,
  });
  const wsRef = useRef(null);
  const inputAudioContextRef = useRef(null);
  const playbackAudioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const isSpeechActiveRef = useRef(false);
  const playbackQueueRef = useRef([]);
  const playbackRunningRef = useRef(false);
  const currentSourceRef = useRef(null);
  const sourceScrollRef = useRef(null);
  const translationScrollRef = useRef(null);
  const timelineCanvasRef = useRef(null);
  const sessionIdRef = useRef(`${Date.now()}-${Math.random().toString(16).slice(2)}`);
  const activeTurnIdRef = useRef(null);
  const turnCounterRef = useRef(0);
  const turnStartedAtRef = useRef(new Map());
  const pendingPlaybackMetricsRef = useRef([]);
  const inputLevelHistoryRef = useRef([]);
  const outputLevelHistoryRef = useRef([]);
  const animationFrameRef = useRef(null);
  const streamStartedPerfRef = useRef(null);
  const recentSessionsRefreshTokenRef = useRef(0);

  const TIMELINE_WINDOW_MS = 16000;
  const TIMELINE_TICK_MS = 2000;

  useEffect(() => () => stopSession(), []);

  useEffect(() => {
    void loadRecentSessions();
  }, []);

  useEffect(() => {
    void loadFrontendConfig();
  }, []);

  useEffect(() => {
    void loadVisitStats();
    const timer = window.setInterval(() => {
      void loadVisitStats();
    }, 15000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!streamStartedAt) return undefined;
    setDurationNow(Date.now());
    const timer = window.setInterval(() => {
      setDurationNow(Date.now());
    }, 1000);
    return () => window.clearInterval(timer);
  }, [streamStartedAt]);

  useEffect(() => {
    if (sourceScrollRef.current) {
      sourceScrollRef.current.scrollTop = sourceScrollRef.current.scrollHeight;
    }
  }, [sourceSpeakerSegments, asrPartial]);

  useEffect(() => {
    if (translationScrollRef.current) {
      translationScrollRef.current.scrollTop = translationScrollRef.current.scrollHeight;
    }
  }, [translationSpeakerSegments]);

  useEffect(() => () => stopTimelineLoop(), []);

  useEffect(() => {
    startTimelineLoop();
    return () => stopTimelineLoop();
  }, []);

  async function startSession() {
    if (wsRef.current) return;
    setStatus("connecting");
    setAsrPartial("...");
    setTranslations([]);
    setSourceSpeakerSegments([]);
    setTranslationSpeakerSegments([]);
    setLatencyMetrics({
      e2e_ms: 0,
      asr_ms: 0,
      mt_ms: 0,
      tts_ms: 0,
    });
    activeTurnIdRef.current = null;
    turnCounterRef.current = 0;
    turnStartedAtRef.current = new Map();
    pendingPlaybackMetricsRef.current = [];
    inputLevelHistoryRef.current = [];
    outputLevelHistoryRef.current = [];
    streamStartedPerfRef.current = nowMs();
    const startedAt = Date.now();
    setStreamStartedAt(startedAt);
    setDurationNow(startedAt);

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = async () => {
      ws.send(
        JSON.stringify({
          action: "session_config",
          target_lang: targetLang,
          sample_rate: 16000,
        })
      );
      ws.send(JSON.stringify({ event: "start", session_id: sessionIdRef.current }));
      await startMicrophone(ws);
      setStatus("streaming");
    };
    ws.onmessage = async (event) => {
      if (typeof event.data === "string") {
        const payload = JSON.parse(event.data);
        if (payload.action === "asr_result") {
          const result = payload.data || {};
          const cleanedText = normalizeDisplayText(result.text || "");
          if (result.type === "partial" || result.type === "blank") {
            setAsrPartial(cleanedText || "...");
          } else if (result.type === "stable") {
            setAsrPartial("...");
            activeTurnIdRef.current = null;
          } else if (result.type === "final") {
            setAsrPartial("...");
            activeTurnIdRef.current = null;
          }
        }
        if (payload.action === "source_segment_ready") {
          const sourceText = payload.data.source_text || "";
          if (sourceText) {
            setSourceSpeakerSegments((prev) =>
              appendSpeakerSegment(prev, payload.data.speaker_id, sourceText)
            );
          }
        }
        if (payload.action === "translation_ready") {
          const translatedText = payload.data.translated_text || "";
          pendingPlaybackMetricsRef.current.push({
            turnId: payload.data.turn_id || "",
            segmentIndex: Number(payload.data.segment_index || 1),
            segmentCount: Number(payload.data.segment_count || 1),
            metrics: payload.data.metrics || {},
          });
          if (translatedText) {
            setTranslationSpeakerSegments((prev) =>
              appendSpeakerSegment(prev, payload.data.speaker_id, translatedText)
            );
          }
          setTranslations((prev) => [payload.data, ...prev].slice(0, 30));
        }
        if (payload.action === "error") {
          setStatus(`error: ${payload.data.message}`);
        }
      } else {
        settleLatencyMetrics();
        enqueueAudio(event.data);
      }
    };
    ws.onclose = () => {
      wsRef.current = null;
      setStatus("idle");
      setStreamStartedAt(null);
      setDurationNow(Date.now());
      void refreshRecentSessionsAfterStop();
    };
    wsRef.current = ws;
  }

  async function startMicrophone(ws) {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        autoGainControl: true,
        noiseSuppression: false,
      },
    });
    mediaStreamRef.current = stream;
    const audioContext = new AudioContext({ sampleRate: 16000 });
    inputAudioContextRef.current = audioContext;
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(512, 1, 1);
    ws.send(
      JSON.stringify({
        event: "config_audio",
        sample_rate: audioContext.sampleRate,
      })
    );
    processor.onaudioprocess = (event) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      if (!activeTurnIdRef.current) {
        turnCounterRef.current += 1;
        const turnId = `turn-${turnCounterRef.current}`;
        activeTurnIdRef.current = turnId;
        turnStartedAtRef.current.set(turnId, nowMs());
        ws.send(
          JSON.stringify({
            event: "turn_started",
            turn_id: turnId,
          })
        );
      }
      const input = event.inputBuffer.getChannelData(0);
      recordAudioLevel(inputLevelHistoryRef, rmsFromFloat32(input), nowMs());
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, input[i]));
        pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      ws.send(pcm.buffer);
    };
    source.connect(processor);
    processor.connect(audioContext.destination);
    processorRef.current = processor;
  }

  async function loadRecentSessions() {
    try {
      const response = await fetch("/api/recent-sessions?limit=3", { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      setRecentSessions(Array.isArray(payload.items) ? payload.items : []);
    } catch (_error) {
    }
  }

  async function loadFrontendConfig() {
    try {
      const response = await fetch("/api/frontend-config", { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      if (typeof payload.show_speaker === "boolean") {
        setShowSpeaker(payload.show_speaker);
      }
    } catch (_error) {
    }
  }

  async function loadVisitStats() {
    try {
      const response = await fetch("/api/visit-stats", { cache: "no-store" });
      if (!response.ok) return;
      const payload = await response.json();
      const totalVisits = Number(payload.total_visits);
      if (Number.isFinite(totalVisits) && totalVisits >= 0) {
        setVisitCount(totalVisits);
      }
    } catch (_error) {
    }
  }

  async function refreshRecentSessionsAfterStop() {
    const token = Date.now();
    recentSessionsRefreshTokenRef.current = token;
    for (let attempt = 0; attempt < 6; attempt += 1) {
      if (recentSessionsRefreshTokenRef.current !== token) return;
      await loadRecentSessions();
      if (attempt < 5) {
        await new Promise((resolve) => setTimeout(resolve, 350));
      }
    }
  }

  function startTimelineLoop() {
    if (animationFrameRef.current) return;
    const draw = () => {
      renderTimeline();
      animationFrameRef.current = requestAnimationFrame(draw);
    };
    animationFrameRef.current = requestAnimationFrame(draw);
  }

  function stopTimelineLoop() {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }

  function renderTimeline() {
    const canvas = timelineCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const width = Math.max(320, Math.round(rect.width || 0));
    const height = 220;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.height = `${height}px`;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const now = nowMs();
    const leftPad = 18;
    const rightPad = 16;
    const topPad = 18;
    const bottomPad = 34;
    const graphWidth = width - leftPad - rightPad;
    const graphHeight = height - topPad - bottomPad;
    const laneGap = 14;
    const laneHeight = (graphHeight - laneGap) / 2;
    const inputTop = topPad;
    const outputTop = topPad + laneHeight + laneGap;

    ctx.fillStyle = "rgba(255,255,255,0.02)";
    ctx.fillRect(0, 0, width, height);

    for (let tick = TIMELINE_WINDOW_MS; tick >= 0; tick -= TIMELINE_TICK_MS) {
      const x = leftPad + (graphWidth * (TIMELINE_WINDOW_MS - tick)) / TIMELINE_WINDOW_MS;
      ctx.strokeStyle = tick === 0 ? "rgba(142, 247, 255, 0.24)" : "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, topPad);
      ctx.lineTo(x, height - bottomPad + 2);
      ctx.stroke();

      ctx.fillStyle = "rgba(135, 161, 193, 0.8)";
      ctx.font = '12px "SFMono-Regular", monospace';
      ctx.textAlign = tick === 0 ? "right" : "center";
      const label = tick === 0 ? "now" : `-${Math.round(tick / 1000)}s`;
      ctx.fillText(label, tick === 0 ? x - 2 : x, height - 10);
    }

    drawLane(ctx, inputLevelHistoryRef.current, now, leftPad, graphWidth, inputTop, laneHeight, "rgba(47, 211, 255, 0.95)", "Input");
    drawLane(ctx, outputLevelHistoryRef.current, now, leftPad, graphWidth, outputTop, laneHeight, "rgba(123, 125, 255, 0.95)", "Output", {
      extendToNow: true,
    });
  }

  function enqueueAudio(arrayBuffer) {
    playbackQueueRef.current.push({
      arrayBuffer,
      waveform: buildPCMChunkLevels(arrayBuffer, 48000),
    });
    if (!playbackRunningRef.current) {
      void drainPlaybackQueue();
    }
  }

  function settleLatencyMetrics() {
    const pending = pendingPlaybackMetricsRef.current.shift();
    if (!pending) return;

    const metrics = pending.metrics || {};
    const asrMs = Number(metrics.asr_ms || 0);
    const mtMs = Number(metrics.mt_ms || 0);
    const ttsMs = Number(metrics.tts_ms || 0);
    const turnStartedAt = pending.turnId ? turnStartedAtRef.current.get(pending.turnId) : null;
    const observedE2EMs =
      turnStartedAt == null ? 0 : Math.max(0, Math.round(nowMs() - turnStartedAt));

    setLatencyMetrics({
      e2e_ms: observedE2EMs,
      asr_ms: asrMs,
      mt_ms: mtMs,
      tts_ms: ttsMs,
    });

    if (pending.turnId && pending.segmentIndex >= pending.segmentCount) {
      turnStartedAtRef.current.delete(pending.turnId);
    }
  }

  async function drainPlaybackQueue() {
    playbackRunningRef.current = true;
    while (playbackQueueRef.current.length > 0) {
      const nextChunk = playbackQueueRef.current.shift();
      if (!nextChunk) {
        continue;
      }
      await playAudioChunk(nextChunk);
    }
    playbackRunningRef.current = false;
  }

  async function playAudioChunk(chunk) {
    const { arrayBuffer, waveform } = chunk;
    let audioContext = playbackAudioContextRef.current;
    if (!audioContext) {
      audioContext = new AudioContext({ sampleRate: 48000 });
      playbackAudioContextRef.current = audioContext;
    }
    const pcm = new Int16Array(arrayBuffer);
    const audioBuffer = audioContext.createBuffer(1, pcm.length, 48000);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i += 1) {
      channel[i] = pcm[i] / 32768;
    }
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    currentSourceRef.current = source;
    const playbackStartMs = nowMs();
    recordScheduledOutputLevels(
      outputLevelHistoryRef,
      waveform,
      playbackStartMs
    );
    await new Promise((resolve) => {
      source.onended = () => {
        if (currentSourceRef.current === source) {
          currentSourceRef.current = null;
        }
        resolve();
      };
      source.start();
    });
  }

  function stopSession() {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (playbackAudioContextRef.current) {
      playbackAudioContextRef.current.close();
      playbackAudioContextRef.current = null;
    }
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
      } catch (_error) {
      }
      currentSourceRef.current = null;
    }
    playbackQueueRef.current = [];
    playbackRunningRef.current = false;
    activeTurnIdRef.current = null;
    turnCounterRef.current = 0;
    turnStartedAtRef.current = new Map();
    pendingPlaybackMetricsRef.current = [];
    inputLevelHistoryRef.current = [];
    outputLevelHistoryRef.current = [];
    streamStartedPerfRef.current = null;
    if (wsRef.current) {
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ event: "stop", session_id: sessionIdRef.current }));
      }
      wsRef.current.close();
      wsRef.current = null;
    }
    sessionIdRef.current = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setStreamStartedAt(null);
    setDurationNow(Date.now());
    setStatus("idle");
  }

  const durationText = streamStartedAt
    ? `${Math.max(0, Math.floor((durationNow - streamStartedAt) / 1000))}s`
    : "0s";

  return (
    <main className="page">
      <section className="panel hero">
        <div className="visit-counter">
          Visitor Count {visitCount == null ? "--" : formatVisitCount(visitCount)}
        </div>
        <p className="eyebrow">Realtime Speech Translation Demo</p>
        <h1>X-Translator</h1>
      </section>

      <section className="panel control-panel">
        <div className="control-meta">
          <p className="label">Voice Route</p>
          <p className="route-hint">Input is auto-detected by ASR.</p>
        </div>
        <div className="language-row compact">
          <div className="route-node">
            <span className="route-node-label">Input</span>
            <span className="route-node-value">Auto Detect</span>
          </div>
          <div className="arrow">→</div>
          <label className="route-select">
            <span className="route-node-label">Output</span>
            <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>
              {LANGUAGE_OPTIONS.map(([value, label]) => (
                <option key={`dst-${value}`} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="controls">
          <button className="primary-action" onClick={startSession} disabled={status !== "idle"}>Start Stream</button>
          <button className="secondary-action" onClick={stopSession}>Stop</button>
        </div>
      </section>

      <section className="stats">
        <div className="panel stat">
          <p className="label">Status</p>
          <p className="mono">{status}</p>
        </div>
        <div className="panel stat">
          <p className="label">Session Duration</p>
          <p className="mono">{durationText}</p>
        </div>
        <div className="panel latency-panel">
          <p className="label">Latency</p>
          <div className="latency-strip">
            <div className="latency-item primary">
              <span className="latency-name">End-to-End</span>
              <span className="latency-value mono">{formatLatency(latencyMetrics.e2e_ms)}</span>
            </div>
            <div className="latency-item">
              <span className="latency-name">ASR</span>
              <span className="latency-value mono">{formatLatency(latencyMetrics.asr_ms)}</span>
            </div>
            <div className="latency-item">
              <span className="latency-name">MT</span>
              <span className="latency-value mono">{formatLatency(latencyMetrics.mt_ms)}</span>
            </div>
            <div className="latency-item">
              <span className="latency-name">TTS</span>
              <span className="latency-value mono">{formatLatency(latencyMetrics.tts_ms)}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace-grid">
        <section className="stack">
          <article className="panel">
            <p className="label">Session Source Text</p>
            <div className="text-box transcript-box session-text-box mono" ref={sourceScrollRef}>
              {renderSpeakerSegments(sourceSpeakerSegments, normalizeLiveText(asrPartial), showSpeaker)}
            </div>
          </article>
          <article className="panel">
            <p className="label">Session Translation Text</p>
            <div className="text-box transcript-box session-text-box" ref={translationScrollRef}>
              {renderSpeakerSegments(translationSpeakerSegments, "", showSpeaker)}
            </div>
          </article>
        </section>
        <article className="panel">
          <p className="label">Translation Feed</p>
          <div className="feed">
            {translations.length === 0 ? (
              <div className="empty">...</div>
            ) : (
              translations.map((item, index) => (
                <article className="card" key={`${index}-${item.translated_text}`}>
                  <p className="src">{item.source_text}</p>
                  <p className="dst">{item.translated_text}</p>
                </article>
              ))
            )}
          </div>
        </article>
      </section>

      <section className="panel timeline-panel">
        <div className="timeline-head">
          <p className="label">Live Audio Timeline</p>
          <div className="timeline-legend">
            <span className="timeline-legend-item">
              <span className="timeline-dot input"></span>
              Input
            </span>
            <span className="timeline-legend-item">
              <span className="timeline-dot output"></span>
              Output
            </span>
          </div>
        </div>
        <div className="timeline-frame">
          <canvas ref={timelineCanvasRef} className="timeline-canvas"></canvas>
        </div>
      </section>

      <section className="panel recent-audio-panel">
        <p className="label">Recent Session Audio</p>
        <div className="recent-session-list">
          {recentSessions.length === 0 ? (
            <div className="empty">...</div>
          ) : (
            recentSessions.map((item) => (
              <article className="recent-session-card" key={item.label}>
                <p className="recent-session-title mono">{formatSessionLabel(item.label)}</p>
                <div className="recent-session-grid">
                  <div className="recent-audio-block">
                    <p className="recent-audio-label">Input</p>
                    <audio controls preload="none" src={item.session_audio_url} className="audio-player" />
                    <a href={item.session_audio_url} download={item.session_audio_name} className="download-link">
                      Download Input
                    </a>
                  </div>
                  <div className="recent-audio-block">
                    <p className="recent-audio-label">Output</p>
                    {item.tts_audio_url ? (
                      <>
                        <audio controls preload="none" src={item.tts_audio_url} className="audio-player" />
                        <a href={item.tts_audio_url} download={item.tts_audio_name} className="download-link">
                          Download Output
                        </a>
                      </>
                    ) : (
                      <div className="empty">Pending</div>
                    )}
                  </div>
                </div>
              </article>
            ))
          )}
        </div>
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

function normalizeDisplayText(text) {
  return text
    .replace(/\s+([，。！？；：、,.!?;:])/g, "$1")
    .replace(/([（【《“‘])\s+/g, "$1")
    .replace(/\s+([）】》”’])/g, "$1")
    .trim();
}

function joinSessionText(previous, next) {
  const left = (previous || "").trim();
  const right = normalizeDisplayText(next || "");
  if (!right) return left;
  if (!left) return right;
  return `${left}${needsInlineSeparator(left, right) ? " " : ""}${right}`;
}

function normalizeLiveText(text) {
  const normalized = normalizeDisplayText(text || "");
  return normalized === "..." ? "" : normalized;
}

function appendSpeakerSegment(segments, speakerId, text) {
  const value = normalizeDisplayText(text || "");
  if (!value) return segments;

  const normalizedSpeakerId = String(speakerId || "").trim();
  const next = [...segments];
  const last = next[next.length - 1];
  if (last && last.speakerId === normalizedSpeakerId) {
    next[next.length - 1] = {
      ...last,
      text: joinSessionText(last.text, value),
    };
  } else {
    next.push({
      speakerId: normalizedSpeakerId,
      text: value,
    });
  }
  return next;
}

function formatSpeakerLabel(speakerId) {
  const match = String(speakerId || "").match(/^speaker_(\d+)$/);
  if (!match) return "speaker ?";
  return `speaker ${Number(match[1]) + 1}`;
}

function renderSpeakerSegments(segments, liveText = "", showSpeaker = true) {
  const live = normalizeLiveText(liveText);
  if (segments.length === 0 && !live) {
    return <p className="transcript-line live-static">...</p>;
  }

  if (!showSpeaker) {
    const merged = segments.reduce((acc, segment) => joinSessionText(acc, segment.text), "");
    return (
      <>
        {merged ? <p className="speaker-transcript-text">{merged}</p> : null}
        {live ? <p className="transcript-line live speaker-live-text">{live}</p> : null}
      </>
    );
  }

  return (
    <>
      {segments.map((segment, index) => (
        <p className="speaker-transcript-block" key={`${index}-${segment.speakerId}`}>
          <span className="speaker-transcript-label">{formatSpeakerLabel(segment.speakerId)}</span>
          <span className="speaker-transcript-text">{segment.text}</span>
        </p>
      ))}
      {live ? <p className="transcript-line live speaker-live-text">{live}</p> : null}
    </>
  );
}

function formatLatency(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) {
    return "--";
  }
  return `${Math.round(number)} ms`;
}

function formatVisitCount(value) {
  return new Intl.NumberFormat("zh-CN").format(Math.max(0, Math.round(value || 0)));
}

function nowMs() {
  return performance.now();
}

function needsInlineSeparator(left, right) {
  return /[A-Za-z0-9]$/.test(left) && /^[A-Za-z0-9]/.test(right);
}

function formatSessionLabel(label) {
  const match = String(label || "").match(/^(\d{8})_(\d{6})_(\d{3})$/);
  if (!match) return label || "--";
  const [, date, time, ms] = match;
  return `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)} ${time.slice(0, 2)}:${time.slice(2, 4)}:${time.slice(4, 6)}.${ms}`;
}

function rmsFromFloat32(samples) {
  if (!samples || samples.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / samples.length);
}

function rmsFromPCMBuffer(arrayBuffer) {
  const pcm = new Int16Array(arrayBuffer);
  if (pcm.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < pcm.length; i += 1) {
    const sample = pcm[i] / 32768;
    sum += sample * sample;
  }
  return Math.sqrt(sum / pcm.length);
}

function buildPCMChunkLevels(arrayBuffer, sampleRate) {
  const pcm = new Int16Array(arrayBuffer);
  if (pcm.length === 0 || !sampleRate) return [];

  const segmentSize = Math.max(1, Math.round(sampleRate * 0.04));
  const points = [];

  for (let start = 0; start < pcm.length; start += segmentSize) {
    const end = Math.min(pcm.length, start + segmentSize);
    let sum = 0;
    for (let index = start; index < end; index += 1) {
      const sample = pcm[index] / 32768;
      sum += sample * sample;
    }
    const rms = Math.sqrt(sum / Math.max(1, end - start));
    points.push({
      relativeMs: (((start + end) / 2) / sampleRate) * 1000,
      value: Math.min(1, Math.max(0, rms * 4.5)),
    });
  }
  return points;
}

function recordScheduledOutputLevels(historyRef, waveform, playbackStartMs) {
  if (!waveform || waveform.length === 0) return;
  const history = historyRef.current;
  for (const point of waveform) {
    history.push({
      time: playbackStartMs + point.relativeMs,
      value: point.value,
    });
  }
  const cutoff = playbackStartMs - 20000;
  while (history.length > 0 && history[0].time < cutoff) {
    history.shift();
  }
}

function recordAudioLevel(historyRef, value, timestamp) {
  const history = historyRef.current;
  history.push({ time: timestamp, value: Math.min(1, Math.max(0, value * 4.5)) });
  const cutoff = timestamp - 20000;
  while (history.length > 0 && history[0].time < cutoff) {
    history.shift();
  }
}

function drawLane(ctx, history, now, leftPad, graphWidth, top, laneHeight, color, label, options = {}) {
  const midY = top + laneHeight / 2;
  const { extendToNow = false } = options;
  ctx.fillStyle = "rgba(255,255,255,0.025)";
  ctx.fillRect(leftPad, top, graphWidth, laneHeight);
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.moveTo(leftPad, midY);
  ctx.lineTo(leftPad + graphWidth, midY);
  ctx.stroke();

  ctx.fillStyle = "rgba(135, 161, 193, 0.9)";
  ctx.font = '12px "SFMono-Regular", monospace';
  ctx.textAlign = "left";
  ctx.fillText(label, leftPad + 8, top + 16);

  const visible = history.filter((point) => point.time <= now && now - point.time <= 16000);
  if (visible.length < 2) {
    return;
  }

  const sorted = [...visible].sort((a, b) => a.time - b.time);
  const points = sorted.map((point) => {
    const age = now - point.time;
    const x = leftPad + graphWidth - (age / 16000) * graphWidth;
    const amplitude = point.value * (laneHeight * 0.44);
    return {
      x,
      y: midY - amplitude,
      mirrorY: midY + amplitude,
      value: point.value,
    };
  });

  if (extendToNow && points.length > 0) {
    const hasFuturePoint = history.some((point) => point.time > now);
    const lastPoint = points[points.length - 1];
    const value = hasFuturePoint ? lastPoint.value : 0;
    const amplitude = value * (laneHeight * 0.44);
    points.push({
      x: leftPad + graphWidth,
      y: midY - amplitude,
      mirrorY: midY + amplitude,
      value,
    });
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.stroke();

  ctx.strokeStyle = color.replace("0.95", "0.35");
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.mirrorY);
    } else {
      ctx.lineTo(point.x, point.mirrorY);
    }
  });
  ctx.stroke();

  ctx.fillStyle = color.replace("0.95", "0.12");
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  for (let index = points.length - 1; index >= 0; index -= 1) {
    const point = points[index];
    ctx.lineTo(point.x, point.mirrorY);
  }
  ctx.closePath();
  ctx.fill();

  if (points.length > 0) {
    const lastPoint = points[points.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, 2.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}
