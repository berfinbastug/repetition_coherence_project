import { useState, useCallback, useRef, useEffect, useMemo } from "react";

// ─── Tone-cloud generation (mirrors tone_cloud_production.py) ───────────────

function buildFreqGrid(lowf, highf, fstep) {
  const grid = [lowf];
  while (grid[grid.length - 1] * Math.pow(2, fstep) <= highf) {
    grid.push(grid[grid.length - 1] * Math.pow(2, fstep));
  }
  return grid;
}

function seededRandom(seed) {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateToneCloud({ unitdur, percentage, nrep, seed }) {
  const rand = seededRandom(seed);
  const lowf = 200, highf = 3000, fstep = 0.4, timestep = 0.05;
  const freqgrid = buildFreqGrid(lowf, highf, fstep);
  const nfsteps = freqgrid.length;
  const timegrid = [];
  for (let t = 0; t < unitdur - timestep + 1e-5; t += timestep) timegrid.push(t);
  const ntsteps = timegrid.length;
  const ntones = nfsteps * ntsteps;

  // Random perturbations for first cycle
  const fnorm = Array.from({ length: nfsteps }, () =>
    Array.from({ length: ntsteps }, () => rand())
  );
  const tnorm = Array.from({ length: nfsteps }, () =>
    Array.from({ length: ntsteps }, () => rand())
  );

  // Nominal gridss
  const bigf = freqgrid.map((f) => Array(ntsteps).fill(f));
  const bigt = Array.from({ length: nfsteps }, () => [...timegrid]);

  // Perturbed grids (first cycle baseline)
  let zf = bigf.map((row, i) =>
    row.map((f, j) => Math.pow(2, Math.log2(f) + fnorm[i][j] * fstep))
  );
  let zt = bigt.map((row, i) =>
    row.map((t, j) => t + tnorm[i][j] * timestep)
  );

  // Determine frozen vs new tones
  let nreptones;
  if (percentage === 0) nreptones = 0;
  else if (percentage === 1) nreptones = ntones;
  else nreptones = Math.ceil(ntones * percentage);

  const idxdraw = Array.from({ length: ntones }, (_, i) => i);
  // Fisher-Yates shuffle with seeded random
  for (let i = idxdraw.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idxdraw[i], idxdraw[j]] = [idxdraw[j], idxdraw[i]];
  }
  const idxrep = new Set(idxdraw.slice(0, nreptones));

  // Build all cycles
  const tones = [];
  for (let cycle = 0; cycle < nrep; cycle++) {
    for (let i = 0; i < nfsteps; i++) {
      for (let j = 0; j < ntsteps; j++) {
        const idx = i * ntsteps + j;
        const isFrozen = idxrep.has(idx);
        if (!isFrozen) {
          zf[i][j] = Math.pow(2, Math.log2(bigf[i][j]) + rand() * fstep);
          zt[i][j] = bigt[i][j] + rand() * timestep;
        }
        tones.push({
          freq: zf[i][j],
          time: zt[i][j] + cycle * unitdur,
          duration: 0.05,
          frozen: isFrozen,
          cycle,
        });
      }
    }
  }

  return { tones, freqgrid, totalDuration: nrep * unitdur, lowf, highf };
}

// ─── Audio synthesis ────────────────────────────────────────────────────────

function synthesizeAudio(tones, totalDuration, sampleRate = 44100) {
  const rtime = 0.025;
  const bufferLen = Math.ceil((totalDuration + 0.4 + 0.05) * sampleRate);
  const buffer = new Float32Array(bufferLen);
  const padSamples = Math.floor(0.2 * sampleRate);

  for (const tone of tones) {
    const toneDur = tone.duration;
    const nSamples = Math.floor(toneDur * sampleRate);
    const istart = padSamples + Math.round(tone.time * sampleRate);
    const rampSamples = Math.floor(rtime * sampleRate);

    for (let n = 0; n < nSamples && istart + n < bufferLen; n++) {
      const t = n / sampleRate;
      let sample = Math.sin(2 * Math.PI * tone.freq * t);

      // Cosine-squared ramp
      if (n < rampSamples) {
        const tr = n / sampleRate;
        const ramp = Math.pow((Math.cos(2 * Math.PI * tr / rtime / 2 + Math.PI) + 1) / 2, 2);
        sample *= ramp;
      } else if (n >= nSamples - rampSamples) {
        const tr = (nSamples - 1 - n) / sampleRate;
        const ramp = Math.pow((Math.cos(2 * Math.PI * tr / rtime / 2 + Math.PI) + 1) / 2, 2);
        sample *= ramp;
      }

      buffer[istart + n] += sample;
    }
  }

  // Normalize (loop instead of spread to avoid stack overflow on large arrays)
  let peak = 0;
  for (let i = 0; i < buffer.length; i++) {
    const abs = Math.abs(buffer[i]);
    if (abs > peak) peak = abs;
  }
  if (peak > 0) {
    const scale = 0.3 / peak;
    for (let i = 0; i < buffer.length; i++) buffer[i] *= scale;
  }

  return buffer;
}

// ─── Visualization component ────────────────────────────────────────────────

const FROZEN_COLOR = "#2576B8";
const NEW_COLOR = "#5c5b5b";

function ToneCloudViz({ tones, totalDuration, lowf, highf, freqgrid, unitdur, playheadPos }) {
  const W = 700;
  const H = 220;
  const PAD = { top: 12, bottom: 28, left: 48, right: 16 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const logLow = Math.log2(lowf);
  const logHigh = Math.log2(highf * 1.15);
  const toX = (t) => PAD.left + (t / totalDuration) * plotW;
  const toY = (f) => H - PAD.bottom - ((Math.log2(f) - logLow) / (logHigh - logLow)) * plotH;

  // Cycle boundary lines
  const boundaries = [];
  for (let c = 1; c < Math.round(totalDuration / unitdur); c++) {
    boundaries.push(c * unitdur);
  }

  // Grid lines (freq)
  const upperLimit = highf + 500;
  const freqgridPlot = [...freqgrid, upperLimit];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: 700, display: "block" }}>
      {/* Background */}
      <rect x={PAD.left} y={PAD.top} width={plotW} height={plotH} fill="rgb(255, 255, 255)" rx="2" />

      {/* Freq grid */}
      {freqgridPlot.map((f, i) => (
        <line key={`fg${i}`} x1={PAD.left} x2={W - PAD.right}
          y1={toY(f)} y2={toY(f)} stroke="#ddd" strokeWidth="0.5" />
      ))}

      {/* Time grid */}
      {Array.from({ length: Math.round(totalDuration / 0.05) + 1 }, (_, i) => i * 0.05).map((t, i) => (
        <line key={`tg${i}`} x1={toX(t)} x2={toX(t)}
          y1={PAD.top} y2={H - PAD.bottom}
          stroke={boundaries.some((b) => Math.abs(t - b) < 0.001) ? "#1a1a1a" : "#eee"}
          strokeWidth={boundaries.some((b) => Math.abs(t - b) < 0.001) ? 1.5 : 0.4}
          strokeDasharray={boundaries.some((b) => Math.abs(t - b) < 0.001) ? "4,3" : "none"} />
      ))}

      {/* Tones */}
      {tones.map((tone, i) => (
        <line key={i}
          x1={toX(tone.time)} x2={toX(tone.time + tone.duration)}
          y1={toY(tone.freq)} y2={toY(tone.freq)}
          stroke={tone.frozen ? FROZEN_COLOR : NEW_COLOR}
          strokeWidth={tone.frozen ? 2.2 : 1.8}
          strokeLinecap="round"
          opacity={tone.frozen ? 1 : 0.55} />
      ))}

      {/* Playhead
            {playheadPos !== null && (
              <line x1={toX(playheadPos)} x2={toX(playheadPos)}
                y1={PAD.top} y2={H - PAD.bottom}
                stroke="#E84545" strokeWidth={2} opacity={0.8} />
            )}
      */} 

      {/* Cycle labels */}
      {Array.from({ length: Math.round(totalDuration / unitdur) }, (_, i) => (
        <text key={`cl${i}`}
          x={toX(i * unitdur + unitdur / 2)} y={H - 8}
          textAnchor="middle" fontSize="11"
          fill="#888" fontFamily="'JetBrains Mono', monospace">
          Cycle {i + 1}
        </text>
      ))}

      {/* Y axis label */}
      <text x={12} y={H / 2} textAnchor="middle" fontSize="11" fill="#888"
        fontFamily="'JetBrains Mono', monospace"
        transform={`rotate(-90, 12, ${H / 2})`}>
        Frequency (Hz)
      </text>
    </svg>
  );
}

// ─── Main App ───────────────────────────────────────────────────────────────
// change this part
const COHERENCE_STEPS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const DURATION_OPTIONS = [
  { value: 0.4, label: "0.4 s" },
  { value: 0.7, label: "0.7 s" },
  { value: 1.0, label: "1.0 s" },
];

export default function App() {
  const [coherence, setCoherence] = useState(6); // index into COHERENCE_STEPS
  const [durIdx, setDurIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playheadPos, setPlayheadPos] = useState(null);
  const audioCtxRef = useRef(null);
  const sourceRef = useRef(null);
  const animRef = useRef(null);
  const startTimeRef = useRef(null);
  const SEED = 42;
  const NREP_VIZ = 4;  // used in generateToneCloud for the vizualization
  const NREP_AUDIO = 10;  // used in synthesizeAudio for the vizualization

  const unitdur = DURATION_OPTIONS[durIdx].value;
  const percentage = COHERENCE_STEPS[coherence];

  const cloud = useMemo(
    () => generateToneCloud({ unitdur, percentage, nrep: NREP_VIZ, seed: SEED }),
    [unitdur, percentage]
  );

  const cloudAudio = useMemo(
    () => generateToneCloud({ unitdur, percentage, nrep: NREP_AUDIO, seed: SEED }),
    [unitdur, percentage]
  );

  const stopPlayback = useCallback(() => {
    if (sourceRef.current) {
      try { sourceRef.current.stop(); } catch (_) {}
      sourceRef.current = null;
    }
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setIsPlaying(false);
    setPlayheadPos(null);
    startTimeRef.current = null;
  }, []);

  const playAudio = useCallback(async () => {
    if (isPlaying) { stopPlayback(); return; }

    const ctx = audioCtxRef.current || new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = ctx;

    // resume() is async — wait for the context to actually be running
    // before starting playback, or the sound is dropped (silence).
    if (ctx.state !== "running") {
      try { await ctx.resume(); } catch (e) { console.error("resume failed", e); }
    }

    const sampleRate = ctx.sampleRate;
    const rawBuffer = synthesizeAudio(cloudAudio.tones, cloudAudio.totalDuration, sampleRate);
    const audioBuffer = ctx.createBuffer(1, rawBuffer.length, sampleRate);
    audioBuffer.copyToChannel(rawBuffer, 0);

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);

    source.onended = () => { stopPlayback(); };

    const padDur = 0.2;
    const totalAudioDur = cloudAudio.totalDuration + 0.4 + 0.05 / sampleRate;
    startTimeRef.current = ctx.currentTime;
    source.start();
    sourceRef.current = source;
    setIsPlaying(true);

    const animate = () => {
      if (!startTimeRef.current) return;
      const elapsed = ctx.currentTime - startTimeRef.current - padDur;
      if (elapsed >= 0 && elapsed <= cloud.totalDuration) {
        setPlayheadPos(elapsed);
      }
      if (ctx.currentTime - startTimeRef.current < totalAudioDur) {
        animRef.current = requestAnimationFrame(animate);
      }
    };
    animRef.current = requestAnimationFrame(animate);
  }, [isPlaying, cloud, stopPlayback]);

  // Cleanup
  useEffect(() => () => { stopPlayback(); }, [stopPlayback]);
  // Stop when params change
  useEffect(() => { stopPlayback(); }, [unitdur, percentage, stopPlayback]);

  const frozenCount = cloud.tones.filter((t) => t.cycle === 0 && t.frozen).length;
  const totalPerCycle = cloud.tones.filter((t) => t.cycle === 0).length;

  return (
    <div style={{
      minHeight: "100vh",
      background: "linear-gradient(168deg, #0D1117 0%, #161B22 50%, #1A1F2B 100%)",
      color: "#C9D1D9",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      padding: "32px 24px",
      boxSizing: "border-box",
    }}>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{
            fontSize: 20, fontWeight: 700, color: "#E6EDF3", margin: 0,
            letterSpacing: "-0.02em", lineHeight: 1.3,
          }}>
            Repetition Coherence Explorer
          </h1>
          <p style={{
            fontSize: 13, color: "#7D8590", margin: "6px 0 0", lineHeight: 1.5,
          }}>
            Interactive tone-cloud stimulus from Bastug et al. — adjust coherence and unit duration to see how auditory objects emerge from noise.
          </p>
        </div>

        {/* Controls */}
        <div style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10,
          padding: "20px 24px",
          marginBottom: 20,
        }}>
          {/* Coherence slider */}
          <div style={{ marginBottom: 18 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
              <label style={{ fontSize: 12, color: "#7D8590", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Repetition Coherence
              </label>
              <span style={{
                fontSize: 18, fontWeight: 700, color: FROZEN_COLOR,
                fontVariantNumeric: "tabular-nums",
              }}>
                {percentage.toFixed(2)}
              </span>
            </div>
            <input type="range" min={0} max={COHERENCE_STEPS.length - 1} step={1}
              value={coherence} onChange={(e) => setCoherence(+e.target.value)}
              style={{ width: "100%", accentColor: FROZEN_COLOR, cursor: "pointer" }} />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#484F58", marginTop: 4 }}>
              <span>Random</span><span>Fully repeating</span>
            </div>
          </div>

          {/* Unit duration selector */}
          <div>
            <label style={{ fontSize: 12, color: "#7D8590", textTransform: "uppercase", letterSpacing: "0.06em", display: "block", marginBottom: 8 }}>
              Unit Duration
            </label>
            <div style={{ display: "flex", gap: 8 }}>
              {DURATION_OPTIONS.map((opt, i) => (
                <button key={opt.value} onClick={() => setDurIdx(i)}
                  style={{
                    flex: 1, padding: "8px 0", fontSize: 14, fontWeight: 600,
                    fontFamily: "inherit", cursor: "pointer",
                    background: durIdx === i ? FROZEN_COLOR : "rgba(255,255,255,0.06)",
                    color: durIdx === i ? "#fff" : "#7D8590",
                    border: durIdx === i ? `1px solid ${FROZEN_COLOR}` : "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 6,
                    transition: "all 0.15s ease",
                  }}>
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Visualization */}
        <div style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10,
          padding: "16px 16px 12px",
          marginBottom: 20,
        }}>
          <ToneCloudViz {...cloud} unitdur={unitdur} playheadPos={playheadPos} />

          {/* Legend */}
          <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 10, fontSize: 12 }}>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 18, height: 3, background: FROZEN_COLOR, borderRadius: 2, display: "inline-block" }} />
              <span style={{ color: "#7D8590" }}>Frozen ({frozenCount} tones)</span>
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 18, height: 3, background: NEW_COLOR, borderRadius: 2, display: "inline-block", opacity: 0.55 }} />
              <span style={{ color: "#7D8590" }}>New each cycle ({totalPerCycle - frozenCount})</span>
            </span>
          </div>
        </div>

        {/* Play button */}
        <button onClick={playAudio}
          style={{
            width: "100%", padding: "14px 0", fontSize: 15, fontWeight: 700,
            fontFamily: "inherit", cursor: "pointer",
            background: isPlaying
              ? "rgba(232, 69, 69, 0.15)"
              : `linear-gradient(135deg, ${FROZEN_COLOR}, #1A5FA0)`,
            color: isPlaying ? "#E84545" : "#fff",
            border: isPlaying ? "1px solid rgba(232,69,69,0.3)" : "1px solid rgba(255,255,255,0.1)",
            borderRadius: 8,
            transition: "all 0.2s ease",
            letterSpacing: "0.04em",
          }}>
          {isPlaying ? "■  Stop" : "▶  Play Stimulus"}
        </button>

        {/* Info panel */}
        <div style={{
          marginTop: 20, padding: "14px 18px",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 8,
          fontSize: 12, lineHeight: 1.6, color: "#484F58",
        }}>
          <strong style={{ color: "#7D8590" }}>How it works:</strong>{" "}
          Each column of the grid is one 50 ms time step. Ten frequencies span 200–3000 Hz (0.4-octave spacing).
          Blue tones are frozen across all four cycles; gray tones are regenerated each cycle.
          At low coherence the sequence sounds random. As coherence increases, a repeating
          pattern emerges — the auditory object.
        </div>
      </div>
    </div>
  );
}
