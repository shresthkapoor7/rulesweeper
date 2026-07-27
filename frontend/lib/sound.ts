// Tiny synthesized sound effects via the Web Audio API — no asset files.
// The AudioContext is created lazily on first use so it starts inside a user
// gesture (satisfies browser autoplay policy).

let ctx: AudioContext | null = null;
let enabled = true;

export function isSoundEnabled(): boolean {
  return enabled;
}

export function setSoundEnabled(on: boolean) {
  enabled = on;
  try {
    localStorage.setItem("sound", on ? "on" : "off");
  } catch {
    /* ignore */
  }
}

export function initSoundPref() {
  try {
    enabled = localStorage.getItem("sound") !== "off";
  } catch {
    /* ignore */
  }
}

function getCtx(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!ctx) {
    const AC =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

// One short tone with an exponential decay envelope.
function tone(
  freq: number,
  start: number,
  dur: number,
  type: OscillatorType,
  gain: number
) {
  const ac = ctx!;
  const osc = ac.createOscillator();
  const g = ac.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, ac.currentTime + start);
  g.gain.setValueAtTime(0.0001, ac.currentTime + start);
  g.gain.exponentialRampToValueAtTime(gain, ac.currentTime + start + 0.008);
  g.gain.exponentialRampToValueAtTime(
    0.0001,
    ac.currentTime + start + dur
  );
  osc.connect(g);
  g.connect(ac.destination);
  osc.start(ac.currentTime + start);
  osc.stop(ac.currentTime + start + dur + 0.02);
}

// Filtered white-noise burst — used for the explosion.
function noise(start: number, dur: number, gain: number, cutoff: number) {
  const ac = ctx!;
  const frames = Math.floor(ac.sampleRate * dur);
  const buffer = ac.createBuffer(1, frames, ac.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < frames; i++) data[i] = Math.random() * 2 - 1;
  const src = ac.createBufferSource();
  src.buffer = buffer;
  const filter = ac.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.value = cutoff;
  const g = ac.createGain();
  g.gain.setValueAtTime(gain, ac.currentTime + start);
  g.gain.exponentialRampToValueAtTime(0.0001, ac.currentTime + start + dur);
  src.connect(filter);
  filter.connect(g);
  g.connect(ac.destination);
  src.start(ac.currentTime + start);
  src.stop(ac.currentTime + start + dur + 0.02);
}

function guard(): boolean {
  if (!enabled) return false;
  return getCtx() != null;
}

// A soft blip when a safe cell is revealed.
export function playReveal() {
  if (!guard()) return;
  tone(440, 0, 0.09, "sine", 0.16);
}

// A crisp two-step for planting/removing a flag.
export function playFlag() {
  if (!guard()) return;
  tone(660, 0, 0.05, "square", 0.09);
  tone(880, 0.05, 0.06, "square", 0.09);
}

// A low thud when a mine is hit but the player survives (multi-life modes).
export function playHit() {
  if (!guard()) return;
  noise(0, 0.18, 0.35, 700);
  tone(120, 0, 0.2, "sawtooth", 0.2);
}

// A bigger explosion when the game is lost.
export function playLose() {
  if (!guard()) return;
  noise(0, 0.5, 0.5, 900);
  tone(90, 0, 0.5, "sawtooth", 0.25);
  tone(60, 0.05, 0.55, "triangle", 0.2);
}

// A short rising arpeggio on a win.
export function playWin() {
  if (!guard()) return;
  const notes = [523.25, 659.25, 783.99, 1046.5];
  notes.forEach((f, i) => tone(f, i * 0.1, 0.18, "triangle", 0.18));
}
