/**
 * MedBrief frontend — voice capture and playback.
 *
 * State machine:  IDLE → RECORDING → PROCESSING → SPEAKING → IDLE
 *
 * Silence detection:
 *   AudioWorkletProcessor runs in a dedicated audio thread, computing
 *   RMS per 128-sample frame. After 1.5 s of RMS < silenceThreshold
 *   AND the user has started speaking, it posts a 'silence_detected'
 *   message back to the main thread which stops the MediaRecorder.
 *
 * Session identity:
 *   A UUID is stored in sessionStorage ('medbrief_session_id').
 *   This is tab-scoped and not persisted across browser sessions,
 *   matching Haptik's stateless-per-tab design (SKILL.md §Voice UX).
 */

'use strict';

// ── AudioWorklet processor source (inlined as a Blob URL) ────────────────────
// This runs in the audio rendering thread — no DOM, no fetch, no closures.
const SILENCE_PROCESSOR_SRC = `
class SilenceDetector extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this._threshold = opts.silenceThreshold || 0.01;
    this._silenceDuration = opts.silenceDuration || 1.5;   // seconds
    this._silentSamples = 0;
    this._hasSpeech = false;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    let sumSq = 0;
    for (let i = 0; i < channel.length; i++) sumSq += channel[i] * channel[i];
    const rms = Math.sqrt(sumSq / channel.length);

    if (rms > this._threshold) {
      this._hasSpeech = true;
      this._silentSamples = 0;
    } else if (this._hasSpeech) {
      this._silentSamples += channel.length;
      if (this._silentSamples / sampleRate >= this._silenceDuration) {
        this.port.postMessage({ type: 'silence_detected' });
        this._hasSpeech = false;
        this._silentSamples = 0;
      }
    }
    return true; // keep processor alive
  }
}
registerProcessor('silence-detector', SilenceDetector);
`;

// ── State constants ───────────────────────────────────────────────────────────
const S = Object.freeze({
  IDLE:       'idle',
  RECORDING:  'recording',
  PROCESSING: 'processing',
  SPEAKING:   'speaking',
});

const BADGE_LABELS = {
  [S.IDLE]:       'Idle',
  [S.RECORDING]:  'Recording…',
  [S.PROCESSING]: 'Processing…',
  [S.SPEAKING]:   'Speaking…',
};

// ── DOM ───────────────────────────────────────────────────────────────────────
const micBtn     = document.getElementById('micBtn');
const badge      = document.getElementById('badge');
const badgeText  = document.getElementById('badgeText');
const errorMsg   = document.getElementById('errorMsg');
const transcript = document.getElementById('transcript');

// ── Session (tab-scoped per SKILL.md) ────────────────────────────────────────
function getSessionId() {
  let sid = sessionStorage.getItem('medbrief_session_id');
  if (!sid) {
    sid = crypto.randomUUID();
    sessionStorage.setItem('medbrief_session_id', sid);
  }
  return sid;
}

// ── App state ─────────────────────────────────────────────────────────────────
let state         = S.IDLE;
let mediaRecorder = null;
let audioChunks   = [];
let audioCtx      = null;
let workletNode   = null;
let mediaStream   = null;

// ── State machine helpers ─────────────────────────────────────────────────────
function setState(s) {
  state = s;
  micBtn.dataset.state  = s;
  badge.dataset.state   = s;
  badgeText.textContent = BADGE_LABELS[s];
  micBtn.disabled       = (s === S.PROCESSING || s === S.SPEAKING);
  errorMsg.textContent  = '';
}

function showError(msg) {
  errorMsg.textContent = msg;
  setState(S.IDLE);
}

// ── Transcript helpers ────────────────────────────────────────────────────────
function addTurn(role, text) {
  const div = document.createElement('div');
  div.className = `turn ${role}`;

  const label = document.createElement('span');
  label.className = 'turn-label';
  label.textContent = role === 'user' ? 'You' : 'MedBrief';

  const body = document.createElement('span');
  body.textContent = text;

  div.appendChild(label);
  div.appendChild(body);
  transcript.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

// ── AudioWorklet silence detection ────────────────────────────────────────────
async function attachSilenceDetector(stream) {
  audioCtx = new AudioContext();
  const source = audioCtx.createMediaStreamSource(stream);

  const blob    = new Blob([SILENCE_PROCESSOR_SRC], { type: 'application/javascript' });
  const blobUrl = URL.createObjectURL(blob);

  try {
    await audioCtx.audioWorklet.addModule(blobUrl);
    workletNode = new AudioWorkletNode(audioCtx, 'silence-detector', {
      processorOptions: { silenceThreshold: 0.01, silenceDuration: 1.5 },
    });
    workletNode.port.onmessage = (ev) => {
      if (ev.data.type === 'silence_detected' && state === S.RECORDING) {
        stopRecording();
      }
    };
    source.connect(workletNode);
    // Do NOT connect to destination — we don't want mic playback
  } catch (err) {
    // AudioWorklet unavailable (unlikely in modern browsers).
    // The user can still tap the button to stop manually.
    console.warn('AudioWorklet unavailable, silence detection disabled:', err);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

// ── Recording ─────────────────────────────────────────────────────────────────
async function startRecording() {
  if (state !== S.IDLE) return;
  errorMsg.textContent = '';

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  } catch (err) {
    showError('Microphone access denied. Allow microphone access and reload.');
    return;
  }

  await attachSilenceDetector(mediaStream);

  audioChunks = [];
  const mimeType =
    MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus'
    : MediaRecorder.isTypeSupported('audio/webm')           ? 'audio/webm'
    :                                                          'audio/ogg';

  mediaRecorder = new MediaRecorder(mediaStream, { mimeType });
  mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
  mediaRecorder.onstop = handleRecordingStop;
  mediaRecorder.start(100); // fire ondataavailable every 100 ms

  setState(S.RECORDING);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  cleanupAudioPipeline();
}

function cleanupAudioPipeline() {
  if (workletNode) { workletNode.disconnect(); workletNode = null; }
  if (audioCtx)    { audioCtx.close().catch(() => {}); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
}

// ── After recording stops: send to gateway ───────────────────────────────────
async function handleRecordingStop() {
  if (!audioChunks.length) { setState(S.IDLE); return; }

  setState(S.PROCESSING);

  const mimeType  = audioChunks[0].type || 'audio/webm';
  const audioBlob = new Blob(audioChunks, { type: mimeType });
  audioChunks     = [];

  const formData = new FormData();
  formData.append('audio', audioBlob, 'audio.webm');

  const sessionId = getSessionId();

  let response;
  try {
    response = await fetch('/voice', {
      method: 'POST',
      body:   formData,
      headers: { 'X-Session-Id': sessionId },
    });
  } catch (err) {
    showError(`Network error: ${err.message}`);
    return;
  }

  if (!response.ok) {
    let detail = `Server error ${response.status}`;
    try {
      const body = await response.json();
      detail = (body.detail && body.detail.error) || detail;
    } catch (_) { /* ignore parse error */ }
    showError(detail);
    return;
  }

  // Read response headers for transcript text
  const returnedSid  = response.headers.get('X-Session-Id');
  const userTextEnc  = response.headers.get('X-User-Text')     || '';
  const replyTextEnc = response.headers.get('X-Response-Text') || '';

  if (returnedSid) sessionStorage.setItem('medbrief_session_id', returnedSid);

  const userText  = userTextEnc  ? decodeURIComponent(userTextEnc)  : '';
  const replyText = replyTextEnc ? decodeURIComponent(replyTextEnc) : '';

  if (userText)  addTurn('user', userText);

  // Play audio stream
  setState(S.SPEAKING);
  try {
    await playAudioStream(response);
  } catch (err) {
    console.error('Audio playback error:', err);
  }

  if (replyText) addTurn('assistant', replyText);
  setState(S.IDLE);
}

// ── Audio streaming playback ──────────────────────────────────────────────────
/**
 * Streams MP3 audio from a fetch Response using MediaSource API.
 * Falls back to collecting all bytes and playing as a Blob URL if
 * MediaSource / audio/mpeg is unsupported (e.g. older Safari).
 */
async function playAudioStream(response) {
  const reader = response.body.getReader();

  // Detect MediaSource support for audio/mpeg (Chrome, Firefox)
  const msSupported =
    typeof MediaSource !== 'undefined' &&
    MediaSource.isTypeSupported('audio/mpeg');

  if (msSupported) {
    return playViaMediaSource(reader);
  } else {
    return playViaBlobUrl(reader);
  }
}

function playViaMediaSource(reader) {
  return new Promise((resolve, reject) => {
    const ms    = new MediaSource();
    const audio = new Audio();
    audio.src   = URL.createObjectURL(ms);

    ms.addEventListener('sourceopen', () => {
      let sb;
      try {
        sb = ms.addSourceBuffer('audio/mpeg');
      } catch (err) {
        // Fallback if addSourceBuffer fails at runtime
        URL.revokeObjectURL(audio.src);
        collectAndPlay(reader).then(resolve).catch(reject);
        return;
      }

      const queue   = [];
      let   reading = true;

      sb.addEventListener('updateend', () => {
        if (queue.length > 0 && !sb.updating) {
          sb.appendBuffer(queue.shift());
        } else if (!reading && queue.length === 0 && !sb.updating) {
          try { ms.endOfStream(); } catch (_) {}
        }
      });

      const pump = async () => {
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) {
              reading = false;
              if (!sb.updating && queue.length === 0) {
                try { ms.endOfStream(); } catch (_) {}
              }
              break;
            }
            if (value) {
              if (!sb.updating) {
                sb.appendBuffer(value.buffer);
              } else {
                queue.push(value.buffer);
              }
            }
          }
        } catch (err) {
          reject(err);
        }
      };

      pump();
      audio.oncanplay = () => audio.play().catch(reject);
      audio.onended   = () => { URL.revokeObjectURL(audio.src); resolve(); };
      audio.onerror   = () => { URL.revokeObjectURL(audio.src); reject(new Error('Audio playback error')); };
    });
  });
}

async function collectAndPlay(reader) {
  return playViaBlobUrl(reader);
}

async function playViaBlobUrl(reader) {
  const chunks = [];
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    if (value) chunks.push(value);
  }
  const blob   = new Blob(chunks, { type: 'audio/mpeg' });
  const url    = URL.createObjectURL(blob);
  const audio  = new Audio(url);
  return new Promise((resolve, reject) => {
    audio.onended = () => { URL.revokeObjectURL(url); resolve(); };
    audio.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Audio playback failed')); };
    audio.play().catch(reject);
  });
}

// ── Controls ──────────────────────────────────────────────────────────────────
micBtn.addEventListener('click', () => {
  if (state === S.IDLE)      startRecording();
  else if (state === S.RECORDING) stopRecording();
});

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target === document.body) {
    e.preventDefault();
    if (state === S.IDLE)           startRecording();
    else if (state === S.RECORDING) stopRecording();
  }
});
