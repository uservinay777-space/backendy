/**
 * PhantomFlow Frontend Controller & Voice Service Client
 * 
 * Integrates Web Speech Recognition (STT), SpeechSynthesis (TTS),
 * animated audio waveform visualizer, and live SwitchSync slot telemetry.
 */

// Application State
const state = {
  sessionId: 'session_demo_01',
  isListening: false,
  isSpeaking: false,
  recognition: null,
  synth: window.speechSynthesis,
  activeUtterance: null,
  selectedVoice: null,
  waveformAnimationId: null,
  audioPlayer: null,
  currentTurnId: null,
};

// DOM Elements
const elements = {
  serverStatusBadge: document.getElementById('serverStatusBadge'),
  statusText: document.getElementById('statusText'),
  sessionIdDisplay: document.getElementById('sessionIdDisplay'),
  resetSessionBtn: document.getElementById('resetSessionBtn'),
  waveformCanvas: document.getElementById('waveformCanvas'),
  voiceStateLabel: document.getElementById('voiceStateLabel'),
  micBtn: document.getElementById('micBtn'),
  autoTtsToggle: document.getElementById('autoTtsToggle'),
  ttsProviderSelect: document.getElementById('ttsProviderSelect'),
  voiceLanguageSelect: document.getElementById('voiceLanguageSelect'),
  interruptAudioBtn: document.getElementById('interruptAudioBtn'),
  dialogueContainer: document.getElementById('dialogueContainer'),
  chatForm: document.getElementById('chatForm'),
  textInput: document.getElementById('textInput'),
  sendBtn: document.getElementById('sendBtn'),
  quickChips: document.querySelectorAll('.chip-btn'),

  // Slots
  valOrigin: document.getElementById('valOrigin'),
  valDestination: document.getElementById('valDestination'),
  valDate: document.getElementById('valDate'),
  valTime: document.getElementById('valTime'),
  valTrainNumber: document.getElementById('valTrainNumber'),
  valPnr: document.getElementById('valPnr'),
  missingFieldsContainer: document.getElementById('missingFieldsContainer'),
  missingFieldsText: document.getElementById('missingFieldsText'),

  // Metrics
  metricLatency: document.getElementById('metricLatency'),
  metricRoute: document.getElementById('metricRoute'),
  metricIntent: document.getElementById('metricIntent'),
  metricLang: document.getElementById('metricLang'),
  metricCurseCheck: document.getElementById('metricCurseCheck'),
};

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
  checkServerHealth();
  initSpeechRecognition();
  initVoices();
  initWaveformVisualizer();
  bindEventListeners();
  loadSessionContext();
});

// ==========================================================================
// Event Listeners
// ==========================================================================
function bindEventListeners() {
  // Audio & Speech unlock gesture listener for browser autoplay policies
  const unlockAudio = () => {
    if (window.speechSynthesis) {
      try {
        window.speechSynthesis.resume();
        window.speechSynthesis.getVoices();
      } catch (e) {}
    }
  };
  document.addEventListener('click', unlockAudio, { once: true });
  document.addEventListener('keydown', unlockAudio, { once: true });

  // Welcome message speak button
  const welcomeBtn = document.getElementById('welcomeSpeakBtn');
  if (welcomeBtn) {
    welcomeBtn.addEventListener('click', () => {
      unlockAudio();
      const card = welcomeBtn.closest('.message-card');
      const text = card?.querySelector('.message-text')?.textContent?.trim();
      if (text) {
        speakResponse(text);
      }
    });
  }

  // Chat form submit
  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    unlockAudio();
    const text = elements.textInput.value.trim();
    if (text) {
      handleUserSendMessage(text);
      elements.textInput.value = '';
    }
  });

  // Mic button click
  elements.micBtn.addEventListener('click', () => {
    unlockAudio();
    toggleMicrophone();
  });

  // Interrupt / Halt Audio (GhostBarge)
  elements.interruptAudioBtn.addEventListener('click', () => {
    haltAudioPlayback('User interrupted playback (GhostBarge)');
  });

  // Reset Session
  elements.resetSessionBtn.addEventListener('click', () => {
    resetSession();
  });

  // Quick Chips
  elements.quickChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      unlockAudio();
      const prompt = chip.getAttribute('data-text');
      if (prompt) {
        elements.textInput.value = '';
        handleUserSendMessage(prompt);
      }
    });
  });
}

// ==========================================================================
// API Interaction
// ==========================================================================
async function checkServerHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      elements.serverStatusBadge.classList.add('online');
      elements.statusText.textContent = 'ONLINE (5001)';
    } else {
      elements.statusText.textContent = 'ERROR';
    }
  } catch (err) {
    elements.statusText.textContent = 'OFFLINE';
  }
}

async function handleUserSendMessage(userText) {
  // Halt any active speech playback before sending a new turn
  haltAudioPlayback('New turn initiated');

  // Render User Message
  renderMessage({
    role: 'user',
    text: userText,
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
  });

  // Set typing indicator / state
  setVoiceState('Processing request via Fast Lane / Smart Path...', 'processing');

  try {
    const res = await fetch('/api/conversation/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: state.sessionId,
        text: userText,
      }),
    });

    const data = await res.json();

    if (data.success) {
      // Render Assistant Response
      renderMessage({
        role: 'assistant',
        text: data.response,
        intent: data.intent,
        latencyMs: data.latencyMs,
        processingPath: data.processingPath,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      });

      // Update SwitchSync slots
      updateSlotsTelemetry(data.entities, data.missingFields);

      // Update Latency & Route Telemetry
      updateMetricsTelemetry(data);

      // Trigger Voice Output if enabled
      if (elements.autoTtsToggle.checked && data.response) {
        speakResponse(data.response, data.turnId || `turn_${Date.now()}`);
      } else {
        setVoiceState('Ready to listen • Tap mic or speak', 'idle');
      }
    } else {
      renderMessage({
        role: 'assistant',
        text: data.response || 'Error processing request.',
        processingPath: 'fast',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      });
      setVoiceState('Ready', 'idle');
    }
  } catch (err) {
    console.error('[Client] Message error:', err);
    renderMessage({
      role: 'assistant',
      text: 'Network error connecting to backend service.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    });
    setVoiceState('Error', 'idle');
  }
}

async function loadSessionContext() {
  try {
    const res = await fetch(`/api/conversation/${state.sessionId}`);
    if (res.ok) {
      const data = await res.json();
      if (data.session && data.session.entities) {
        updateSlotsTelemetry(data.session.entities, []);
      }
    }
  } catch (err) {
    console.warn('[Client] Could not fetch initial session context');
  }
}

async function resetSession() {
  haltAudioPlayback('Session reset');
  try {
    await fetch(`/api/conversation/${state.sessionId}`, { method: 'DELETE' });
  } catch (e) {}

  // Generate new session ID
  state.sessionId = 'session_' + Math.random().toString(36).substring(2, 9);
  elements.sessionIdDisplay.textContent = state.sessionId;

  // Clear slots
  updateSlotsTelemetry({}, []);

  // Clear dialogue except welcome
  elements.dialogueContainer.innerHTML = `
    <div class="message-card assistant">
      <div class="message-meta">
        <span class="sender-tag">👻 PHANTOM OPERATOR</span>
        <span class="time-tag">Reset</span>
        <span class="path-badge badge-fast">⚡ FAST LANE</span>
      </div>
      <div class="message-text">
        Session context has been reset! Main PhantomFlow Indian Railway assistant hoon. Kahan ki yatra plan karni hai?
      </div>
    </div>
  `;

  setVoiceState('Session reset • Ready to listen', 'idle');
}

// ==========================================================================
// Voice Services: Speech-to-Text (STT) & Text-to-Speech (TTS)
// ==========================================================================
function initSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    console.warn('[Speech] Web Speech API not supported in this browser.');
    elements.voiceStateLabel.textContent = 'Voice input unavailable (Browser lacks Web Speech API). Please use text input.';
    elements.micBtn.style.opacity = '0.4';
    elements.micBtn.disabled = true;
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = elements.voiceLanguageSelect.value || 'en-IN';

  elements.voiceLanguageSelect.addEventListener('change', () => {
    if (state.recognition) {
      state.recognition.lang = elements.voiceLanguageSelect.value;
    }
  });

  state.recognition.onstart = () => {
    state.isListening = true;
    elements.micBtn.classList.add('active');
    setVoiceState('Listening for speech... (speak now)', 'listening');
  };

  state.recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    if (interimTranscript) {
      setVoiceState(`Hearing: "${interimTranscript}"`, 'listening');
    }

    if (finalTranscript) {
      setVoiceState(`Understood: "${finalTranscript}"`, 'idle');
      elements.textInput.value = '';
      handleUserSendMessage(finalTranscript);
    }
  };

  state.recognition.onerror = (event) => {
    console.warn('[Speech] Recognition error:', event.error);
    state.isListening = false;
    elements.micBtn.classList.remove('active');
    setVoiceState(`Speech error: ${event.error}`, 'idle');
  };

  state.recognition.onend = () => {
    state.isListening = false;
    elements.micBtn.classList.remove('active');
  };
}

function toggleMicrophone() {
  if (!state.recognition) {
    alert('Web Speech API is not supported in this browser. Please use the text input below.');
    return;
  }

  if (state.isListening) {
    state.recognition.stop();
  } else {
    // If assistant is currently speaking, stop it immediately (GhostBarge)
    haltAudioPlayback('User started speaking (GhostBarge)');
    try {
      state.recognition.start();
    } catch (e) {
      console.warn('[Speech] Already active');
    }
  }
}

function initVoices() {
  if (!state.synth) return;

  const populateVoices = () => {
    const voices = state.synth.getVoices();
    // Prefer Indian English (en-IN) or Hindi (hi-IN)
    state.selectedVoice = 
      voices.find((v) => v.lang === 'en-IN' || v.lang === 'hi-IN') ||
      voices.find((v) => v.lang.startsWith('en')) ||
      voices[0];
  };

  populateVoices();
  if (state.synth.onvoiceschanged !== undefined) {
    state.synth.onvoiceschanged = populateVoices;
  }
}

async function speakResponse(text, turnId = null) {
  haltAudioPlayback('New playback initiated');

  const effectiveTurnId = turnId || `turn_${Date.now()}`;
  state.currentTurnId = effectiveTurnId;

  const ttsProvider = elements.ttsProviderSelect?.value || 'sarvam';

  // 1. Sarvam AI / ElevenLabs Cloud Audio Pipeline
  if (ttsProvider === 'sarvam' || ttsProvider === 'elevenlabs') {
    try {
      const providerLabel = ttsProvider === 'sarvam' ? 'Sarvam AI (Bulbul v4 Flash)' : 'ElevenLabs Flash v2.5';
      setVoiceState(`Synthesizing voice response with ${providerLabel}...`, 'speaking');
      const startTts = performance.now();

      const res = await fetch('/api/speech/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId: state.sessionId,
          turnId: effectiveTurnId,
          text,
          language: elements.voiceLanguageSelect?.value || 'hi-IN',
        }),
      });

      const data = await res.json();

      // Check if user cancelled while audio was being generated
      if (state.currentTurnId !== effectiveTurnId) {
        console.log(`[TTS] Turn ${effectiveTurnId} was cancelled before playback.`);
        return;
      }

      if (data.success && data.audioBase64 && data.audioBase64.length > 200 && !data.isFallback && !data.isMock) {
        const mime = data.audioMimeType || (data.provider === 'elevenlabs' ? 'audio/mpeg' : 'audio/wav');
        try {
          const byteCharacters = atob(data.audioBase64);
          const byteNumbers = new Uint8Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const blob = new Blob([byteNumbers], { type: mime });
          const audioUrl = URL.createObjectURL(blob);
          const player = new Audio(audioUrl);
          state.audioPlayer = player;
          state.isSpeaking = true;

          const ttfb = data.metrics?.timeToFirstAudioChunkMs || Math.round(performance.now() - startTts);
          const engineLabel = data.provider === 'sarvam' ? 'Sarvam AI (Bulbul v4 Flash)' : 'ElevenLabs (Flash v2.5)';
          setVoiceState(`Speaking (${engineLabel} • ${ttfb}ms) • Tap "Halt Audio" to interrupt`, 'speaking');

          player.onended = () => {
            URL.revokeObjectURL(audioUrl);
            state.isSpeaking = false;
            state.audioPlayer = null;
            setVoiceState('Ready to listen • Tap mic or speak', 'idle');
          };

          player.onerror = (e) => {
            URL.revokeObjectURL(audioUrl);
            console.warn('[Audio Player] Audio decode failed, falling back to Browser Voice:', e);
            state.isSpeaking = false;
            state.audioPlayer = null;
            speakBrowserWebSpeech(data.fallbackText || text);
          };

          player.play().catch((e) => {
            console.warn('[Audio Player] Playback blocked by browser autoplay, falling back to Browser Voice:', e);
            speakBrowserWebSpeech(data.fallbackText || text);
          });
          return;
        } catch (decodeErr) {
          console.warn('[Audio Player] Blob creation error, falling back:', decodeErr);
          speakBrowserWebSpeech(data.fallbackText || text);
          return;
        }
      } else {
        // Fallback to high-quality browser voice
        speakBrowserWebSpeech(data.fallbackText || text);
        return;
      }
    } catch (err) {
      console.warn('[Audio Client] Error:', err);
      speakBrowserWebSpeech(text);
      return;
    }
  }

  // 2. Direct Browser Web Speech Synthesis
  speakBrowserWebSpeech(text);
}

function speakBrowserWebSpeech(text) {
  const synth = window.speechSynthesis;
  if (!synth) {
    console.warn('[TTS] window.speechSynthesis is not supported in this browser.');
    setVoiceState('Browser speech not supported • Use text input', 'idle');
    return;
  }

  // Clear any existing stalled speech safely
  try {
    if (synth.speaking || synth.pending) {
      synth.cancel();
    }
    synth.resume();
  } catch (e) {}

  // Clean technical labels or markdown symbols from speech
  const cleanSpoken = text
    .replace(/[#*`_~]/g, '')
    .replace(/₹/g, 'Rupees ')
    .trim();

  if (!cleanSpoken) return;

  const utterance = new SpeechSynthesisUtterance(cleanSpoken);

  // Pick best available Indian English or Hindi voice
  const voices = synth.getVoices();
  const langChoice = (elements.voiceLanguageSelect?.value || 'en-IN').toLowerCase();
  
  let matchedVoice = null;
  if (voices && voices.length > 0) {
    matchedVoice = 
      voices.find(v => {
        const n = v.name.toLowerCase();
        return n.includes('aman') || n.includes('rishi');
      }) ||
      voices.find(v => v.lang.replace('_', '-').toLowerCase() === langChoice) ||
      voices.find(v => {
        const n = v.name.toLowerCase();
        return n.includes('lekha') || n.includes('tara');
      }) ||
      voices.find(v => v.lang.toLowerCase().includes('in')) ||
      voices.find(v => v.lang.toLowerCase().startsWith('en')) ||
      voices[0];
  }

  if (matchedVoice) {
    utterance.voice = matchedVoice;
    utterance.lang = matchedVoice.lang;
  } else {
    // Avoid setting unsupported 'en-IN' if browser has no en-IN voice, which causes Chrome to drop it
    utterance.lang = langChoice.includes('hi') ? 'hi-IN' : 'en-US';
  }

  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  // Periodic heartbeat to prevent Chrome 15-second speech stall bug
  let resumeTimer = null;

  utterance.onstart = () => {
    state.isSpeaking = true;
    setVoiceState('Speaking response aloud... (Tap "Halt Audio" to interrupt)', 'speaking');
    resumeTimer = setInterval(() => {
      if (!state.isSpeaking || !synth.speaking) {
        clearInterval(resumeTimer);
      } else {
        synth.resume();
      }
    }, 2000);
  };

  utterance.onend = () => {
    if (resumeTimer) clearInterval(resumeTimer);
    state.isSpeaking = false;
    window.__currentUtterance = null;
    setVoiceState('Ready to listen • Tap mic or speak', 'idle');
  };

  utterance.onerror = (e) => {
    if (resumeTimer) clearInterval(resumeTimer);
    console.warn('[TTS] Synthesis error event:', e.error);
    state.isSpeaking = false;
    window.__currentUtterance = null;
    setVoiceState('Ready to listen • Tap mic or speak', 'idle');
  };

  // Prevent V8 garbage collection bug
  window.__currentUtterance = utterance;
  state.activeUtterance = utterance;

  // Slight timeout before speak to prevent Chrome cancel race condition
  setTimeout(() => {
    try {
      synth.speak(utterance);
      synth.resume();
    } catch (e) {
      console.warn('[TTS] Synth speak error:', e);
    }
  }, 30);
}

function haltAudioPlayback(reason = '') {
  // 1. Cancel Browser Web Speech
  if (state.synth && state.synth.speaking) {
    state.synth.cancel();
  }

  // 2. Halt ElevenLabs Audio Player
  if (state.audioPlayer) {
    state.audioPlayer.pause();
    state.audioPlayer.src = '';
    state.audioPlayer = null;
  }

  // 3. Notify Backend of GhostBarge Interruption
  if (state.currentTurnId) {
    const turnToCancel = state.currentTurnId;
    state.currentTurnId = null;
    fetch('/api/speech/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: state.sessionId,
        turnId: turnToCancel,
      }),
    }).catch(() => {});
  }

  state.isSpeaking = false;
  console.log(`[GhostBarge] Assistant audio halted: ${reason}`);
  setVoiceState('Audio halted (GhostBarge) • Ready to listen', 'idle');
}

function setVoiceState(message, mode = 'idle') {
  elements.voiceStateLabel.textContent = message;
  elements.voiceStateLabel.className = `voice-state-label ${mode}`;
}

// ==========================================================================
// Dynamic Audio Waveform Visualizer
// ==========================================================================
function initWaveformVisualizer() {
  const canvas = elements.waveformCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let phase = 0;

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.width;
    const height = canvas.height;
    const midY = height / 2;

    let amplitude = 4;
    let frequency = 0.03;
    let strokeColor = 'rgba(255, 255, 255, 0.12)';

    if (state.isListening) {
      amplitude = 18;
      frequency = 0.06;
      strokeColor = '#00f5ff'; // Cyan for listening
    } else if (state.isSpeaking) {
      amplitude = 22;
      frequency = 0.05;
      strokeColor = '#a855f7'; // Purple for speaking
    }

    // Draw primary wave
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = strokeColor;

    for (let x = 0; x < width; x++) {
      const y = midY + Math.sin(x * frequency + phase) * amplitude * Math.sin((x / width) * Math.PI);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw secondary lighter harmonic
    if (state.isListening || state.isSpeaking) {
      ctx.beginPath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = state.isListening ? 'rgba(0, 245, 255, 0.4)' : 'rgba(236, 72, 153, 0.4)';
      for (let x = 0; x < width; x++) {
        const y = midY + Math.sin(x * (frequency * 1.5) - phase) * (amplitude * 0.6) * Math.sin((x / width) * Math.PI);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    phase += 0.08;
    requestAnimationFrame(draw);
  }

  draw();
}

// ==========================================================================
// Telemetry & UI Renderers
// ==========================================================================
function renderMessage({ role, text, intent, latencyMs, processingPath, timestamp }) {
  const card = document.createElement('div');
  card.className = `message-card ${role}`;

  const isAssistant = role === 'assistant';
  const senderTitle = isAssistant ? '👻 PHANTOM OPERATOR' : '👤 CALLER';

  let pathBadge = '';
  if (isAssistant && processingPath) {
    const isFast = processingPath === 'fast';
    const badgeClass = isFast ? 'badge-fast' : 'badge-smart';
    const badgeText = isFast ? `⚡ FAST LANE (${latencyMs}ms)` : `🧠 SMART PATH (${latencyMs}ms)`;
    pathBadge = `<span class="path-badge ${badgeClass}">${badgeText}</span>`;
  }

  let speakBtn = '';
  if (isAssistant) {
    speakBtn = `<button class="speak-again-btn" title="Replay Audio">🔊 Speak</button>`;
  }

  card.innerHTML = `
    <div class="message-meta">
      <span class="sender-tag">${senderTitle}</span>
      <span class="time-tag">${timestamp || 'Now'}</span>
      ${pathBadge}
    </div>
    <div class="message-text">${escapeHtml(text)}</div>
    ${speakBtn}
  `;

  if (isAssistant) {
    const btn = card.querySelector('.speak-again-btn');
    if (btn) {
      btn.addEventListener('click', () => speakResponse(text));
    }
  }

  elements.dialogueContainer.appendChild(card);
  elements.dialogueContainer.scrollTop = elements.dialogueContainer.scrollHeight;
}

function updateSlotsTelemetry(entities = {}, missingFields = []) {
  const updateSlot = (element, val) => {
    const box = element.closest('.slot-box');
    const prevVal = element.textContent;

    if (val) {
      element.textContent = val;
      box.classList.add('has-val');
      if (prevVal !== val && prevVal !== '—') {
        box.classList.add('updated');
        setTimeout(() => box.classList.remove('updated'), 1000);
      }
    } else {
      element.textContent = '—';
      box.classList.remove('has-val');
    }
  };

  updateSlot(elements.valOrigin, entities.origin);
  updateSlot(elements.valDestination, entities.destination);
  updateSlot(elements.valDate, entities.date);
  updateSlot(elements.valTime, entities.time);
  updateSlot(elements.valTrainNumber, entities.trainNumber);
  updateSlot(elements.valPnr, entities.pnr);

  // Missing Fields Alert
  if (missingFields && missingFields.length > 0) {
    elements.missingFieldsContainer.style.display = 'flex';
    elements.missingFieldsText.textContent = `Pending information: ${missingFields.join(', ')}`;
  } else {
    elements.missingFieldsContainer.style.display = 'none';
  }
}

function updateMetricsTelemetry(data) {
  elements.metricLatency.textContent = data.latencyMs;

  const isFast = data.processingPath === 'fast';
  elements.metricRoute.className = `route-badge ${isFast ? 'badge-fast' : 'badge-smart'}`;
  elements.metricRoute.textContent = isFast ? '⚡ FAST LANE' : '🧠 SMART PATH';

  elements.metricIntent.textContent = data.intent || 'railway_search';
  elements.metricLang.textContent = data.language || 'hinglish';

  const meetsTarget = data.latencyMs < 600;
  elements.metricCurseCheck.className = `meta-val ${meetsTarget ? 'text-success' : 'text-danger'}`;
  elements.metricCurseCheck.textContent = meetsTarget ? '✅ PASS (< 600ms)' : '⚠️ EXCEEDS 600ms';
}

function escapeHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
