/**
 * ElevenLabs Streaming Text-to-Speech (TTS) Client
 * 
 * Implements low-latency streaming audio synthesis via ElevenLabs REST API.
 * Supports:
 * - Hindi, English, and Hinglish pronunciation via eleven_multilingual_v2
 * - Streaming latency optimization (optimize_streaming_latency=3)
 * - Microsecond latency profiling (time-to-first-audio, API latency, total generation time)
 * - AbortController signal handling for immediate GhostBarge interruption
 * - Deterministic fallback mode when API key is missing
 */

const { performance } = require('perf_hooks');
const config = require('../config/config');

class ElevenLabsTTS {
  constructor() {
    this.apiKey = config.tts.apiKey;
    this.voiceId = config.tts.voiceId;
    this.modelId = config.tts.modelId;
    this.optimizeLatency = config.tts.optimizeStreamingLatency;
    this.stability = config.tts.stability;
    this.similarityBoost = config.tts.similarityBoost;
    this.timeoutMs = config.tts.timeoutMs;
  }

  /**
   * Stream synthesized audio chunks from ElevenLabs.
   * @param {Object} params
   * @param {string} params.text - The text to synthesize into speech
   * @param {string} params.turnId - Unique identifier for the speech turn
   * @param {string} [params.voiceId] - Optional voice override
   * @param {string} [params.modelId] - Optional model override
   * @param {AbortSignal} [params.signal] - Abort signal for GhostBarge interruption
   * @returns {AsyncGenerator<{
   *   turnId: string,
   *   chunkIndex: number,
   *   data: Buffer,
   *   isFinal: boolean,
   *   metrics: {
   *     ttsApiLatencyMs: number,
   *     timeToFirstAudioChunkMs: number,
   *     totalAudioGenerationMs?: number,
   *     totalBytes: number
   *   }
   * }>}
   */
  async *synthesizeStream({ text, turnId, language, voiceId, modelId, signal }) {
    const startTime = performance.now();
    const activeVoiceId = voiceId || this.voiceId;
    const activeModelId = modelId || this.modelId;

    // Fallback to Mock Streaming if no API key is configured
    if (!this.apiKey || config.tts.provider === 'mock') {
      yield* this._mockSynthesizeStream({ text, turnId, startTime, signal, language });
      return;
    }

    const endpoint = `https://api.elevenlabs.io/v1/text-to-speech/${activeVoiceId}/stream?optimize_streaming_latency=${this.optimizeLatency}`;

    let firstChunkTime = null;
    let apiLatency = null;
    let totalBytes = 0;
    let chunkIndex = 0;

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'audio/mpeg',
          'xi-api-key': this.apiKey,
        },
        body: JSON.stringify({
          text,
          model_id: activeModelId,
          voice_settings: {
            stability: this.stability,
            similarity_boost: this.similarityBoost,
            style: 0.0,
            use_speaker_boost: true,
          },
        }),
        signal,
      });

      apiLatency = Math.round((performance.now() - startTime) * 100) / 100;

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown API error');
        console.error(`[ElevenLabsTTS] API responded with HTTP ${response.status}: ${errorText}`);
        yield* this._mockSynthesizeStream({ text, turnId, startTime, signal, errorFallback: true, language });
        return;
      }

      if (!response.body) {
        throw new Error('ElevenLabs response did not contain a readable audio stream.');
      }

      const reader = response.body.getReader();

      while (true) {
        if (signal && signal.aborted) {
          console.log(`[ElevenLabsTTS] Generation aborted for turn ${turnId} via signal.`);
          break;
        }

        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        if (value && value.length > 0) {
          if (firstChunkTime === null) {
            firstChunkTime = Math.round((performance.now() - startTime) * 100) / 100;
          }

          const chunkBuffer = Buffer.from(value);
          totalBytes += chunkBuffer.length;

          yield {
            turnId,
            chunkIndex: chunkIndex++,
            data: chunkBuffer,
            isFinal: false,
            metrics: {
              ttsApiLatencyMs: apiLatency,
              timeToFirstAudioChunkMs: firstChunkTime,
              totalBytes,
            },
          };
        }
      }

      const totalGenerationTime = Math.round((performance.now() - startTime) * 100) / 100;

      // Final empty chunk to signal completion
      yield {
        turnId,
        chunkIndex: chunkIndex++,
        data: Buffer.alloc(0),
        isFinal: true,
        metrics: {
          ttsApiLatencyMs: apiLatency,
          timeToFirstAudioChunkMs: firstChunkTime || totalGenerationTime,
          totalAudioGenerationMs: totalGenerationTime,
          totalBytes,
        },
      };
    } catch (err) {
      if (err.name === 'AbortError' || (signal && signal.aborted)) {
        console.log(`[ElevenLabsTTS] Audio generation halted mid-stream for turn ${turnId}`);
        return;
      }

      console.error(`[ElevenLabsTTS] Network or API error during synthesis:`, err.message);
      yield* this._mockSynthesizeStream({ text, turnId, startTime, signal, errorFallback: true, language });
    }
  }

  /**
   * Generates a complete, decodable WAV buffer.
   * If on macOS and no ElevenLabs API key is configured, uses high-quality native voice (Aman/Lekha/system)
   * converted to 16-bit PCM WAV. Otherwise falls back to synthetic gentle tone WAV.
   */
  _generateDecodableWav(text, language = 'en-IN') {
    const { execSync } = require('child_process');
    const fs = require('fs');
    const path = require('path');
    const os = require('os');

    if (process.platform === 'darwin') {
      try {
        const randId = `${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
        const tmpAiff = path.join(os.tmpdir(), `phantom_speech_${randId}.aiff`);
        const tmpWav = path.join(os.tmpdir(), `phantom_speech_${randId}.wav`);

        // Clean text for command line
        const cleanText = (text || 'PhantomFlow railway query response')
          .replace(/["\\`$]/g, '')
          .replace(/[#*_~]/g, '')
          .replace(/₹/g, 'Rupees ')
          .trim();

        // Determine best native macOS voice (Aman for Indian English/Hinglish male, Rishi as backup male)
        let voiceFlag = '';
        try {
          const available = execSync('say -v "?"', { encoding: 'utf8', timeout: 1000 });
          if (available.includes('Aman')) {
            voiceFlag = '-v Aman';
          } else if (available.includes('Rishi')) {
            voiceFlag = '-v Rishi';
          } else if (available.includes('Lekha')) {
            voiceFlag = '-v Lekha';
          }
        } catch (e) {
          voiceFlag = '';
        }

        execSync(`say ${voiceFlag} "${cleanText}" -o "${tmpAiff}"`, { timeout: 3000 });
        execSync(`afconvert -f WAVE -d LEI16 "${tmpAiff}" "${tmpWav}"`, { timeout: 2000 });

        const wavData = fs.readFileSync(tmpWav);

        try { fs.unlinkSync(tmpAiff); } catch (e) {}
        try { fs.unlinkSync(tmpWav); } catch (e) {}

        if (wavData && wavData.length > 44) {
          return wavData;
        }
      } catch (err) {
        // Fall through to synthetic PCM WAV
      }
    }

    // Generic 16kHz mono PCM WAV fallback
    const sampleRate = 16000;
    const durationSeconds = 0.8;
    const sampleCount = Math.floor(sampleRate * durationSeconds);
    const dataSize = sampleCount * 2;
    const wavBuffer = Buffer.alloc(44 + dataSize);

    wavBuffer.write('RIFF', 0);
    wavBuffer.writeUInt32LE(36 + dataSize, 4);
    wavBuffer.write('WAVE', 8);
    wavBuffer.write('fmt ', 12);
    wavBuffer.writeUInt32LE(16, 16);
    wavBuffer.writeUInt16LE(1, 20); // PCM format
    wavBuffer.writeUInt16LE(1, 22); // Mono
    wavBuffer.writeUInt32LE(sampleRate, 24); // 16kHz
    wavBuffer.writeUInt32LE(sampleRate * 2, 28); // ByteRate
    wavBuffer.writeUInt16LE(2, 32);  // BlockAlign
    wavBuffer.writeUInt16LE(16, 34); // 16-bit
    wavBuffer.write('data', 36);
    wavBuffer.writeUInt32LE(dataSize, 40);

    for (let s = 0; s < sampleCount; s++) {
      const val = Math.sin((s / sampleRate) * 440 * 2 * Math.PI) * 0.15 * 32767;
      wavBuffer.writeInt16LE(Math.round(val), 44 + s * 2);
    }

    return wavBuffer;
  }

  /**
   * Deterministic mock audio generator for testing without an API key or on API failure.
   */
  async *this_mockSynthesizeStream({ text, turnId, startTime, signal, errorFallback = false, language = 'en-IN' }) {
    const apiLatency = Math.round((performance.now() - startTime) * 100) / 100;
    const fullWav = this._generateDecodableWav(text, language);

    // Stream the complete WAV in realistic chunks (16KB chunks)
    const chunkSize = 16384;
    const totalChunks = Math.max(1, Math.ceil(fullWav.length / chunkSize));
    let firstChunkTime = null;
    let offset = 0;

    for (let i = 0; i < totalChunks; i++) {
      if (signal && signal.aborted) {
        return;
      }

      await new Promise((r) => setTimeout(r, 15));

      if (firstChunkTime === null) {
        firstChunkTime = Math.round((performance.now() - startTime) * 100) / 100;
      }

      const end = Math.min(offset + chunkSize, fullWav.length);
      const chunkData = fullWav.slice(offset, end);
      offset = end;

      yield {
        turnId,
        chunkIndex: i,
        data: chunkData,
        audioMimeType: 'audio/wav',
        isFinal: false,
        isMock: true,
        errorFallback,
        metrics: {
          ttsApiLatencyMs: apiLatency,
          timeToFirstAudioChunkMs: firstChunkTime,
          totalBytes: offset,
        },
      };
    }

    const totalGenerationTime = Math.round((performance.now() - startTime) * 100) / 100;

    yield {
      turnId,
      chunkIndex: totalChunks,
      data: Buffer.alloc(0),
      audioMimeType: 'audio/wav',
      isFinal: true,
      isMock: true,
      errorFallback,
      metrics: {
        ttsApiLatencyMs: apiLatency,
        timeToFirstAudioChunkMs: firstChunkTime || totalGenerationTime,
        totalAudioGenerationMs: totalGenerationTime,
        totalBytes: fullWav.length,
      },
    };
  }

  // Bind the internal mock helper
  _mockSynthesizeStream(params) {
    return this.this_mockSynthesizeStream(params);
  }
}

module.exports = new ElevenLabsTTS();

