/**
 * Speech Adapter
 * 
 * Provides a clean, decoupled interface for Text-to-Speech (TTS) generation and cancellation.
 * Routes through the central TTSService (Sarvam Bulbul v4 Flash as primary, ElevenLabs as optional).
 * Adheres strictly to the team interface:
 * 
 * ```javascript
 * speechAdapter.generateSpeech({ sessionId, turnId, text, language });
 * speechAdapter.cancelSpeech({ sessionId, turnId });
 * ```
 */

const { performance } = require('perf_hooks');
const ttsService = require('../services/tts/ttsService');
const audioBufferManager = require('./audioBufferManager');

class SpeechAdapter {
  constructor() {
    this.telemetryHistory = [];
  }

  /**
   * Generate speech audio for a conversational response.
   * Collects all streaming audio chunks and returns the complete audio buffer + latency metrics.
   * @param {Object} params
   * @param {string} params.sessionId - Active session identifier
   * @param {string} params.turnId - Unique identifier for the speech turn
   * @param {string} params.text - Response text to synthesize
   * @param {string} [params.language] - Language code (hinglish, hindi, english)
   * @returns {Promise<{
   *   success: boolean,
   *   sessionId: string,
   *   turnId: string,
   *   audioBuffer: Buffer,
   *   audioMimeType: string,
   *   interrupted: boolean,
   *   metrics: {
   *     ttsApiLatencyMs: number,
   *     timeToFirstAudioChunkMs: number,
   *     totalAudioGenerationMs: number,
   *     totalBytes: number,
   *     chunkCount: number
   *   },
   *   error?: string
   * }>}
   */
  async generateSpeech({ sessionId, turnId, text, language = 'hinglish', provider = null, voiceId = null, modelId = null }) {
    const effectiveTurnId = turnId || `turn_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const abortController = new AbortController();

    // Register with GhostBarge buffer manager
    audioBufferManager.registerTurn(sessionId, effectiveTurnId, abortController);

    try {
      const result = await ttsService.synthesize({
        text,
        turnId: effectiveTurnId,
        language,
        provider,
        voiceId,
        modelId,
        signal: abortController.signal,
      });

      if (audioBufferManager.isCancelled(effectiveTurnId)) {
        return {
          success: false,
          sessionId,
          turnId: effectiveTurnId,
          audioBuffer: Buffer.alloc(0),
          audioMimeType: 'audio/wav',
          interrupted: true,
          provider: result.provider,
          metrics: result.metrics,
        };
      }

      const resObj = {
        success: Boolean(result.success),
        sessionId,
        turnId: effectiveTurnId,
        audioBuffer: result.audioBuffer || Buffer.alloc(0),
        audioMimeType: result.audioMimeType || 'audio/wav',
        isFallback: Boolean(result.isFallback),
        fallbackText: result.fallbackText || text,
        interrupted: Boolean(result.interrupted),
        provider: result.provider,
        metrics: result.metrics || {},
      };

      this._recordTelemetry(resObj);
      return resObj;
    } catch (err) {
      console.error(`[SpeechAdapter] Error generating speech:`, err.message);
      return {
        success: true,
        sessionId,
        turnId: effectiveTurnId,
        isFallback: true,
        fallbackText: text,
        audioBuffer: Buffer.alloc(0),
        audioMimeType: null,
        interrupted: false,
        provider: 'browser_speech',
        error: err.message,
        metrics: {
          ttsApiLatencyMs: 0,
          timeToFirstAudioChunkMs: 0,
          totalAudioGenerationMs: 0,
          totalBytes: 0,
        },
      };
    } finally {
      audioBufferManager.completeTurn(effectiveTurnId);
    }
  }

  /**
   * Real-time streaming generator for WebSockets, chunked HTTP, or LiveKit output.
   * Emits chunks incrementally as soon as synthesized.
   * @param {Object} params
   * @returns {AsyncGenerator<Object>}
   */
  async *generateSpeechStream({ sessionId, turnId, text, language }) {
    const effectiveTurnId = turnId || `turn_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const abortController = new AbortController();

    audioBufferManager.registerTurn(sessionId, effectiveTurnId, abortController);

    try {
      const stream = ttsService.synthesizeStream({
        text,
        turnId: effectiveTurnId,
        signal: abortController.signal,
      });

      for await (const chunk of stream) {
        if (audioBufferManager.isCancelled(effectiveTurnId)) {
          console.log(`[SpeechAdapter] Suppressing stale audio chunk for cancelled turn ${effectiveTurnId}`);
          break;
        }

        yield {
          sessionId,
          turnId: effectiveTurnId,
          ...chunk,
        };
      }
    } finally {
      audioBufferManager.completeTurn(effectiveTurnId);
    }
  }

  /**
   * GhostBarge Action: Cancel active speech synthesis / audio playback.
   * Halts generation mid-stream and suppresses pending chunks immediately.
   * @param {Object} params
   * @param {string} [params.sessionId]
   * @param {string} [params.turnId]
   * @returns {{ success: boolean, cancelled: boolean, turnId?: string, sessionId?: string, timestamp: number }}
   */
  cancelSpeech({ sessionId, turnId }) {
    const cancelled = audioBufferManager.cancelTurn(sessionId, turnId);
    console.log(`[SpeechAdapter] cancelSpeech called: sessionId=${sessionId}, turnId=${turnId}, cancelled=${cancelled}`);
    return {
      success: true,
      cancelled,
      turnId,
      sessionId,
      timestamp: Date.now(),
    };
  }

  /**
   * Retrieve rolling latency telemetry.
   */
  getTelemetry() {
    return this.telemetryHistory.slice(-20);
  }

  _recordTelemetry(record) {
    this.telemetryHistory.push({
      turnId: record.turnId,
      sessionId: record.sessionId,
      interrupted: record.interrupted,
      bytes: record.metrics.totalBytes,
      timeToFirstAudioChunkMs: record.metrics.timeToFirstAudioChunkMs,
      totalAudioGenerationMs: record.metrics.totalAudioGenerationMs,
      timestamp: Date.now(),
    });

    if (this.telemetryHistory.length > 50) {
      this.telemetryHistory.shift();
    }
  }
}

module.exports = new SpeechAdapter();
