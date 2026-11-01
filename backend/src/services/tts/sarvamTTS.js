/**
 * Sarvam AI Text-to-Speech (TTS) Adapter
 * Primary Voice Provider for PHANTOMFLOW using Bulbul v4 Flash (bulbul:v4-flash).
 * 
 * Delivers natural Indian voice synthesis with low-latency (< 250ms TTFB)
 * and native Hindi/Hinglish/English code-mixing.
 */

const { performance } = require('perf_hooks');
const config = require('../../config/config');
const { normalizeForSpeech } = require('./textNormalizer');

class SarvamTTS {
  constructor() {
    this.endpoint = 'https://api.sarvam.ai/text-to-speech';
    this.model = config.sarvam.model;
    this.speaker = config.sarvam.speaker;
    this.languageCode = config.sarvam.languageCode;
    this.pace = config.sarvam.pace;
    this.timeoutMs = config.sarvam.timeoutMs;
  }

  /**
   * Synthesize text into complete audio buffer using Sarvam AI Bulbul v4 Flash.
   * @param {Object} params
   * @param {string} params.text - Response text to synthesize
   * @param {string} params.turnId - Unique turn identifier
   * @param {AbortSignal} [params.signal] - For GhostBarge interruption
   * @returns {Promise<{
   *   success: boolean,
   *   audioBuffer: Buffer,
   *   audioMimeType: string,
   *   metrics: {
   *     ttsApiLatencyMs: number,
   *     timeToFirstAudioChunkMs: number,
   *     totalAudioGenerationMs: number,
   *     totalBytes: number
   *   },
   *   error?: string
   * }>}
   */
  async synthesize({ text, turnId, signal }) {
    const startTime = performance.now();

    // 1. Check Configuration
    if (!config.sarvam.isConfigured) {
      throw new Error(
        'Sarvam AI is not configured. Please set SARVAM_API_KEY in .env to enable Bulbul v4 Flash text-to-speech.'
      );
    }

    try {
      const normalizedText = normalizeForSpeech(text);
      const targetModel = config.sarvam.model || this.model || 'bulbul:v4-flash';
      let targetSpeaker = config.sarvam.speaker || this.speaker || 'shubh_hi_customer';
      if (targetSpeaker === 'shubh' && targetModel.includes('v4')) {
        targetSpeaker = 'shubh_hi_customer';
      }

      const response = await fetch(this.endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-subscription-key': config.sarvam.apiKey,
        },
        body: JSON.stringify({
          model: targetModel,
          inputs: [normalizedText],
          target_language_code: config.sarvam.languageCode || this.languageCode,
          speaker: targetSpeaker,
          enable_preprocessing: true,
          pace: config.sarvam.pace || this.pace,
          speech_sample_rate: 24000,
        }),
        signal,
      });

      const apiLatency = Math.round((performance.now() - startTime) * 100) / 100;

      if (!response.ok) {
        const errorBody = await response.text().catch(() => 'Unknown error');
        throw new Error(`Sarvam API error (HTTP ${response.status}): ${errorBody}`);
      }

      const data = await response.json();

      if (!data.audios || !data.audios.length || !data.audios[0]) {
        throw new Error('Sarvam API returned an empty audio array.');
      }

      const audioBuffer = Buffer.from(data.audios[0], 'base64');
      const totalTime = Math.round((performance.now() - startTime) * 100) / 100;

      return {
        success: true,
        turnId,
        audioBuffer,
        audioMimeType: 'audio/wav',
        metrics: {
          ttsApiLatencyMs: apiLatency,
          timeToFirstAudioChunkMs: apiLatency,
          totalAudioGenerationMs: totalTime,
          totalBytes: audioBuffer.length,
        },
      };
    } catch (err) {
      if (err.name === 'AbortError' || (signal && signal.aborted)) {
        console.log(`[SarvamTTS] Synthesis cancelled via GhostBarge for turn ${turnId}`);
        return {
          success: false,
          turnId,
          interrupted: true,
          audioBuffer: Buffer.alloc(0),
          audioMimeType: 'audio/wav',
          metrics: {
            ttsApiLatencyMs: Math.round((performance.now() - startTime) * 100) / 100,
            timeToFirstAudioChunkMs: 0,
            totalAudioGenerationMs: Math.round((performance.now() - startTime) * 100) / 100,
            totalBytes: 0,
          },
        };
      }

      console.error(`[SarvamTTS] Error synthesizing speech for turn ${turnId}:`, err.message);
      throw err;
    }
  }

  /**
   * Streaming synthesis generator for incremental audio delivery and GhostBarge interruption.
   * Splits synthesized audio into real-time 16KB playback chunks.
   * @param {Object} params
   * @returns {AsyncGenerator<Object>}
   */
  async *synthesizeStream({ text, turnId, signal }) {
    const startTime = performance.now();

    if (!config.sarvam.isConfigured) {
      throw new Error(
        'Sarvam AI is not configured. Please set SARVAM_API_KEY in .env to enable Bulbul v4 Flash text-to-speech.'
      );
    }

    const result = await this.synthesize({ text, turnId, signal });

    if (!result.success || !result.audioBuffer || result.audioBuffer.length === 0) {
      return;
    }

    const fullBuffer = result.audioBuffer;
    const chunkSize = 16384;
    const totalChunks = Math.ceil(fullBuffer.length / chunkSize);
    let offset = 0;
    let chunkIndex = 0;

    for (let i = 0; i < totalChunks; i++) {
      if (signal && signal.aborted) {
        console.log(`[SarvamTTS] Stream interrupted for turn ${turnId}`);
        return;
      }

      const end = Math.min(offset + chunkSize, fullBuffer.length);
      const chunkData = fullBuffer.slice(offset, end);
      offset = end;

      yield {
        turnId,
        chunkIndex: chunkIndex++,
        data: chunkData,
        audioMimeType: 'audio/wav',
        isFinal: false,
        metrics: {
          ttsApiLatencyMs: result.metrics.ttsApiLatencyMs,
          timeToFirstAudioChunkMs: result.metrics.timeToFirstAudioChunkMs,
          totalBytes: offset,
        },
      };

      // Gentle pause to simulate continuous stream timing for audio playback buffering
      await new Promise((r) => setTimeout(r, 10));
    }

    yield {
      turnId,
      chunkIndex: chunkIndex++,
      data: Buffer.alloc(0),
      audioMimeType: 'audio/wav',
      isFinal: true,
      metrics: {
        ...result.metrics,
        totalAudioGenerationMs: Math.round((performance.now() - startTime) * 100) / 100,
      },
    };
  }
}

module.exports = new SarvamTTS();
