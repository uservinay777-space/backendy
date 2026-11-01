/**
 * ElevenLabs Text-to-Speech (TTS) Adapter
 * Optional / Alternative Voice Provider for PHANTOMFLOW.
 * 
 * Uses ElevenLabs streaming endpoint (eleven_flash_v2_5 / eleven_multilingual_v2).
 */

const { performance } = require('perf_hooks');
const config = require('../../config/config');
const { normalizeForSpeech } = require('./textNormalizer');

class ElevenLabsTTS {
  constructor() {
    this.voiceId = config.elevenlabs.voiceId || '21m00Tcm4TlvDq8ikWAM';
    this.modelId = config.elevenlabs.modelId || 'eleven_flash_v2_5';
    this.optimizeLatency = config.elevenlabs.optimizeLatency || 3;
    this.stability = config.elevenlabs.stability || 0.40;
    this.similarityBoost = config.elevenlabs.similarityBoost || 0.75;
    this.style = config.elevenlabs.style || 0.10;
    this.speed = config.elevenlabs.speed || 1.0;
  }

  /**
   * Stream synthesized audio chunks from ElevenLabs.
   * @param {Object} params
   * @param {string} params.text
   * @param {string} params.turnId
   * @param {string} [params.voiceId]
   * @param {string} [params.modelId]
   * @param {AbortSignal} [params.signal]
   * @returns {AsyncGenerator<Object>}
   */
  async *synthesizeStream({ text, turnId, voiceId, modelId, signal }) {
    const startTime = performance.now();

    if (!config.elevenlabs.isConfigured) {
      throw new Error(
        'ElevenLabs is not configured. Please set ELEVENLABS_API_KEY in .env to enable ElevenLabs speech synthesis.'
      );
    }

    const effectiveVoiceId = voiceId || this.voiceId;
    const effectiveModelId = modelId || this.modelId;
    const normalizedText = normalizeForSpeech(text);
    const endpoint = `https://api.elevenlabs.io/v1/text-to-speech/${effectiveVoiceId}/stream?optimize_streaming_latency=${this.optimizeLatency}&output_format=mp3_44100_128`;

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
          'xi-api-key': config.elevenlabs.apiKey,
        },
        body: JSON.stringify({
          text: normalizedText,
          model_id: effectiveModelId,
          voice_settings: {
            stability: this.stability,
            similarity_boost: this.similarityBoost,
            style: this.style,
            use_speaker_boost: true,
            speed: this.speed,
          },
        }),
        signal,
      });

      apiLatency = Math.round((performance.now() - startTime) * 100) / 100;

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown API error');
        throw new Error(`ElevenLabs API error (HTTP ${response.status}): ${errorText}`);
      }

      if (!response.body) {
        throw new Error('ElevenLabs response did not contain an audio stream.');
      }

      const reader = response.body.getReader();

      while (true) {
        if (signal && signal.aborted) {
          console.log(`[ElevenLabsTTS] Generation aborted via GhostBarge for turn ${turnId}`);
          break;
        }

        const { done, value } = await reader.read();
        if (done) break;

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
            audioMimeType: 'audio/mpeg',
            isFinal: false,
            metrics: {
              ttsApiLatencyMs: apiLatency,
              timeToFirstAudioChunkMs: firstChunkTime,
              totalBytes,
            },
          };
        }
      }

      const totalTime = Math.round((performance.now() - startTime) * 100) / 100;

      yield {
        turnId,
        chunkIndex: chunkIndex++,
        data: Buffer.alloc(0),
        audioMimeType: 'audio/mpeg',
        isFinal: true,
        metrics: {
          ttsApiLatencyMs: apiLatency,
          timeToFirstAudioChunkMs: firstChunkTime || totalTime,
          totalAudioGenerationMs: totalTime,
          totalBytes,
        },
      };
    } catch (err) {
      if (err.name === 'AbortError' || (signal && signal.aborted)) {
        return;
      }
      console.error(`[ElevenLabsTTS] Synthesis error for turn ${turnId}:`, err.message);
      throw err;
    }
  }

  /**
   * Synthesize complete audio buffer (collects streaming chunks).
   * @param {Object} params
   * @returns {Promise<Object>}
   */
  async synthesize({ text, turnId, voiceId, modelId, signal }) {
    const chunks = [];
    let finalMetrics = {};
    let isInterrupted = false;

    try {
      for await (const chunk of this.synthesizeStream({ text, turnId, voiceId, modelId, signal })) {
        if (chunk.data && chunk.data.length > 0) {
          chunks.push(chunk.data);
        }
        if (chunk.metrics) {
          finalMetrics = chunk.metrics;
        }
      }
    } catch (err) {
      if (err.name === 'AbortError' || (signal && signal.aborted)) {
        isInterrupted = true;
      } else {
        throw err;
      }
    }

    const audioBuffer = Buffer.concat(chunks);
    return {
      success: !isInterrupted && audioBuffer.length > 0,
      turnId,
      audioBuffer,
      audioMimeType: 'audio/mpeg',
      interrupted: isInterrupted,
      metrics: {
        ...finalMetrics,
        totalBytes: audioBuffer.length,
      },
    };
  }
}

module.exports = new ElevenLabsTTS();
