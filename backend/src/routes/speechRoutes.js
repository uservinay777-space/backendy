/**
 * Speech REST API Routes
 * 
 * Endpoints for Text-to-Speech generation, streaming, and GhostBarge interruption:
 * - POST /api/speech/synthesize
 * - POST /api/speech/cancel
 * - GET  /api/speech/telemetry
 */

const express = require('express');
const router = express.Router();
const speechAdapter = require('../speech/speechAdapter');

/**
 * POST /api/speech/synthesize
 * Converts response text into speech audio.
 * Query / header support:
 * - If ?format=audio or Accept: audio/* -> streams raw MP3 audio directly
 * - Otherwise returns JSON with base64 audio and latency metrics
 */
router.post('/synthesize', async (req, res, next) => {
  try {
    const { sessionId, turnId, text, language, provider, voiceId, modelId } = req.body;

    if (!text || typeof text !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Validation Error: text is required.',
      });
    }

    const wantsStream = req.query.format === 'audio' || req.headers.accept?.includes('audio');

    if (wantsStream) {
      res.setHeader('Content-Type', 'audio/mpeg');
      res.setHeader('Transfer-Encoding', 'chunked');

      const stream = speechAdapter.generateSpeechStream({
        sessionId: sessionId || 'default_session',
        turnId,
        text,
        language,
        provider,
        voiceId,
        modelId,
      });

      for await (const chunk of stream) {
        if (chunk.data && chunk.data.length > 0) {
          res.write(chunk.data);
        }
      }

      res.end();
      return;
    }

    // Default JSON response mode
    const result = await speechAdapter.generateSpeech({
      sessionId: sessionId || 'default_session',
      turnId,
      text,
      language,
      provider,
      voiceId,
      modelId,
    });

    return res.status(200).json({
      success: result.success,
      sessionId: result.sessionId,
      turnId: result.turnId,
      audioBase64: result.audioBuffer && result.audioBuffer.length > 0 ? result.audioBuffer.toString('base64') : '',
      audioMimeType: result.audioMimeType,
      isFallback: Boolean(result.isFallback),
      fallbackText: result.fallbackText || text,
      provider: result.provider,
      interrupted: result.interrupted,
      metrics: result.metrics,
    });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/speech/providers
 * Returns public provider configuration status without leaking keys
 */
router.get('/providers', (req, res) => {
  const ttsService = require('../services/tts/ttsService');
  const status = ttsService.checkConfiguration();
  return res.status(200).json({
    success: true,
    activeProvider: status.activeProvider,
    providers: {
      sarvam: {
        isConfigured: status.sarvam.isConfigured,
        model: status.sarvam.model,
        speaker: status.sarvam.speaker,
      },
      elevenlabs: {
        isConfigured: status.elevenlabs.isConfigured,
        modelId: status.elevenlabs.modelId,
        voiceId: status.elevenlabs.voiceId,
      },
    },
  });
});

/**
 * POST /api/speech/cancel
 * GhostBarge Action: Intercept and halt active audio generation
 */
router.post('/cancel', (req, res) => {
  const { sessionId, turnId } = req.body;
  const result = speechAdapter.cancelSpeech({ sessionId, turnId });
  return res.status(200).json(result);
});

/**
 * GET /api/speech/telemetry
 * Retrieve recent TTS latency telemetry and stats
 */
router.get('/telemetry', (req, res) => {
  const telemetry = speechAdapter.getTelemetry();
  return res.status(200).json({
    success: true,
    telemetry,
  });
});

module.exports = router;
