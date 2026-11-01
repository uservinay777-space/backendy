/**
 * Central Application Configuration
 * Reads environment variables, validates settings, and exposes provider statuses.
 * Avoids exposing secrets to API responses or logs.
 */

const path = require('path');
require('dotenv').config({ path: path.resolve(__dirname, '../../.env') });

class ConfigurationError extends Error {
  constructor(service, message) {
    super(message);
    this.name = 'ConfigurationError';
    this.code = 'CONFIG_MISSING';
    this.service = service;
  }
}

const config = {
  // Server Configuration
  port: parseInt(process.env.PORT, 10) || 5001,
  env: process.env.NODE_ENV || 'development',
  corsOrigin: process.env.CORS_ORIGIN || '*',

  // RailKit Indian Railway SDK Configuration
  railkit: {
    apiKey: process.env.RAILKIT_API_KEY || '',
    isConfigured: Boolean(process.env.RAILKIT_API_KEY && process.env.RAILKIT_API_KEY.trim()),
    baseUrl: 'https://api.railkit.in',
    timeoutMs: parseInt(process.env.RAILKIT_TIMEOUT_MS, 10) || 6000,
  },

  // Sarvam AI Configuration (Primary Voice / TTS)
  sarvam: {
    apiKey: process.env.SARVAM_API_KEY || '',
    model: process.env.SARVAM_TTS_MODEL || 'bulbul:v4-flash',
    isConfigured: Boolean(process.env.SARVAM_API_KEY && process.env.SARVAM_API_KEY.trim()),
    languageCode: process.env.SARVAM_LANG_CODE || 'hi-IN',
    speaker: process.env.SARVAM_SPEAKER || 'shubh_hi_customer',
    pace: parseFloat(process.env.SARVAM_PACE) || 1.0,
    timeoutMs: parseInt(process.env.SARVAM_TIMEOUT_MS, 10) || 5000,
  },

  // ElevenLabs Configuration (Natural Conversational Voice)
  elevenlabs: {
    apiKey: process.env.ELEVENLABS_API_KEY || process.env.ELEVEN_API_KEY || '',
    modelId: process.env.ELEVENLABS_MODEL_ID || 'eleven_flash_v2_5',
    voiceId: process.env.ELEVENLABS_VOICE_ID || '21m00Tcm4TlvDq8ikWAM', // Rachel - Natural Conversational Voice
    isConfigured: Boolean((process.env.ELEVENLABS_API_KEY || process.env.ELEVEN_API_KEY) && (process.env.ELEVENLABS_API_KEY || process.env.ELEVEN_API_KEY).trim()),
    optimizeLatency: parseInt(process.env.ELEVENLABS_OPTIMIZE_LATENCY, 10) || 3,
    stability: parseFloat(process.env.ELEVENLABS_STABILITY) || 0.40,
    similarityBoost: parseFloat(process.env.ELEVENLABS_SIMILARITY) || 0.75,
    style: parseFloat(process.env.ELEVENLABS_STYLE) || 0.10,
    useSpeakerBoost: true,
    speed: parseFloat(process.env.ELEVENLABS_SPEED) || 1.0,
    timeoutMs: parseInt(process.env.TTS_TIMEOUT_MS, 10) || 6000,
  },

  // TTS Provider Selection
  tts: {
    provider: process.env.TTS_PROVIDER || 'sarvam',
  },

  // LLM / Smart Path Configuration
  llm: {
    provider: process.env.LLM_PROVIDER || 'mock',
    apiKey: process.env.LLM_API_KEY || process.env.GROQ_API_KEY || '',
    model: process.env.LLM_MODEL || 'openai/gpt-oss-120b',
    timeoutMs: parseInt(process.env.LLM_TIMEOUT_MS, 10) || 3000,
    isConfigured: Boolean(
      (process.env.LLM_API_KEY && process.env.LLM_API_KEY.trim()) ||
      (process.env.GROQ_API_KEY && process.env.GROQ_API_KEY.trim())
    ),
    maxTokens: parseInt(process.env.LLM_MAX_TOKENS, 10) || 250,
    temperature: parseFloat(process.env.LLM_TEMPERATURE) || 0.2,
  },

  // Session & Context Management
  session: {
    ttlSeconds: 3600,
    maxHistoryTurns: 20,
  },

  // MongoDB Atlas Database Configuration
  mongodb: {
    uri: process.env.MONGODB_URI || '',
    dbName: process.env.MONGODB_DB_NAME || 'phantomflow',
    isConfigured: Boolean(process.env.MONGODB_URI && process.env.MONGODB_URI.trim()),
  },

  /**
   * Asserts that a required external service is properly configured.
   * Throws a clear ConfigurationError if key is missing without failing on app startup.
   * @param {'railkit' | 'sarvam' | 'elevenlabs' | 'llm'} serviceName
   */
  assertConfigured(serviceName) {
    const serviceMap = {
      railkit: {
        configured: this.railkit.isConfigured,
        envVar: 'RAILKIT_API_KEY',
        msg: 'RailKit Railway API is not configured. Set RAILKIT_API_KEY in .env to enable verified railway queries.',
      },
      sarvam: {
        configured: this.sarvam.isConfigured,
        envVar: 'SARVAM_API_KEY',
        msg: 'Sarvam AI TTS is not configured. Set SARVAM_API_KEY in .env to enable Bulbul v4 Flash voice synthesis.',
      },
      elevenlabs: {
        configured: this.elevenlabs.isConfigured,
        envVar: 'ELEVENLABS_API_KEY',
        msg: 'ElevenLabs TTS is not configured. Set ELEVENLABS_API_KEY in .env to enable ElevenLabs synthesis.',
      },
      llm: {
        configured: this.llm.isConfigured,
        envVar: 'LLM_API_KEY',
        msg: 'LLM provider is not configured. Set LLM_API_KEY in .env to enable Smart Path reasoning.',
      },
    };

    const target = serviceMap[serviceName];
    if (!target) {
      throw new ConfigurationError(serviceName, `Unknown service: ${serviceName}`);
    }

    if (!target.configured) {
      throw new ConfigurationError(serviceName, target.msg);
    }
  },

  /**
   * Safe public configuration snapshot (free of sensitive credentials).
   */
  getPublicStatus() {
    return {
      port: this.port,
      env: this.env,
      ttsProvider: this.tts.provider,
      providers: {
        railkit: { isConfigured: this.railkit.isConfigured },
        sarvam: { isConfigured: this.sarvam.isConfigured, model: this.sarvam.model },
        elevenlabs: { isConfigured: this.elevenlabs.isConfigured, modelId: this.elevenlabs.modelId },
        llm: { isConfigured: this.llm.isConfigured, provider: this.llm.provider, model: this.llm.model },
        mongodb: { isConfigured: this.mongodb.isConfigured, dbName: this.mongodb.dbName },
      },
    };
  },
};

module.exports = config;
module.exports.ConfigurationError = ConfigurationError;
