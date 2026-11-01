/**
 * Conversation Engine
 * 
 * Core coordinator for Member 3 (AI Conversation & Backend Engineer).
 * Integrates Intent Recognition, Entity Extraction, Context Management (SwitchSync),
 * Phantom Fast Lane (< 30ms), and Smart Path (LLM).
 * 
 * Exposes a clean, independent interface that Member 1 (Central Orchestrator)
 * can call directly or via REST API.
 */

const { performance } = require('perf_hooks');
const { recognizeIntent } = require('./intentRecognizer');
const { extractEntities } = require('./entityExtractor');
const contextManager = require('./contextManager');
const fastLane = require('../fastLane/fastLane');
const smartPath = require('../smartPath/smartPath');
const mongoManager = require('../database/mongoClient');

class ConversationEngine {
  /**
   * Process a single conversational message turn.
   * @param {Object} params
   * @param {string} params.sessionId - Unique session identifier
   * @param {string} params.text - User utterance text
   * @returns {Promise<{
   *   success: boolean,
   *   sessionId: string,
   *   intent: string,
   *   entities: Object,
   *   missingFields: string[],
   *   processingPath: 'fast' | 'smart',
   *   response: string,
   *   latencyMs: number,
   *   error?: string
   * }>}
   */
  async processMessage({ sessionId, text }) {
    const startTime = performance.now();

    // 1. Input Validation
    if (!sessionId || typeof sessionId !== 'string') {
      return {
        success: false,
        sessionId: sessionId || 'unknown',
        intent: 'unknown',
        entities: {},
        missingFields: [],
        processingPath: 'fast',
        response: 'Error: sessionId must be a valid non-empty string.',
        latencyMs: Math.round((performance.now() - startTime) * 100) / 100,
      };
    }

    if (!text || typeof text !== 'string' || !text.trim()) {
      return {
        success: false,
        sessionId,
        intent: 'unknown',
        entities: {},
        missingFields: [],
        processingPath: 'fast',
        response: 'Please say something or ask your railway question.',
        latencyMs: Math.round((performance.now() - startTime) * 100) / 100,
      };
    }

    const trimmedText = text.trim();

    try {
      // 2. Retrieve Existing Session Context
      const currentSession = await contextManager.getOrCreateSession(sessionId);

      // 3. Recognize Intent with context awareness
      const { intent, language, bhashaSpectrum, isEmergency } = recognizeIntent(trimmedText, currentSession.entities);

      // 4. Extract Entities and detect corrections
      const { entities: newEntities, isCorrection, correctedField, missingFields } = extractEntities(
        trimmedText,
        currentSession.entities
      );

      // If user provides a fresh pair of stations in this turn (new route),
      // clear any old date unless explicitly provided in this turn
      if (newEntities.origin && newEntities.destination && !newEntities.date) {
        currentSession.entities.date = null;
        currentSession.entities.time = null;
      }

      // 5. Update Context non-destructively (SwitchSync)
      const updatedSession = await contextManager.updateContext(
        sessionId,
        newEntities,
        intent
      );

      // Recalculate missingFields against newly merged session entities
      const mergedEntities = updatedSession.entities;
      const finalMissingFields = [];
      if (intent === 'TRAIN_SEARCH' || intent === 'railway_search') {
        if (!mergedEntities.origin) finalMissingFields.push('origin');
        if (!mergedEntities.destination) finalMissingFields.push('destination');
        if (!mergedEntities.date) finalMissingFields.push('date');
      }

      // 6. Dual-Path Routing: Fast Lane (< 30ms) vs Smart Path (LLM)
      let processingPath = 'fast';
      let responseText = '';
      let railwayData = null;

      const fastLaneResult = await fastLane.process({
        intent,
        text: trimmedText,
        entities: mergedEntities,
        missingFields: finalMissingFields,
        isCorrection,
        correctedField,
      });

      if (fastLaneResult.canHandle) {
        processingPath = 'fast';
        responseText = fastLaneResult.response;
        railwayData = fastLaneResult.railwayData || null;
      } else {
        // Fallback to Smart Path LLM
        processingPath = 'smart';
        const smartResult = await smartPath.process({
          text: trimmedText,
          entities: mergedEntities,
          history: updatedSession.history,
        });
        responseText = smartResult.response;
      }

      // 7. Record turn in session history
      await contextManager.recordTurn(sessionId, trimmedText, responseText);

      // Asynchronously log turn telemetry to MongoDB (non-blocking)
      mongoManager.logTurn({
        sessionId,
        userQuery: trimmedText,
        intent,
        language,
        entities: mergedEntities,
        processingPath,
        responseText,
        railwayDataAvailable: Boolean(railwayData),
        timestamp: Date.now(),
      });

      const endTime = performance.now();
      const latencyMs = Math.round((endTime - startTime) * 100) / 100;

      // 8. Return structured result adhering to API contract
      return {
        success: true,
        sessionId,
        intent,
        language,
        entities: {
          origin: mergedEntities.origin,
          destination: mergedEntities.destination,
          date: mergedEntities.date,
          time: mergedEntities.time,
          trainNumber: mergedEntities.trainNumber || null,
          pnr: mergedEntities.pnr || null,
          travelClass: mergedEntities.travelClass || null,
          quota: mergedEntities.quota || null,
        },
        missingFields: finalMissingFields,
        processingPath,
        response: responseText,
        responseText: responseText,
        railwayData,
        isEmergency: Boolean(isEmergency || fastLaneResult.isEmergency),
        emergencyDossier: fastLaneResult.emergencyDossier || null,
        bhashaSpectrum: bhashaSpectrum || null,
        metadata: {
          language,
          provider: 'sarvam',
          bhashaSpectrum: bhashaSpectrum || null,
          emergencyDossier: fastLaneResult.emergencyDossier || null,
        },
        latencyMs,
      };
    } catch (err) {
      console.error(`[ConversationEngine] Error processing turn for session ${sessionId}:`, err);
      const endTime = performance.now();

      return {
        success: false,
        sessionId,
        intent: 'unknown',
        entities: {},
        missingFields: [],
        processingPath: 'fast',
        response: 'Maaf kijiye, kuch technical problem hui hai. Kripya dobara try karein.',
        latencyMs: Math.round((endTime - startTime) * 100) / 100,
        error: err.message,
      };
    }
  }

  /**
   * Get session data by ID.
   * @param {string} sessionId 
   */
  async getSession(sessionId) {
    return contextManager.getSession(sessionId);
  }

  /**
   * Reset / clear session.
   * @param {string} sessionId 
   */
  async resetSession(sessionId) {
    return contextManager.deleteSession(sessionId);
  }
}

module.exports = new ConversationEngine();
