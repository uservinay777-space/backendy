/**
 * Conversation REST API Routes
 * 
 * Endpoints for Member 1 (Central Orchestrator) or Frontend clients:
 * - POST   /api/conversation/message
 * - GET    /api/conversation/:sessionId
 * - DELETE /api/conversation/:sessionId
 */

const express = require('express');
const router = express.Router();
const conversationEngine = require('../conversation/conversationEngine');

/**
 * POST /api/conversation/message
 * Main conversational turn endpoint
 */
router.post('/message', async (req, res, next) => {
  try {
    const { sessionId, text } = req.body;

    if (!sessionId) {
      return res.status(400).json({
        success: false,
        error: 'Validation Error: sessionId is required in request body.',
      });
    }

    if (!text || typeof text !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Validation Error: text must be a non-empty string in request body.',
      });
    }

    const result = await conversationEngine.processMessage({ sessionId, text });
    return res.status(200).json(result);
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/conversation/:sessionId
 * Inspect session context, active entities, and turn history
 */
router.get('/:sessionId', async (req, res, next) => {
  try {
    const { sessionId } = req.params;
    const session = await conversationEngine.getSession(sessionId);

    if (!session) {
      return res.status(404).json({
        success: false,
        message: `Session ${sessionId} not found.`,
      });
    }

    return res.status(200).json({
      success: true,
      session,
    });
  } catch (err) {
    next(err);
  }
});

/**
 * DELETE /api/conversation/:sessionId
 * Reset and clear a session
 */
router.delete('/:sessionId', async (req, res, next) => {
  try {
    const { sessionId } = req.params;
    const deleted = await conversationEngine.resetSession(sessionId);

    return res.status(200).json({
      success: true,
      message: `Session ${sessionId} has been reset.`,
      deleted,
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
