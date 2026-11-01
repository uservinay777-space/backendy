/**
 * Audio Buffer & Interruption Manager (GhostBarge)
 * 
 * Tracks active generation turns, manages AbortControllers, and ensures
 * that stale/outdated audio chunks are immediately suppressed when an interruption occurs.
 */

class AudioBufferManager {
  constructor() {
    // Map of turnId -> { sessionId, abortController, cancelled: boolean, registeredAt: number }
    this.activeTurns = new Map();
  }

  /**
   * Register a new audio turn.
   * @param {string} sessionId
   * @param {string} turnId
   * @param {AbortController} abortController
   */
  registerTurn(sessionId, turnId, abortController) {
    if (!turnId) return;
    this.activeTurns.set(turnId, {
      sessionId,
      abortController,
      cancelled: false,
      registeredAt: Date.now(),
    });
  }

  /**
   * GhostBarge Action: Cancel an active speech turn immediately.
   * Aborts network streaming and marks the turn as cancelled.
   * @param {string} sessionId
   * @param {string} turnId
   * @returns {boolean} True if turn was found and cancelled
   */
  cancelTurn(sessionId, turnId) {
    let cancelled = false;

    if (turnId && this.activeTurns.has(turnId)) {
      const record = this.activeTurns.get(turnId);
      record.cancelled = true;
      if (record.abortController && !record.abortController.signal.aborted) {
        record.abortController.abort('GhostBarge User Interruption');
      }
      this.activeTurns.delete(turnId);
      cancelled = true;
    } else if (sessionId) {
      // Cancel all turns associated with this session if turnId is not specified
      for (const [tId, record] of this.activeTurns.entries()) {
        if (record.sessionId === sessionId) {
          record.cancelled = true;
          if (record.abortController && !record.abortController.signal.aborted) {
            record.abortController.abort('GhostBarge User Interruption');
          }
          this.activeTurns.delete(tId);
          cancelled = true;
        }
      }
    }

    return cancelled;
  }

  /**
   * Check whether a turn has been cancelled.
   * @param {string} turnId
   * @returns {boolean}
   */
  isCancelled(turnId) {
    if (!turnId) return false;
    const record = this.activeTurns.get(turnId);
    return !record || record.cancelled;
  }

  /**
   * Clean up a completed turn.
   * @param {string} turnId
   */
  completeTurn(turnId) {
    if (turnId) {
      this.activeTurns.delete(turnId);
    }
  }

  /**
   * Number of currently active turns.
   */
  getActiveCount() {
    return this.activeTurns.size;
  }
}

module.exports = new AudioBufferManager();
