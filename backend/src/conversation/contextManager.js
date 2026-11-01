/**
 * Conversation Context Manager
 * 
 * Implements session state persistence, multi-turn history, and SwitchSync slot retention.
 * Ensures non-destructive updates so that uncorrected entities (origin, destination)
 * are cleanly preserved when a user updates a single field (e.g., date).
 * 
 * Designed with a clean storage interface that can easily transition from in-memory
 * to Redis or MongoDB in production.
 */

const config = require('../config/config');

/**
 * Storage Interface for Session State
 */
class InMemorySessionStore {
  constructor() {
    this.store = new Map();
  }

  async get(sessionId) {
    return this.store.get(sessionId) || null;
  }

  async set(sessionId, sessionData) {
    this.store.set(sessionId, sessionData);
    return sessionData;
  }

  async delete(sessionId) {
    return this.store.delete(sessionId);
  }

  async clear() {
    this.store.clear();
  }

  async count() {
    return this.store.size;
  }
}

/**
 * MongoDB-Backed Session Store with in-memory write-through fallback
 */
class MongoSessionStore {
  constructor(fallbackStore = new InMemorySessionStore()) {
    this.fallbackStore = fallbackStore;
  }

  _getCollection() {
    try {
      const mongoManager = require('../database/mongoClient');
      const db = mongoManager.getDb();
      return db ? db.collection('sessions') : null;
    } catch {
      return null;
    }
  }

  async get(sessionId) {
    const coll = this._getCollection();
    if (coll) {
      try {
        const doc = await coll.findOne({ sessionId });
        if (doc) {
          const { _id, ...sessionData } = doc;
          await this.fallbackStore.set(sessionId, sessionData);
          return sessionData;
        }
      } catch (err) {
        console.warn(`[MongoSessionStore] Read fallback for ${sessionId}: ${err.message}`);
      }
    }
    return this.fallbackStore.get(sessionId);
  }

  async set(sessionId, sessionData) {
    await this.fallbackStore.set(sessionId, sessionData);
    const coll = this._getCollection();
    if (coll) {
      try {
        await coll.replaceOne(
          { sessionId },
          { ...sessionData, sessionId, updatedAt: Date.now() },
          { upsert: true }
        );
      } catch (err) {
        console.warn(`[MongoSessionStore] Write warning for ${sessionId}: ${err.message}`);
      }
    }
    return sessionData;
  }

  async delete(sessionId) {
    await this.fallbackStore.delete(sessionId);
    const coll = this._getCollection();
    if (coll) {
      try {
        await coll.deleteOne({ sessionId });
      } catch (err) {
        console.warn(`[MongoSessionStore] Delete warning for ${sessionId}: ${err.message}`);
      }
    }
    return true;
  }

  async clear() {
    await this.fallbackStore.clear();
    const coll = this._getCollection();
    if (coll) {
      try {
        await coll.deleteMany({});
      } catch (err) {
        console.warn(`[MongoSessionStore] Clear warning: ${err.message}`);
      }
    }
  }

  async count() {
    const coll = this._getCollection();
    if (coll) {
      try {
        return await coll.countDocuments();
      } catch {
        // fallback
      }
    }
    return this.fallbackStore.count();
  }
}

class ContextManager {
  /**
   * @param {InMemorySessionStore|MongoSessionStore} store - Pluggable session store
   */
  constructor(store = new MongoSessionStore()) {
    this.store = store;
  }

  /**
   * Retrieve existing session or initialize a fresh one for the given sessionId.
   * @param {string} sessionId
   * @returns {Promise<Object>} Session object
   */
  async getOrCreateSession(sessionId) {
    if (!sessionId) {
      throw new Error('sessionId is required for context management.');
    }

    let session = await this.store.get(sessionId);
    if (!session) {
      session = {
        sessionId,
        entities: {
          origin: null,
          destination: null,
          date: null,
          time: null,
          trainNumber: null,
          pnr: null,
        },
        lastIntent: null,
        turnCount: 0,
        history: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      await this.store.set(sessionId, session);
    }

    return session;
  }

  /**
   * Non-destructive entity merge (SwitchSync).
   * Updates only fields present in newEntities, preserving all other existing slots.
   * @param {string} sessionId 
   * @param {Object} newEntities - Extracted entities for this turn
   * @param {string|null} intent - Detected intent
   * @returns {Promise<Object>} Updated session
   */
  async updateContext(sessionId, newEntities = {}, intent = null) {
    const session = await this.getOrCreateSession(sessionId);

    // Apply SwitchSync non-destructive merge
    for (const [key, value] of Object.entries(newEntities)) {
      if (value !== undefined && value !== null && value !== '') {
        session.entities[key] = value;
      }
    }

    if (intent) {
      session.lastIntent = intent;
    }

    session.turnCount += 1;
    session.updatedAt = Date.now();

    await this.store.set(sessionId, session);
    return session;
  }

  /**
   * Record a conversational turn into the session history.
   * @param {string} sessionId 
   * @param {string} userText 
   * @param {string} assistantText 
   */
  async recordTurn(sessionId, userText, assistantText) {
    const session = await this.getOrCreateSession(sessionId);

    session.history.push({
      turn: session.turnCount,
      user: userText,
      assistant: assistantText,
      timestamp: Date.now(),
    });

    // Cap history to prevent unbounded memory growth
    const maxHistory = config.session.maxHistoryTurns || 20;
    if (session.history.length > maxHistory) {
      session.history = session.history.slice(-maxHistory);
    }

    session.updatedAt = Date.now();
    await this.store.set(sessionId, session);
  }

  /**
   * Fetch current session data.
   * @param {string} sessionId 
   * @returns {Promise<Object|null>}
   */
  async getSession(sessionId) {
    return this.store.get(sessionId);
  }

  /**
   * Delete / reset session.
   * @param {string} sessionId 
   * @returns {Promise<boolean>}
   */
  async deleteSession(sessionId) {
    return this.store.delete(sessionId);
  }
}

// Export singleton with Mongo/In-Memory store
const contextManagerInstance = new ContextManager();
module.exports = contextManagerInstance;
module.exports.ContextManager = ContextManager;
module.exports.InMemorySessionStore = InMemorySessionStore;
module.exports.MongoSessionStore = MongoSessionStore;
