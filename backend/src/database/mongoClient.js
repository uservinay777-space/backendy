/**
 * MongoDB Atlas Connection and Data Access Layer
 * 
 * Provides:
 * - Resilient connection handling with auto-reconnect
 * - Persistent session storage for multi-turn conversations (SwitchSync)
 * - Conversation turn logging for analytics and evaluation
 * - Non-blocking asynchronous writes that never degrade response latency
 */

const { MongoClient } = require('mongodb');
const config = require('../config/config');

class MongoDBManager {
  constructor() {
    this.client = null;
    this.db = null;
    this.isConnected = false;
    this.connectionPromise = null;
    this.initAttempted = false;
  }

  /**
   * Connect to MongoDB Atlas cluster.
   * Gracefully falls back to null if unreachable or authentication fails.
   */
  async connect() {
    if (!config.mongodb.isConfigured) {
      return null;
    }

    if (this.isConnected && this.db) {
      return this.db;
    }

    if (this.connectionPromise) {
      return this.connectionPromise;
    }

    this.connectionPromise = (async () => {
      this.initAttempted = true;
      try {
        this.client = new MongoClient(config.mongodb.uri, {
          serverSelectionTimeoutMS: 5000,
          connectTimeoutMS: 10000,
          maxPoolSize: 10,
        });

        await this.client.connect();
        this.db = this.client.db(config.mongodb.dbName);
        this.isConnected = true;
        console.log(`[MongoDB] Connected successfully to Atlas database: "${config.mongodb.dbName}"`);

        await this._setupIndexes();
        return this.db;
      } catch (err) {
        console.warn(`[MongoDB] Connection warning (${err.message}). Continuing with in-memory storage fallback.`);
        this.isConnected = false;
        this.client = null;
        this.db = null;
        return null;
      } finally {
        this.connectionPromise = null;
      }
    })();

    return this.connectionPromise;
  }

  /**
   * Set up TTL and unique indexes on collections.
   */
  async _setupIndexes() {
    try {
      if (!this.db) return;

      const sessions = this.db.collection('sessions');
      await sessions.createIndex({ sessionId: 1 }, { unique: true });
      await sessions.createIndex(
        { updatedAt: 1 },
        { expireAfterSeconds: config.session.ttlSeconds || 86400 }
      );

      const logs = this.db.collection('conversation_logs');
      await logs.createIndex({ sessionId: 1, timestamp: -1 });
      await logs.createIndex({ timestamp: -1 });
    } catch (err) {
      console.warn('[MongoDB] Index creation note:', err.message);
    }
  }

  /**
   * Log a conversation turn asynchronously.
   * @param {Object} turnData
   */
  async logTurn(turnData) {
    if (!this.isConnected || !this.db) {
      return;
    }

    // Fire-and-forget: do not block conversation engine response latency
    setImmediate(async () => {
      try {
        const collection = this.db.collection('conversation_logs');
        await collection.insertOne({
          ...turnData,
          loggedAt: new Date(),
        });
      } catch (err) {
        console.warn('[MongoDB] Failed to log turn:', err.message);
      }
    });
  }

  /**
   * Retrieve conversation history logs for a session.
   * @param {string} sessionId
   * @param {number} limit
   */
  async getTurnLogs(sessionId, limit = 50) {
    if (!this.isConnected || !this.db) {
      return [];
    }

    try {
      const collection = this.db.collection('conversation_logs');
      return await collection
        .find({ sessionId })
        .sort({ timestamp: -1 })
        .limit(limit)
        .toArray();
    } catch (err) {
      console.warn('[MongoDB] Error reading turn logs:', err.message);
      return [];
    }
  }

  /**
   * Close active MongoDB client.
   */
  async close() {
    if (this.client) {
      try {
        await this.client.close();
      } catch (e) {
        // ignore
      }
      this.isConnected = false;
      this.db = null;
      this.client = null;
    }
  }
}

module.exports = new MongoDBManager();
