const { test, describe, before, after } = require('node:test');
const assert = require('node:assert');
const { InMemorySessionStore, MongoSessionStore, ContextManager } = require('../src/conversation/contextManager');
const mongoManager = require('../src/database/mongoClient');
const config = require('../src/config/config');

describe('MongoDB Atlas Integration & Session Store Tests', () => {
  test('1. InMemorySessionStore correctly handles get, set, delete, and count', async () => {
    const store = new InMemorySessionStore();
    const sessionData = { sessionId: 'test_s1', entities: { origin: 'NDLS' } };

    await store.set('test_s1', sessionData);
    const retrieved = await store.get('test_s1');
    assert.deepStrictEqual(retrieved.entities, { origin: 'NDLS' });

    const count = await store.count();
    assert.strictEqual(count, 1);

    await store.delete('test_s1');
    const afterDelete = await store.get('test_s1');
    assert.strictEqual(afterDelete, null);
  });

  test('2. MongoSessionStore seamlessly falls back to memory when DB is disconnected', async () => {
    const fallbackStore = new InMemorySessionStore();
    const store = new MongoSessionStore(fallbackStore);
    const sessionData = {
      sessionId: 'test_mongo_fallback',
      entities: { origin: 'CSMT', destination: 'HWH' },
    };

    // Should not throw even if Mongo is disconnected
    await store.set('test_mongo_fallback', sessionData);
    const result = await store.get('test_mongo_fallback');
    assert.strictEqual(result.sessionId, 'test_mongo_fallback');
    assert.strictEqual(result.entities.origin, 'CSMT');
    assert.strictEqual(result.entities.destination, 'HWH');
  });

  test('3. ContextManager persists multi-turn slots non-destructively with MongoSessionStore', async () => {
    const context = new ContextManager(new MongoSessionStore());
    const sessionId = 'session_turn_switchsync';

    // Turn 1: User says Delhi to Mumbai
    await context.updateContext(sessionId, { origin: 'NDLS', destination: 'MMCT' }, 'TRAIN_SEARCH');
    let session = await context.getSession(sessionId);
    assert.strictEqual(session.entities.origin, 'NDLS');
    assert.strictEqual(session.entities.destination, 'MMCT');
    assert.strictEqual(session.entities.date, null);

    // Turn 2: User adds travel date (SwitchSync preservation)
    await context.updateContext(sessionId, { date: '25-12-2026' });
    session = await context.getSession(sessionId);
    assert.strictEqual(session.entities.origin, 'NDLS');
    assert.strictEqual(session.entities.destination, 'MMCT');
    assert.strictEqual(session.entities.date, '25-12-2026');
  });

  test('4. mongoManager.logTurn handles turn telemetry asynchronously without throwing', async () => {
    // Should never throw, even if offline
    assert.doesNotThrow(() => {
      mongoManager.logTurn({
        sessionId: 'test_log_session',
        userQuery: 'Delhi to Mumbai train',
        intent: 'TRAIN_SEARCH',
        latencyMs: 14.5,
        timestamp: Date.now(),
      });
    });
  });

  test('5. config.getPublicStatus includes mongodb provider metadata', () => {
    const publicStatus = config.getPublicStatus();
    assert.ok(publicStatus.providers.mongodb);
    assert.strictEqual(typeof publicStatus.providers.mongodb.isConfigured, 'boolean');
    assert.strictEqual(publicStatus.providers.mongodb.dbName, 'phantomflow');
  });
});
