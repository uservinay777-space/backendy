/**
 * REST API Routes Integration Tests
 * Verifies HTTP endpoints using Node.js fetch against the Express app.
 */

const { test, describe, before, after } = require('node:test');
const assert = require('node:assert');
const app = require('../src/server');

describe('Conversation Engine HTTP REST API Tests', () => {
  let server;
  let baseUrl;

  before(async () => {
    // Start server on an ephemeral free port (port 0)
    await new Promise((resolve) => {
      server = app.listen(0, () => {
        const port = server.address().port;
        baseUrl = `http://127.0.0.1:${port}`;
        resolve();
      });
    });
  });

  after(async () => {
    await new Promise((resolve) => server.close(resolve));
  });

  test('GET /health returns 200 OK and status', async () => {
    const res = await fetch(`${baseUrl}/health`);
    assert.strictEqual(res.status, 200);

    const body = await res.json();
    assert.strictEqual(body.status, 'ok');
    assert.strictEqual(body.service, 'phantomflow-conversation-engine');
  });

  test('POST /api/conversation/message rejects missing payload with 400', async () => {
    const res = await fetch(`${baseUrl}/api/conversation/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    assert.strictEqual(res.status, 400);

    const body = await res.json();
    assert.strictEqual(body.success, false);
    assert.ok(body.error.includes('sessionId'));
  });

  test('POST /api/conversation/message processes valid turn and returns 200', async () => {
    const res = await fetch(`${baseUrl}/api/conversation/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: 'http_session_01',
        text: 'Mujhe Bengaluru se Chennai jaana hai Saturday morning mein',
      }),
    });
    assert.strictEqual(res.status, 200);

    const body = await res.json();
    assert.strictEqual(body.success, true);
    assert.strictEqual(body.sessionId, 'http_session_01');
    assert.strictEqual(body.intent, 'railway_search');
    assert.strictEqual(body.entities.origin, 'Bengaluru');
    assert.strictEqual(body.entities.destination, 'Chennai');
    assert.strictEqual(body.entities.date, 'Saturday');
    assert.strictEqual(body.processingPath, 'fast');
    assert.ok(typeof body.latencyMs === 'number');
  });

  test('GET /api/conversation/:sessionId inspects session context', async () => {
    const res = await fetch(`${baseUrl}/api/conversation/http_session_01`);
    assert.strictEqual(res.status, 200);

    const body = await res.json();
    assert.strictEqual(body.success, true);
    assert.strictEqual(body.session.sessionId, 'http_session_01');
    assert.strictEqual(body.session.entities.origin, 'Bengaluru');
    assert.strictEqual(body.session.history.length, 1);
  });

  test('DELETE /api/conversation/:sessionId clears session', async () => {
    const res = await fetch(`${baseUrl}/api/conversation/http_session_01`, {
      method: 'DELETE',
    });
    assert.strictEqual(res.status, 200);

    const checkRes = await fetch(`${baseUrl}/api/conversation/http_session_01`);
    assert.strictEqual(checkRes.status, 404);
  });
});
