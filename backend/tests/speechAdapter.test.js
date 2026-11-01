/**
 * Speech Adapter & ElevenLabs TTS Test Suite
 * 
 * Verifies:
 * 1. Audio generation via speechAdapter.generateSpeech
 * 2. Streaming audio generation via speechAdapter.generateSpeechStream
 * 3. GhostBarge interruption and cancellation via speechAdapter.cancelSpeech
 * 4. Stale audio chunk suppression on cancelled turns
 * 5. Latency metrics tracking (time-to-first-chunk, API latency, total generation time)
 * 6. REST API /api/speech/synthesize and /api/speech/cancel integration
 */

const { test, describe, before, after } = require('node:test');
const assert = require('node:assert');
const speechAdapter = require('../src/speech/speechAdapter');
const audioBufferManager = require('../src/speech/audioBufferManager');
const app = require('../src/server');

describe('Speech Adapter & ElevenLabs TTS Integration Tests', () => {
  let server;
  let baseUrl;

  before(async () => {
    // Start HTTP server on an ephemeral port
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

  test('1. speechAdapter.generateSpeech produces audio buffer and latency metrics', async () => {
    const turnId = `test_turn_${Date.now()}`;
    const result = await speechAdapter.generateSpeech({
      sessionId: 'session_test_tts',
      turnId,
      text: 'Bengaluru se Chennai ke liye Shatabdi Express subah 6 baje nikalti hai.',
      language: 'hinglish',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.turnId, turnId);
    assert.strictEqual(result.interrupted, false);
    assert.ok(result.audioBuffer instanceof Buffer);
    assert.ok(result.audioBuffer.length > 0);
    assert.ok(['audio/mpeg', 'audio/wav'].includes(result.audioMimeType));

    // Latency metrics verification
    assert.ok(typeof result.metrics.ttsApiLatencyMs === 'number');
    assert.ok(typeof result.metrics.timeToFirstAudioChunkMs === 'number');
    assert.ok(result.metrics.timeToFirstAudioChunkMs > 0);
    assert.ok(result.metrics.totalBytes > 0);
    assert.ok(result.metrics.chunkCount > 0);
  });

  test('2. speechAdapter.generateSpeechStream emits incremental audio chunks', async () => {
    const turnId = `test_stream_${Date.now()}`;
    const stream = speechAdapter.generateSpeechStream({
      sessionId: 'session_test_stream',
      turnId,
      text: 'Vande Bharat Express at 05:45 AM right time par chal rahi hai.',
      language: 'hinglish',
    });

    const receivedChunks = [];
    for await (const chunk of stream) {
      receivedChunks.push(chunk);
    }

    assert.ok(receivedChunks.length >= 2, 'Stream should have emitted multiple chunks');
    assert.strictEqual(receivedChunks[0].chunkIndex, 0);
    assert.strictEqual(receivedChunks[0].isFinal, false);
    assert.strictEqual(receivedChunks[receivedChunks.length - 1].isFinal, true);
  });

  test('3. GhostBarge interruption: speechAdapter.cancelSpeech halts active turn mid-stream', async () => {
    const turnId = `test_interrupt_${Date.now()}`;
    const sessionId = 'session_test_ghostbarge';

    let chunksEmitted = 0;
    const stream = speechAdapter.generateSpeechStream({
      sessionId,
      turnId,
      text: 'This is a very long response that will be interrupted while the assistant is speaking.',
      language: 'english',
    });

    for await (const chunk of stream) {
      chunksEmitted++;
      if (chunksEmitted === 1) {
        // Trigger GhostBarge interruption after first chunk
        const cancelResult = speechAdapter.cancelSpeech({ sessionId, turnId });
        assert.strictEqual(cancelResult.cancelled, true);
      }
    }

    // Should have terminated early without emitting remaining chunks
    assert.ok(chunksEmitted <= 2, `Emitted ${chunksEmitted} chunks; expected early termination.`);
    assert.strictEqual(audioBufferManager.isCancelled(turnId), true);
  });

  test('4. REST API: POST /api/speech/synthesize returns JSON with base64 audio and telemetry', async () => {
    const res = await fetch(`${baseUrl}/api/speech/synthesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: 'session_http_tts',
        turnId: 'turn_http_01',
        text: 'Aapka PNR status confirmed hai.',
        language: 'hinglish',
      }),
    });

    assert.strictEqual(res.status, 200);
    const body = await res.json();

    assert.strictEqual(body.success, true);
    assert.strictEqual(body.turnId, 'turn_http_01');
    assert.ok(body.audioBase64.length > 50);
    assert.ok(['audio/mpeg', 'audio/wav'].includes(body.audioMimeType));
    assert.ok(typeof body.metrics.timeToFirstAudioChunkMs === 'number');
  });

  test('5. REST API: POST /api/speech/cancel halts audio turn', async () => {
    const res = await fetch(`${baseUrl}/api/speech/cancel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        sessionId: 'session_http_tts',
        turnId: 'turn_to_cancel',
      }),
    });

    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.success, true);
  });

  test('6. REST API: GET /api/speech/telemetry returns rolling metrics', async () => {
    const res = await fetch(`${baseUrl}/api/speech/telemetry`);
    assert.strictEqual(res.status, 200);
    const body = await res.json();
    assert.strictEqual(body.success, true);
    assert.ok(Array.isArray(body.telemetry));
    assert.ok(body.telemetry.length > 0);
  });
});
