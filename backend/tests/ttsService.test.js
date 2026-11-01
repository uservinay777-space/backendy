/**
 * TTS Service & Text Normalizer Unit Tests
 * 
 * Verifies:
 * 1. Sarvam configuration detection and missing key error messages
 * 2. Active provider selection (Sarvam vs ElevenLabs)
 * 3. Text normalization (train numbers, PNRs, currency, time)
 * 4. Streaming audio chunk generation with latency telemetry
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');
const ttsService = require('../src/services/tts/ttsService');
const { normalizeForSpeech } = require('../src/services/tts/textNormalizer');
const config = require('../src/config/config');

describe('TTS Service & Text Normalization Unit Tests', () => {
  test('1. Text Normalizer expands 5-digit train numbers into spaced digits', () => {
    const raw = 'Train number 12301 right time par hai.';
    const normalized = normalizeForSpeech(raw);
    assert.strictEqual(normalized.includes('1 2 3 0 1'), true);
  });

  test('2. Text Normalizer spaces out 10-digit PNR for speech clarity', () => {
    const raw = 'Aapka PNR 1234567890 confirmed hai.';
    const normalized = normalizeForSpeech(raw);
    assert.strictEqual(normalized.includes('1 2 3 4 5 6 7 8 9 0'), true);
  });

  test('3. Text Normalizer expands currency symbol ₹ to Rupees', () => {
    const raw = 'Total ticket fare ₹1250 hai.';
    const normalized = normalizeForSpeech(raw);
    assert.strictEqual(normalized.includes('1250 Rupees'), true);
  });

  test('4. Text Normalizer converts 12-hour AM/PM times to natural Hinglish', () => {
    const morningText = normalizeForSpeech('Shatabdi Express 06:00 AM par nikalti hai.');
    assert.strictEqual(morningText.includes('subah 6 baje'), true);

    const afternoonText = normalizeForSpeech('Arrival at 02:30 PM today.');
    assert.strictEqual(afternoonText.includes('dopahar 2:30'), true);
  });

  test('5. ttsService detects configuration state under strict mode', () => {
    const check = ttsService.checkConfiguration(true);
    if (config.sarvam.isConfigured) {
      assert.strictEqual(check.isConfigured, true);
      assert.strictEqual(check.provider, 'sarvam');
    } else {
      assert.strictEqual(check.isConfigured, false);
      assert.ok(check.error.includes('SARVAM_API_KEY'));
    }
  });

  test('6. ttsService provider selector defaults to sarvam', () => {
    const provider = ttsService.getActiveProvider();
    assert.strictEqual(provider, 'sarvam');
  });

  test('7. ttsService.synthesize produces a valid audio buffer and latency metrics', async () => {
    const res = await ttsService.synthesize({
      text: 'Train number 12301 New Delhi se depart ho chuki hai.',
      turnId: 'unit_test_turn_tts_1',
    });

    assert.strictEqual(res.success, true);
    assert.ok(res.audioBuffer instanceof Buffer);
    assert.ok(res.audioBuffer.length > 0);
    assert.ok(typeof res.metrics.ttsApiLatencyMs === 'number');
    assert.ok(typeof res.metrics.totalAudioGenerationMs === 'number');
  });

  test('8. ttsService.synthesizeStream emits sequential chunks with chunkIndex and final flag', async () => {
    const stream = ttsService.synthesizeStream({
      text: 'Shatabdi Express on time.',
      turnId: 'unit_test_turn_tts_2',
    });

    const chunks = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    assert.ok(chunks.length >= 2);
    assert.strictEqual(chunks[0].chunkIndex, 0);
    assert.strictEqual(chunks[chunks.length - 1].isFinal, true);
  });
});
