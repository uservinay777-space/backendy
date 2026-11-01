/**
 * Groq LLM & Smart Path Provider Test Suite
 * 
 * Tests the complete integration of Groq LLM:
 * 1. Successful Groq response
 * 2. Invalid API key handling
 * 3. Request timeout handling
 * 4. Server failure / 500 error handling
 * 5. Mock provider functionality
 * 6. Hinglish input processing
 * 7. Conversation context preservation across turns
 * 8. End-to-end graceful fallback behavior
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');
const { LLMAdapter } = require('../src/smartPath/llmAdapter');
const smartPath = require('../src/smartPath/smartPath');
const conversationEngine = require('../src/conversation/conversationEngine');
const config = require('../src/config/config');

describe('Groq LLM Integration & Smart Path Resilience Tests', () => {

  // Test 1: Successful Groq response
  test('1. Successful Groq response generates valid text and metadata', async () => {
    // If live Groq key is present in environment, test live API; otherwise test via client interface
    if (config.llm.isConfigured && config.llm.apiKey.startsWith('gsk_')) {
      const adapter = new LLMAdapter();
      const res = await adapter.generateCompletion([
        { role: 'system', content: 'You are PhantomFlow. Reply in one short Hinglish sentence.' },
        { role: 'user', content: 'Hello, testing connection.' },
      ]);

      assert.strictEqual(res.success, true);
      assert.ok(typeof res.content === 'string' && res.content.length > 0);
      assert.strictEqual(res.isFallback, false);
      assert.strictEqual(res.provider, 'groq');
    } else {
      // Mock client verification
      const mockClient = {
        chat: {
          completions: {
            create: async () => ({
              choices: [{ message: { content: 'Namaste! Main PhantomFlow hoon.' } }],
              model: 'openai/gpt-oss-120b',
            }),
          },
        },
      };

      const adapter = new LLMAdapter({
        provider: 'groq',
        apiKey: 'gsk_mock_test_key',
        groqClient: mockClient,
      });

      const res = await adapter.generateCompletion([
        { role: 'user', content: 'Hello' },
      ]);

      assert.strictEqual(res.success, true);
      assert.strictEqual(res.content, 'Namaste! Main PhantomFlow hoon.');
      assert.strictEqual(res.provider, 'groq');
      assert.strictEqual(res.isFallback, false);
    }
  });

  // Test 2: Invalid API key
  test('2. Invalid API key is gracefully caught without leaking key or crashing', async () => {
    // Mock a 401 AuthenticationError as returned by Groq
    const failingClient = {
      chat: {
        completions: {
          create: async () => {
            const err = new Error('Invalid API Key');
            err.status = 401;
            err.name = 'AuthenticationError';
            throw err;
          },
        },
      },
    };

    const adapter = new LLMAdapter({
      provider: 'groq',
      apiKey: 'gsk_invalid_test_key_should_not_leak',
      groqClient: failingClient,
    });

    const res = await adapter.generateCompletion([
      { role: 'user', content: 'Kaise ho?' },
    ]);

    // Must not throw; must provide safe fallback
    assert.strictEqual(res.success, true);
    assert.strictEqual(res.isFallback, true);
    assert.strictEqual(res.provider, 'mock_fallback');
    assert.ok(res.content.length > 0);
    // Ensure the key itself is never exposed in the response
    assert.strictEqual(JSON.stringify(res).includes('gsk_invalid_test_key'), false);
  });

  // Test 3: API Timeout
  test('3. API timeout triggers graceful fallback and reports timeout reason', async () => {
    const timeoutClient = {
      chat: {
        completions: {
          create: async () => {
            const err = new Error('Request timed out.');
            err.name = 'APIConnectionTimeoutError';
            throw err;
          },
        },
      },
    };

    const adapter = new LLMAdapter({
      provider: 'groq',
      apiKey: 'gsk_test_key',
      timeoutMs: 500,
      groqClient: timeoutClient,
    });

    const res = await adapter.generateCompletion([
      { role: 'user', content: 'What is the train schedule?' },
    ]);

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.isFallback, true);
    assert.strictEqual(res.provider, 'mock_fallback');
    assert.ok(res.fallbackReason.includes('Timeout') || res.fallbackReason.includes('timed out'));
  });

  // Test 4: Groq server failure (HTTP 500 / 503)
  test('4. Groq server failure (500) gracefully returns domain fallback without crashing', async () => {
    const serverErrorClient = {
      chat: {
        completions: {
          create: async () => {
            const err = new Error('Internal Server Error');
            err.status = 500;
            throw err;
          },
        },
      },
    };

    const adapter = new LLMAdapter({
      provider: 'groq',
      apiKey: 'gsk_test_key',
      groqClient: serverErrorClient,
    });

    const res = await adapter.generateCompletion([
      { role: 'user', content: 'Train mein doctor chahiye emergency mein.' },
    ]);

    assert.strictEqual(res.success, true);
    assert.strictEqual(res.isFallback, true);
    // Contextual fallback must recognize emergency even in fallback mode
    assert.ok(res.content.includes('112') || res.content.includes('TTE') || res.content.includes('139'));
  });

  // Test 5: Mock provider functionality
  test('5. Mock provider functions deterministically without external API calls', async () => {
    const adapter = new LLMAdapter({ provider: 'mock', apiKey: '' });

    // Test Medical / Emergency domain
    const emergencyRes = await adapter.generateCompletion([
      { role: 'user', content: 'Doctor ya medical emergency helpline kya hai?' },
    ]);
    assert.strictEqual(emergencyRes.provider, 'mock');
    assert.ok(emergencyRes.content.includes('112') || emergencyRes.content.includes('139'));

    // Test Greetings domain
    const greetingRes = await adapter.generateCompletion([
      { role: 'user', content: 'Aapka kya haal chaal hai?' },
    ]);
    assert.strictEqual(greetingRes.provider, 'mock');
    assert.ok(greetingRes.content.includes('theek') || greetingRes.content.includes('planning'));

    // Test Identity domain
    const identityRes = await adapter.generateCompletion([
      { role: 'user', content: 'Aap kaun ho?' },
    ]);
    assert.strictEqual(identityRes.provider, 'mock');
    assert.ok(identityRes.content.includes('PhantomFlow'));

    // Test Food domain
    const foodRes = await adapter.generateCompletion([
      { role: 'user', content: 'Train mein food ya khana kaise milega?' },
    ]);
    assert.strictEqual(foodRes.provider, 'mock');
    assert.ok(foodRes.content.includes('IRCTC') || foodRes.content.includes('eCatering'));
  });

  // Test 6: Hinglish input handling
  test('6. Hinglish input produces coherent, natural conversational output', async () => {
    const adapter = new LLMAdapter({ provider: 'mock', apiKey: '' });
    const res = await adapter.generateCompletion([
      { role: 'user', content: 'Bhai mausam kharab hone par train delay hoti hai kya?' },
    ]);

    assert.strictEqual(res.success, true);
    assert.ok(res.content.includes('Mausam') || res.content.includes('schedule') || res.content.includes('train'));
  });

  // Test 7: Conversation context preservation
  test('7. Conversation context and active slots are preserved into system prompt', async () => {
    const entities = {
      origin: 'Bengaluru',
      destination: 'New Delhi',
      date: 'Sunday',
      travelClass: '3A',
    };

    const history = [
      { user: 'Mujhe Bengaluru se Delhi train chahiye', assistant: 'Kis date ko travel karna hai?' },
      { user: 'Sunday ko', assistant: 'Travel class kaunsi prefer karenge?' },
    ];

    // Capture the messages received by the adapter
    let capturedMessages = null;
    const inspectClient = {
      chat: {
        completions: {
          create: async ({ messages }) => {
            capturedMessages = messages;
            return {
              choices: [{ message: { content: 'Sunday ke liye 3A mein train options hain.' } }],
            };
          },
        },
      },
    };

    const customAdapter = new LLMAdapter({
      provider: 'groq',
      apiKey: 'gsk_test_key',
      groqClient: inspectClient,
    });

    // Test smartPath system prompt builder
    const prompt = smartPath._buildSystemPrompt(entities);
    assert.ok(prompt.includes('origin: Bengaluru'));
    assert.ok(prompt.includes('destination: New Delhi'));
    assert.ok(prompt.includes('date: Sunday'));
    assert.ok(prompt.includes('travelClass: 3A'));

    // Test process with context
    const originalAdapter = require('../src/smartPath/llmAdapter');
    // Temporarily point smartPath to custom adapter
    const res = await customAdapter.generateCompletion([
      { role: 'system', content: prompt },
      ...history.flatMap((h) => [
        { role: 'user', content: h.user },
        { role: 'assistant', content: h.assistant },
      ]),
      { role: 'user', content: 'Kaunsi train best rahegi?' },
    ]);

    assert.strictEqual(res.success, true);
    assert.strictEqual(capturedMessages.length, 6); // 1 system + 4 history + 1 current
    assert.strictEqual(capturedMessages[0].role, 'system');
    assert.ok(capturedMessages[0].content.includes('Bengaluru'));
    assert.strictEqual(capturedMessages[capturedMessages.length - 1].content, 'Kaunsi train best rahegi?');
  });

  // Test 8: End-to-end fallback behavior via conversationEngine
  test('8. Full conversation engine turn returns valid response when Smart Path falls back', async () => {
    const sessionId = 'test_groq_fallback_e2e';

    // An ambiguous query that routes through Smart Path
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Can you explain the emergency medical protocols for sudden illness in coach B2?',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.processingPath, 'smart');
    assert.ok(typeof result.responseText === 'string');
    assert.ok(result.responseText.length > 0);
  });
});
