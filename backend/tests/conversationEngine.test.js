/**
 * Conversation Engine Comprehensive Test Suite
 * 
 * Verifies all 11 required scenarios:
 * 1. Basic English railway queries.
 * 2. Hindi railway queries.
 * 3. Hinglish railway queries.
 * 4. Missing destination prompt.
 * 5. Missing travel date prompt.
 * 6. User correcting Saturday to Sunday (SwitchSync).
 * 7. User changing destination.
 * 8. Follow-up questions using existing context.
 * 9. Separate session isolation.
 * 10. Fast Lane versus Smart Path routing.
 * 11. LLM API failure fallback.
 */

const { test, describe, beforeEach } = require('node:test');
const assert = require('node:assert');
const conversationEngine = require('../src/conversation/conversationEngine');
const contextManager = require('../src/conversation/contextManager');
const llmAdapter = require('../src/smartPath/llmAdapter');

describe('PHANTOMFLOW Conversation Engine Test Suite', () => {
  beforeEach(async () => {
    // Clear session store between test runs for repeatability
    await contextManager.store.clear();
  });

  // Test 1: Basic English railway queries
  test('1. Basic English railway query extracts all entities and routes to Fast Lane', async () => {
    const sessionId = 'test_english_01';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Find trains from Bengaluru to Chennai on Saturday morning',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.intent, 'railway_search');
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, 'Chennai');
    assert.strictEqual(result.entities.date, 'Saturday');
    assert.strictEqual(result.entities.time, 'morning');
    assert.deepStrictEqual(result.missingFields, []);
    assert.strictEqual(result.processingPath, 'fast');
    assert.ok(result.response.includes('Bengaluru'));
    assert.ok(result.response.includes('Chennai'));
  });

  // Test 2: Hindi railway queries (Devanagari script)
  test('2. Hindi railway query in Devanagari detects Hindi and extracts slots', async () => {
    const sessionId = 'test_hindi_02';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'बेंगलुरु से चेन्नई के लिए शनिवार की ट्रेन चाहिए',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.intent, 'railway_search');
    assert.strictEqual(result.language, 'hindi');
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, 'Chennai');
    assert.strictEqual(result.entities.date, 'Saturday');
    assert.deepStrictEqual(result.missingFields, []);
  });

  // Test 3: Hinglish railway queries
  test('3. Hinglish railway query with fluid code-mixing resolves correctly', async () => {
    const sessionId = 'test_hinglish_03';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Mujhe Bengaluru se Chennai jaana hai Saturday morning mein',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.intent, 'railway_search');
    assert.strictEqual(result.language, 'hinglish');
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, 'Chennai');
    assert.strictEqual(result.entities.date, 'Saturday');
    assert.strictEqual(result.entities.time, 'morning');
    assert.deepStrictEqual(result.missingFields, []);
    assert.strictEqual(result.processingPath, 'fast');
  });

  // Test 4: Missing destination
  test('4. Incomplete query with missing destination prompts user specifically for destination', async () => {
    const sessionId = 'test_missing_dest_04';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Mujhe Bengaluru se train chahiye Saturday ko',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.intent, 'railway_search');
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, null);
    assert.ok(result.missingFields.includes('destination'));
    assert.strictEqual(result.processingPath, 'fast');
    // Verifies prompt explicitly asks for destination
    assert.ok(result.response.toLowerCase().includes('destination') || result.response.toLowerCase().includes('kahan'));
  });

  // Test 5: Missing travel date
  test('5. Incomplete query with missing travel date prompts user for date', async () => {
    const sessionId = 'test_missing_date_05';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Bengaluru se Chennai ki train chahiye',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.intent, 'railway_search');
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, 'Chennai');
    assert.strictEqual(result.entities.date, null);
    assert.deepStrictEqual(result.missingFields, ['date']);
    assert.strictEqual(result.processingPath, 'fast');
    // Verifies prompt explicitly asks for date
    assert.ok(result.response.toLowerCase().includes('din') || result.response.toLowerCase().includes('travel'));
  });

  // Test 6: User correcting Saturday to Sunday (SwitchSync)
  test('6. SwitchSync: User correcting Saturday to Sunday updates date while preserving origin and destination', async () => {
    const sessionId = 'test_correction_06';

    // Turn 1: Initial query
    const turn1 = await conversationEngine.processMessage({
      sessionId,
      text: 'Mujhe Bengaluru se Chennai jaana hai Saturday morning mein',
    });
    assert.strictEqual(turn1.entities.date, 'Saturday');
    assert.strictEqual(turn1.entities.origin, 'Bengaluru');
    assert.strictEqual(turn1.entities.destination, 'Chennai');

    // Turn 2: Correction
    const turn2 = await conversationEngine.processMessage({
      sessionId,
      text: 'Actually Sunday ko jaana hai',
    });

    assert.strictEqual(turn2.success, true);
    assert.strictEqual(turn2.intent, 'railway_search');
    assert.strictEqual(turn2.entities.date, 'Sunday', 'Date was not updated to Sunday!');
    assert.strictEqual(turn2.entities.origin, 'Bengaluru', 'Origin was wiped out during correction!');
    assert.strictEqual(turn2.entities.destination, 'Chennai', 'Destination was wiped out during correction!');
    assert.strictEqual(turn2.entities.time, 'morning', 'Time was wiped out during correction!');
    assert.deepStrictEqual(turn2.missingFields, []);
    assert.ok(turn2.response.includes('Sunday'));
  });

  // Test 7: User changing destination
  test('7. SwitchSync: User changing destination updates destination while preserving origin and date', async () => {
    const sessionId = 'test_dest_change_07';

    // Turn 1: Mumbai trip
    await conversationEngine.processMessage({
      sessionId,
      text: 'Mujhe Delhi se Mumbai jaana hai Saturday',
    });

    // Turn 2: User changes mind to Pune
    const turn2 = await conversationEngine.processMessage({
      sessionId,
      text: 'Change destination to Pune',
    });

    assert.strictEqual(turn2.success, true);
    assert.strictEqual(turn2.entities.origin, 'New Delhi', 'Origin was lost!');
    assert.strictEqual(turn2.entities.destination, 'Pune', 'Destination was not updated to Pune!');
    assert.strictEqual(turn2.entities.date, 'Saturday', 'Date was lost!');
  });

  // Test 8: Follow-up questions using existing context
  test('8. Multi-turn slot accumulation: user provides missing date in follow-up turn', async () => {
    const sessionId = 'test_multiturn_08';

    // Turn 1: User provides origin and destination only
    const turn1 = await conversationEngine.processMessage({
      sessionId,
      text: 'Bengaluru se Chennai train ticket book karni hai',
    });
    assert.deepStrictEqual(turn1.missingFields, ['date']);

    // Turn 2: User provides date and time in follow-up
    const turn2 = await conversationEngine.processMessage({
      sessionId,
      text: 'Saturday morning',
    });

    assert.strictEqual(turn2.success, true);
    assert.strictEqual(turn2.entities.origin, 'Bengaluru');
    assert.strictEqual(turn2.entities.destination, 'Chennai');
    assert.strictEqual(turn2.entities.date, 'Saturday');
    assert.strictEqual(turn2.entities.time, 'morning');
    assert.deepStrictEqual(turn2.missingFields, []);
    assert.ok(turn2.response.includes('Vande Bharat') || turn2.response.includes('Shatabdi') || turn2.response.includes('trains') || turn2.response.includes('Chennai'));
  });

  // Test 9: Separate session isolation
  test('9. Session Isolation: distinct sessions do not leak entities or context', async () => {
    const sessionA = 'user_session_A';
    const sessionB = 'user_session_B';

    // User A searches Bengaluru to Chennai
    await conversationEngine.processMessage({
      sessionId: sessionA,
      text: 'Bengaluru se Chennai train Saturday',
    });

    // User B searches Delhi to Mumbai
    await conversationEngine.processMessage({
      sessionId: sessionB,
      text: 'Delhi se Mumbai train Sunday',
    });

    const dataA = await conversationEngine.getSession(sessionA);
    const dataB = await conversationEngine.getSession(sessionB);

    assert.strictEqual(dataA.entities.origin, 'Bengaluru');
    assert.strictEqual(dataA.entities.destination, 'Chennai');
    assert.strictEqual(dataA.entities.date, 'Saturday');

    assert.strictEqual(dataB.entities.origin, 'New Delhi');
    assert.strictEqual(dataB.entities.destination, 'Mumbai');
    assert.strictEqual(dataB.entities.date, 'Sunday');
  });

  // Test 10: Fast Lane vs Smart Path routing
  test('10. Fast Lane resolves predictable railway FAQ in < 30ms; complex reasoning routes to Smart Path', async () => {
    const sessionId = 'test_routing_10';

    // A. Predictable Railway FAQ -> Fast Lane
    const fastResult = await conversationEngine.processMessage({
      sessionId,
      text: 'Tatkal booking timing kya hai?',
    });

    assert.strictEqual(fastResult.intent, 'railway_information');
    assert.strictEqual(fastResult.processingPath, 'fast');
    assert.ok(fastResult.response.includes('10:00 AM'));
    assert.ok(fastResult.latencyMs < 30, `Fast lane latency (${fastResult.latencyMs}ms) exceeded 30ms`);

    // B. Complex medical/unstructured inquiry -> Smart Path
    const smartResult = await conversationEngine.processMessage({
      sessionId,
      text: 'Can you advise me which doctor to consult for chest pain during travel?',
    });

    assert.strictEqual(smartResult.processingPath, 'smart');
    assert.ok(smartResult.response.length > 10);
  });

  // Test 11: LLM API failure fallback
  test('11. Smart Path gracefully falls back when external LLM API throws or fails', async () => {
    const sessionId = 'test_llm_fallback_11';

    // Temporarily stub generateCompletion to simulate network/API crash
    const originalGenerate = llmAdapter.generateCompletion;
    llmAdapter.generateCompletion = async () => {
      return {
        success: false,
        content: 'Maaf kijiye, abhi network issue ke kaaran main information process nahi kar pa raha hoon. Kripya thodi der baad dobara koshish karein ya seedhe apna origin aur destination station batayein.',
        provider: 'groq',
        error: 'Simulated API Network Timeout (504 Gateway Timeout)',
      };
    };

    try {
      const result = await conversationEngine.processMessage({
        sessionId,
        text: 'What are the complex philosophical reasons to take a train?',
      });

      assert.strictEqual(result.success, true);
      assert.strictEqual(result.processingPath, 'smart');
      assert.ok(result.response.includes('Maaf kijiye') || result.response.includes('network issue'));
    } finally {
      // Restore original function
      llmAdapter.generateCompletion = originalGenerate;
    }
  });

  // Test 12: Negative date expression
  test('12. Date negation: "Saturday ke liye nahin mujhe Sunday ke liye chahie" resolves to Sunday', async () => {
    const sessionId = 'test_negation_12';
    // Turn 1: Initial query
    await conversationEngine.processMessage({
      sessionId,
      text: 'Mujhe Bengaluru se Chennai train chahiye Saturday ko',
    });

    // Turn 2: Correction with negation
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Saturday ke liye nahin mujhe Sunday ke liye chahie',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.entities.date, 'Sunday');
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, 'Chennai');
    assert.ok(result.response.includes('Sunday'));
    assert.ok(!result.response.includes('Saturday kar di gayi hai'));
  });

  // Test 13: X to Y pair extraction
  test('13. Station pair: "Bangalore to Delhi train" extracts both origin and destination', async () => {
    const sessionId = 'test_pair_13';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Bangalore to Delhi train',
    });

    assert.strictEqual(result.success, true);
    assert.strictEqual(result.entities.origin, 'Bengaluru');
    assert.strictEqual(result.entities.destination, 'New Delhi');
    assert.ok(result.missingFields.includes('date'));
  });

  // Test 14: Non-railway destination
  test('14. Non-railway destination: "Kolkata se America" explains lack of train connectivity', async () => {
    const sessionId = 'test_foreign_14';
    const result = await conversationEngine.processMessage({
      sessionId,
      text: 'Mujhe Kolkata se America jana hai train bataen',
    });

    assert.strictEqual(result.success, true);
    assert.ok(result.response.includes('America'));
    assert.ok(result.response.includes('connectivity nahi hai') || result.response.includes('domestic'));
  });

  // Test 15: Chit-chat & Objections
  test('15. Conversational check-ins and objections receive natural contextual replies', async () => {
    const sessionId = 'test_chitchat_15';
    const res1 = await conversationEngine.processMessage({
      sessionId,
      text: 'kya hal Chaal',
    });
    assert.strictEqual(res1.success, true);
    assert.ok(res1.response.length > 5);

    const res2 = await conversationEngine.processMessage({
      sessionId,
      text: 'per Maine to mention hi nahin kiya',
    });
    assert.strictEqual(res2.success, true);
    const textLower = res2.response.toLowerCase();
    assert.ok(
      textLower.includes('misunderstanding') ||
      textLower.includes('kahan') ||
      textLower.includes('origin') ||
      textLower.includes('destination') ||
      textLower.includes('journey') ||
      textLower.includes('madad') ||
      textLower.includes('help') ||
      textLower.includes('train') ||
      textLower.includes('station') ||
      textLower.includes('jaankari') ||
      textLower.includes('jankari') ||
      textLower.includes('railway') ||
      textLower.includes('yatra')
    );
  });
});
