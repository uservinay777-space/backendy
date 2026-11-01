/**
 * Smart Path Controller
 * 
 * Invoked when queries are ambiguous, complex, or require general conversational reasoning.
 * Injects session context and active entities into the system prompt for context-aware responses.
 */

const llmAdapter = require('./llmAdapter');

class SmartPath {
  /**
   * Execute reasoning via LLM adapter.
   * @param {Object} params
   * @param {string} params.text - Current user utterance
   * @param {Object} params.entities - Current session entities
   * @param {Array<Object>} params.history - Prior conversation turns
   * @returns {Promise<{ response: string, processingPath: 'smart' }>}
   */
  async process({ text, entities, history = [] }) {
    const systemPrompt = this._buildSystemPrompt(entities);

    const messages = [
      { role: 'system', content: systemPrompt },
    ];

    // Append last few conversation history turns for conversational context
    const recentHistory = history.slice(-4);
    for (const turn of recentHistory) {
      if (turn.user) messages.push({ role: 'user', content: turn.user });
      if (turn.assistant) messages.push({ role: 'assistant', content: turn.assistant });
    }

    // Append current turn
    messages.push({ role: 'user', content: text });

    const completion = await llmAdapter.generateCompletion(messages);

    return {
      response: completion.content,
      responseText: completion.content,
      processingPath: 'smart',
      provider: completion.provider,
      isFallback: Boolean(completion.isFallback),
    };
  }

  /**
   * Build persona system prompt with injected context slots.
   * @param {Object} entities 
   * @returns {string} System prompt
   */
  _buildSystemPrompt(entities) {
    const activeSlots = Object.entries(entities)
      .filter(([_, v]) => v !== null && v !== undefined)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ') || 'None';

    return `You are PhantomFlow, a friendly, ultra-fast Indian Railway voice assistant.
You speak naturally in conversational Hinglish (a fluid blend of Hindi and English written in Latin script).

ACTIVE SESSION CONTEXT:
Known journey entities: [${activeSlots}]

CORE GUIDELINES:
1. Keep your spoken responses concise (1 to 3 sentences maximum) suitable for audio voice output.
2. If the user is asking about trains, guide them with natural suggestions.
3. If information like origin, destination, or date is missing, ask for it politely.
4. If the user corrects any detail, acknowledge the change seamlessly.
5. FACTUAL INTEGRITY: You are NOT the source of railway facts. NEVER invent or hallucinate train schedules, delays, platform numbers, PNR records, or seat availability. If verified data is not in the context, politely guide the user to provide their train number or station name.
6. Tone: Helpful, courteous, Indian conversational cadence, natural Hinglish.`;
  }
}

module.exports = new SmartPath();
