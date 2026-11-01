/**
 * LLM Provider Adapter
 * 
 * Provides an isolated, resilient interface to LLM providers:
 * - Primary: Official Groq SDK (`groq-sdk`) with ultra-fast inference
 * - Fallback: Rich, deterministic Mock Provider producing domain-aware Hinglish responses
 * - Safety: Timeout protection, sanitized error logging, and graceful fallback
 */

const Groq = require('groq-sdk');
const config = require('../config/config');

class LLMAdapter {
  /**
   * @param {Object} [options] - Optional overrides for testing
   * @param {string} [options.provider]
   * @param {string} [options.apiKey]
   * @param {string} [options.model]
   * @param {number} [options.timeoutMs]
   * @param {number} [options.maxTokens]
   * @param {number} [options.temperature]
   * @param {Object} [options.groqClient] - Mockable Groq client instance
   */
  constructor(options = {}) {
    this.provider = options.provider || config.llm.provider;
    this.apiKey = options.apiKey || config.llm.apiKey;
    this.model = options.model || config.llm.model;
    this.timeoutMs = options.timeoutMs || config.llm.timeoutMs;
    this.maxTokens = options.maxTokens || config.llm.maxTokens;
    this.temperature = options.temperature !== undefined ? options.temperature : config.llm.temperature;
    this._groqClient = options.groqClient || null;
  }

  /**
   * Factory / accessor for the official Groq client.
   * @param {string} apiKey 
   * @param {number} timeoutMs 
   * @returns {Groq}
   */
  getGroqClient(apiKey, timeoutMs) {
    if (this._groqClient) return this._groqClient;
    return new Groq({
      apiKey: apiKey || this.apiKey,
      timeout: timeoutMs || this.timeoutMs,
    });
  }

  /**
   * Send chat completion request to the configured LLM provider.
   * Seamlessly falls back to mock responses if Groq fails or times out.
   * 
   * @param {Array<{ role: string, content: string }>} messages
   * @param {Object} [overrideOptions] - Dynamic overrides for testing
   * @returns {Promise<{
   *   success: boolean,
   *   content: string,
   *   provider: string,
   *   model?: string,
   *   isFallback?: boolean,
   *   fallbackReason?: string,
   *   error?: string
   * }>}
   */
  async generateCompletion(messages, overrideOptions = {}) {
    const provider = overrideOptions.provider || this.provider || config.llm.provider;
    const apiKey = overrideOptions.apiKey !== undefined 
      ? overrideOptions.apiKey 
      : (this.apiKey !== undefined ? this.apiKey : config.llm.apiKey);
    const model = overrideOptions.model || this.model || config.llm.model;
    const timeoutMs = overrideOptions.timeoutMs || this.timeoutMs || config.llm.timeoutMs;
    const maxTokens = overrideOptions.maxTokens || this.maxTokens || config.llm.maxTokens;
    const temperature = overrideOptions.temperature !== undefined ? overrideOptions.temperature : this.temperature;

    // 1. Mock Provider or Missing API Key Fallback
    if (provider === 'mock' || !apiKey || !apiKey.trim()) {
      return this._handleMockCompletion(messages);
    }

    // 2. Groq Provider (Official Groq SDK)
    if (provider === 'groq') {
      return this._callGroq({
        messages,
        apiKey,
        model,
        timeoutMs,
        maxTokens,
        temperature,
        overrideOptions,
      });
    }

    // 3. Generic OpenAI-compatible Provider Fallback
    return this._callOpenAICompatible({
      messages,
      apiKey,
      model,
      timeoutMs,
      maxTokens,
      temperature,
    });
  }

  /**
   * Executes inference via official Groq SDK.
   * @private
   */
  async _callGroq({ messages, apiKey, model, timeoutMs, maxTokens, temperature, overrideOptions }) {
    const client = overrideOptions.groqClient || this._groqClient || this.getGroqClient(apiKey, timeoutMs);

    try {
      const chatCompletion = await client.chat.completions.create({
        messages,
        model,
        max_tokens: maxTokens,
        temperature,
      });

      const content = chatCompletion.choices?.[0]?.message?.content?.trim();
      if (!content) {
        return this._getSafeFallback(messages, 'Empty response from Groq');
      }

      return {
        success: true,
        content,
        provider: 'groq',
        model: chatCompletion.model || model,
        usage: chatCompletion.usage || null,
        isFallback: false,
      };
    } catch (err) {
      // If requested model is not found on account, attempt fallback to openai/gpt-oss-120b
      if (err.status === 404 && model !== 'openai/gpt-oss-120b') {
        try {
          console.warn(`[LLMAdapter] Model ${model} not available on Groq, attempting openai/gpt-oss-120b...`);
          const retryCompletion = await client.chat.completions.create({
            messages,
            model: 'openai/gpt-oss-120b',
            max_tokens: maxTokens,
            temperature,
          });
          const content = retryCompletion.choices?.[0]?.message?.content?.trim();
          if (content) {
            return {
              success: true,
              content,
              provider: 'groq',
              model: 'openai/gpt-oss-120b',
              usage: retryCompletion.usage || null,
              isFallback: false,
            };
          }
        } catch (retryErr) {
          console.error(`[LLMAdapter] Groq secondary model failed: ${retryErr.message}`);
        }
      }

      const isTimeout =
        err.name === 'APIConnectionTimeoutError' ||
        err.message?.includes('timed out') ||
        err.code === 'ETIMEDOUT';
      const reason = isTimeout ? `Timeout after ${timeoutMs}ms` : (err.message || 'Groq API error');

      console.error(`[LLMAdapter] Groq request failed (${reason}). Falling back to safe response.`);
      return this._getSafeFallback(messages, reason);
    }
  }

  /**
   * Fallback for standard OpenAI-compatible REST endpoints.
   * @private
   */
  async _callOpenAICompatible({ messages, apiKey, model, timeoutMs, maxTokens, temperature }) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);

    try {
      let endpoint = 'https://api.openai.com/v1/chat/completions';
      if (config.llm.baseUrl) {
        const base = config.llm.baseUrl.replace(/\/+$/, '');
        endpoint = base.endsWith('/chat/completions') ? base : `${base}/chat/completions`;
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages,
          max_tokens: maxTokens,
          temperature,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeout);

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown API error');
        console.error(`[LLMAdapter] Provider returned status ${response.status}: ${errorText}`);
        return this._getSafeFallback(messages, `Provider returned status ${response.status}`);
      }

      const data = await response.json();
      const content = data.choices?.[0]?.message?.content?.trim();

      if (!content) {
        return this._getSafeFallback(messages, 'Empty response from LLM');
      }

      return {
        success: true,
        content,
        provider: 'openai',
        model,
        isFallback: false,
      };
    } catch (err) {
      clearTimeout(timeout);
      const isTimeout = err.name === 'AbortError';
      const reason = isTimeout ? `Timeout after ${timeoutMs}ms` : err.message;
      console.error(`[LLMAdapter] OpenAI-compatible request failed (${reason})`);
      return this._getSafeFallback(messages, reason);
    }
  }

  /**
   * Deterministic mock completion for local testing or when no API key is provided.
   * Produces rich, context-aware Hinglish responses across conversational domains.
   * @param {Array<{ role: string, content: string }>} messages 
   * @returns {{ success: boolean, content: string, provider: string, isFallback: boolean }}
   */
  _handleMockCompletion(messages) {
    const lastUserMessage = [...messages].reverse().find((m) => m.role === 'user')?.content || '';
    const lower = lastUserMessage.toLowerCase();

    // 1. Health / Emergency / Medical
    if (
      lower.includes('doctor') ||
      lower.includes('medical') ||
      lower.includes('health') ||
      lower.includes('chest pain') ||
      lower.includes('emergency')
    ) {
      return {
        success: true,
        content:
          'Agar health concern ya emergency hai, toh turant 112 par call karein. Running train mein emergency ke liye TTE ko inform karein ya 139 dial karein.',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 2. Greetings & How-are-you
    if (
      lower.includes('hal') ||
      lower.includes('haal') ||
      lower.includes('kaise ho') ||
      lower.includes('kaisa hai') ||
      lower.includes('how are you') ||
      lower.includes('sup')
    ) {
      return {
        success: true,
        content: 'Main bilkul theek aur on-time hoon! Aap bataiye, aaj railway yatra ki kya planning chal rahi hai?',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 3. User Identity / What are you / Who made you
    if (
      lower.includes('tum kaun') ||
      lower.includes('aap kaun') ||
      lower.includes('who are you') ||
      lower.includes('tera naam') ||
      lower.includes('name')
    ) {
      return {
        success: true,
        content:
          'Main PhantomFlow Indian Railway voice assistant hoon. Main trains search, live status, PNR aur rules ki information Hinglish mein provide karta hoon.',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 4. Sarcasm / Casual / Srishti / Jokes
    if (
      lower.includes('pagal') ||
      lower.includes('joke') ||
      lower.includes('chutkula') ||
      lower.includes('funny') ||
      lower.includes('srishti')
    ) {
      return {
        success: true,
        content:
          'Haha, mera dhyan toh sirf Indian Railway tracks aur trains par rehta hai! Aap kis station ya train ke baare mein jaanna chahte hain?',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 5. Food / Catering / Pantry in Train
    if (
      lower.includes('food') ||
      lower.includes('khana') ||
      lower.includes('pantry') ||
      lower.includes('catering') ||
      lower.includes('meal')
    ) {
      return {
        success: true,
        content:
          'Trains mein aap IRCTC eCatering app ya 1323 par call karke apni seat par khana manga sakte hain. Vande Bharat aur Rajdhani mein catering booking time par bhi add hoti hai.',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 6. Non-Railway Transport (Flight, Bus, Taxi)
    if (
      lower.includes('flight') ||
      lower.includes('airplane') ||
      lower.includes('bus') ||
      lower.includes('hotel')
    ) {
      return {
        success: true,
        content:
          'Main specifically Indian Railway network handle karta hoon. Agar aapko train se yatra karni hai, toh please origin aur destination station batayein.',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 7. Clarifications / Objections (e.g. "maine to mention nahi kiya")
    if (
      lower.includes('mention') ||
      lower.includes('maine kab') ||
      lower.includes('galat') ||
      lower.includes('wrong') ||
      lower.includes('nahi bola')
    ) {
      return {
        success: true,
        content:
          'Maaf kijiye misunderstanding ke liye! Kripya bataiye aap kahan se kahan tak ki train dhoondh rahe hain aur kis din jaana chahte hain?',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 8. Polite Thank You
    if (lower.includes('thank') || lower.includes('shukriya') || lower.includes('dhanyawad')) {
      return {
        success: true,
        content: 'Aapka swagat hai! Kripya batayein agar kisi aur train ya PNR status ki jaankari chahiye.',
        provider: 'mock',
        isFallback: false,
      };
    }

    // 9. Weather
    if (
      lower.includes('weather') ||
      lower.includes('mausam') ||
      lower.includes('rain') ||
      lower.includes('baarish')
    ) {
      return {
        success: true,
        content:
          'Mausam ke kaaran trains ke schedule par asar pad sakta hai. Aap train number bata kar live running status verify kar sakte hain.',
        provider: 'mock',
        isFallback: false,
      };
    }

    // Default conversational reply
    return {
      success: true,
      content:
        'Aapka sawal mil gaya hai. Indian Railway train availability, ticket booking timing, ya running status ke liye details batayein.',
      provider: 'mock',
      isFallback: false,
    };
  }

  /**
   * Returns a safe, graceful fallback message along with domain-aware content.
   * Ensures the assistant never crashes or leaves the user hanging.
   * 
   * @param {Array<{ role: string, content: string }>} messages 
   * @param {string} reason 
   * @returns {{ success: boolean, content: string, provider: string, isFallback: boolean, fallbackReason: string }}
   */
  _getSafeFallback(messages, reason) {
    const mock = this._handleMockCompletion(messages);
    return {
      success: true,
      content: mock.content,
      provider: 'mock_fallback',
      isFallback: true,
      fallbackReason: reason,
    };
  }
}

module.exports = new LLMAdapter();
module.exports.LLMAdapter = LLMAdapter;
