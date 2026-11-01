/**
 * Railway Rules & Deterministic Knowledge Base
 * Answers common Indian Railway informational inquiries in < 5ms without LLM overhead.
 */

const RAILWAY_RULES_DATA = [
  {
    topic: 'tatkal',
    keywords: ['tatkal', 'tatkal timing', 'tatkal booking', 'tatkal time', 'tatkal charge'],
    response: 'Tatkal booking timings: AC classes ke liye subah 10:00 AM aur Non-AC (Sleeper) ke liye subah 11:00 AM booking shuru hoti hai, journey date se ek din pehle.',
  },
  {
    topic: 'cancellation',
    keywords: ['cancellation', 'refund', 'cancel charges', 'cancel ticket', 'refund rules'],
    response: 'Ticket cancellation charges: 48 ghante pehle cancel karne par flat charges lagte hain (1A: ₹240, 2A: ₹200, 3A: ₹180, Sleeper: ₹120). 48 se 12 ghante ke beech 25% aur 12 se 4 ghante ke beech 50% katauti hoti hai. Chart banne ke baad cancellation nahi hota.',
  },
  {
    topic: 'luggage',
    keywords: ['luggage', 'baggage', 'weight limit', 'kitna saman', 'luggage allowance'],
    response: 'Free luggage allowance: AC 1st Class mein 70 kg, AC 2-Tier mein 50 kg, AC 3-Tier/Chair Car mein 40 kg, aur Sleeper Class mein 40 kg tak luggage bina extra charges le jaa sakte hain.',
  },
  {
    topic: 'chart_preparation',
    keywords: ['chart', 'chart preparation', 'charting time', 'chart kab banta hai'],
    response: 'Indian Railways mein first reservation chart train departure se 4 ghante pehle banta hai. Second chart train departure se 30 minutes pehle prepare hota hai.',
  },
  {
    topic: 'platform_ticket',
    keywords: ['platform ticket', 'platform fee', 'platform price'],
    response: 'Platform ticket ki validity aamtaur par 2 ghante hoti hai aur iski price station ke hisaab se ₹10 se ₹30 hoti hai.',
  },
];

/**
 * Match a deterministic rule based on user query
 * @param {string} text 
 * @returns {string|null} Rule response or null if not found
 */
function findRuleAnswer(text) {
  if (!text) return null;
  const lower = text.toLowerCase();

  for (const rule of RAILWAY_RULES_DATA) {
    if (rule.keywords.some((kw) => lower.includes(kw))) {
      return rule.response;
    }
  }

  return null;
}

module.exports = {
  findRuleAnswer,
  RAILWAY_RULES_DATA,
};
