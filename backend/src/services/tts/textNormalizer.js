/**
 * Text Normalizer for Voice Speech Synthesis
 * Pre-processes conversational railway responses for natural Hinglish pronunciation in Sarvam / ElevenLabs.
 * 
 * Expands:
 * - 5-digit train numbers (e.g. 12301 -> "1 2 3 0 1")
 * - 10-digit PNRs (separated digits)
 * - Platform numbers
 * - Currency symbols (₹ -> "Rupees" or "Rupaye")
 * - Time formats ("06:00 AM" -> "subah 6 baje", "10:45 PM" -> "raat 10:45")
 * - Common station code abbreviations (e.g. "NDLS" -> "New Delhi", "SBC" -> "Bengaluru City")
 * 
 * Preserves underlying facts without alteration.
 */

const { CANONICAL_STATIONS } = require('../stationResolver');

// Map of station codes to spoken names
const STATION_CODE_TO_SPOKEN = {};
for (const [_, station] of Object.entries(CANONICAL_STATIONS)) {
  STATION_CODE_TO_SPOKEN[station.code] = station.name;
}

/**
 * Normalizes text for human-like spoken Hinglish.
 * @param {string} text
 * @returns {string}
 */
function normalizeForSpeech(text) {
  if (!text || typeof text !== 'string') return '';

  let normalized = text;

  // 1. Remove Markdown formatting (*, _, `, #, ~, [])
  normalized = normalized
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/__(.*?)__/g, '$1')
    .replace(/_(.*?)_/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/^#+\s+/gm, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

  // 2. Currency symbol normalization
  normalized = normalized
    .replace(/₹\s*(\d+(?:\.\d+)?)/g, '$1 Rupees')
    .replace(/Rs\.?\s*(\d+(?:\.\d+)?)/gi, '$1 Rupees');

  // 3. Platform number formatting
  normalized = normalized
    .replace(/\bPF\s*#?\s*(\d+)\b/gi, 'Platform number $1')
    .replace(/\bPlatform\s*#?\s*(\d+)\b/gi, 'Platform number $1');

  // 4. PNR number formatting (10-digit spaced out for digit-by-digit clarity)
  normalized = normalized.replace(/\bPNR\s*(?:number\s*)?(\d{10})\b/gi, (_, pnr) => {
    return `PNR ${pnr.split('').join(' ')}`;
  });

  // 5. Train number formatting (5 digits spoken digit-by-digit)
  // Match "Train 12301" or "Train number 12301" or standalone 5-digit numbers after keywords
  normalized = normalized.replace(/\b(?:train|train number)\s*(\d{5})\b/gi, (_, num) => {
    return `Train number ${num.split('').join(' ')}`;
  });

  // 6. Time normalization (12-hour AM/PM to conversational Hinglish)
  // E.g. "06:00 AM" -> "subah 6 baje", "08:30 PM" -> "raat 8:30 baje"
  normalized = normalized.replace(/\b0?([1-9]|1[0-2]):(00)\s*(AM|am)\b/g, 'subah $1 baje');
  normalized = normalized.replace(/\b0?([1-9]|1[0-2]):([0-5][0-9])\s*(AM|am)\b/g, 'subah $1:$2');
  normalized = normalized.replace(/\b0?([1-5]):(00)\s*(PM|pm)\b/g, 'dopahar $1 baje');
  normalized = normalized.replace(/\b0?([1-5]):([0-5][0-9])\s*(PM|pm)\b/g, 'dopahar $1:$2');
  normalized = normalized.replace(/\b0?([6-9]|1[0-1]):(00)\s*(PM|pm)\b/g, 'shaam $1 baje');
  normalized = normalized.replace(/\b0?([6-9]|1[0-1]):([0-5][0-9])\s*(PM|pm)\b/g, 'shaam $1:$2');
  normalized = normalized.replace(/\b12:(00)\s*(PM|pm)\b/g, 'dopahar 12 baje');
  normalized = normalized.replace(/\b12:(00)\s*(AM|am)\b/g, 'raat 12 baje');

  // 7. Delay formatting (e.g. "20m late" -> "20 minutes late")
  normalized = normalized.replace(/\b(\d+)\s*(?:mins?|m)\s+late\b/gi, '$1 minutes late');

  // 8. Station code expansion in brackets: e.g. "New Delhi (NDLS)" -> "New Delhi"
  normalized = normalized.replace(/\((NDLS|SBC|MAS|HWH|CSMT|MMCT|PNBE|PUNE|HYB|BSB)\)/gi, '');

  // 9. Clean excess whitespace and newlines
  normalized = normalized.replace(/\s+/g, ' ').trim();

  return normalized;
}

module.exports = {
  normalizeForSpeech,
};
