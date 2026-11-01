/**
 * Intent Recognizer
 * Classifies user utterances into Indian Railway intents across English, Hindi, and Hinglish.
 * 
 * Supported Intents:
 * - TRAIN_STATUS: Live train location, running delay, on-time status
 * - PNR_STATUS: Passenger booking status, coach, berth, chart preparation
 * - TRAIN_SEARCH: Finding trains between stations, travel planning
 * - TRAIN_INFO: Train route, stoppages, train name/number lookup
 * - STATION_STATUS: Upcoming trains at station, platform numbers
 * - FARE_LOOKUP: Ticket prices, ticket fare breakdown
 * - SEAT_AVAILABILITY: Seat availability, waiting list, confirmation chances
 * - TIMETABLE: Schedule, arrival/departure timings
 * - GENERAL_RAILWAY_QUERY: Rules (Tatkal, refund, luggage), greetings, FAQs
 * - UNKNOWN: Open-ended, conversational objections, or non-railway questions
 */

const INTENTS = {
  TRAIN_SEARCH: 'railway_search',
  TRAIN_STATUS: 'train_status',
  PNR_STATUS: 'pnr_status',
  TRAIN_INFO: 'train_info',
  STATION_STATUS: 'station_status',
  FARE_LOOKUP: 'fare_lookup',
  SEAT_AVAILABILITY: 'seat_availability',
  TIMETABLE: 'timetable',
  GENERAL_RAILWAY_QUERY: 'railway_information',
  GENERAL_QUERY: 'general_query',
  EMERGENCY_SOS: 'emergency_sos',
  UNKNOWN: 'unknown',

  // Backward compatibility aliases
  RAILWAY_SEARCH: 'railway_search',
  RAILWAY_INFORMATION: 'railway_information',
};

// Language detection patterns
const DEVANAGARI_REGEX = /[\u0900-\u097F]/;
const HINDI_ROMAN_KEYWORDS = [
  'mujhe', 'chahiye', 'jaana', 'jana', 'hai', 'karna', 'karni', 'batao', 'bataiye',
  'kahan', 'pahuchi', 'pahunchi', 'kitna', 'kitne', 'raat', 'subah', 'shaam', 'dopahar',
  'aaj', 'kal', 'parso', 'mein', 'se', 'ko', 'chal', 'rahi', 'kya', 'namaste', 'shukriya',
  'actually', 'nahi', 'change', 'kiraya', 'seat', 'berth', 'dibba'
];

/**
 * Detect language of utterance
 * @param {string} text 
 * @returns {'hindi'|'english'|'hinglish'}
 */
function detectLanguage(text) {
  if (!text) return 'english';

  if (DEVANAGARI_REGEX.test(text)) {
    return 'hindi';
  }

  const lower = text.toLowerCase();
  const words = lower.split(/\s+/);
  const hindiCount = words.filter((w) => HINDI_ROMAN_KEYWORDS.includes(w)).length;

  if (hindiCount >= 2 || (hindiCount >= 1 && words.length <= 4)) {
    return 'hinglish';
  }

  return 'english';
}

/**
 * Analyze dynamic code-mixing ratio, dialect nuance, and normalized tokens (Feature 3: Bhasha Spectrum).
 * @param {string} text
 */
function analyzeBhashaSpectrum(text) {
  if (!text) {
    return {
      hindiPercent: 50,
      englishPercent: 50,
      dialectNuance: 'Metro Hinglish',
      script: 'Roman',
      normalizedTokens: [],
    };
  }

  const isDevanagari = DEVANAGARI_REGEX.test(text);
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const totalWords = words.length || 1;

  const hindiMatches = words.filter(w => HINDI_ROMAN_KEYWORDS.includes(w) || DEVANAGARI_REGEX.test(w));
  let hindiPercent = Math.round((hindiMatches.length / totalWords) * 100);
  if (isDevanagari) hindiPercent = Math.max(hindiPercent, 88);
  if (hindiMatches.length === 0 && !isDevanagari) hindiPercent = 12;
  if (hindiPercent > 88) hindiPercent = 88;
  if (hindiPercent < 12) hindiPercent = 12;
  const englishPercent = 100 - hindiPercent;

  let dialectNuance = 'Metro Hinglish';
  const lower = text.toLowerCase();
  if (isDevanagari) {
    dialectNuance = 'Standard Devanagari';
  } else if (/ka\s*ho|baani|humko|kahan\s*ba|kabo/.test(lower)) {
    dialectNuance = 'Regional Bhojpuri / Purvanchal';
  } else if (/apun|kidhar\s*ko|bantai|khopche/.test(lower)) {
    dialectNuance = 'Bambaiya Colloquial';
  } else if (/actually|schedule|confirm|booking|timing|delay/.test(lower) && hindiPercent > 20) {
    dialectNuance = 'Metro Hinglish (Bilingual Urban)';
  } else if (englishPercent >= 80) {
    dialectNuance = 'English Conversational';
  }

  // Token Normalizations
  const normalizedTokens = [];
  const trainMatch = text.match(/\b\d{4,5}\b/);
  if (trainMatch) {
    normalizedTokens.push({ original: 'Spoken Digits', normalized: `Train #${trainMatch[0]}` });
  }
  const pnrMatch = text.match(/\b\d{10}\b/);
  if (pnrMatch) {
    normalizedTokens.push({ original: '10-Digit Utterance', normalized: `PNR: ${pnrMatch[0]}` });
  }
  if (/₹|\b(rupees|rs|rupya|rupaye)\b/i.test(text)) {
    normalizedTokens.push({ original: 'Currency', normalized: 'INR ₹ Standardized' });
  }

  return {
    hindiPercent,
    englishPercent,
    dialectNuance,
    script: isDevanagari ? 'Devanagari' : 'Latin/Romanized',
    normalizedTokens,
  };
}

/**
 * Classify user utterance intent
 * @param {string} text - User's input text
 * @param {Object} contextEntities - Current session entities for follow-ups
 * @returns {{ intent: string, confidence: number, language: string, isEmergency?: boolean, bhashaSpectrum: Object }}
 */
function recognizeIntent(text, contextEntities = {}) {
  if (!text || typeof text !== 'string') {
    return {
      intent: INTENTS.UNKNOWN,
      confidence: 0,
      language: 'english',
      bhashaSpectrum: analyzeBhashaSpectrum(''),
    };
  }

  const trimmed = text.trim();
  const lower = trimmed.toLowerCase();
  const language = detectLanguage(trimmed);
  const bhashaSpectrum = analyzeBhashaSpectrum(trimmed);

  // If user is asking for general consultation, explanation, protocols, or medical advice, let Smart Path handle it
  const isAdviceOrInquiry = /advise|consult|which doctor|recommend|suggestion|what should i do|kya karu|explain|protocol|procedure|rules|guideline/i.test(lower);

  // 0a. Emergency Distress / SOS Detection (Feature 4: Red Alert Hotline)
  const isExplicitEmergency = /\b(sos|emergency|urgent medical|doctor bulao|ambulance|accident|railmadad)\b/i.test(lower);
  const isCriticalDistress = /\b(behosh|unconscious|bleeding|khoon beh|saans nahi|heart attack)\b/i.test(lower);

  if (!isAdviceOrInquiry && (isExplicitEmergency || isCriticalDistress)) {
    return {
      intent: INTENTS.EMERGENCY_SOS,
      confidence: 0.99,
      language,
      isEmergency: true,
      bhashaSpectrum,
    };
  }

  // 0b. Exclude non-emergency medical / non-railway / general chit-chat queries from railway routing
  const complexNonRailwayKeywords = [
    'doctor', 'hospital', 'medicine', 'chest pain',
    'fever', 'philosophical', 'philosophy', 'poem', 'joke', 'weather',
    'hotel', 'restaurant', 'flight', 'airplane', 'srishti', 'pagal'
  ];
  if (complexNonRailwayKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.UNKNOWN, confidence: 0.9, language, bhashaSpectrum };
  }

  // Conversational objections & corrections (e.g. "maine to mention hi nahi kiya")
  const objectionKeywords = [
    'mention', 'maine kab', 'maine to nahi', 'nahi bola', 'galat', 'wrong', 'kisne bola'
  ];
  if (objectionKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.UNKNOWN, confidence: 0.9, language, bhashaSpectrum };
  }

  // 1. PNR STATUS INTENT
  // E.g. "Mera PNR check karo", "PNR status kya hai", "Check PNR 1234567890"
  if (
    lower.includes('pnr') ||
    /\b\d{10}\b/.test(trimmed) ||
    lower.includes('पीएनआर')
  ) {
    return { intent: INTENTS.PNR_STATUS, confidence: 0.98, language };
  }

  // 2. FARE LOOKUP INTENT
  // E.g. "How much is the fare?", "Kiraya kitna hai", "Ticket price kya hai", "3rd AC fare"
  const fareKeywords = [
    'fare', 'price', 'ticket price', 'kiraya', 'ticket kitne ka', 'cost', 'charge',
    'किराया', 'टिकट का दाम', 'दाम'
  ];
  if (fareKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.FARE_LOOKUP, confidence: 0.95, language };
  }

  // 3. SEAT AVAILABILITY INTENT
  // E.g. "Seat available hai kya?", "Waiting list kitni hai", "Seat confirmation chances"
  const seatKeywords = [
    'seat available', 'availability', 'seats', 'berth available', 'seat hai kya',
    'waiting list', 'wl status', 'rac', 'confirm hogi kya', 'सीट उपलब्ध', 'सीट'
  ];
  if (seatKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.SEAT_AVAILABILITY, confidence: 0.95, language };
  }

  // 4. LIVE TRAIN RUNNING STATUS INTENT
  // E.g. "Train 12301 kahan pahuchi?", "Where is train right now?", "Is train running late?"
  const statusKeywords = [
    'running status', 'status', 'kahan pahuchi', 'kahan hai', 'late hai kya',
    'delay', 'on time', 'where is train', 'train status', 'live status',
    'स्थिति', 'कहाँ पहुँची', 'कहाँ है', 'लेट है क्या'
  ];
  if (
    statusKeywords.some((kw) => lower.includes(kw)) ||
    (/train\s*\d{4,5}/i.test(lower) && (lower.includes('status') || lower.includes('kahan') || lower.includes('late')))
  ) {
    return { intent: INTENTS.TRAIN_STATUS, confidence: 0.95, language };
  }

  // 5. STATION STATUS / LIVE AT STATION INTENT
  // E.g. "Platform number kya hai?", "Station par kaunsi train aane wali hai", "New Delhi par upcoming trains"
  const stationStatusKeywords = [
    'platform number', 'platform kya hai', 'platform par', 'upcoming trains', 'station par',
    'platform no', 'प्लेटफॉर्म नंबर', 'प्लेटफॉर्म'
  ];
  if (stationStatusKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.STATION_STATUS, confidence: 0.93, language };
  }

  // 6. TIMETABLE & SCHEDULE INTENT
  // E.g. "Time table batao", "Departure time kya hai", "Kab aayegi train", "Schedule"
  const timetableKeywords = [
    'timetable', 'time table', 'schedule', 'kab nikalti hai', 'departure time',
    'arrival time', 'kab aayegi', 'kab pahuchegi', 'समय सारिणी'
  ];
  if (timetableKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.TIMETABLE, confidence: 0.92, language };
  }

  // 7. TRAIN INFO (Route, stoppages, general train lookup)
  // E.g. "Rajdhani express route", "Train 12301 info", "Stoppages kya hain"
  const infoKeywords = [
    'train info', 'route', 'stoppages', 'stops', 'kahan rukti hai', 'route kya hai',
    'train details'
  ];
  if (infoKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.TRAIN_INFO, confidence: 0.9, language };
  }

  // 8. GENERAL RAILWAY QUERY & RULES (Tatkal, Cancellation, Baggage, Senior Citizen)
  const ruleKeywords = [
    'tatkal', 'cancellation', 'refund', 'baggage', 'luggage', 'rules',
    'senior citizen', 'concession', 'chart preparation', 'charting', 'charges',
    'tatkal timing', 'cancellation charges', 'kitna refund', 'rules kya hai',
    'तत्काल', 'रिफंड', 'नियम'
  ];
  if (ruleKeywords.some((kw) => lower.includes(kw))) {
    return { intent: INTENTS.GENERAL_RAILWAY_QUERY, confidence: 0.92, language };
  }

  // 9. GREETINGS & CHIT-CHAT
  const greetingKeywords = [
    'hello', 'hi', 'hey', 'namaste', 'namaskar', 'pranam', 'kaise ho', 'kaisa hai',
    'kya hal', 'kya haal', 'how are you', 'sup',
    'good morning', 'good evening', 'who are you', 'tum kaun ho', 'aap kaun ho',
    'kya kar sakte ho', 'what can you do', 'thank you', 'shukriya', 'dhanyawad',
    'नमस्ते', 'नमस्कार', 'प्रणाम'
  ];
  const tokenWords = lower.split(/[\s,!?.]+/);
  const isPureGreeting = greetingKeywords.some((kw) => {
    if (kw.includes(' ')) {
      return lower === kw || lower.startsWith(kw + ' ') || lower.endsWith(' ' + kw) || lower.includes(' ' + kw + ' ');
    }
    return tokenWords.includes(kw);
  });
  const hasTravelTerms = lower.includes('train') || lower.includes('jaana') || lower.includes('ticket') ||
    lower.includes('sunday') || lower.includes('saturday') || lower.includes('monday') || lower.includes('delhi') ||
    lower.includes('chennai') || lower.includes('bengaluru') || lower.includes('mumbai') || lower.includes('kolkata');

  if (isPureGreeting && !hasTravelTerms) {
    return { intent: INTENTS.GENERAL_RAILWAY_QUERY, confidence: 0.95, language };
  }

  // 10. TRAIN SEARCH INTENT (Find trains, ticket search, travel)
  // E.g. "Delhi se Patna train batao", "Mujhe Bengaluru se Chennai jaana hai"
  const searchKeywords = [
    'train', 'trains', 'jaana', 'jana', 'ticket', 'tickets', 'booking', 'chahiye',
    'bengaluru', 'chennai', 'mumbai', 'delhi', 'pune', 'kolkata', 'hyderabad',
    'patna', 'varanasi', 'chandigarh',
    'ट्रेन', 'टिकट', 'जाना', 'चाहिए', 'सफर', 'रेल'
  ];
  const matchesSearch = searchKeywords.some((kw) => lower.includes(kw));

  // Context-aware follow-up or correction (e.g. "Actually Sunday", "Saturday morning", "Change destination to Pune")
  const isCorrectionOrFollowUp = (
    (contextEntities.origin || contextEntities.destination) &&
    (
      lower.includes('actually') || lower.includes('change') ||
      lower.includes('sunday') || lower.includes('saturday') || lower.includes('monday') ||
      lower.includes('tuesday') || lower.includes('wednesday') || lower.includes('thursday') ||
      lower.includes('friday') || lower.includes('morning') || lower.includes('evening') ||
      lower.includes('subah') || lower.includes('shaam') ||
      lower.includes('destination') || lower.includes('pune') || lower.includes('mumbai')
    )
  );

  if (matchesSearch || isCorrectionOrFollowUp) {
    return { intent: INTENTS.TRAIN_SEARCH, confidence: 0.9, language };
  }

  // Default fallback
  return { intent: INTENTS.UNKNOWN, confidence: 0.4, language, bhashaSpectrum };
}

module.exports = {
  INTENTS,
  recognizeIntent,
  detectLanguage,
  analyzeBhashaSpectrum,
};
