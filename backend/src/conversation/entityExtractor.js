/**
 * Entity Extractor
 * Extracts railway entities (origin, destination, date, time, trainNumber, pnr)
 * across English, Hindi, and Hinglish.
 * Accurately detects corrections and computes missing fields.
 */

// Canonical Station Mapping (Normalizes slang / variants / Hindi names)
const STATION_CANONICAL_MAP = {
  // Bengaluru
  bengaluru: 'Bengaluru',
  bangalore: 'Bengaluru',
  blore: 'Bengaluru',
  bengalur: 'Bengaluru',
  बेंगलुरु: 'Bengaluru',
  बैंगलोर: 'Bengaluru',

  // Chennai
  chennai: 'Chennai',
  madras: 'Chennai',
  चेन्नई: 'Chennai',
  मद्रास: 'Chennai',

  // Delhi
  delhi: 'New Delhi',
  'new delhi': 'New Delhi',
  dilli: 'New Delhi',
  दिल्ली: 'New Delhi',
  'नई दिल्ली': 'New Delhi',

  // Mumbai
  mumbai: 'Mumbai',
  bombay: 'Mumbai',
  csmt: 'Mumbai',
  बंबई: 'Mumbai',
  मुंबई: 'Mumbai',

  // Pune
  pune: 'Pune',
  poona: 'Pune',
  पुणे: 'Pune',

  // Kolkata
  kolkata: 'Kolkata',
  calcutta: 'Kolkata',
  howrah: 'Kolkata',
  sealdah: 'Kolkata',
  कोलकाता: 'Kolkata',

  // Hyderabad
  hyderabad: 'Hyderabad',
  secunderabad: 'Hyderabad',
  हैदराबाद: 'Hyderabad',

  // Lucknow
  lucknow: 'Lucknow',
  लखनऊ: 'Lucknow',

  // Jaipur
  jaipur: 'Jaipur',
  जयपुर: 'Jaipur',

  // Ahmedabad
  ahmedabad: 'Ahmedabad',
  अहमदाबाद: 'Ahmedabad',

  // Other major Indian rail hubs
  patna: 'Patna',
  chandigarh: 'Chandigarh',
  varanasi: 'Varanasi',
  bhopal: 'Bhopal',
  kanpur: 'Kanpur',
  amritsar: 'Amritsar',
  surat: 'Surat',
  agra: 'Agra',
  nagpur: 'Nagpur',
  goa: 'Goa',
};

// Common non-railway countries/places that users might ask about
const NON_RAILWAY_PLACES = [
  'america', 'usa', 'united states', 'london', 'uk', 'dubai', 'uae', 'paris',
  'canada', 'singapore', 'australia', 'japan', 'china', 'germany'
];

// Reserved words that must never be treated as station names
const STOPWORDS = new Set([
  'train', 'trains', 'ticket', 'tickets', 'booking', 'destination', 'origin',
  'from', 'to', 'se', 'ko', 'mein', 'chahiye', 'jaana', 'jana', 'hai', 'karna',
  'change', 'actually', 'today', 'tomorrow', 'saturday', 'sunday', 'monday',
  'morning', 'evening', 'subah', 'shaam', 'night', 'flight', 'bus', 'station',
  'please', 'mujhe', 'humko', 'kahan', 'batao', 'status', 'pnr', 'time',
  'mention', 'travel', 'trip', 'leave', 'reach', 'arrive', 'depart', 'book',
  'check', 'show', 'tell', 'know', 'help', 'think', 'where', 'when', 'what',
  'which', 'there', 'here', 'with', 'about', 'some', 'any', 'none', 'make',
  'give', 'take', 'karni', 'karke', 'karte', 'kya', 'nahi', 'nahin', 'pata',
  'maloom', 'bhi', 'hi', 'per', 'par', 'aur', 'ek', 'do', 'shukriya', 'dhanyawad',
  'kaun', 'kaisa', 'kaise', 'kuch', 'kisi', 'liye', 'suno', 'bolo', 'bol',
  'ट्रेन', 'टिकट', 'चाहिए', 'जाना', 'है'
]);

// Days of the week mapping (English & Hindi)
const DAYS_OF_WEEK_MAP = {
  monday: 'Monday',
  somvar: 'Monday',
  सोमवार: 'Monday',

  tuesday: 'Tuesday',
  mangalwar: 'Tuesday',
  मंगलवार: 'Tuesday',

  wednesday: 'Wednesday',
  budhwar: 'Wednesday',
  बुधवार: 'Wednesday',

  thursday: 'Thursday',
  guruwar: 'Thursday',
  brihaspatiwar: 'Thursday',
  गुरुवार: 'Thursday',

  friday: 'Friday',
  shukrawar: 'Friday',
  शुक्रवार: 'Friday',

  saturday: 'Saturday',
  shanivar: 'Saturday',
  शनिवार: 'Saturday',

  sunday: 'Sunday',
  ravivar: 'Sunday',
  itwar: 'Sunday',
  रविवार: 'Sunday',
  इतवार: 'Sunday',

  // Relative dates
  today: 'Today',
  aaj: 'Today',
  आज: 'Today',

  tomorrow: 'Tomorrow',
  kal: 'Tomorrow',
  कल: 'Tomorrow',

  'day after tomorrow': 'Day After Tomorrow',
  parso: 'Day After Tomorrow',
  परसों: 'Day After Tomorrow',
};

// Time slot mapping
const TIME_SLOT_MAP = {
  morning: 'morning',
  subah: 'morning',
  सुबह: 'morning',

  afternoon: 'afternoon',
  dopahar: 'afternoon',
  दोपहर: 'afternoon',

  evening: 'evening',
  shaam: 'evening',
  शाम: 'evening',

  night: 'night',
  raat: 'night',
  रात: 'night',
};

/**
 * Normalizes station names to their canonical title.
 * Rejects common stopwords and non-station verbs.
 * @param {string} stationRaw
 * @returns {string|null}
 */
function normalizeStation(stationRaw) {
  if (!stationRaw) return null;
  const clean = stationRaw.trim().toLowerCase();
  if (STOPWORDS.has(clean)) return null;

  if (STATION_CANONICAL_MAP[clean]) {
    return STATION_CANONICAL_MAP[clean];
  }

  // Preserve non-railway countries for explicit error messaging
  if (NON_RAILWAY_PLACES.includes(clean)) {
    return clean.charAt(0).toUpperCase() + clean.slice(1);
  }

  // Proper noun check ONLY if at least 4 letters, capitalized in input or Devanagari, and not an English dictionary verb
  if (clean.length >= 4 && /^[a-zA-Z\u0900-\u097F]+$/.test(clean)) {
    return clean.charAt(0).toUpperCase() + clean.slice(1);
  }

  return null;
}

/**
 * Extracts entities from a user utterance.
 * @param {string} text
 * @param {Object} existingContext - Already established session entities
 * @returns {{
 *   entities: { origin?: string, destination?: string, date?: string, time?: string, trainNumber?: string, pnr?: string },
 *   isCorrection: boolean,
 *   correctedField: string|null,
 *   missingFields: string[]
 * }}
 */
function extractEntities(text, existingContext = {}) {
  if (!text || typeof text !== 'string') {
    return {
      entities: {},
      isCorrection: false,
      correctedField: null,
      missingFields: [],
    };
  }

  const trimmed = text.trim();
  const lower = trimmed.toLowerCase();

  const extracted = {};
  let isCorrection = false;
  let correctedField = null;

  // 1. Detect Explicit Correction markers
  const correctionMarkers = ['actually', 'nahi', 'change to', 'instead of', 'badal do', 'change destination'];
  if (correctionMarkers.some((m) => lower.includes(m))) {
    isCorrection = true;
  }

  // 2. Extract Train Number (5 digits)
  const trainNumberMatch = trimmed.match(/\b\d{5}\b/);
  if (trainNumberMatch) {
    extracted.trainNumber = trainNumberMatch[0];
  }

  // 3. Extract PNR Number (10 digits)
  const pnrMatch = trimmed.match(/\b\d{10}\b/);
  if (pnrMatch) {
    extracted.pnr = pnrMatch[0];
  }

  // 3b. Extract Travel Class
  const classPatterns = [
    { pattern: /\b(1st\s*ac|first\s*ac|1a)\b/i, value: '1A' },
    { pattern: /\b(2nd\s*ac|second\s*ac|2a)\b/i, value: '2A' },
    { pattern: /\b(3rd\s*ac|third\s*ac|3a|3e)\b/i, value: '3A' },
    { pattern: /\b(sleeper|sl)\b/i, value: 'SL' },
    { pattern: /\b(chair\s*car|cc)\b/i, value: 'CC' },
    { pattern: /\b(exec\s*chair|ec)\b/i, value: 'EC' },
    { pattern: /\b(second\s*seating|2s)\b/i, value: '2S' },
  ];
  for (const { pattern, value } of classPatterns) {
    if (pattern.test(lower)) {
      extracted.travelClass = value;
      break;
    }
  }

  // 3c. Extract Quota
  if (lower.includes('tatkal')) {
    extracted.quota = 'TQ';
  } else if (lower.includes('ladies') || lower.includes('mahila')) {
    extracted.quota = 'LD';
  } else if (lower.includes('senior citizen') || lower.includes('buzurg')) {
    extracted.quota = 'SS';
  } else if (lower.includes('general')) {
    extracted.quota = 'GN';
  }

  // 4. Extract Travel Date with Negation Handling ("Saturday ke liye nahi mujhe Sunday chahiye")
  const negatedDays = new Set();
  for (const [key, val] of Object.entries(DAYS_OF_WEEK_MAP)) {
    const negPattern = new RegExp(`(?:nahi|nahin|not|mat|instead of)\\s+(?:ke\\s+liye\\s+|ko\\s+)?${key}|${key}\\s+(?:ke\\s+liye\\s+|ko\\s+)?(?:nahi|nahin|not|mat)`, 'i');
    if (negPattern.test(lower)) {
      negatedDays.add(val);
    }
  }

  for (const [key, val] of Object.entries(DAYS_OF_WEEK_MAP)) {
    const regex = new RegExp(`(?:^|\\s)${key}(?:$|\\s|[.,!?])`, 'i');
    if (regex.test(lower) && !negatedDays.has(val)) {
      extracted.date = val;
      if (existingContext.date && existingContext.date !== val) {
        isCorrection = true;
        correctedField = 'date';
      }
      break;
    }
  }

  // 5. Extract Time Slot
  for (const [key, val] of Object.entries(TIME_SLOT_MAP)) {
    const regex = new RegExp(`(?:^|\\s)${key}(?:$|\\s|[.,!?])`, 'i');
    if (regex.test(lower)) {
      extracted.time = val;
      if (existingContext.time && existingContext.time !== val) {
        isCorrection = true;
        correctedField = 'time';
      }
      break;
    }
  }

  // 6. Extract Origin and Destination
  // Detect known station mentions
  const detectedStations = [];
  for (const [variant, canonical] of Object.entries(STATION_CANONICAL_MAP)) {
    const regex = new RegExp(`(?:^|\\s)${variant}(?:$|\\s|[.,!?])`, 'i');
    if (regex.test(lower)) {
      if (!detectedStations.some((s) => s.canonical === canonical)) {
        const match = lower.match(regex);
        detectedStations.push({
          canonical,
          index: match ? match.index : -1,
        });
      }
    }
  }
  detectedStations.sort((a, b) => a.index - b.index);

  // Pattern A: "from X to Y" or "X to Y" (e.g., "Bangalore to Delhi train")
  const pairToMatch = lower.match(/(?:from\s+)?([a-zA-Z\u0900-\u097F]+)\s+to\s+([a-zA-Z\u0900-\u097F]+)/i);
  if (pairToMatch) {
    const p1 = normalizeStation(pairToMatch[1]);
    const p2 = normalizeStation(pairToMatch[2]);
    if (p1 && p2) {
      extracted.origin = p1;
      extracted.destination = p2;
    }
  }

  // Pattern B: Hinglish "X se Y" or Hindi "X से Y" (e.g. "Kolkata se Delhi", "Bengaluru se Chennai")
  if (!extracted.origin || !extracted.destination) {
    const seMatch = lower.match(/([a-zA-Z\u0900-\u097F]+)\s+(?:se|से)\s+([a-zA-Z\u0900-\u097F]+)/i);
    if (seMatch) {
      const p1 = normalizeStation(seMatch[1]);
      const p2 = normalizeStation(seMatch[2]);
      if (p1 && p2) {
        extracted.origin = p1;
        extracted.destination = p2;
      }
    }
  }

  // Pattern C: If two distinct known stations were found in order
  if (!extracted.origin && !extracted.destination && detectedStations.length >= 2) {
    extracted.origin = detectedStations[0].canonical;
    extracted.destination = detectedStations[1].canonical;
  }

  // Pattern D: Single station extractions
  if (!extracted.destination) {
    // Explicit destination markers
    const destExplicitMatch = lower.match(/(?:change destination to|destination to|heading to|travel to|go to)\s+([a-zA-Z\u0900-\u097F]+)/i);
    if (destExplicitMatch) {
      const norm = normalizeStation(destExplicitMatch[1]);
      if (norm) {
        extracted.destination = norm;
        if (existingContext.destination && existingContext.destination !== norm) {
          isCorrection = true;
          correctedField = 'destination';
        }
      }
    } else {
      // Bare "to X" ONLY IF X is an explicitly recognized station
      const bareToMatch = lower.match(/\bto\s+([a-zA-Z\u0900-\u097F]+)/i);
      if (bareToMatch && STATION_CANONICAL_MAP[bareToMatch[1].toLowerCase()]) {
        extracted.destination = STATION_CANONICAL_MAP[bareToMatch[1].toLowerCase()];
        if (existingContext.destination && existingContext.destination !== extracted.destination) {
          isCorrection = true;
          correctedField = 'destination';
        }
      }
    }
  }

  // If only 1 known station in detectedStations
  if (!extracted.origin && !extracted.destination && detectedStations.length === 1) {
    const single = detectedStations[0].canonical;
    if (lower.includes(`${single.toLowerCase()} se`) || lower.includes(`${single.toLowerCase()} से`) || lower.includes(`from ${single.toLowerCase()}`)) {
      extracted.origin = single;
    } else if (lower.includes(`to ${single.toLowerCase()}`) || lower.includes(`${single.toLowerCase()} ke liye`) || lower.includes(`${single.toLowerCase()} jaana`) || lower.includes(`${single.toLowerCase()} जाना`)) {
      extracted.destination = single;
    } else if (existingContext.origin && !existingContext.destination) {
      extracted.destination = single;
    } else if (!existingContext.origin) {
      extracted.origin = single;
    } else if (isCorrection) {
      extracted.destination = single;
    }
  }

  // Check if destination was corrected
  if (extracted.destination && existingContext.destination && extracted.destination !== existingContext.destination) {
    isCorrection = true;
    correctedField = 'destination';
  }

  // Check if origin was corrected
  if (extracted.origin && existingContext.origin && extracted.origin !== existingContext.origin) {
    isCorrection = true;
    correctedField = 'origin';
  }

  // 7. Calculate Missing Fields for Railway Search
  const combined = { ...existingContext, ...extracted };
  const missingFields = [];
  if (!combined.origin) missingFields.push('origin');
  if (!combined.destination) missingFields.push('destination');
  if (!combined.date) missingFields.push('date');

  return {
    entities: extracted,
    isCorrection,
    correctedField,
    missingFields,
  };
}

module.exports = {
  extractEntities,
  normalizeStation,
  STATION_CANONICAL_MAP,
  DAYS_OF_WEEK_MAP,
  TIME_SLOT_MAP,
};
