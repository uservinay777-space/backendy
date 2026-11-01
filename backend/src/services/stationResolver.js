/**
 * Indian Railway Station Code Resolver
 * Normalizes user-supplied station names (Hindi, English, Hinglish) to canonical 2-5 letter station codes.
 * Detects ambiguous stations and prompts for clarification instead of guessing randomly.
 */

// Popular and canonical Indian Railway station mappings
const CANONICAL_STATIONS = {
  // Delhi NCR
  'delhi': { code: 'NDLS', name: 'New Delhi', aliases: ['new delhi', 'ndls', 'dilli', 'नई दिल्ली', 'दिल्ली'] },
  'old delhi': { code: 'DLI', name: 'Old Delhi Jn', aliases: ['dli', 'purani dilli', 'पुरानी दिल्ली'] },
  'hazrat nizamuddin': { code: 'NZM', name: 'Hazrat Nizamuddin', aliases: ['nizamuddin', 'nzm', 'निज़ामुद्दीन'] },
  'anand vihar': { code: 'ANVT', name: 'Anand Vihar Terminal', aliases: ['anvt', 'आनंद विहार'] },

  // Mumbai
  'mumbai': { code: 'MMCT', name: 'Mumbai Central', aliases: ['bombay', 'mmct', 'mumbai central', 'bct', 'मुम्बई'] },
  'csmt': { code: 'CSMT', name: 'Chhatrapati Shivaji Maharaj Terminus', aliases: ['cst', 'vt', 'victoria terminus', 'csmt'] },
  'bandra': { code: 'BDTS', name: 'Bandra Terminus', aliases: ['bdts'] },

  // Bengaluru
  'bengaluru': { code: 'SBC', name: 'KSR Bengaluru City', aliases: ['bangalore', 'sbc', 'ksr bengaluru', 'बैंगलोर', 'बेंगलुरु'] },
  'yesvantpur': { code: 'YPR', name: 'Yesvantpur Jn', aliases: ['ypr', 'यशवंतपुर'] },
  'smvb': { code: 'SMVB', name: 'Sir M. Visvesvaraya Terminal', aliases: ['smvb', 'baiyappanahalli'] },

  // Chennai
  'chennai': { code: 'MAS', name: 'MGR Chennai Central', aliases: ['madras', 'mas', 'chennai central', 'चेन्नई', 'मद्रास'] },
  'chennai egmore': { code: 'MS', name: 'Chennai Egmore', aliases: ['ms', 'egmore'] },

  // Kolkata
  'kolkata': { code: 'HWH', name: 'Howrah Jn', aliases: ['howrah', 'hwh', 'calcutta', 'कोलकाता', 'हावड़ा'] },
  'sealdah': { code: 'SDAH', name: 'Sealdah', aliases: ['sdah', 'सियालदह'] },

  // Other Major Junctions
  'patna': { code: 'PNBE', name: 'Patna Jn', aliases: ['pnbe', 'पटना'] },
  'pune': { code: 'PUNE', name: 'Pune Jn', aliases: ['pune', 'पूना', 'पुणे'] },
  'hyderabad': { code: 'HYB', name: 'Hyderabad Deccan', aliases: ['hyb', 'हैदराबाद'] },
  'secunderabad': { code: 'SC', name: 'Secunderabad Jn', aliases: ['sc', 'सिकंदराबाद'] },
  'ahmedabad': { code: 'ADI', name: 'Ahmedabad Jn', aliases: ['adi', 'अहमदाबाद'] },
  'jaipur': { code: 'JP', name: 'Jaipur Jn', aliases: ['jp', 'जयपुर'] },
  'lucknow': { code: 'LKO', name: 'Lucknow Charbagh', aliases: ['lko', 'लखनऊ'] },
  'chandigarh': { code: 'CDG', name: 'Chandigarh', aliases: ['cdg', 'चंडीगढ़'] },
  'varanasi': { code: 'BSB', name: 'Varanasi Jn', aliases: ['bsb', 'banaras', 'kashi', 'वाराणसी', 'बनारस'] },
  'kanpur': { code: 'CNB', name: 'Kanpur Central', aliases: ['cnb', 'कानपुर'] },
  'bhopal': { code: 'BPL', name: 'Bhopal Jn', aliases: ['bpl', 'भोपाल'] },
  'surat': { code: 'ST', name: 'Surat', aliases: ['st', 'सूरत'] },
  'nagpur': { code: 'NGP', name: 'Nagpur Jn', aliases: ['ngp', 'नागपुर'] },
  'amritsar': { code: 'ASR', name: 'Amritsar Jn', aliases: ['asr', 'अमृतसर'] },
  'agra': { code: 'AGC', name: 'Agra Cantt', aliases: ['agc', 'आगरा'] },
  'goa': { code: 'MAO', name: 'Madgaon (Goa)', aliases: ['madgaon', 'mao', 'vasco', 'गोवा', 'मडगांव'] },
  'gorakhpur': { code: 'GKP', name: 'Gorakhpur Jn', aliases: ['gkp', 'गोरखपुर'] },
  'prayagraj': { code: 'PRYJ', name: 'Prayagraj Jn', aliases: ['pryj', 'allahabad', 'प्रयागराज', 'इलाहाबाद'] },
  'gwalior': { code: 'GWL', name: 'Gwalior Jn', aliases: ['gwl', 'ग्वालियर'] },
  'indore': { code: 'INDB', name: 'Indore Jn', aliases: ['indb', 'इंदौर'] },
  'guwahati': { code: 'GHY', name: 'Guwahati', aliases: ['ghy', 'गुवाहाटी'] },
  'ranchi': { code: 'RNC', name: 'Ranchi Jn', aliases: ['rnc', 'राँची', 'रांची'] },
  'bhubaneswar': { code: 'BBS', name: 'Bhubaneswar', aliases: ['bbs', 'भुवनेश्वर'] },
  'dehradun': { code: 'DDN', name: 'Dehradun', aliases: ['ddn', 'देहरादून'] },
  'jammu': { code: 'JAT', name: 'Jammu Tawi', aliases: ['jat', 'jammu tawi', 'जम्मू'] },
  'coimbatore': { code: 'CBE', name: 'Coimbatore Jn', aliases: ['cbe', 'कोयंबटूर'] },
  'kochi': { code: 'ERS', name: 'Ernakulam Jn (Kochi)', aliases: ['ers', 'ernakulam', 'cochin', 'कोच्चि'] },
  'trivandrum': { code: 'TVC', name: 'Thiruvananthapuram Central', aliases: ['tvc', 'thiruvananthapuram', 'त्रिवेंद्रम'] },
};

// Build fast lookup map (both lowercase aliases and uppercase codes)
const LOOKUP_MAP = new Map();

for (const [key, station] of Object.entries(CANONICAL_STATIONS)) {
  LOOKUP_MAP.set(key.toLowerCase(), station);
  LOOKUP_MAP.set(station.code.toUpperCase(), station);
  for (const alias of station.aliases) {
    LOOKUP_MAP.set(alias.toLowerCase(), station);
  }
}

/**
 * Resolve a user-supplied station string into an official station code and name.
 * @param {string} input 
 * @returns {{
 *   resolved: boolean,
 *   stationCode: string | null,
 *   stationName: string | null,
 *   isAmbiguous: boolean,
 *   suggestedClarification?: string
 * }}
 */
function resolveStation(input) {
  if (!input || typeof input !== 'string') {
    return { resolved: false, stationCode: null, stationName: null, isAmbiguous: false };
  }

  const clean = input.trim();
  const lower = clean.toLowerCase();
  const upper = clean.toUpperCase();

  // 1. Direct match in canonical map
  if (LOOKUP_MAP.has(lower)) {
    const station = LOOKUP_MAP.get(lower);
    return {
      resolved: true,
      stationCode: station.code,
      stationName: station.name,
      isAmbiguous: false,
    };
  }

  // 2. Exact 2-5 letter station code pattern (e.g. NDLS, SBC, MAS, PNBE)
  if (/^[A-Z]{2,5}$/.test(upper)) {
    return {
      resolved: true,
      stationCode: upper,
      stationName: upper,
      isAmbiguous: false,
    };
  }

  // 3. Substring / Prefix match
  for (const [key, station] of LOOKUP_MAP.entries()) {
    if (key.length >= 3 && (lower.includes(key) || key.includes(lower))) {
      return {
        resolved: true,
        stationCode: station.code,
        stationName: station.name,
        isAmbiguous: false,
      };
    }
  }

  // 4. Ambiguity detection
  if (lower.includes('station') || lower.includes('jn') || lower.includes('cantt')) {
    const baseWord = lower.replace(/\b(station|jn|cantt|railway|city|terminal)\b/g, '').trim();
    if (LOOKUP_MAP.has(baseWord)) {
      const station = LOOKUP_MAP.get(baseWord);
      return {
        resolved: true,
        stationCode: station.code,
        stationName: station.name,
        isAmbiguous: false,
      };
    }
  }

  return {
    resolved: false,
    stationCode: null,
    stationName: clean,
    isAmbiguous: true,
    suggestedClarification: `Station '${clean}' ka code verify nahi ho paya. Kripya railway station ka poora naam ya station code batayein.`,
  };
}

module.exports = {
  CANONICAL_STATIONS,
  resolveStation,
};
