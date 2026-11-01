/**
 * Response Generator
 * Formats verified Indian Railway data into concise, natural, voice-friendly Hinglish.
 * 
 * CRITICAL RULES:
 * 1. Never invent fake delays, platforms, seat numbers, or train names.
 * 2. If data is unavailable or unconfigured, report it honestly in natural Hinglish.
 * 3. Keep responses brief (1-3 sentences) suitable for streaming audio output.
 */

class ResponseGenerator {
  /**
   * Generate conversational prompt when required search fields are missing.
   * @param {string[]} missingFields 
   * @param {Object} currentEntities 
   * @returns {string} Natural voice prompt
   */
  generateMissingFieldsPrompt(missingFields, currentEntities) {
    if (!missingFields || missingFields.length === 0) {
      return '';
    }

    const { origin, destination } = currentEntities;

    // Both origin and destination missing
    if (missingFields.includes('origin') && missingFields.includes('destination')) {
      return 'Aapko kahan se kahan tak ki train chahiye? Please apna departure aur arrival station batayein.';
    }

    // Only destination missing
    if (missingFields.includes('destination')) {
      return `Theek hai, ${origin} se aapko kahan jaana hai? Destination station batayein.`;
    }

    // Only origin missing
    if (missingFields.includes('origin')) {
      return `${destination} ke liye aap kahan se train board karna chahte hain?`;
    }

    // Only date missing
    if (missingFields.includes('date')) {
      return `${origin} se ${destination} ke liye aap kis din travel karna chahte hain? (Jaise Saturday ya kal).`;
    }

    return 'Please apni journey ki poori details batayein.';
  }

  /**
   * Format train search results into concise, spoken Hinglish.
   * @param {Object} searchResult 
   * @param {Object} entities 
   * @param {boolean} wasCorrection 
   * @param {string|null} correctedField 
   * @returns {string} Spoken summary
   */
  formatTrainSearchResults(searchResult, entities, wasCorrection = false, correctedField = null) {
    const { origin, destination, date, time } = entities;
    const timeText = time ? ` ${time}` : '';
    let prefix = '';

    if (wasCorrection) {
      if (correctedField === 'date') {
        prefix = `Date update karke ${date} kar di gayi hai. `;
      } else if (correctedField === 'destination') {
        prefix = `Destination update karke ${destination} kar diya gaya hai. `;
      } else if (correctedField === 'origin') {
        prefix = `Origin update karke ${origin} kar diya gaya hai. `;
      } else {
        prefix = `Details update kar di gayi hain. `;
      }
    }

    // Handle Unconfigured / Error States
    if (!searchResult || !searchResult.success) {
      if (searchResult?.error?.code === 'RAILWAY_API_UNCONFIGURED') {
        return `${prefix}${origin} se ${destination} ke beech direct train search karne ke liye RAILKIT_API_KEY configure karein. Filhal live data connect nahi hai.`;
      }
      if (searchResult?.error?.message) {
        return `${prefix}${searchResult.error.message}`;
      }
      return `${prefix}${origin} se ${destination} ke beech train jaankari abhi uplabdh nahi ho paayi hai.`;
    }

    // Extract trains list (supports both RailKit SDK output and legacy schema)
    const rawData = searchResult.data || searchResult.trains || [];
    const trainsList = Array.isArray(rawData) ? rawData : (rawData.trains || rawData.data || []);

    if (!trainsList || trainsList.length === 0) {
      return `${prefix}${origin} se ${destination} ke liye ${date || 'chuni hui date'}${timeText} koi direct train nahi mili. Aap different date ya time slot try kar sakte hain.`;
    }

    const count = trainsList.length;
    const top = trainsList[0];
    const topNum = top.train_no || top.trainNumber || top.number || '';
    const topName = top.train_name || top.trainName || top.name || (topNum ? `Train ${topNum}` : 'Express');
    const topDep = top.from_time || top.departureTime || top.departure || top.src_departure_time || '';
    const seatsInfo = top.availableSeats
      ? Object.entries(top.availableSeats).map(([c, s]) => `${c}: ${s} seats`).join(', ')
      : '';

    if (count === 1) {
      const depText = topDep ? `, jo ${topDep} nikalti hai` : '';
      const seatText = seatsInfo ? `. Availability: ${seatsInfo}` : '';
      return `${prefix}${origin} se ${destination} ke liye ek direct train mili hai: ${topName} (${topNum})${depText}${seatText}.`;
    }

    const second = trainsList[1];
    const secondNum = second.train_no || second.trainNumber || second.number || '';
    const secondName = second.train_name || second.trainName || second.name || (secondNum ? `Train ${secondNum}` : 'Express');
    const secondDep = second.from_time || second.departureTime || second.departure || second.src_departure_time || '';

    const firstDepText = topDep ? ` at ${topDep}` : '';
    const secondDepText = secondDep ? ` at ${secondDep}` : '';
    const secondNumText = secondNum ? ` (${secondNum})` : '';

    return `${prefix}${origin} se ${destination} ke liye ${count} direct trains mili hain. Pehli option hai ${topName} (${topNum})${firstDepText}, aur doosri hai ${secondName}${secondNumText}${secondDepText}.`;
  }

  /**
   * Format running status response.
   * @param {Object} statusResult 
   * @returns {string}
   */
  formatTrainStatus(statusResult) {
    if (!statusResult || !statusResult.success) {
      if (statusResult?.error?.code === 'RAILWAY_API_UNCONFIGURED') {
        return 'Train ka live running status dekhne ke liye RAILKIT_API_KEY configure karein. Live tracking credentials ke bina available nahi hai.';
      }
      return statusResult?.error?.message || statusResult?.message || 'Train running status ki jaankari abhi uplabdh nahi hai.';
    }

    const raw = statusResult.data || statusResult.status || {};
    const data = (raw && typeof raw === 'object' && raw.data) ? raw.data : raw;

    // RailKit trackTrain / trackTrainV2 response normalization
    const trainNum = data.trainNo || data.trainNumber || data.train_no || data.trainNo || '';
    const trainName = data.trainName || data.train_name || 'Express';
    const statusNote = data.statusNote || data.status_note || '';
    const curStation = data.currentStationName || data.currentStation || data.currentStationCode || data.current_station_name || data.station_name || '';
    const delayMins = data.delay_in_arrival !== undefined 
      ? data.delay_in_arrival 
      : (data.delayMinutes !== undefined ? data.delayMinutes : (data.delay || 0));

    if (statusNote) {
      return `Train ${trainNum} (${trainName}): ${statusNote}.`;
    }

    let delayText = 'right time par chal rahi hai';
    if (delayMins > 0) {
      delayText = `lagbhag ${delayMins} minutes late chal rahi hai`;
    }

    const nextStation = data.next_station_name || data.nextStation || '';
    const nextStationText = nextStation ? ` Next station ${nextStation} aane wala hai.` : '';
    const stationLoc = curStation ? ` abhi ${curStation} ke paas hai aur` : '';

    return `Train ${trainNum} (${trainName})${stationLoc} ${delayText}.${nextStationText}`;
  }

  /**
   * Format PNR status response.
   * @param {Object} pnrResult 
   * @returns {string}
   */
  formatPNRStatus(pnrResult) {
    if (!pnrResult || !pnrResult.success) {
      if (pnrResult?.error?.code === 'RAILWAY_API_UNCONFIGURED') {
        return 'Real PNR status verify karne ke liye RAILKIT_API_KEY configure karein. Fake PNR data generate nahi kiya jata.';
      }
      return pnrResult?.error?.message || pnrResult?.message || 'PNR status nahi mil paaya. Kripya apna 10-digit PNR number check karein.';
    }

    const r = pnrResult.data || pnrResult.pnrRecord || {};
    const pnrNum = r.pnr_number || r.pnrNumber || r.pnr || '';
    const trainName = r.train_name || r.trainName || 'Train';
    const chartPrepared = (r.chart_prepared || r.chartPrepared) ? 'Chart ban chuka hai' : 'Chart abhi nahi bana hai';

    const passengers = r.passenger_list || r.passengers || [];
    let passText = '';
    if (passengers.length > 0) {
      const p1 = passengers[0];
      const curStatus = p1.current_status || p1.currentStatus || p1.status || 'Confirmed';
      const coach = p1.coach || p1.booking_coach ? `Coach ${p1.coach || p1.booking_coach}` : '';
      const berth = p1.berth || p1.booking_berth_no ? `Berth ${p1.berth || p1.booking_berth_no}` : '';
      passText = `Status: ${curStatus} ${coach} ${berth}.`.trim();
    }

    return `PNR ${pnrNum} (${trainName}): ${passText} ${chartPrepared}.`;
  }

  /**
   * Format seat availability response.
   * @param {Object} availResult 
   * @returns {string}
   */
  formatSeatAvailability(availResult) {
    if (!availResult || !availResult.success) {
      if (availResult?.error?.code === 'RAILWAY_API_UNCONFIGURED') {
        return 'Live seat availability check karne ke liye RAILKIT_API_KEY configure karein.';
      }
      return availResult?.error?.message || 'Seat availability jaankari abhi uplabdh nahi hai.';
    }

    const data = availResult.data || {};
    return `Train mein seat status: ${data.status || 'Available'}. Booking quota: ${data.quota || 'General'}.`;
  }

  /**
   * Format fare lookup response.
   * @param {Object} fareResult 
   * @returns {string}
   */
  formatFareLookup(fareResult) {
    if (!fareResult || !fareResult.success) {
      if (fareResult?.error?.code === 'RAILWAY_API_UNCONFIGURED') {
        return 'Ticket fare verify karne ke liye RAILKIT_API_KEY configure karein.';
      }
      return fareResult?.error?.message || 'Fare jaankari uplabdh nahi ho paayi.';
    }

    const data = fareResult.data || {};
    const total = data.totalFare || data.fare || 'Standard IRCTC rates';
    return `Is journey ka total fare lagbhag ${total} Rupees hai.`;
  }

  /**
   * Format general greetings and bot capabilities.
   * @param {string} [text]
   * @returns {string}
   */
  formatGreeting(text = '') {
    const lower = (text || '').toLowerCase();
    if (lower.includes('hal') || lower.includes('haal') || lower.includes('kaise ho') || lower.includes('how are you')) {
      return 'Namaste! Main bilkul theek aur on-time hoon. Aap bataiye, aaj railway yatra ki kya planning chal rahi hai?';
    }
    if (lower.includes('who are you') || lower.includes('tum kaun') || lower.includes('aap kaun')) {
      return 'Main PhantomFlow hoon, aapka real-time Indian Railway voice assistant. Aap mujhse train search, running status, PNR aur rules pooch sakte hain.';
    }
    if (lower.includes('thank') || lower.includes('shukriya') || lower.includes('dhanyawad')) {
      return 'Aapka swagat hai! Agar kisi aur train ya ticket ki jaankari chahiye toh zaroor batayein.';
    }
    return 'Namaste! Main PhantomFlow railway voice assistant hoon. Aap mujhse Indian Railway trains search, running status, PNR ya rules ke baare mein Hinglish ya English mein pooch sakte hain.';
  }
}

module.exports = new ResponseGenerator();
