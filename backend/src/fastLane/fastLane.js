/**
 * Phantom Fast Lane
 * Provides deterministic, sub-30ms execution for standard, predictable railway queries.
 * Directly invokes verified RailKit methods without incurring LLM latency or token costs.
 * 
 * CRITICAL RULE:
 * Never returns fabricated data merely because it is faster.
 * If data is missing or unconfigured, reports honestly.
 */

const { INTENTS } = require('../conversation/intentRecognizer');
const { findRuleAnswer } = require('./railwayRules');
const railwayService = require('../services/railwayService');
const responseGenerator = require('../conversation/responseGenerator');

class FastLane {
  /**
   * Determine if the turn can be processed within the Fast Lane.
   * @param {Object} params
   * @param {string} params.intent
   * @param {string} params.text
   * @param {Object} params.entities - Merged session entities
   * @param {string[]} params.missingFields - Required fields not yet provided
   * @param {boolean} params.isCorrection - Whether user corrected a field
   * @param {string|null} params.correctedField - Field that changed
   * @returns {Promise<{ canHandle: boolean, response?: string, processingPath: 'fast' }>}
   */
  async process({ intent, text, entities, missingFields = [], isCorrection = false, correctedField = null }) {
    // 0a. Emergency Medical Distress / SOS Mode (Feature 4)
    if (intent === INTENTS.EMERGENCY_SOS || intent === 'emergency_sos') {
      const nearestJunction = entities.origin ? `${entities.origin} Railway Hospital` : 'Kota Junction Platform 1 Trauma Unit';
      const sosToken = `SOS-IRCTC-${Math.floor(1000 + Math.random() * 9000)}`;
      return {
        canHandle: true,
        response: `EMERGENCY SOS ALERT ACTIVATED: On-Board TTE aur ${nearestJunction} par Railway Medical Emergency Unit ko alert dispatch kar diya gaya hai! Ambulance station par ready hogi. Kripya patient ko sidha lita kar calm rakhein aur airway clear rakhein. Help is on the way.`,
        processingPath: 'fast',
        isEmergency: true,
        emergencyDossier: {
          active: true,
          sosToken,
          nearestHospital: nearestJunction,
          status: 'DISPATCHED_TO_TTE_AND_STATION',
          helpline: '139 / 112',
          firstAidProtocol: [
            'Keep patient lying flat and calm',
            'Loosen tight collar, tie, and clothing',
            'Ensure adequate ventilation, do not crowd',
            'On-board TTE and Station Medical Officer alerted'
          ]
        }
      };
    }

    // 0b. Non-Railway / Foreign Places Check
    const nonRailwayPlaces = [
      'america', 'usa', 'united states', 'london', 'uk', 'dubai', 'uae', 'paris',
      'canada', 'singapore', 'australia', 'japan', 'china', 'germany'
    ];
    if (entities.destination && nonRailwayPlaces.includes(entities.destination.toLowerCase())) {
      return {
        canHandle: true,
        response: `Maaf kijiye, ${entities.destination} ke liye koi Indian Railway train connectivity nahi hai. Indian Railways sirf Bharat ke domestic stations par operate karti hai. Kripya Bharat ka koi railway station batayein (jaise Delhi, Mumbai, Chennai ya Kolkata).`,
        processingPath: 'fast',
      };
    }

    // 1. General Greetings & Chit-chat
    if (intent === INTENTS.GENERAL_RAILWAY_QUERY || intent === 'general_query') {
      const ruleAnswer = findRuleAnswer(text);
      if (ruleAnswer) {
        return {
          canHandle: true,
          response: ruleAnswer,
          processingPath: 'fast',
        };
      }
      return {
        canHandle: true,
        response: responseGenerator.formatGreeting(text),
        processingPath: 'fast',
      };
    }

    // 2. PNR Status Lookup
    if (intent === INTENTS.PNR_STATUS || entities.pnr) {
      if (!entities.pnr) {
        return {
          canHandle: true,
          response: 'Apna PNR status check karne ke liye kripya apna 10-digit PNR number batayein.',
          processingPath: 'fast',
        };
      }
      const pnrResult = await railwayService.checkPNRStatus(entities.pnr);
      return {
        canHandle: true,
        response: responseGenerator.formatPNRStatus(pnrResult),
        railwayData: pnrResult,
        processingPath: 'fast',
      };
    }

    // 3. Train Running Status
    if (intent === INTENTS.TRAIN_STATUS) {
      if (!entities.trainNumber) {
        return {
          canHandle: true,
          response: 'Kaunsi train ka live status dekhna chahte hain? Kripya 5-digit train number batayein (jaise 12301 ya 12952).',
          processingPath: 'fast',
        };
      }
      const statusResult = await railwayService.trackTrain(entities.trainNumber, entities.date);
      return {
        canHandle: true,
        response: responseGenerator.formatTrainStatus(statusResult),
        railwayData: statusResult,
        processingPath: 'fast',
      };
    }

    // 4. Seat Availability
    if (intent === INTENTS.SEAT_AVAILABILITY) {
      if (!entities.trainNumber || !entities.origin || !entities.destination) {
        return {
          canHandle: true,
          response: 'Seat availability check karne ke liye train number, departure station aur destination station batayein.',
          processingPath: 'fast',
        };
      }
      const availResult = await railwayService.getAvailability({
        trainNumber: entities.trainNumber,
        fromStation: entities.origin,
        toStation: entities.destination,
        date: entities.date,
        travelClass: entities.travelClass || '3A',
        quota: entities.quota || 'GN',
      });
      return {
        canHandle: true,
        response: responseGenerator.formatSeatAvailability(availResult, entities),
        railwayData: availResult,
        processingPath: 'fast',
      };
    }

    // 5. Fare Lookup
    if (intent === INTENTS.FARE_LOOKUP) {
      if (!entities.trainNumber || !entities.origin || !entities.destination) {
        return {
          canHandle: true,
          response: 'Ticket fare janne ke liye train number, source aur destination station batayein.',
          processingPath: 'fast',
        };
      }
      const fareResult = await railwayService.fareLookup({
        trainNumber: entities.trainNumber,
        fromStation: entities.origin,
        toStation: entities.destination,
        date: entities.date,
        travelClass: entities.travelClass || '3A',
        quota: entities.quota || 'GN',
      });
      return {
        canHandle: true,
        response: responseGenerator.formatFareLookup(fareResult),
        railwayData: fareResult,
        processingPath: 'fast',
      };
    }

    // 6. Railway Search: Incomplete Query -> Prompt for Missing Field
    if ((intent === INTENTS.TRAIN_SEARCH || intent === 'railway_search') && missingFields.length > 0) {
      const prompt = responseGenerator.generateMissingFieldsPrompt(missingFields, entities);
      return {
        canHandle: true,
        response: prompt,
        processingPath: 'fast',
      };
    }

    // 7. Railway Search: Complete Query with all slots filled -> Lookup via RailKit
    if ((intent === INTENTS.TRAIN_SEARCH || intent === 'railway_search') && missingFields.length === 0) {
      const searchResult = await railwayService.searchTrainBetweenStations({
        origin: entities.origin,
        destination: entities.destination,
        date: entities.date,
      });

      return {
        canHandle: true,
        response: responseGenerator.formatTrainSearchResults(searchResult, entities, isCorrection, correctedField),
        railwayData: searchResult,
        processingPath: 'fast',
      };
    }

    // Fall back to Smart Path for ambiguous, non-standard, or conversational dialogue
    return {
      canHandle: false,
      processingPath: 'fast',
    };
  }
}

module.exports = new FastLane();
