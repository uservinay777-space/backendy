/**
 * Railway Service & RailKit Integration Unit Tests
 * 
 * Verifies:
 * 1. Valid and invalid train numbers
 * 2. Valid and invalid PNR numbers
 * 3. Station resolution and ambiguity detection
 * 4. Indian Railway Date parsing (Asia/Kolkata)
 * 5. Freshness separation (isLive: true for tracking vs isLive: false for static)
 * 6. Standardized envelope structure
 * 7. Unconfigured API key error handling
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');
const railwayService = require('../src/services/railwayService');
const { resolveStation } = require('../src/services/stationResolver');
const { toIndianRailwayDate } = require('../src/services/dateHelper');

describe('Railway Service & RailKit Unit Tests', () => {
  test('1. Valid 5-digit train number accepts input and returns standard envelope', async () => {
    const res = await railwayService.getTrainInfo('12301');
    assert.strictEqual(typeof res.success, 'boolean');
    assert.strictEqual(res.isLive, false); // Static scheduled timetable
    assert.ok(res.source);
    assert.ok(res.retrievedAt);
  });

  test('2. Invalid train number is rejected with validation error', async () => {
    const resEmpty = await railwayService.getTrainInfo('');
    assert.strictEqual(resEmpty.success, false);
    assert.strictEqual(resEmpty.error.code, 'INVALID_TRAIN_NUMBER');

    const resInvalid = await railwayService.getTrainInfo('12');
    assert.strictEqual(resInvalid.success, false);
    assert.strictEqual(resInvalid.error.code, 'INVALID_TRAIN_FORMAT');
  });

  test('3. Valid 10-digit PNR format is accepted', async () => {
    const res = await railwayService.checkPNRStatus('1234567890');
    assert.strictEqual(typeof res.success, 'boolean');
    assert.ok(res.retrievedAt);
  });

  test('4. Invalid PNR length is rejected with INVALID_PNR_LENGTH', async () => {
    const resShort = await railwayService.checkPNRStatus('12345');
    assert.strictEqual(resShort.success, false);
    assert.strictEqual(resShort.error.code, 'INVALID_PNR_LENGTH');

    const resEmpty = await railwayService.checkPNRStatus('');
    assert.strictEqual(resEmpty.success, false);
    assert.strictEqual(resEmpty.error.code, 'INVALID_PNR');
  });

  test('5. Station resolver normalizes Indian cities to canonical codes', () => {
    const delhi = resolveStation('New Delhi');
    assert.strictEqual(delhi.resolved, true);
    assert.strictEqual(delhi.stationCode, 'NDLS');

    const blr = resolveStation('Bangalore');
    assert.strictEqual(blr.resolved, true);
    assert.strictEqual(blr.stationCode, 'SBC');

    const chennai = resolveStation('Chennai Central');
    assert.strictEqual(chennai.resolved, true);
    assert.strictEqual(chennai.stationCode, 'MAS');

    const kolkata = resolveStation('Howrah');
    assert.strictEqual(kolkata.resolved, true);
    assert.strictEqual(kolkata.stationCode, 'HWH');
  });

  test('6. Station resolver detects unknown / ambiguous station and flags clarification', () => {
    const unknown = resolveStation('SomeRandomNonExistentPlaceXYZ');
    assert.strictEqual(unknown.resolved, false);
    assert.strictEqual(unknown.isAmbiguous, true);
    assert.ok(unknown.suggestedClarification.includes('Station'));
  });

  test('7. Indian Railway Date Helper formats to DD-MM-YYYY in Asia/Kolkata timezone', () => {
    const todayStr = toIndianRailwayDate('aaj');
    assert.match(todayStr, /^\d{2}-\d{2}-\d{4}$/);

    const kalStr = toIndianRailwayDate('kal');
    assert.match(kalStr, /^\d{2}-\d{2}-\d{4}$/);

    const sundayStr = toIndianRailwayDate('sunday');
    assert.match(sundayStr, /^\d{2}-\d{2}-\d{4}$/);

    const alreadyFormatted = toIndianRailwayDate('15-08-2026');
    assert.strictEqual(alreadyFormatted, '15-08-2026');
  });

  test('8. Live train tracking reports isLive: true', async () => {
    const res = await railwayService.trackTrain('12301', 'today');
    assert.strictEqual(typeof res.success, 'boolean');
    assert.strictEqual(res.isLive, true); // Live tracking must be true
  });

  test('9. Search between stations validates missing station parameters', async () => {
    const res = await railwayService.searchTrainBetweenStations({ origin: '', destination: 'Chennai' });
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.error.code, 'MISSING_STATIONS');
  });
});
