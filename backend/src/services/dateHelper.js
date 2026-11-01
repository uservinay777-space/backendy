/**
 * Date & Timezone Helper for Indian Railways (Asia/Kolkata)
 * Converts relative terms ("aaj", "kal", "parso", day names) and standard dates
 * into the DD-MM-YYYY format required by Indian Railways & RailKit.
 */

const TIMEZONE = 'Asia/Kolkata';

/**
 * Get current date in Asia/Kolkata timezone.
 * @returns {Date}
 */
function getNowInKolkata() {
  const now = new Date();
  const kolkataTimeString = now.toLocaleString('en-US', { timeZone: TIMEZONE });
  return new Date(kolkataTimeString);
}

/**
 * Format a Date object to DD-MM-YYYY
 * @param {Date} dateObj
 * @returns {string} e.g. "15-04-2026"
 */
function formatDateToDDMMYYYY(dateObj) {
  const day = String(dateObj.getDate()).padStart(2, '0');
  const month = String(dateObj.getMonth() + 1).padStart(2, '0');
  const year = dateObj.getFullYear();
  return `${day}-${month}-${year}`;
}

/**
 * Convert user input or date string into Indian Railways DD-MM-YYYY format.
 * Supports:
 * - "aaj", "today", "आज" -> today
 * - "kal", "tomorrow", "कल" -> tomorrow
 * - "parso", "day after tomorrow", "परसों" -> day after tomorrow
 * - Weekday names: "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
 * - Existing DD-MM-YYYY, YYYY-MM-DD, DD/MM/YYYY
 * 
 * @param {string|Date} inputDate
 * @returns {string} Formatted DD-MM-YYYY string
 */
function toIndianRailwayDate(inputDate) {
  const now = getNowInKolkata();

  if (!inputDate || typeof inputDate !== 'string') {
    if (inputDate instanceof Date) {
      return formatDateToDDMMYYYY(inputDate);
    }
    return formatDateToDDMMYYYY(now);
  }

  const clean = inputDate.trim().toLowerCase();

  // 1. Relative today
  if (['aaj', 'today', 'आज'].includes(clean)) {
    return formatDateToDDMMYYYY(now);
  }

  // 2. Relative tomorrow
  if (['kal', 'tomorrow', 'कल'].includes(clean)) {
    const tomorrow = new Date(now);
    tomorrow.setDate(tomorrow.getDate() + 1);
    return formatDateToDDMMYYYY(tomorrow);
  }

  // 3. Relative day after tomorrow
  if (['parso', 'parson', 'day after tomorrow', 'परसों'].includes(clean)) {
    const dayAfter = new Date(now);
    dayAfter.setDate(dayAfter.getDate() + 2);
    return formatDateToDDMMYYYY(dayAfter);
  }

  // 4. Weekday names (next occurrence of weekday)
  const daysOfWeek = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'];
  const dayIndex = daysOfWeek.indexOf(clean);
  if (dayIndex !== -1) {
    const currentDay = now.getDay();
    let daysToAdd = (dayIndex - currentDay + 7) % 7;
    if (daysToAdd === 0) {
      daysToAdd = 7; // User asking for "Sunday" on a Sunday means the upcoming Sunday
    }
    const targetDate = new Date(now);
    targetDate.setDate(targetDate.getDate() + daysToAdd);
    return formatDateToDDMMYYYY(targetDate);
  }

  // 5. Already DD-MM-YYYY
  if (/^\d{2}-\d{2}-\d{4}$/.test(clean)) {
    return clean;
  }

  // 6. Already DD/MM/YYYY
  if (/^\d{2}\/\d{2}\/\d{4}$/.test(clean)) {
    const [d, m, y] = clean.split('/');
    return `${d}-${m}-${y}`;
  }

  // 7. YYYY-MM-DD
  if (/^\d{4}-\d{2}-\d{2}$/.test(clean)) {
    const [y, m, d] = clean.split('-');
    return `${d}-${m}-${y}`;
  }

  // 8. General date parse fallback
  const parsed = new Date(inputDate);
  if (!isNaN(parsed.getTime())) {
    return formatDateToDDMMYYYY(parsed);
  }

  // Default to today if unparseable
  return formatDateToDDMMYYYY(now);
}

module.exports = {
  getNowInKolkata,
  formatDateToDDMMYYYY,
  toIndianRailwayDate,
  TIMEZONE,
};
