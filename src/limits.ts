/**
 * Questions a browser-local model must not be allowed to answer.
 *
 * Every one of these was learned from a model doing it. Told plainly in its
 * system prompt that it could not search, a 0.5B model still answered "Yes, I
 * can search the web. What would you like to search for?" — and then, asked
 * what was airing, invented a title and a release date. At this size an
 * instruction is a suggestion and agreement is the likeliest next token.
 *
 * So this is not left to the model. The rules are deliberately blunt: a false
 * positive costs somebody one honest sentence, a false negative is a machine
 * lying fluently about something nobody in a browser tab can check.
 *
 * Reasons rather than sentences, because the words belong to the application.
 * A portfolio, a docs site and a support widget should each say this in their
 * own voice.
 */

export type Limit =
  /** Asked to search, browse, or otherwise reach the internet. */
  | "no-internet"
  /** Asked about the present: news, prices, weather, what is out now. */
  | "no-present"
  /** Asked the date or time. */
  | "no-clock"
  /** Asked where the last answer came from — about the conversation, not the subject. */
  | "asks-for-sources";

const LOOKUP =
  /\b(search|google|look (it |this |that )?up|browse|internet access|the web|online)\b/i;

const PRESENT =
  /\b(today|tonight|right now|currently|this (week|month|year)|latest|newest|recent|news|weather|price of|stock|airing|out now|release[ds]? (this|last)?)\b/i;

const CLOCK = /\b(what (day|date|time)|todays? date|what year)\b/i;

const SOURCES =
  /\b(based on|where (did|do) (you|that) (get|come)|what.{0,12}(source|sources)|how do you know|says who)\b/i;

/**
 * What, if anything, stops this question reaching the model.
 *
 * Order matters: a question can be about both the clock and the present, and
 * "what is today's date" deserves the answer about having no clock rather than
 * an offer to search for it.
 */
export function limit(question: string): Limit | undefined {
  const q = question.trim();
  if (!q) return undefined;
  if (SOURCES.test(q)) return "asks-for-sources";
  if (CLOCK.test(q)) return "no-clock";
  if (LOOKUP.test(q)) return "no-internet";
  if (PRESENT.test(q)) return "no-present";
  return undefined;
}
