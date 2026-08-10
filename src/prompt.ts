import type { Document } from "./retrieval";

/**
 * Putting what is known in front of the question.
 *
 * The format is not a detail. The first version of this opened with "Use only
 * the notes below to answer. If they do not cover it, say you do not know." —
 * and a small model, with the answer sitting directly above the question,
 * replied "I'm sorry, but I can't assist with that request." It reads a
 * prohibition as a policy it is being tested on rather than as material.
 *
 * What works is stating what the notes are and then asking. The
 * staying-inside-them half is carried by them being the only relevant thing in
 * the window, which at this size is enough.
 */

export interface GroundOptions {
  /** What the notes are, in the app's own words. */
  heading?: string;
  /** How to end: the instruction the question is attached to. */
  instruction?: string;
  /**
   * What to do when nothing was retrieved.
   *
   * `"ask-anyway"` passes the question through untouched, which is right when
   * the model may reasonably know — general knowledge, definitions, code.
   * `"say-unknown"` tells it outright that there is nothing, which is the one
   * instruction that reliably produces "I do not know" rather than fiction. Use
   * it when the corpus is the only thing that could have answered.
   */
  whenEmpty?: "ask-anyway" | "say-unknown";
}

export function ground(
  question: string,
  found: Document[],
  options: GroundOptions = {}
): string {
  const heading = options.heading ?? "Notes:";
  const instruction = options.instruction ?? "Using the notes above, answer briefly:";

  if (!found.length) {
    if (options.whenEmpty !== "say-unknown") return question;
    return [
      "There are no notes about this.",
      "",
      `Say plainly that you do not know, in one sentence. Question: ${question}`
    ].join("\n");
  }

  return [
    heading,
    ...found.map((document) => `- ${document.text}`),
    "",
    `${instruction} ${question}`
  ].join("\n");
}
