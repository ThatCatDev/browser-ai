import type { ChatOptions, Message } from "../chat";
import type { Device, DevicePreference } from "../device";

/**
 * What the two sides say to each other.
 *
 * Deliberately small and explicit: a worker boundary is the one place in a
 * library where a shape mismatch turns into silence rather than a type error,
 * so every message is one of these and every reply carries the id it answers.
 */

export type Request =
  | { id: number; kind: "load-chat"; model: string; device: DevicePreference; options?: ChatOptions }
  | { id: number; kind: "generate"; messages: Message[] }
  | { id: number; kind: "load-embed"; model?: string }
  | { id: number; kind: "embed"; texts: string[] }
  | { id: number; kind: "abort" };

export type Response =
  /** A download, as a fraction. Sent many times against one request. */
  | { id: number; kind: "progress"; fraction: number }
  /** A piece of an answer, as it is generated. */
  | { id: number; kind: "token"; text: string }
  /** The request is finished, with whatever it produced. */
  | {
      id: number;
      kind: "done";
      text?: string;
      vectors?: number[][];
      device?: Device;
      fellBack?: boolean;
    }
  | { id: number; kind: "error"; message: string };
