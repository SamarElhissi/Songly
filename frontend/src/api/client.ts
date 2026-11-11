import { isSongStatus, type Song, type SongCreate } from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function listSongs(): Promise<Song[]> {
  const body = await request("/api/songs");
  if (!Array.isArray(body)) {
    throw new Error("Expected a list of songs");
  }
  return body.map(parseSong);
}

export async function getSong(id: string): Promise<Song> {
  return parseSong(await request(`/api/songs/${encodeURIComponent(id)}`));
}

export async function createSong(input: SongCreate): Promise<Song> {
  return parseSong(
    await request("/api/songs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
  );
}

export async function deleteSong(id: string): Promise<void> {
  await request(`/api/songs/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function audioUrl(id: string, track: "final" | "instrumental"): string {
  return `/api/songs/${encodeURIComponent(id)}/audio?track=${track}`;
}

async function request(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(path, init);
  if (response.status === 204) {
    return null;
  }
  const text = await response.text();
  const body = text ? parseJson(text) : null;
  if (!response.ok) {
    const detail = isRecord(body) ? readDetail(body.detail) : null;
    throw new ApiError(detail ?? response.statusText, response.status);
  }
  return body;
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error("The server returned invalid JSON");
  }
}

function readDetail(detail: unknown): string | null {
  if (typeof detail === "string") {
    return detail;
  }
  if (!Array.isArray(detail)) {
    return null;
  }
  const messages = detail.flatMap((item) => {
    if (isRecord(item) && typeof item.msg === "string") {
      return [item.msg];
    }
    return [];
  });
  return messages.length > 0 ? messages.join(" ") : null;
}

export function parseSong(value: unknown): Song {
  if (!isRecord(value)) {
    throw new Error("Invalid song");
  }
  const status = value.status;
  if (typeof status !== "string" || !isSongStatus(status)) {
    throw new Error("Invalid song status");
  }
  return {
    id: requiredString(value.id, "id"),
    prompt: requiredString(value.prompt, "prompt"),
    duration_seconds: requiredNumber(value.duration_seconds, "duration"),
    status,
    error_message: nullableString(value.error_message, "error"),
    lyrics: nullableString(value.lyrics, "lyrics"),
    tempo_bpm: nullableNumber(value.tempo_bpm),
    step_timings: parseTimings(value.step_timings),
    engine: value.engine === "acestep" ? "acestep" : "staged",
    tracks: parseTracks(value.tracks),
    created_at: requiredString(value.created_at, "created_at"),
    updated_at: requiredString(value.updated_at, "updated_at"),
  };
}

function parseTimings(value: unknown): Record<string, number> {
  if (!isRecord(value)) {
    throw new Error("Invalid step timings");
  }
  const timings: Record<string, number> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item !== "number") {
      throw new Error("Invalid step timing");
    }
    timings[key] = item;
  }
  return timings;
}

function parseTracks(value: unknown): Song["tracks"] {
  if (
    !isRecord(value) ||
    typeof value.instrumental !== "boolean" ||
    typeof value.final !== "boolean"
  ) {
    throw new Error("Invalid tracks");
  }
  return { instrumental: value.instrumental, final: value.final };
}

function requiredString(value: unknown, label: string): string {
  if (typeof value !== "string") {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

function requiredNumber(value: unknown, label: string): number {
  if (typeof value !== "number") {
    throw new Error(`Invalid ${label}`);
  }
  return value;
}

function nullableString(value: unknown, label: string): string | null {
  if (value === null) {
    return null;
  }
  return requiredString(value, label);
}

function nullableNumber(value: unknown): number | null {
  if (value === null) {
    return null;
  }
  if (typeof value !== "number") {
    throw new Error("Invalid tempo");
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
