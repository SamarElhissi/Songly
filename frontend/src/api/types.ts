export const SONG_STATUSES = [
  "queued",
  "generating_music",
  "analyzing_beats",
  "writing_lyrics",
  "synthesizing_vocals",
  "mixing",
  "completed",
  "failed",
] as const;

export type SongStatus = (typeof SONG_STATUSES)[number];

export type Song = {
  id: string;
  prompt: string;
  duration_seconds: number;
  status: SongStatus;
  error_message: string | null;
  lyrics: string | null;
  tempo_bpm: number | null;
  step_timings: Record<string, number>;
  engine: "acestep" | "staged";
  tracks: {
    instrumental: boolean;
    final: boolean;
  };
  created_at: string;
  updated_at: string;
};

export type SongCreate = {
  prompt: string;
  duration_seconds: number;
};

export const PIPELINE_STEPS: { status: SongStatus; label: string }[] = [
  { status: "queued", label: "Queued" },
  { status: "generating_music", label: "Generating music" },
  { status: "analyzing_beats", label: "Analyzing beats" },
  { status: "writing_lyrics", label: "Writing lyrics" },
  { status: "synthesizing_vocals", label: "Synthesizing vocals" },
  { status: "mixing", label: "Mixing" },
];

export function isFinished(status: SongStatus): boolean {
  return status === "completed" || status === "failed";
}

export function isSongStatus(value: string): value is SongStatus {
  return (SONG_STATUSES as readonly string[]).includes(value);
}
