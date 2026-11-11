import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Song } from "../api/types";
import { ProgressSteps } from "./ProgressSteps";

function song(overrides: Partial<Song> = {}): Song {
  return {
    id: "song-1",
    prompt: "rain",
    duration_seconds: 30,
    status: "generating_music",
    error_message: null,
    lyrics: null,
    tempo_bpm: null,
    step_timings: {},
    engine: "acestep",
    tracks: { instrumental: false, final: false },
    created_at: "2026-11-11T15:00:00Z",
    updated_at: new Date(Date.now() - 2500).toISOString(),
    ...overrides,
  };
}

describe("ProgressSteps", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows one Generating step for ACE-Step", () => {
    render(<ProgressSteps song={song()} />);
    expect(screen.getByText("Generating")).toBeTruthy();
    expect(screen.queryByText("Analyzing beats")).toBeNull();
    expect(screen.queryByText("Loading model (first song only)")).toBeNull();
  });

  it("shows the load line when this song loaded the model", () => {
    render(<ProgressSteps song={song({ step_timings: { loading_models: 12.4 } })} />);
    expect(screen.getByText("Loading model (first song only)")).toBeTruthy();
    expect(screen.getByText("12.4s")).toBeTruthy();
    expect(screen.getByText("Generating")).toBeTruthy();
  });

  it("keeps the five steps for the older pipeline", () => {
    render(<ProgressSteps song={song({ engine: "staged", status: "writing_lyrics" })} />);
    expect(screen.getByText("Writing lyrics")).toBeTruthy();
    expect(screen.getByText("Generating music")).toBeTruthy();
    expect(screen.queryByText("Generating")).toBeNull();
  });
});
