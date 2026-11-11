import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Song } from "../api/types";
import { useSongPolling } from "./useSongPolling";

function song(status: Song["status"]): Song {
  return {
    id: "song-1",
    prompt: "rain",
    duration_seconds: 5,
    status,
    error_message: null,
    lyrics: null,
    tempo_bpm: null,
    step_timings: {},
    engine: "staged",
    tracks: { instrumental: false, final: false },
    created_at: "2026-11-06T00:00:00Z",
    updated_at: "2026-11-06T00:00:00Z",
  };
}

describe("useSongPolling", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("polls every 2 seconds until the song is finished", async () => {
    vi.useFakeTimers();
    const statuses: Song["status"][] = ["generating_music", "generating_music", "completed"];
    let call = 0;
    const fetchMock = vi.fn(async () => {
      const status = statuses[Math.min(call, statuses.length - 1)] ?? "completed";
      call += 1;
      return new Response(JSON.stringify(song(status)), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(() => useSongPolling("song-1", true), { wrapper });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(result.current.data?.status).toBe("generating_music");
    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(result.current.data?.status).toBe("completed");
    expect(fetchMock).toHaveBeenCalledTimes(3);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });
});
