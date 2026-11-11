import { useEffect, useState } from "react";
import { PIPELINE_STEPS, type Song, type SongStatus } from "../api/types";

type StepState = "done" | "current" | "upcoming";

export function ProgressSteps({ song }: { song: Song }) {
  if (song.engine === "acestep") {
    return <AceProgress song={song} />;
  }
  return (
    <ol className="steps">
      {PIPELINE_STEPS.map((step) => {
        const state = stepState(step.status, song.status);
        const timing = song.step_timings[step.status];
        return (
          <li key={step.status} className={`step ${state}`}>
            <span className="step-mark" aria-hidden="true" />
            <span>{step.label}</span>
            {state === "done" && timing !== undefined && (
              <span className="step-time">{timing.toFixed(1)}s</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function AceProgress({ song }: { song: Song }) {
  const elapsed = useElapsed(song.updated_at);
  const loading = song.step_timings.loading_models;
  return (
    <ol className="steps">
      {loading !== undefined && (
        <li className="step done">
          <span className="step-mark" aria-hidden="true" />
          <span>Loading model (first song only)</span>
          <span className="step-time">{loading.toFixed(1)}s</span>
        </li>
      )}
      <li className="step current">
        <span className="step-mark" aria-hidden="true" />
        <span>Generating</span>
        <span className="step-time">{elapsed.toFixed(1)}s</span>
      </li>
    </ol>
  );
}

function useElapsed(updatedAt: string): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  const started = Date.parse(updatedAt);
  if (Number.isNaN(started)) {
    return 0;
  }
  return Math.max(0, (now - started) / 1000);
}

function stepState(step: SongStatus, current: SongStatus): StepState {
  const order = PIPELINE_STEPS.map((item) => item.status);
  const stepIndex = order.indexOf(step);
  const currentIndex = order.indexOf(current);
  if (currentIndex === -1 || stepIndex < currentIndex) {
    return "done";
  }
  if (stepIndex === currentIndex) {
    return "current";
  }
  return "upcoming";
}
