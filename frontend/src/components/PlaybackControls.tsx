import type React from "react";
import { Play, Pause, RotateCcw, FastForward } from "lucide-react";

interface PlaybackControlsProps {
  isPlaying: boolean;
  currentTime: number;
  maxTime: number;
  playbackSpeed: number;
  onPlayToggle: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  onSeek: (time: number) => void;
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  isPlaying,
  currentTime,
  maxTime,
  playbackSpeed,
  onPlayToggle,
  onReset,
  onSpeedChange,
  onSeek,
}) => {
  const speeds = [1, 2, 5, 10];

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="playback-bar">
      <button className="btn btn-primary" onClick={onPlayToggle} aria-pressed={isPlaying}>
        {isPlaying ? <Pause size={18} aria-hidden="true" /> : <Play size={18} aria-hidden="true" />}
        <span>{isPlaying ? "Pause" : "Play"}</span>
      </button>

      <button className="btn" onClick={onReset} aria-label="Reset to start" title="Reset to start">
        <RotateCcw size={16} aria-hidden="true" />
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }} role="group" aria-label="Playback speed">
        <FastForward size={16} style={{ color: "var(--text-muted)" }} aria-hidden="true" />
        {speeds.map((s) => (
          <button
            key={s}
            className={`btn ${playbackSpeed === s ? "btn-primary" : ""}`}
            style={{ padding: "0.25rem 0.6rem", fontSize: "0.8rem" }}
            onClick={() => onSpeedChange(s)}
            aria-pressed={playbackSpeed === s}
          >
            {s}x
          </button>
        ))}
      </div>

      <input
        type="range"
        className="timeline-slider"
        aria-label="Seek replay timeline"
        min={0}
        max={maxTime || 3600}
        value={currentTime}
        onChange={(e) => onSeek(Number(e.target.value))}
      />

      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--accent-cyan)" }}>
        {formatTime(currentTime)} / {formatTime(maxTime || 3600)}
      </div>
    </div>
  );
};
