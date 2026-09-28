import React from "react";
import type { QueueTrendPoint } from "../utils/trajectory";

interface QueueTrendSparklineProps {
  data: QueueTrendPoint[];
  currentTime: number;
  color: string;
  height?: number;
}

/**
 * Queue-count-over-time trend for the currently fetched replay window — the
 * visual answer to "is this controller actually clearing traffic, or does
 * the queue just sit there while the average-wait number looks good?" A
 * single instantaneous "Queued: N" readout can't show that; this can,
 * across the whole window at a glance, without reading any numbers.
 */
export const QueueTrendSparkline: React.FC<QueueTrendSparklineProps> = ({
  data,
  currentTime,
  color,
  height = 46,
}) => {
  if (data.length < 2) return null;

  const width = 100; // viewBox units; scales to container via width="100%"
  const maxQueued = Math.max(1, ...data.map((d) => d.queued));
  const minTime = data[0].time;
  const maxTime = data[data.length - 1].time;
  const timeSpan = Math.max(1, maxTime - minTime);

  const x = (t: number) => ((t - minTime) / timeSpan) * width;
  const y = (q: number) => height - (q / maxQueued) * (height - 4) - 2;

  const points = data.map((d) => `${x(d.time).toFixed(1)},${y(d.queued).toFixed(1)}`).join(" ");
  const areaPoints = `0,${height} ${points} ${width},${height}`;
  const cursorX = x(Math.min(Math.max(currentTime, minTime), maxTime));

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      width="100%"
      height={height}
      role="img"
      aria-label={`Queue trend over the replay window, peaking at ${maxQueued} queued vehicles`}
    >
      <polygon points={areaPoints} fill={color} fillOpacity={0.12} stroke="none" />
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      <line x1={cursorX} y1={0} x2={cursorX} y2={height} stroke={color} strokeWidth={1} strokeDasharray="2,2" opacity={0.6} />
    </svg>
  );
};
