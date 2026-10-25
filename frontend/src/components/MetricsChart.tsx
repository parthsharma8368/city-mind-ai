import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, ReferenceLine,
} from "recharts";
import type { ComparisonEntry } from "../api/client";
import { CONTROLLER_LABELS, CONTROLLER_HEX, controllerShortLabel } from "../constants/controllers";
import { formatChartValue } from "../utils/format";

type MetricKey =
  | "mean_waiting_time"
  | "throughput"
  | "mean_queue"
  | "mean_speed"
  | "trips_completed"
  | "mean_trip_timeloss"
  | "lag_per_scheduled_car_s"
  | "never_inserted";

interface ComparisonChartProps {
  data: ComparisonEntry[];
  metric: MetricKey;
  title: string;
  unit?: string;
  /** Integer counts (trips completed): no decimals, thousands separators. */
  isCount?: boolean;
}

function ChartTooltip({ active, payload, unit, isCount }: {
  active?: boolean;
  payload?: Array<{ value: number; payload?: { name?: string } }>;
  unit: string;
  isCount: boolean;
}) {
  if (active && payload?.length) {
    return (
      <div style={{
        background: "rgba(13,17,23,0.95)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 8, padding: "8px 14px",
        fontSize: 13, color: "#f0f6ff",
      }}>
        {payload[0].payload?.name && <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 2 }}>{payload[0].payload.name}</div>}
        <strong>{formatChartValue(payload[0].value, { isCount })}{unit}</strong>
      </div>
    );
  }
  return null;
}

export const ComparisonBarChart: React.FC<ComparisonChartProps> = ({
  data,
  metric,
  title,
  unit = "",
  isCount = false,
}) => {
  const chartData = data.map((d) => ({
    name: CONTROLLER_LABELS[d.controller] ?? d.controller,
    short: controllerShortLabel(d.controller),
    // Missing/non-finite values plot as 0 height but are not silently "real" zeros in the tooltip path
    value: Number.isFinite(Number(d[metric])) ? Number(d[metric]) : 0,
    controller: d.controller,
  }));

  return (
    <div className="chart-container">
      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.75rem", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
        {title}
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} barCategoryGap="35%">
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="short"
            interval={0}
            tick={{ fill: "var(--text-muted)", fontSize: 10 }}
            axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
            width={55}
            tickFormatter={(v: number) => `${formatChartValue(v, { isCount, decimals: 1 })}${unit}`}
          />
          <Tooltip content={<ChartTooltip unit={unit} isCount={isCount} />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={70}>
            {chartData.map((entry) => (
              <Cell
                key={entry.controller}
                fill={CONTROLLER_HEX[entry.controller] ?? "#64748b"}
                fillOpacity={0.9}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

interface RewardLineChartProps {
  data: Array<{ step: number; reward: number }>;
  title?: string;
}

export const RewardLineChart: React.FC<RewardLineChartProps> = ({ data, title = "Cumulative Reward" }) => {
  return (
    <div className="chart-container">
      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.75rem", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
        {title}
      </p>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="step" tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fill: "var(--text-muted)", fontSize: 10 }} axisLine={false} tickLine={false} width={50} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
          <Line
            type="monotone" dataKey="reward"
            stroke="var(--accent-cyan)" strokeWidth={2}
            dot={false} animationDuration={800}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};
