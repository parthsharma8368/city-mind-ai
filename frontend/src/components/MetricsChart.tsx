import React from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, ReferenceLine,
} from "recharts";
import type { ComparisonEntry } from "../api/client";

const CONTROLLER_COLORS: Record<string, string> = {
  ppo:           "#22d3ee",
  ippo:          "#a855f7",
  fixed_time:    "#f59e0b",
  max_pressure:  "#818cf8",
};

const CONTROLLER_LABELS: Record<string, string> = {
  ppo:          "PPO",
  ippo:         "IPPO",
  fixed_time:   "Fixed-Time",
  max_pressure: "Max-Pressure",
};

type MetricKey = "mean_waiting_time" | "throughput" | "mean_queue" | "mean_speed";

interface ComparisonChartProps {
  data: ComparisonEntry[];
  metric: MetricKey;
  title: string;
  unit?: string;
}

export const ComparisonBarChart: React.FC<ComparisonChartProps> = ({
  data,
  metric,
  title,
  unit = "",
}) => {
  const chartData = data.map((d) => ({
    name: CONTROLLER_LABELS[d.controller] ?? d.controller,
    value: Number(d[metric] ?? 0),
    controller: d.controller,
  }));

  const CustomTooltip = ({ active, payload }: { active?: boolean; payload?: Array<{ value: number }> }) => {
    if (active && payload?.length) {
      return (
        <div style={{
          background: "rgba(13,17,23,0.95)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 8, padding: "8px 14px",
          fontSize: 13, color: "#f0f6ff",
        }}>
          <strong>{payload[0].value.toFixed(3)}{unit}</strong>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="chart-container">
      <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "0.75rem", textTransform: "uppercase", letterSpacing: "0.06em", fontWeight: 600 }}>
        {title}
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} barCategoryGap="35%">
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis
            dataKey="name"
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
            tickFormatter={(v: number) => `${v.toFixed(1)}${unit}`}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={70}>
            {chartData.map((entry) => (
              <Cell
                key={entry.controller}
                fill={CONTROLLER_COLORS[entry.controller] ?? "#64748b"}
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
