import { useMemo, useState } from 'react';
import { useParams, Link } from 'react-router';
import { Bar, Scatter } from 'react-chartjs-2';
import type { ChartOptions } from 'chart.js';
import { useRuns, useRun, useScores, useMetrics, useSingleTurnTimings, useMultiTurnTimings } from '../api/hooks';
import { modelColor, hexAlpha } from '../lib/colors';
import { LoadingState } from '../components/LoadingState';
import { ChartCard } from '../components/ChartCard';
import { CAT_LABELS } from '../components/FilterBar';
import '../lib/chartjs-setup';

const SINGLE_TURN = new Set(['simple', 'multiple', 'parallel', 'parallel_multiple', 'irrelevance', 'java', 'javascript']);

export function ModelDetail() {
  const { modelKey } = useParams<{ modelKey: string }>();
  const { data: runs } = useRuns(modelKey);
  const finishedRuns = useMemo(() => runs?.filter(r => r.finished_at) ?? [], [runs]);
  const [selectedRunId, setSelectedRunId] = useState<number | undefined>(undefined);
  const activeRunId = selectedRunId ?? finishedRuns[0]?.id;

  const { data: run } = useRun(activeRunId);
  const { data: scores } = useScores(activeRunId);
  const { data: metrics } = useMetrics(activeRunId);
  const { data: singleTimings } = useSingleTurnTimings(activeRunId);
  const { data: multiTimings } = useMultiTurnTimings(activeRunId);

  if (!modelKey) return null;
  const color = modelColor(modelKey);

  if (!run || !scores) return <LoadingState label={`Loading ${modelKey}...`} />;

  const sortedScores = [...scores].sort((a, b) => b.accuracy - a.accuracy);
  const overallAcc = scores.length ? (scores.reduce((s, sc) => s + sc.accuracy, 0) / scores.length * 100) : 0;

  return (
    <>
      {/* Header */}
      <header className="pt-10">
        <Link to="/" className="text-sm text-accent hover:underline">&larr; Back to comparison</Link>
        <h1 className="mt-2 font-heading text-[34px] font-normal text-text" style={{ color }}>
          {run.model_display}
        </h1>
        {finishedRuns.length > 1 && (
          <select
            value={activeRunId}
            onChange={e => setSelectedRunId(Number(e.target.value))}
            className="mt-2 rounded-[var(--radius-sm)] border border-border bg-surface px-3 py-1.5 text-sm"
          >
            {finishedRuns.map(r => (
              <option key={r.id} value={r.id}>
                Run #{r.id} — {new Date(r.started_at).toLocaleDateString()}
                {r.notes ? ` (${r.notes})` : ''}
              </option>
            ))}
          </select>
        )}
        <hr className="mt-5 border-0 border-t border-border" />
      </header>

      {/* 2x3 Chart Grid */}
      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* 1. Accuracy by Category */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Accuracy by Category</h3>
          <Bar
            data={{
              labels: sortedScores.map(s => CAT_LABELS[s.category] || s.category),
              datasets: [{
                label: 'Accuracy',
                data: sortedScores.map(s => s.accuracy * 100),
                backgroundColor: sortedScores.map(s =>
                  SINGLE_TURN.has(s.category) ? hexAlpha(color, 0.55) : hexAlpha(color, 0.3)
                ),
                borderColor: color,
                borderWidth: 1,
                borderRadius: 4,
              }],
            }}
            options={{
              indexAxis: 'y',
              responsive: true,
              plugins: { legend: { display: false } },
              scales: {
                x: { beginAtZero: true, max: 100, title: { display: true, text: '%' } },
                y: { grid: { display: false } },
              },
            } as ChartOptions<'bar'>}
          />
        </ChartCard>

        {/* 2. Throughput Distribution */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Throughput Distribution</h3>
          {metrics ? (
            <Bar
              data={{
                labels: ['Mean', 'Median', 'P95', 'P99'],
                datasets: [
                  {
                    label: 'Prompt tok/s',
                    data: [
                      metrics.prompt_tokens_per_sec.mean,
                      metrics.prompt_tokens_per_sec.median,
                      metrics.prompt_tokens_per_sec.p95,
                      metrics.prompt_tokens_per_sec.p99,
                    ],
                    backgroundColor: hexAlpha(color, 0.55),
                    borderColor: color,
                    borderWidth: 1,
                    borderRadius: 4,
                  },
                  {
                    label: 'Gen tok/s',
                    data: [
                      metrics.generation_tokens_per_sec.mean,
                      metrics.generation_tokens_per_sec.median,
                      metrics.generation_tokens_per_sec.p95,
                      metrics.generation_tokens_per_sec.p99,
                    ],
                    backgroundColor: hexAlpha(color, 0.2),
                    borderColor: hexAlpha(color, 0.6),
                    borderWidth: 1,
                    borderRadius: 4,
                  },
                ],
              }}
              options={{
                responsive: true,
                plugins: { legend: { position: 'top' } },
                scales: {
                  x: { grid: { display: false } },
                  y: { beginAtZero: true, title: { display: true, text: 'tok/s' } },
                },
              }}
            />
          ) : <p className="text-sm text-text-muted">No metrics data</p>}
        </ChartCard>

        {/* 3. TTFT Distribution */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">TTFT Distribution</h3>
          <TTFTHistogram timings={singleTimings ?? []} color={color} medianMs={metrics?.ttft_ms?.median} />
        </ChartCard>

        {/* 4. Gen Speed vs Prompt Length */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Generation Speed vs Prompt Length</h3>
          <GenScatter timings={singleTimings ?? []} color={color} />
        </ChartCard>

        {/* 5. Memory Usage */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Memory Usage</h3>
          {metrics?.memory ? (
            <Bar
              data={{
                labels: ['Baseline RSS', 'Peak RSS', 'Baseline VRAM', 'Peak VRAM'],
                datasets: [{
                  label: 'MB',
                  data: [
                    metrics.memory.baseline_rss_mb,
                    metrics.memory.peak_rss_mb,
                    metrics.memory.baseline_gpu_vram_mb,
                    metrics.memory.peak_gpu_vram_mb,
                  ],
                  backgroundColor: [
                    hexAlpha(color, 0.3), hexAlpha(color, 0.55),
                    hexAlpha(color, 0.2), hexAlpha(color, 0.45),
                  ],
                  borderColor: color,
                  borderWidth: 1,
                  borderRadius: 4,
                }],
              }}
              options={{
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                  x: { grid: { display: false } },
                  y: { beginAtZero: true, title: { display: true, text: 'MB' } },
                },
              }}
            />
          ) : <p className="text-sm text-text-muted">No memory data</p>}
        </ChartCard>

        {/* 6. Summary Card */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Summary</h3>
          <div className="grid grid-cols-2 gap-4">
            <StatItem label="Overall Accuracy" value={`${overallAcc.toFixed(1)}%`} color={color} />
            <StatItem label="Prompt tok/s (mean)" value={metrics?.prompt_tokens_per_sec?.mean?.toFixed(1) ?? 'N/A'} color={color} />
            <StatItem label="Gen tok/s (mean)" value={metrics?.generation_tokens_per_sec?.mean?.toFixed(1) ?? 'N/A'} color={color} />
            <StatItem label="TTFT median" value={metrics?.ttft_ms?.median ? `${metrics.ttft_ms.median.toFixed(1)} ms` : 'N/A'} color={color} />
            <StatItem label="Peak RSS" value={metrics?.memory?.peak_rss_mb ? `${(metrics.memory.peak_rss_mb / 1024).toFixed(2)} GB` : 'N/A'} color={color} />
            <StatItem label="Peak VRAM" value={metrics?.memory?.peak_gpu_vram_mb ? `${(metrics.memory.peak_gpu_vram_mb / 1024).toFixed(2)} GB` : 'N/A'} color={color} />
            <StatItem label="Categories" value={`${scores.length}`} color={color} />
            <StatItem label="Multi-turn entries" value={`${multiTimings?.length ?? 0}`} color={color} />
          </div>
        </ChartCard>
      </div>
    </>
  );
}

function StatItem({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div>
      <div className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">{label}</div>
      <div className="mt-1 font-mono text-lg font-bold" style={{ color }}>{value}</div>
    </div>
  );
}

function TTFTHistogram({ timings, color, medianMs }: { timings: { prompt_ms: number }[]; color: string; medianMs?: number }) {
  const { labels, counts } = useMemo(() => {
    if (!timings.length) return { labels: [] as string[], counts: [] as number[] };
    const vals = timings.map(t => t.prompt_ms).filter(v => v > 0);
    if (!vals.length) return { labels: [] as string[], counts: [] as number[] };
    const max = Math.min(Math.max(...vals), vals.sort((a, b) => a - b)[Math.floor(vals.length * 0.98)] * 1.5);
    const binCount = 20;
    const binSize = max / binCount;
    const bins = new Array(binCount).fill(0);
    vals.forEach(v => {
      const idx = Math.min(Math.floor(v / binSize), binCount - 1);
      bins[idx]++;
    });
    return {
      labels: bins.map((_, i) => `${(i * binSize).toFixed(0)}`),
      counts: bins,
    };
  }, [timings]);

  if (!labels.length) return <p className="text-sm text-text-muted">No timing data</p>;

  return (
    <Bar
      data={{
        labels,
        datasets: [{
          label: 'Requests',
          data: counts,
          backgroundColor: hexAlpha(color, 0.55),
          borderColor: color,
          borderWidth: 1,
          borderRadius: 2,
        }],
      }}
      options={{
        responsive: true,
        plugins: {
          legend: { display: false },
          annotation: medianMs != null ? {
            annotations: {
              median: {
                type: 'line', xMin: medianMs, xMax: medianMs,
                borderColor: '#dc2626', borderWidth: 2, borderDash: [4, 4],
                label: { display: true, content: `Median: ${medianMs.toFixed(1)}ms`, position: 'start' },
              },
            },
          } : undefined,
        },
        scales: {
          x: { title: { display: true, text: 'ms' }, grid: { display: false } },
          y: { beginAtZero: true, title: { display: true, text: 'Count' } },
        },
      } as any}
    />
  );
}

function GenScatter({ timings, color }: { timings: { prompt_n: number; predicted_per_second: number; category: string }[]; color: string }) {
  const data = useMemo(() => {
    if (!timings.length) return { datasets: [] };
    const stPoints = timings
      .filter(t => SINGLE_TURN.has(t.category) && t.predicted_per_second > 0)
      .map(t => ({ x: t.prompt_n, y: t.predicted_per_second }));
    return {
      datasets: [{
        label: 'Generation speed',
        data: stPoints,
        backgroundColor: hexAlpha(color, 0.4),
        borderColor: color,
        pointRadius: 3,
        pointHoverRadius: 5,
      }],
    };
  }, [timings, color]);

  if (!data.datasets.length || !data.datasets[0].data.length)
    return <p className="text-sm text-text-muted">No timing data</p>;

  return (
    <Scatter
      data={data}
      options={{
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Prompt tokens' } },
          y: { beginAtZero: true, title: { display: true, text: 'Gen tok/s' } },
        },
      } as ChartOptions<'scatter'>}
    />
  );
}
