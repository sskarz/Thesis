import { useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import type { ChartOptions } from 'chart.js';
import type { ModelResult } from '../api/types';
import { modelColor, hexAlpha } from '../lib/colors';
import { ChartCard } from './ChartCard';

export function ThroughputChart({
  models,
  metric,
}: {
  models: ModelResult[];
  metric: 'mean' | 'median';
}) {
  const labels = [`Prompt (${metric})`, `Generation (${metric})`];

  const data = useMemo(() => ({
    labels,
    datasets: models.map(m => {
      const color = modelColor(m.key);
      return {
        label: m.display.split(' (')[0],
        data: [
          m.metrics?.prompt_tokens_per_sec?.[metric] ?? null,
          m.metrics?.generation_tokens_per_sec?.[metric] ?? null,
        ],
        backgroundColor: hexAlpha(color, 0.55),
        borderColor: color,
        borderWidth: 1,
        borderRadius: 4,
      };
    }),
  }), [models, metric, labels]);

  const options = useMemo((): ChartOptions<'bar'> => ({
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2.4,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: { grid: { display: false } },
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Tokens / sec', font: { size: 12 } },
        ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
      },
    },
    datasets: { bar: { barPercentage: 0.78, categoryPercentage: 0.72 } },
    _barLabelFmt: (v: number) => v.toFixed(1),
  } as any), []);

  return (
    <section className="mt-10">
      <h2 className="mb-3.5 font-heading text-[22px] font-normal text-text">Throughput</h2>
      <ChartCard>
        <Bar data={data} options={options} />
      </ChartCard>
    </section>
  );
}
