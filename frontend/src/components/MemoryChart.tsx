import { useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import type { ChartOptions } from 'chart.js';
import type { ModelResult } from '../api/types';
import { modelColor, hexAlpha } from '../lib/colors';
import { ChartCard } from './ChartCard';

export function MemoryChart({ models }: { models: ModelResult[] }) {
  const data = useMemo(() => ({
    labels: ['Baseline RSS', 'Peak RSS'],
    datasets: models.map(m => {
      const color = modelColor(m.key);
      const mem = m.metrics?.memory;
      return {
        label: m.display.split(' (')[0],
        data: [mem?.baseline_rss_mb ?? null, mem?.peak_rss_mb ?? null],
        backgroundColor: hexAlpha(color, 0.55),
        borderColor: color,
        borderWidth: 1,
        borderRadius: 4,
      };
    }),
  }), [models]);

  const options = useMemo((): ChartOptions<'bar'> => ({
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 1.6,
    plugins: { legend: { position: 'top' } },
    scales: {
      x: { grid: { display: false } },
      y: {
        beginAtZero: true,
        title: { display: true, text: 'MB', font: { size: 12 } },
        ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
      },
    },
    datasets: { bar: { barPercentage: 0.65, categoryPercentage: 0.55 } },
    _barLabelFmt: (v: number) => v.toFixed(0),
  } as any), []);

  return (
    <section className="mt-10">
      <h2 className="mb-3.5 font-heading text-[22px] font-normal text-text">Memory Usage</h2>
      <ChartCard narrow>
        <Bar data={data} options={options} />
      </ChartCard>
    </section>
  );
}
