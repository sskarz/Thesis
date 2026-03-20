import { useMemo } from 'react';
import { Bar } from 'react-chartjs-2';
import type { ChartOptions } from 'chart.js';
import type { ModelResult } from '../api/types';
import { modelColor, hexAlpha } from '../lib/colors';
import { CAT_LABELS } from './FilterBar';
import { ChartCard } from './ChartCard';
import '../lib/chartjs-setup';

export function AccuracyChart({
  models,
  selectedCats,
}: {
  models: ModelResult[];
  selectedCats: Set<string>;
}) {
  const cats = useMemo(() => [...selectedCats], [selectedCats]);

  const data = useMemo(() => ({
    labels: cats.map(c => CAT_LABELS[c] || c),
    datasets: models.map(m => {
      const color = modelColor(m.key);
      const scoreMap = Object.fromEntries(m.scores.map(s => [s.category, s.accuracy * 100]));
      return {
        label: m.display.split(' (')[0],
        data: cats.map(c => scoreMap[c] ?? null),
        backgroundColor: hexAlpha(color, 0.55),
        borderColor: color,
        borderWidth: 1,
        borderRadius: 4,
      };
    }),
  }), [models, cats]);

  const options = useMemo((): ChartOptions<'bar'> => ({
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2.2,
    plugins: {
      legend: { position: 'top' },
    },
    scales: {
      x: { grid: { display: false } },
      y: {
        beginAtZero: true,
        max: 100,
        title: { display: true, text: 'Accuracy (%)', font: { size: 12 } },
        ticks: { font: { family: "'JetBrains Mono', monospace", size: 11 } },
      },
    },
    datasets: { bar: { barPercentage: 0.78, categoryPercentage: 0.72 } },
    _barLabelFmt: (v: number) => v === 0 ? '0' : v.toFixed(1) + '%',
  } as any), []);

  return (
    <section className="mt-10">
      <h2 className="mb-3.5 font-heading text-[22px] font-normal text-text">
        Accuracy by Category
      </h2>
      <ChartCard>
        <Bar data={data} options={options} />
      </ChartCard>
      <p className="mt-2.5 text-[13px] italic text-text-muted">
        All models score 0% on parallel and parallel_multiple — a known limitation of small (&le;4B) models.
        Multi-turn categories (MT *) show N/A when not yet benchmarked.
      </p>
    </section>
  );
}
