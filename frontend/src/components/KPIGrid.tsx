import type { ModelResult } from '../api/types';
import { modelColor } from '../lib/colors';

interface KPICardData {
  label: string;
  value: string;
  modelName: string;
  color: string;
}

function bestModel(
  models: ModelResult[],
  fn: (m: ModelResult) => number | null,
  higher: boolean,
): { model: ModelResult; value: number } | null {
  let best: ModelResult | null = null;
  let bestVal = higher ? -Infinity : Infinity;
  for (const m of models) {
    const v = fn(m);
    if (v == null) continue;
    if (higher ? v > bestVal : v < bestVal) { best = m; bestVal = v; }
  }
  return best ? { model: best, value: bestVal } : null;
}

function avg(vals: (number | null)[]): number {
  const valid = vals.filter((v): v is number => v != null);
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
}

export function KPIGrid({
  models,
  selectedCats,
  metric,
}: {
  models: ModelResult[];
  selectedCats: Set<string>;
  metric: 'mean' | 'median';
}) {
  const cats = [...selectedCats];
  const cards: KPICardData[] = [];

  // Best accuracy
  const accBest = models.length && cats.length
    ? bestModel(models, m => {
        const scores = m.scores.filter(s => cats.includes(s.category));
        return scores.length ? avg(scores.map(s => s.accuracy * 100)) : null;
      }, true)
    : null;
  cards.push(accBest
    ? { label: 'Best Avg. Accuracy', value: accBest.value.toFixed(1) + '%', modelName: accBest.model.display.split(' (')[0], color: modelColor(accBest.model.key) }
    : { label: 'Best Avg. Accuracy', value: '--', modelName: '--', color: 'var(--color-text-muted)' },
  );

  // Fastest prompt throughput
  const tpBest = bestModel(models, m => m.metrics?.prompt_tokens_per_sec?.[metric] ?? null, true);
  cards.push(tpBest
    ? { label: 'Fastest Prompt Throughput', value: tpBest.value.toFixed(1) + ' tok/s', modelName: tpBest.model.display.split(' (')[0], color: modelColor(tpBest.model.key) }
    : { label: 'Fastest Prompt Throughput', value: '--', modelName: '--', color: 'var(--color-text-muted)' },
  );

  // Lowest TTFT
  const ttftBest = bestModel(models, m => m.metrics?.ttft_ms?.[metric] ?? null, false);
  cards.push(ttftBest
    ? { label: 'Lowest TTFT', value: ttftBest.value.toFixed(1) + ' ms', modelName: ttftBest.model.display.split(' (')[0], color: modelColor(ttftBest.model.key) }
    : { label: 'Lowest TTFT', value: '--', modelName: '--', color: 'var(--color-text-muted)' },
  );

  // Most memory efficient
  const memBest = bestModel(models, m => m.metrics?.memory?.peak_rss_mb ?? null, false);
  cards.push(memBest
    ? { label: 'Most Memory Efficient', value: (memBest.value / 1024).toFixed(1) + ' GB', modelName: memBest.model.display.split(' (')[0], color: modelColor(memBest.model.key) }
    : { label: 'Most Memory Efficient', value: '--', modelName: '--', color: 'var(--color-text-muted)' },
  );

  return (
    <div className="mt-7 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map(c => (
        <div
          key={c.label}
          className="relative overflow-hidden rounded-[var(--radius-card)] border border-border-light bg-surface p-5 shadow-[var(--shadow-card)]"
        >
          <div className="absolute top-0 right-0 left-0 h-0.5" style={{ background: c.color }} />
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            {c.label}
          </div>
          <div className="font-mono text-[26px] font-bold leading-tight" style={{ color: c.color }}>
            {c.value}
          </div>
          <div className="mt-1.5 text-[13px] font-semibold" style={{ color: c.color }}>
            {c.modelName}
          </div>
        </div>
      ))}
    </div>
  );
}
