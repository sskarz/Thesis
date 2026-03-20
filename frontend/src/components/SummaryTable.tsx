import { useMemo } from 'react';
import type { ModelResult } from '../api/types';
import { modelColor } from '../lib/colors';
import { CAT_LABELS } from './FilterBar';
import { ChartCard } from './ChartCard';

const SINGLE_TURN = ['simple', 'multiple', 'parallel', 'parallel_multiple', 'irrelevance', 'java', 'javascript'];
const MULTI_TURN = ['multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context'];

function avg(vals: (number | null)[]): number | null {
  const valid = vals.filter((v): v is number => v != null);
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : null;
}

interface RowDef {
  name: string;
  vals: (number | null)[];
  higher: boolean;
  unit: string;
  sep?: boolean;
}

export function SummaryTable({
  models,
  selectedCats,
  metric,
}: {
  models: ModelResult[];
  selectedCats: Set<string>;
  metric: 'mean' | 'median';
}) {
  const cats = useMemo(() => [...selectedCats], [selectedCats]);

  const rows = useMemo(() => {
    const r: RowDef[] = [];
    const scoreMap = (m: ModelResult) => Object.fromEntries(m.scores.map(s => [s.category, s.accuracy * 100]));

    // Per-category accuracy
    cats.forEach(c => {
      r.push({
        name: (CAT_LABELS[c] || c) + ' Accuracy',
        vals: models.map(m => scoreMap(m)[c] ?? null),
        higher: true, unit: '%',
      });
    });

    // Averages
    const stCats = cats.filter(c => SINGLE_TURN.includes(c));
    const mtCats = cats.filter(c => MULTI_TURN.includes(c));
    if (stCats.length)
      r.push({ name: 'Avg Accuracy (Single-Turn)', vals: models.map(m => avg(stCats.map(c => scoreMap(m)[c] ?? null))), higher: true, unit: '%', sep: true });
    if (mtCats.length)
      r.push({ name: 'Avg Accuracy (Multi-Turn)', vals: models.map(m => avg(mtCats.map(c => scoreMap(m)[c] ?? null))), higher: true, unit: '%' });
    if (cats.length)
      r.push({ name: 'Avg Accuracy (Overall)', vals: models.map(m => avg(cats.map(c => scoreMap(m)[c] ?? null))), higher: true, unit: '%', sep: !mtCats.length });

    // Throughput
    r.push({ name: `Prompt Throughput (${metric})`, vals: models.map(m => m.metrics?.prompt_tokens_per_sec?.[metric] ?? null), higher: true, unit: ' tok/s' });
    r.push({ name: `Gen Throughput (${metric})`, vals: models.map(m => m.metrics?.generation_tokens_per_sec?.[metric] ?? null), higher: true, unit: ' tok/s' });

    // Latency
    const mLabel = metric[0].toUpperCase() + metric.slice(1);
    r.push({ name: `TTFT ${mLabel}`, vals: models.map(m => m.metrics?.ttft_ms?.[metric] ?? null), higher: false, unit: ' ms' });
    r.push({ name: 'TTFT P95', vals: models.map(m => m.metrics?.ttft_ms?.p95 ?? null), higher: false, unit: ' ms' });
    r.push({ name: 'TTFT P99', vals: models.map(m => m.metrics?.ttft_ms?.p99 ?? null), higher: false, unit: ' ms' });

    // Memory
    r.push({ name: 'Memory Baseline', vals: models.map(m => m.metrics?.memory?.baseline_rss_mb ?? null), higher: false, unit: ' MB' });
    r.push({ name: 'Memory Peak', vals: models.map(m => m.metrics?.memory?.peak_rss_mb ?? null), higher: false, unit: ' MB' });

    return r;
  }, [models, cats, metric]);

  const shortName = (m: ModelResult) => m.display.split(' (')[0];

  return (
    <section className="mt-10">
      <h2 className="mb-3.5 font-heading text-[22px] font-normal text-text">Summary</h2>
      <ChartCard>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                  Metric
                </th>
                {models.map(m => (
                  <th key={m.key} className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-right text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    {shortName(m)}
                  </th>
                ))}
                <th className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                  Best
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => {
                const numVals = row.vals.filter((v): v is number => v != null);
                let bestIdx = -1;
                if (numVals.length > 1) {
                  let bv = row.higher ? -Infinity : Infinity;
                  row.vals.forEach((v, i) => {
                    if (v != null && (row.higher ? v > bv : v < bv)) { bv = v; bestIdx = i; }
                  });
                  const allSame = numVals.every(v => v === numVals[0]);
                  if (allSame) bestIdx = -1;
                }

                return (
                  <tr key={ri} className={`hover:bg-black/[0.012] ${row.sep ? 'border-b-2 border-border' : ''}`}>
                    <td className={`border-b border-border-light px-3.5 py-2 text-text ${row.sep ? 'font-semibold' : 'font-medium'}`}>
                      {row.name}
                    </td>
                    {row.vals.map((v, vi) => (
                      <td key={vi} className="border-b border-border-light px-3.5 py-2 text-right font-mono text-[13px] text-text-secondary">
                        {v != null ? v.toFixed(2) + row.unit : 'N/A'}
                      </td>
                    ))}
                    <td className="border-b border-border-light px-3.5 py-2 text-center text-[13px] font-semibold">
                      {bestIdx >= 0 ? (
                        <span style={{ color: modelColor(models[bestIdx].key) }}>
                          {shortName(models[bestIdx])}
                        </span>
                      ) : (
                        <span className="text-text-muted">{numVals.length > 1 ? 'Tie' : '--'}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </section>
  );
}
