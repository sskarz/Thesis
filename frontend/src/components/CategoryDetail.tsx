import type { ModelResult } from '../api/types';
import { modelColor } from '../lib/colors';
import { CAT_LABELS } from './FilterBar';
import { ChartCard } from './ChartCard';

const TOTALS: Record<string, number> = {
  simple: 400, multiple: 200, parallel: 200, parallel_multiple: 200,
  irrelevance: 240, java: 100, javascript: 50,
  multi_turn_base: 200, multi_turn_miss_func: 200,
  multi_turn_miss_param: 200, multi_turn_long_context: 200,
};

export function CategoryDetail({
  models,
  allCats,
  detailCat,
  setDetailCat,
}: {
  models: ModelResult[];
  allCats: string[];
  detailCat: string;
  setDetailCat: (c: string) => void;
}) {
  const total = TOTALS[detailCat] || 0;

  const ranked = models
    .map(m => {
      const s = m.scores.find(sc => sc.category === detailCat);
      return { model: m, pct: s ? s.accuracy * 100 : null, correct: s?.correct ?? null };
    })
    .sort((a, b) => (b.pct ?? -1) - (a.pct ?? -1));

  const maxPct = Math.max(...ranked.map(r => r.pct ?? 0), 1);

  return (
    <section className="mt-10">
      <div className="mb-3.5 flex items-center justify-between">
        <h2 className="font-heading text-[22px] font-normal text-text">
          Category Detail
        </h2>
        <select
          value={detailCat}
          onChange={e => setDetailCat(e.target.value)}
          className="cursor-pointer appearance-none rounded-[var(--radius-sm)] border border-border bg-surface px-3.5 py-1.5 pr-9 text-sm font-medium text-text transition-colors hover:border-[#ccc] focus:border-accent focus:outline-none"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 16 16'%3E%3Cpath d='M4 6l4 4 4-4' stroke='%238a8a8a' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E")`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'right 12px center',
          }}
        >
          {allCats.map(c => (
            <option key={c} value={c}>{CAT_LABELS[c] || c}</option>
          ))}
        </select>
      </div>
      <ChartCard>
        {ranked.length === 0 ? (
          <p className="py-3 italic text-text-muted">No models selected.</p>
        ) : (
          <div className="flex flex-col gap-3.5">
            {ranked.map((r, i) => {
              const color = modelColor(r.model.key);
              const name = r.model.display.split(' (')[0];
              const barWidth = r.pct != null
                ? Math.max((r.pct / maxPct) * 100, r.pct > 0 ? 3 : 0.5)
                : 0;

              return (
                <div key={r.model.key} className="grid items-center gap-3" style={{ gridTemplateColumns: '30px 120px 1fr 72px 72px' }}>
                  <div className="text-center font-mono text-xs font-semibold text-text-muted">
                    #{i + 1}
                  </div>
                  <div className="truncate text-sm font-semibold" style={{ color }}>
                    {name}
                  </div>
                  <div className="h-[26px] overflow-hidden rounded-[6px] bg-border-light">
                    <div
                      className="h-full rounded-[6px] transition-all duration-200"
                      style={{ width: `${barWidth}%`, background: color }}
                    />
                  </div>
                  <div className="text-right font-mono text-sm font-semibold" style={{ color: r.pct != null ? color : 'var(--color-text-muted)' }}>
                    {r.pct != null ? r.pct.toFixed(2) + '%' : 'N/A'}
                  </div>
                  <div className="text-right font-mono text-xs text-text-muted">
                    {r.correct != null ? `${r.correct}/${total}` : `--/${total}`}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </ChartCard>
    </section>
  );
}
