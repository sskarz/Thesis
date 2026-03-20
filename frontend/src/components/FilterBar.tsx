import { modelColor, hexAlpha } from '../lib/colors';
import type { ModelResult } from '../api/types';

const CAT_LABELS: Record<string, string> = {
  simple: 'Simple', multiple: 'Multiple', parallel: 'Parallel',
  parallel_multiple: 'Parallel Multiple', irrelevance: 'Irrelevance',
  java: 'Java', javascript: 'JavaScript',
  multi_turn_base: 'MT Base', multi_turn_miss_func: 'MT Miss Func',
  multi_turn_miss_param: 'MT Miss Param', multi_turn_long_context: 'MT Long Context',
};

export { CAT_LABELS };

interface FilterBarProps {
  allCats: string[];
  selectedCats: Set<string>;
  toggleCat: (c: string) => void;
  selectAllCats: () => void;
  clearCats: () => void;
  models: ModelResult[];
  selectedModels: Set<string>;
  toggleModel: (k: string) => void;
  metric: 'mean' | 'median';
  setMetric: (m: 'mean' | 'median') => void;
}

export function FilterBar({
  allCats, selectedCats, toggleCat, selectAllCats, clearCats,
  models, selectedModels, toggleModel,
  metric, setMetric,
}: FilterBarProps) {
  return (
    <div className="sticky top-0 z-50 -mx-7 border-b border-border bg-bg/88 px-7 py-3.5 backdrop-blur-[12px]">
      <div className="flex flex-col gap-2.5">
        {/* Category pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="min-w-[76px] shrink-0 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            Categories
          </span>
          <div className="flex flex-wrap items-center gap-1.5">
            {allCats.map(c => (
              <button
                key={c}
                onClick={() => toggleCat(c)}
                className={`rounded-full border px-3.5 py-1 text-[13px] font-medium leading-snug transition-all select-none ${
                  selectedCats.has(c)
                    ? 'border-accent bg-accent-dim text-accent'
                    : 'border-border text-text-muted hover:border-[#ccc] hover:text-text-secondary'
                }`}
              >
                {CAT_LABELS[c] || c}
              </button>
            ))}
          </div>
          <button onClick={selectAllCats} className="px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-text-muted hover:text-accent">All</button>
          <button onClick={clearCats} className="px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-text-muted hover:text-accent">Clear</button>
        </div>

        {/* Model pills */}
        <div className="flex flex-wrap items-center gap-2">
          <span className="min-w-[76px] shrink-0 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            Models
          </span>
          <div className="flex flex-wrap items-center gap-1.5">
            {models.map(m => {
              const color = modelColor(m.key);
              const on = selectedModels.has(m.key);
              return (
                <button
                  key={m.key}
                  onClick={() => toggleModel(m.key)}
                  className="rounded-full border px-3.5 py-1 text-[13px] font-medium leading-snug transition-all select-none"
                  style={on ? {
                    background: hexAlpha(color, 0.08),
                    borderColor: color,
                    color,
                  } : {}}
                >
                  {m.display.split(' (')[0]}
                </button>
              );
            })}
          </div>
        </div>

        {/* Metric toggle */}
        <div className="flex items-center gap-2">
          <span className="min-w-[76px] shrink-0 text-[11px] font-semibold uppercase tracking-wider text-text-muted">
            Metric
          </span>
          <div className="inline-flex overflow-hidden rounded-[var(--radius-sm)] border border-border">
            {(['mean', 'median'] as const).map(m => (
              <button
                key={m}
                onClick={() => setMetric(m)}
                className={`border-r border-border px-4 py-1 text-[13px] font-medium transition-all last:border-r-0 ${
                  metric === m
                    ? 'bg-accent-dim text-accent'
                    : 'text-text-muted hover:text-text-secondary'
                }`}
              >
                {m[0].toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
