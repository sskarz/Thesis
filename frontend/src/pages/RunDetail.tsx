import { useState } from 'react';
import { useParams, Link } from 'react-router';
import { useRun, useScores, useMetrics, useErrors } from '../api/hooks';
import { LoadingState } from '../components/LoadingState';
import { ChartCard } from '../components/ChartCard';
import { CAT_LABELS } from '../components/FilterBar';
import { shortDate, duration, num } from '../lib/formatters';
import { modelColor } from '../lib/colors';

export function RunDetail() {
  const { runId: rawId } = useParams<{ runId: string }>();
  const runId = rawId ? Number(rawId) : undefined;

  const { data: run, isLoading: loadingRun } = useRun(runId);
  const { data: scores } = useScores(runId);
  const { data: metrics } = useMetrics(runId);
  const { data: errors } = useErrors(runId);

  const [expandedError, setExpandedError] = useState<number | null>(null);
  const [errorCatFilter, setErrorCatFilter] = useState<string>('');

  if (loadingRun || !run) return <LoadingState label="Loading run..." />;

  const hw = typeof run.hardware === 'string' ? JSON.parse(run.hardware) : run.hardware;
  const color = modelColor(run.model_key);
  const filteredErrors = errors?.filter(e => !errorCatFilter || e.category === errorCatFilter) ?? [];
  const errorCats = [...new Set(errors?.map(e => e.category) ?? [])];

  return (
    <>
      <header className="pt-10">
        <Link to="/runs" className="text-sm text-accent hover:underline">&larr; All runs</Link>
        <h1 className="mt-2 font-heading text-[34px] font-normal text-text">
          Run <span className="font-mono">#{run.id}</span>
        </h1>
        <p className="mt-1 text-sm text-text-muted">
          <Link to={`/model/${run.model_key}`} className="font-semibold hover:underline" style={{ color }}>
            {run.model_display}
          </Link>
        </p>
        <hr className="mt-5 border-0 border-t border-border" />
      </header>

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Run Metadata */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Run Metadata</h3>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
            <Meta label="Model" value={run.model_display} />
            <Meta label="Started" value={shortDate(run.started_at)} />
            <Meta label="Duration" value={duration(run.started_at, run.finished_at)} />
            <Meta label="GPU" value={hw?.gpu ?? '--'} />
            <Meta label="CPU" value={hw?.cpu ?? '--'} />
            <Meta label="RAM" value={hw?.ram_gb ? `${hw.ram_gb} GB` : '--'} />
            <Meta label="OS" value={hw?.os ?? '--'} />
            <Meta label="llama.cpp" value={run.llama_cpp_version ?? '--'} />
            <Meta label="BFCL" value={run.bfcl_version ?? '--'} />
            <Meta label="Notes" value={run.notes || '--'} />
          </dl>
        </ChartCard>

        {/* Inference Metrics */}
        <ChartCard>
          <h3 className="mb-4 text-base font-semibold text-text">Inference Metrics</h3>
          {metrics && Object.keys(metrics).length > 0 ? (
            <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
              <Meta label="Requests" value={String(metrics.num_requests)} />
              <Meta label="Prompt tok/s (mean)" value={num(metrics.prompt_tokens_per_sec?.mean)} />
              <Meta label="Prompt tok/s (median)" value={num(metrics.prompt_tokens_per_sec?.median)} />
              <Meta label="Gen tok/s (mean)" value={num(metrics.generation_tokens_per_sec?.mean)} />
              <Meta label="Gen tok/s (median)" value={num(metrics.generation_tokens_per_sec?.median)} />
              <Meta label="TTFT mean" value={`${num(metrics.ttft_ms?.mean)} ms`} />
              <Meta label="TTFT median" value={`${num(metrics.ttft_ms?.median)} ms`} />
              <Meta label="TTFT P95" value={`${num(metrics.ttft_ms?.p95)} ms`} />
              <Meta label="Peak RSS" value={`${num(metrics.memory?.peak_rss_mb, 0)} MB`} />
              <Meta label="Peak VRAM" value={`${num(metrics.memory?.peak_gpu_vram_mb, 0)} MB`} />
            </dl>
          ) : <p className="text-sm text-text-muted">No metrics available</p>}
        </ChartCard>
      </div>

      {/* Scores Table */}
      {scores && scores.length > 0 && (
        <div className="mt-6">
          <ChartCard>
            <h3 className="mb-4 text-base font-semibold text-text">Accuracy by Category</h3>
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="border-b-2 border-border px-3.5 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">Category</th>
                  <th className="border-b-2 border-border px-3.5 py-2 text-right text-[11px] font-semibold uppercase tracking-wider text-text-muted">Correct</th>
                  <th className="border-b-2 border-border px-3.5 py-2 text-right text-[11px] font-semibold uppercase tracking-wider text-text-muted">Total</th>
                  <th className="border-b-2 border-border px-3.5 py-2 text-right text-[11px] font-semibold uppercase tracking-wider text-text-muted">Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {[...scores].sort((a, b) => b.accuracy - a.accuracy).map(s => (
                  <tr key={s.category} className="hover:bg-black/[0.012]">
                    <td className="border-b border-border-light px-3.5 py-2 font-medium text-text">{CAT_LABELS[s.category] || s.category}</td>
                    <td className="border-b border-border-light px-3.5 py-2 text-right font-mono text-[13px] text-text-secondary">{s.correct}</td>
                    <td className="border-b border-border-light px-3.5 py-2 text-right font-mono text-[13px] text-text-secondary">{s.total}</td>
                    <td className="border-b border-border-light px-3.5 py-2 text-right font-mono text-[13px] font-semibold" style={{ color }}>
                      {(s.accuracy * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ChartCard>
        </div>
      )}

      {/* Eval Errors */}
      {errors && errors.length > 0 && (
        <div className="mt-6">
          <ChartCard>
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-base font-semibold text-text">Eval Errors ({errors.length})</h3>
              {errorCats.length > 1 && (
                <select
                  value={errorCatFilter}
                  onChange={e => setErrorCatFilter(e.target.value)}
                  className="rounded-[var(--radius-sm)] border border-border bg-surface px-2 py-1 text-sm"
                >
                  <option value="">All categories</option>
                  {errorCats.map(c => <option key={c} value={c}>{CAT_LABELS[c] || c}</option>)}
                </select>
              )}
            </div>
            <div className="max-h-96 overflow-y-auto">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr>
                    <th className="sticky top-0 border-b-2 border-border bg-surface px-3.5 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">Entry</th>
                    <th className="sticky top-0 border-b-2 border-border bg-surface px-3.5 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">Category</th>
                    <th className="sticky top-0 border-b-2 border-border bg-surface px-3.5 py-2 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">Error</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredErrors.slice(0, 100).map(e => (
                    <tr
                      key={e.id}
                      className="cursor-pointer hover:bg-black/[0.012]"
                      onClick={() => setExpandedError(expandedError === e.id ? null : e.id)}
                    >
                      <td className="border-b border-border-light px-3.5 py-2 font-mono text-xs text-text-secondary">{e.entry_id}</td>
                      <td className="border-b border-border-light px-3.5 py-2 text-text-secondary">{CAT_LABELS[e.category] || e.category}</td>
                      <td className="border-b border-border-light px-3.5 py-2 text-text-secondary">
                        <div className={expandedError === e.id ? '' : 'max-w-md truncate'}>
                          {e.error}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filteredErrors.length > 100 && (
                <p className="mt-2 text-xs text-text-muted">Showing first 100 of {filteredErrors.length} errors</p>
              )}
            </div>
          </ChartCard>
        </div>
      )}
    </>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-text-muted">{label}</dt>
      <dd className="font-medium text-text">{value}</dd>
    </>
  );
}
