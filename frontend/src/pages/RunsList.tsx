import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import { useRuns } from '../api/hooks';
import { LoadingState } from '../components/LoadingState';
import { ChartCard } from '../components/ChartCard';
import { shortDate, duration } from '../lib/formatters';

type SortKey = 'id' | 'model_key' | 'started_at';

export function RunsList() {
  const { data: runs, isLoading } = useRuns();
  const [sortKey, setSortKey] = useState<SortKey>('id');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  const sorted = useMemo(() => {
    if (!runs) return [];
    return [...runs].sort((a, b) => {
      let cmp = 0;
      if (sortKey === 'id') cmp = a.id - b.id;
      else if (sortKey === 'model_key') cmp = a.model_key.localeCompare(b.model_key);
      else cmp = a.started_at.localeCompare(b.started_at);
      return sortDir === 'desc' ? -cmp : cmp;
    });
  }, [runs, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };

  if (isLoading) return <LoadingState label="Loading runs..." />;

  const sortIcon = (key: SortKey) => sortKey === key ? (sortDir === 'desc' ? ' ↓' : ' ↑') : '';

  return (
    <>
      <header className="pt-10">
        <h1 className="font-heading text-[34px] font-normal text-text">All Runs</h1>
        <p className="mt-1 text-sm text-text-muted">{sorted.length} runs total</p>
        <hr className="mt-5 border-0 border-t border-border" />
      </header>

      <div className="mt-8">
        <ChartCard>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th onClick={() => toggleSort('id')} className="cursor-pointer border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted hover:text-accent">
                    Run{sortIcon('id')}
                  </th>
                  <th onClick={() => toggleSort('model_key')} className="cursor-pointer border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted hover:text-accent">
                    Model{sortIcon('model_key')}
                  </th>
                  <th onClick={() => toggleSort('started_at')} className="cursor-pointer border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted hover:text-accent">
                    Started{sortIcon('started_at')}
                  </th>
                  <th className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    Duration
                  </th>
                  <th className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    GPU
                  </th>
                  <th className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    Notes
                  </th>
                  <th className="border-b-2 border-border bg-surface px-3.5 py-2.5 text-center text-[11px] font-semibold uppercase tracking-wider text-text-muted">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody>
                {sorted.map(r => {
                  const hw = typeof r.hardware === 'string' ? JSON.parse(r.hardware) : r.hardware;
                  return (
                    <tr key={r.id} className="hover:bg-black/[0.012]">
                      <td className="border-b border-border-light px-3.5 py-2 font-mono text-[13px] font-medium text-accent">
                        <Link to={`/runs/${r.id}`} className="hover:underline">#{r.id}</Link>
                      </td>
                      <td className="border-b border-border-light px-3.5 py-2 font-medium text-text">
                        <Link to={`/model/${r.model_key}`} className="hover:underline">
                          {r.model_display}
                        </Link>
                      </td>
                      <td className="border-b border-border-light px-3.5 py-2 text-text-secondary">
                        {shortDate(r.started_at)}
                      </td>
                      <td className="border-b border-border-light px-3.5 py-2 font-mono text-[13px] text-text-secondary">
                        {duration(r.started_at, r.finished_at)}
                      </td>
                      <td className="border-b border-border-light px-3.5 py-2 text-text-secondary">
                        {hw?.gpu ?? '--'}
                      </td>
                      <td className="border-b border-border-light px-3.5 py-2 text-text-muted">
                        {r.notes || '--'}
                      </td>
                      <td className="border-b border-border-light px-3.5 py-2 text-center">
                        {r.finished_at ? (
                          <span className="inline-block rounded-full bg-green-100 px-2.5 py-0.5 text-xs font-semibold text-green-700">
                            Done
                          </span>
                        ) : (
                          <span className="inline-block rounded-full bg-yellow-100 px-2.5 py-0.5 text-xs font-semibold text-yellow-700">
                            Running
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </ChartCard>
      </div>
    </>
  );
}
