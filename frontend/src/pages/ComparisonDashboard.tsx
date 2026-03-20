import { useMemo } from 'react';
import { useComparison } from '../api/hooks';
import { useFilterState } from '../hooks/useFilterState';
import { modelColor } from '../lib/colors';
import { LoadingState } from '../components/LoadingState';
import { FilterBar } from '../components/FilterBar';
import { KPIGrid } from '../components/KPIGrid';
import { AccuracyChart } from '../components/AccuracyChart';
import { CategoryDetail } from '../components/CategoryDetail';
import { ThroughputChart } from '../components/ThroughputChart';
import { LatencyChart } from '../components/LatencyChart';
import { MemoryChart } from '../components/MemoryChart';
import { SummaryTable } from '../components/SummaryTable';

const SINGLE_TURN_CATS = ['simple', 'multiple', 'parallel', 'parallel_multiple', 'irrelevance', 'java', 'javascript'];
const MULTI_TURN_CATS = ['multi_turn_base', 'multi_turn_miss_func', 'multi_turn_miss_param', 'multi_turn_long_context'];
const ALL_CATS = [...SINGLE_TURN_CATS, ...MULTI_TURN_CATS];

export function ComparisonDashboard() {
  const { data: allModels, isLoading, error } = useComparison();

  const modelKeys = useMemo(() => allModels?.map(m => m.key) ?? [], [allModels]);

  const {
    selectedCats, selectedModels, metric, detailCat,
    toggleCat, toggleModel, setMetric, setDetailCat,
    selectAllCats, clearCats,
  } = useFilterState(ALL_CATS, modelKeys);

  const visibleModels = useMemo(
    () => (allModels ?? []).filter(m => selectedModels.has(m.key)),
    [allModels, selectedModels],
  );

  if (isLoading) return <LoadingState label="Loading benchmark data..." />;
  if (error) return <div className="py-20 text-center text-red-600">Failed to load data.</div>;
  if (!allModels?.length) return <div className="py-20 text-center text-text-muted">No benchmark results found.</div>;

  const subtitle = allModels.map(m => {
    const color = modelColor(m.key);
    return `<span style="color:${color};font-weight:600">${m.display}</span>`;
  }).join(' &middot; ');

  const hw = allModels[0]?.run?.hardware;
  const hwLabel = hw ? `${hw.gpu || hw.cpu} &middot; ${hw.os}` : '';

  return (
    <>
      {/* Header */}
      <header className="pt-13 text-center">
        <h1 className="font-heading text-[38px] font-normal leading-tight tracking-tight text-text">
          BFCL Benchmark Comparison
        </h1>
        <p
          className="mt-2.5 text-[15px] text-text-muted"
          dangerouslySetInnerHTML={{ __html: subtitle + (hwLabel ? ` &middot; ${hwLabel}` : '') }}
        />
        <hr className="mt-7 border-0 border-t border-border" />
      </header>

      {/* Filters */}
      <FilterBar
        allCats={ALL_CATS}
        selectedCats={selectedCats}
        toggleCat={toggleCat}
        selectAllCats={selectAllCats}
        clearCats={clearCats}
        models={allModels}
        selectedModels={selectedModels}
        toggleModel={toggleModel}
        metric={metric}
        setMetric={setMetric}
      />

      {/* KPI cards */}
      <KPIGrid models={visibleModels} selectedCats={selectedCats} metric={metric} />

      {/* Charts */}
      <AccuracyChart models={visibleModels} selectedCats={selectedCats} />
      <CategoryDetail models={visibleModels} allCats={ALL_CATS} detailCat={detailCat} setDetailCat={setDetailCat} />
      <ThroughputChart models={visibleModels} metric={metric} />
      <LatencyChart models={visibleModels} metric={metric} />
      <MemoryChart models={visibleModels} />
      <SummaryTable models={visibleModels} selectedCats={selectedCats} metric={metric} />
    </>
  );
}
