import useSWR from 'swr';
import { fetchJson } from './client';
import type {
  Run, Score, InferenceMetrics, SingleTurnTiming, MultiTurnTiming,
  EvalError, ModelResult, CategoriesResponse,
} from './types';

const opts = { revalidateOnFocus: false } as const;

export function useComparison() {
  return useSWR<ModelResult[]>('/comparison', () => fetchJson('/comparison'), opts);
}

export function useRuns(modelKey?: string) {
  const key = modelKey ? `/runs?model_key=${modelKey}` : '/runs';
  return useSWR<Run[]>(key, () => fetchJson(key), opts);
}

export function useRun(runId: number | undefined) {
  return useSWR<Run>(
    runId != null ? `/runs/${runId}` : null,
    () => fetchJson(`/runs/${runId}`),
    opts,
  );
}

export function useLatestRun(modelKey: string | undefined) {
  return useSWR<Run>(
    modelKey ? `/runs/latest/${modelKey}` : null,
    () => fetchJson(`/runs/latest/${modelKey}`),
    opts,
  );
}

export function useModels() {
  return useSWR<Record<string, Record<string, string>>>('/models', () => fetchJson('/models'), opts);
}

export function useCategories() {
  return useSWR<CategoriesResponse>('/categories', () => fetchJson('/categories'), opts);
}

export function useScores(runId: number | undefined) {
  return useSWR<Score[]>(
    runId != null ? `/scores/${runId}` : null,
    () => fetchJson(`/scores/${runId}`),
    opts,
  );
}

export function useMetrics(runId: number | undefined) {
  return useSWR<InferenceMetrics>(
    runId != null ? `/metrics/${runId}` : null,
    () => fetchJson(`/metrics/${runId}`),
    opts,
  );
}

export function useSingleTurnTimings(runId: number | undefined) {
  return useSWR<SingleTurnTiming[]>(
    runId != null ? `/timings/single/${runId}` : null,
    () => fetchJson(`/timings/single/${runId}`),
    opts,
  );
}

export function useMultiTurnTimings(runId: number | undefined) {
  return useSWR<MultiTurnTiming[]>(
    runId != null ? `/timings/multi/${runId}` : null,
    () => fetchJson(`/timings/multi/${runId}`),
    opts,
  );
}

export function useErrors(runId: number | undefined, category?: string) {
  const path = category
    ? `/errors/${runId}?category=${category}`
    : `/errors/${runId}`;
  return useSWR<EvalError[]>(
    runId != null ? path : null,
    () => fetchJson(path),
    opts,
  );
}

export function useMetricsPerRequest(runId: number | undefined) {
  return useSWR<SingleTurnTiming[]>(
    runId != null ? `/metrics/per-request/${runId}` : null,
    () => fetchJson(`/metrics/per-request/${runId}`),
    opts,
  );
}
