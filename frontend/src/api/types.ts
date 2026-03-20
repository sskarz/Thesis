export interface Run {
  id: number;
  model_key: string;
  model_slug: string;
  model_display: string;
  started_at: string;
  finished_at: string | null;
  hardware: Record<string, any> | null;
  server_params: Record<string, any> | null;
  llama_cpp_version: string | null;
  bfcl_version: string | null;
  notes: string | null;
}

export interface Score {
  id: number;
  run_id: number;
  category: string;
  correct: number;
  total: number;
  accuracy: number;
}

export interface StatBlock {
  mean: number;
  median: number;
  p95: number;
  p99: number;
}

export interface InferenceMetrics {
  num_requests: number;
  prompt_tokens_per_sec: StatBlock;
  generation_tokens_per_sec: StatBlock;
  ttft_ms: StatBlock;
  memory: {
    baseline_rss_mb: number;
    peak_rss_mb: number;
    baseline_gpu_vram_mb: number;
    peak_gpu_vram_mb: number;
  };
  power_watts: number | null;
}

export interface SingleTurnTiming {
  id: number;
  run_id: number;
  entry_id: string;
  category: string;
  prompt_n: number;
  prompt_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_second: number;
}

export interface MultiTurnTiming {
  id: number;
  run_id: number;
  entry_id: string;
  category: string;
  turns: number;
  api_calls: number;
  total_prompt_n: number;
  total_prompt_ms: number;
  total_predicted_n: number;
  total_predicted_ms: number;
  first_prompt_ms: number;
  flags: string;
}

export interface EvalError {
  id: number;
  run_id: number;
  entry_id: string;
  category: string;
  error: string;
}

export interface ModelResult {
  key: string;
  display: string;
  slug: string;
  run_id: number;
  scores: Score[];
  metrics: InferenceMetrics;
  run: Run;
}

export interface CategoryDef {
  language?: string;
  file: string;
}

export interface CategoriesResponse {
  single_turn: Record<string, CategoryDef>;
  multi_turn: Record<string, CategoryDef>;
}
