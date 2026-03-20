export function LoadingState({ label = 'Loading...' }: { label?: string }) {
  return (
    <div className="flex items-center justify-center py-20 text-text-muted">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-border border-t-accent" />
        <span className="text-sm font-medium">{label}</span>
      </div>
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="animate-pulse rounded-[var(--radius-card)] border border-border-light bg-surface p-6 shadow-[var(--shadow-card)]">
      <div className="mb-4 h-4 w-32 rounded bg-border-light" />
      <div className="h-64 rounded bg-border-light" />
    </div>
  );
}
