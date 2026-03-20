export function pct(v: number | null | undefined): string {
  if (v == null) return 'N/A';
  return v.toFixed(1) + '%';
}

export function num(v: number | null | undefined, decimals = 1): string {
  if (v == null) return 'N/A';
  return v.toFixed(decimals);
}

export function duration(started: string, finished: string | null): string {
  if (!finished) return 'In progress';
  const ms = new Date(finished).getTime() - new Date(started).getTime();
  const mins = Math.floor(ms / 60000);
  const secs = Math.floor((ms % 60000) / 1000);
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
}

export function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}
