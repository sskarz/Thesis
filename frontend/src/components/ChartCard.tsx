import type { ReactNode } from 'react';

export function ChartCard({
  children,
  narrow,
}: {
  children: ReactNode;
  narrow?: boolean;
}) {
  return (
    <div
      className={`rounded-[var(--radius-card)] border border-border-light bg-surface p-6 shadow-[var(--shadow-card)] ${
        narrow ? 'mx-auto max-w-[640px]' : ''
      }`}
    >
      {children}
    </div>
  );
}
