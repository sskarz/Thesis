import { Outlet, NavLink } from 'react-router';

export function Layout() {
  return (
    <div className="min-h-screen bg-bg">
      {/* Nav */}
      <nav className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-[1140px] items-center gap-6 px-7 py-3">
          <NavLink to="/" className="font-heading text-lg text-text hover:text-accent">
            BFCL Benchmark
          </NavLink>
          <div className="flex gap-4 text-sm font-medium">
            <NavLink
              to="/"
              end
              className={({ isActive }) => isActive ? 'text-accent' : 'text-text-muted hover:text-text-secondary'}
            >
              Comparison
            </NavLink>
            <NavLink
              to="/runs"
              className={({ isActive }) => isActive ? 'text-accent' : 'text-text-muted hover:text-text-secondary'}
            >
              Runs
            </NavLink>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="mx-auto max-w-[1140px] px-7 pb-20">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-6 text-center text-[13px] leading-relaxed text-text-muted">
        BFCL Benchmark Dashboard &middot; Thesis Project
      </footer>
    </div>
  );
}
