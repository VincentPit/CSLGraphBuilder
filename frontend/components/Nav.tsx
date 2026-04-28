'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import {
  LayoutDashboard,
  Network,
  FileText,
  Database,
  ClipboardCheck,
  ShieldCheck,
  Download,
  Hexagon,
  Layers,
  Menu,
  X,
} from 'lucide-react';
import HealthDot from './HealthDot';

const groups = [
  {
    label: 'Insight',
    links: [
      { href: '/',      label: 'Dashboard', icon: LayoutDashboard, tip: 'Overview of the knowledge graph' },
      { href: '/graph', label: 'Graph',     icon: Network,         tip: 'Explore the knowledge graph visually' },
    ],
  },
  {
    label: 'Build',
    links: [
      { href: '/process',   label: 'Process',     icon: FileText, tip: 'Run a document through the LLM extraction pipeline' },
      { href: '/ingest',    label: 'Ingest',      icon: Database, tip: 'Pull from Open Targets, PubMed, or crawl the web' },
      { href: '/documents', label: 'Job History', icon: Layers,   tip: 'Replay any past run with full timeline + log' },
    ],
  },
  {
    label: 'Quality',
    links: [
      { href: '/curation',     label: 'Curation',     icon: ClipboardCheck, tip: 'Approve, reject, or correct extracted items' },
      { href: '/verification', label: 'Verification', icon: ShieldCheck,    tip: 'Cascading text → embedding → LLM verification' },
      { href: '/export',       label: 'Export',       icon: Download,       tip: 'Download as JSON, Cytoscape, GraphML or HTML' },
    ],
  },
];

export default function Nav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    setOpen(false);
  }, [path]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  return (
    <>
      {/* Mobile top bar — dark to match the slide-over panel. */}
      <div
        className="lg:hidden sticky top-0 z-30 flex items-center justify-between px-4 py-3"
        style={{
          background: 'var(--bg-sidebar-grad)',
          color: 'var(--text-sidebar-active)',
          borderBottom: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="flex items-center gap-2 text-[14px] font-extrabold"
          aria-label="Open navigation menu"
          aria-expanded={open}
        >
          <Menu size={20} aria-hidden="true" />
          <span>GraphBuilder</span>
        </button>
        <HealthDot />
      </div>

      {open && (
        <div
          className="fixed inset-0 z-40 lg:hidden"
          style={{ background: 'rgba(15, 11, 9, 0.55)' }}
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={[
          'w-[240px] min-h-screen flex flex-col',
          // Mobile: fixed slide-over off-canvas. Desktop: in flex-row flow.
          'fixed inset-y-0 left-0 z-50 lg:static lg:translate-x-0 transition-transform duration-200 ease-out',
          open ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
        ].join(' ')}
        style={{ background: 'var(--bg-sidebar-grad)' }}
        role="navigation"
        aria-label="Primary navigation"
      >
        {/* Brand */}
        <div className="px-5 pt-6 pb-5 flex items-center gap-3">
          <div className="relative h-11 w-11 shrink-0">
            <div
              className="absolute inset-0"
              style={{
                background: 'linear-gradient(135deg,#d5212c 0%,#9a131c 100%)',
                borderRadius: 'var(--radius-md)',
                boxShadow: '0 3px 0 #6a0d12',
              }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <Hexagon size={20} className="text-white" strokeWidth={2.6} aria-hidden="true" />
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-[16px] font-extrabold text-white tracking-tight leading-none">
              GraphBuilder
            </p>
            <p
              className="text-[10px] mt-1 font-bold uppercase tracking-widest"
              style={{ color: 'rgba(255,255,255,0.45)' }}
            >
              CSL Behring
            </p>
          </div>
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="lg:hidden p-1"
            style={{ color: 'rgba(255,255,255,0.6)' }}
            aria-label="Close navigation menu"
          >
            <X size={18} aria-hidden="true" />
          </button>
        </div>

        <div className="mx-4 h-px" style={{ background: 'rgba(255,255,255,0.08)' }} />

        <nav className="px-3 py-4 flex-1 flex flex-col gap-5">
          {groups.map((group) => (
            <div key={group.label} className="flex flex-col gap-1">
              <p
                className="px-3 mb-1.5 text-[10px] font-extrabold uppercase tracking-widest"
                style={{ color: 'rgba(255,255,255,0.32)' }}
              >
                {group.label}
              </p>
              {group.links.map(({ href, label, icon: Icon, tip }) => {
                const active = path === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    title={tip}
                    className="relative flex items-center gap-3 px-3 py-2.5 text-[14px] font-bold transition-colors"
                    style={{
                      color: active ? 'var(--text-sidebar-active)' : 'var(--text-sidebar)',
                      background: active ? 'var(--bg-sidebar-active)' : 'transparent',
                      borderRadius: 'var(--radius-md)',
                    }}
                    onMouseEnter={(e) => {
                      if (!active) {
                        e.currentTarget.style.background = 'var(--bg-sidebar-hover)';
                        e.currentTarget.style.color = '#ffffff';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!active) {
                        e.currentTarget.style.background = 'transparent';
                        e.currentTarget.style.color = 'var(--text-sidebar)';
                      }
                    }}
                  >
                    {active && (
                      <span
                        className="absolute left-0 top-1/2 -translate-y-1/2 h-7 w-[3px]"
                        style={{ background: 'var(--accent)' }}
                        aria-hidden="true"
                      />
                    )}
                    <span
                      className="flex items-center justify-center h-7 w-7 shrink-0"
                      style={{
                        background: active ? 'rgba(255,255,255,0.10)' : 'transparent',
                        borderRadius: 'var(--radius-sm)',
                      }}
                    >
                      <Icon
                        size={16}
                        strokeWidth={active ? 2.4 : 2.1}
                        style={{ color: active ? '#fca5a5' : 'currentColor' }}
                        aria-hidden="true"
                      />
                    </span>
                    <span className="relative">{label}</span>
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>

        <div
          className="px-5 py-4 flex flex-col gap-2.5"
          style={{ borderTop: '1px solid rgba(255,255,255,0.08)' }}
        >
          <HealthDot />
          <p
            className="text-[10px] font-bold uppercase tracking-wider"
            style={{ color: 'rgba(255,255,255,0.28)' }}
          >
            v2.1 · CSL GraphBuilder
          </p>
        </div>
      </aside>
    </>
  );
}
