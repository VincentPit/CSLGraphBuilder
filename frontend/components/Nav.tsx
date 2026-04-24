'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
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
  Flame,
} from 'lucide-react';
import HealthDot from './HealthDot';

// Each link gets a tip + a friendly emoji-style label hint to lean into
// the Duolingo "playful, encouraging" tone. The emoji is kept *out* of
// the label itself so screen readers don't trip over it; we put it in
// `prefix` only.
const groups = [
  {
    label: 'Insight',
    links: [
      { href: '/',      label: 'Dashboard',  icon: LayoutDashboard, tip: 'Your daily overview — stats, streaks, recent activity' },
      { href: '/graph', label: 'Graph',      icon: Network,         tip: 'Explore the knowledge graph visually' },
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

  return (
    <aside
      className="w-[240px] min-h-screen flex flex-col sticky top-0"
      style={{ background: 'var(--bg-sidebar-grad)' }}
    >
      {/* Logo — chunkier, with a subtle wobble on hover */}
      <div className="px-5 pt-6 pb-5 flex items-center gap-3 group">
        <div className="relative h-11 w-11 group-hover:[&>div]:wobble-loop">
          <div
            className="absolute inset-0 rounded-2xl bg-animated"
            style={{
              backgroundImage: 'linear-gradient(135deg,#d5212c 0%,#ef4444 50%,#f59e0b 100%)',
              backgroundSize: '200% 200%',
              boxShadow: '0 4px 0 #7a0d14',
            }}
          />
          <div className="absolute inset-0 rounded-2xl flex items-center justify-center">
            <Hexagon size={20} className="text-white drop-shadow" strokeWidth={2.6} />
          </div>
          <div
            className="absolute -inset-1 rounded-2xl opacity-50 blur-md -z-0"
            style={{
              background: 'linear-gradient(135deg,#d5212c 0%,#f59e0b 100%)',
            }}
          />
        </div>
        <div>
          <p
            className="text-[16px] font-extrabold text-white tracking-tight leading-none"
            style={{ fontFamily: 'Nunito, sans-serif' }}
          >
            GraphBuilder
          </p>
          <p
            className="text-[10px] mt-1 font-bold uppercase tracking-widest"
            style={{ color: 'rgba(255,255,255,0.45)' }}
          >
            CSL · Knowledge Quest
          </p>
        </div>
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
                  className="group relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-[14px] font-bold transition-all overflow-hidden"
                  style={{
                    color: active ? 'var(--text-sidebar-active)' : 'var(--text-sidebar)',
                    background: active ? 'var(--bg-sidebar-active)' : 'transparent',
                  }}
                  onMouseEnter={(e) => {
                    if (!active) e.currentTarget.style.background = 'var(--bg-sidebar-hover)';
                  }}
                  onMouseLeave={(e) => {
                    if (!active) e.currentTarget.style.background = 'transparent';
                  }}
                >
                  {active && (
                    <span
                      className="absolute left-0 top-1/2 -translate-y-1/2 h-7 w-[4px] rounded-r-full"
                      style={{
                        background: 'linear-gradient(180deg,#d5212c 0%,#f59e0b 100%)',
                      }}
                    />
                  )}
                  <span
                    className="flex items-center justify-center h-8 w-8 rounded-lg shrink-0 transition-transform group-hover:scale-110"
                    style={{
                      background: active ? 'rgba(255,255,255,0.10)' : 'transparent',
                    }}
                  >
                    <Icon
                      size={17}
                      strokeWidth={active ? 2.6 : 2.2}
                      style={{ color: active ? '#fca5a5' : 'currentColor' }}
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
        <span className="badge badge-streak inline-flex w-fit">
          <Flame size={11} />
          On a roll!
        </span>
        <HealthDot />
        <p
          className="text-[10px] font-bold uppercase tracking-wider"
          style={{ color: 'rgba(255,255,255,0.28)' }}
        >
          v2.1 · CSL GraphBuilder
        </p>
      </div>
    </aside>
  );
}
