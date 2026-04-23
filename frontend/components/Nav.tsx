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
} from 'lucide-react';
import HealthDot from './HealthDot';

// Each link carries a one-line `tip` so hovering a sidebar item explains
// what the page does — a small affordance that helps first-time users.
const groups = [
  {
    label: 'Insight',
    links: [
      { href: '/',      label: 'Dashboard', icon: LayoutDashboard, tip: 'Live counters, breakdowns, and pipeline metrics' },
      { href: '/graph', label: 'Graph',     icon: Network,         tip: 'Interactive force-directed knowledge graph' },
    ],
  },
  {
    label: 'Pipeline',
    links: [
      { href: '/process',   label: 'Process',     icon: FileText, tip: 'Run the LLM extraction pipeline on a URL or text' },
      { href: '/ingest',    label: 'Ingest',      icon: Database, tip: 'Open Targets · PubMed · Web crawl' },
      { href: '/documents', label: 'Job History', icon: Layers,   tip: 'Inspect any past run with stage timeline + log' },
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
      className="w-[232px] min-h-screen flex flex-col sticky top-0"
      style={{ background: 'var(--bg-sidebar-grad)' }}
    >
      {/* Logo */}
      <div className="px-5 pt-6 pb-5 flex items-center gap-3">
        <div className="relative h-9 w-9">
          <div
            className="absolute inset-0 rounded-xl bg-animated"
            style={{
              backgroundImage:
                'linear-gradient(135deg,#d5212c 0%,#ef4444 50%,#f59e0b 100%)',
              backgroundSize: '200% 200%',
            }}
          />
          <div className="absolute inset-0 rounded-xl flex items-center justify-center">
            <Hexagon size={16} className="text-white" strokeWidth={2.2} />
          </div>
          <div
            className="absolute -inset-1 rounded-xl opacity-40 blur-md -z-0"
            style={{
              background:
                'linear-gradient(135deg,#d5212c 0%,#ef4444 50%,#f59e0b 100%)',
            }}
          />
        </div>
        <div>
          <p className="text-[14px] font-semibold text-white tracking-tight leading-none">
            GraphBuilder
          </p>
          <p
            className="text-[10px] mt-1 font-medium uppercase tracking-wider"
            style={{ color: 'rgba(255,255,255,0.40)' }}
          >
            CSL Behring · Knowledge
          </p>
        </div>
      </div>

      <div className="mx-4 h-px" style={{ background: 'rgba(255,255,255,0.06)' }} />

      <nav className="px-3 py-4 flex-1 flex flex-col gap-5">
        {groups.map((group) => (
          <div key={group.label} className="flex flex-col gap-0.5">
            <p
              className="px-3 mb-1.5 text-[10px] font-semibold uppercase tracking-widest"
              style={{ color: 'rgba(255,255,255,0.28)' }}
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
                  className="group relative flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium transition-all overflow-hidden"
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
                      className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-[3px] rounded-r-full"
                      style={{
                        background:
                          'linear-gradient(180deg,#d5212c 0%,#f59e0b 100%)',
                      }}
                    />
                  )}
                  <Icon
                    size={15}
                    style={{
                      opacity: active ? 1 : 0.55,
                      color: active ? '#fca5a5' : 'currentColor',
                    }}
                  />
                  <span className="relative">{label}</span>
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      <div
        className="px-5 py-4 flex flex-col gap-2"
        style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
      >
        <HealthDot />
        <p
          className="text-[10px] font-medium"
          style={{ color: 'rgba(255,255,255,0.22)' }}
        >
          CSL GraphBuilder v2.1
        </p>
      </div>
    </aside>
  );
}
