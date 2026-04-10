'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard, Network, FileText, Database,
  ClipboardCheck, ShieldCheck, Download, Hexagon,
} from 'lucide-react';

const links = [
  { href: '/',             label: 'Dashboard',    icon: LayoutDashboard },
  { href: '/graph',        label: 'Graph',        icon: Network },
  { href: '/process',      label: 'Process',      icon: FileText },
  { href: '/ingest',       label: 'Ingest',       icon: Database },
  { href: '/curation',     label: 'Curation',     icon: ClipboardCheck },
  { href: '/verification', label: 'Verification', icon: ShieldCheck },
  { href: '/export',       label: 'Export',       icon: Download },
];

export default function Nav() {
  const path = usePathname();

  return (
    <aside className="w-[220px] min-h-screen flex flex-col" style={{ background: 'var(--bg-sidebar)' }}>
      {/* Logo */}
      <div className="px-5 pt-6 pb-5 flex items-center gap-2.5">
        <div className="h-7 w-7 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
          <Hexagon size={14} className="text-white" />
        </div>
        <div>
          <p className="text-[13px] font-semibold text-white tracking-tight leading-none">GraphBuilder</p>
          <p className="text-[10px] mt-0.5 font-medium" style={{ color: 'var(--text-sidebar)' }}>Knowledge Platform</p>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-4 h-px" style={{ background: 'rgba(255,255,255,0.08)' }} />

      {/* Navigation */}
      <nav className="px-3 py-4 flex-1 flex flex-col gap-0.5">
        <p className="px-3 mb-2 text-[10px] font-semibold uppercase tracking-widest" style={{ color: 'rgba(255,255,255,0.3)' }}>
          Operations
        </p>
        {links.map(({ href, label, icon: Icon }) => {
          const active = path === href;
          return (
            <Link
              key={href}
              href={href}
              className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium transition-all"
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
              <Icon size={15} style={{ opacity: active ? 1 : 0.6 }} />
              {label}
              {active && <div className="ml-auto h-1.5 w-1.5 rounded-full bg-indigo-400" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <p className="text-[10px] font-medium" style={{ color: 'rgba(255,255,255,0.25)' }}>
          CSL GraphBuilder v2.0
        </p>
      </div>
    </aside>
  );
}
