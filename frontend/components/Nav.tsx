'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard, Network, FileText, Database,
  ClipboardCheck, ShieldCheck, Download, Sparkles,
} from 'lucide-react';

const links = [
  { href: '/',             label: 'Dashboard',    icon: LayoutDashboard, desc: 'Overview & stats' },
  { href: '/graph',        label: 'Graph',        icon: Network,         desc: 'Visualize nodes' },
  { href: '/process',      label: 'Process',      icon: FileText,        desc: 'Extract from docs' },
  { href: '/ingest',       label: 'Ingest',       icon: Database,        desc: 'Import data sources' },
  { href: '/curation',     label: 'Curation',     icon: ClipboardCheck,  desc: 'Review & approve' },
  { href: '/verification', label: 'Verification', icon: ShieldCheck,     desc: 'Validate relations' },
  { href: '/export',       label: 'Export',       icon: Download,        desc: 'Download graph' },
];

export default function Nav() {
  const path = usePathname();
  return (
    <aside className="w-[272px] min-h-screen bg-gradient-to-b from-[#0d1526] to-[#0a1120] border-r border-white/[0.06] flex flex-col shrink-0">
      {/* Logo */}
      <div className="px-6 pt-8 pb-8">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 via-violet-500 to-sky-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Sparkles size={16} className="text-white" />
          </div>
          <div>
            <span className="text-white font-bold text-[15px] tracking-tight block leading-tight">GraphBuilder</span>
            <span className="text-[11px] text-slate-500 font-medium">Knowledge Ops</span>
          </div>
        </div>
      </div>

      {/* Section label */}
      <div className="px-6 mb-2">
        <span className="text-[10px] font-semibold text-slate-600 uppercase tracking-[0.12em]">Navigation</span>
      </div>

      {/* Nav Links */}
      <nav className="flex flex-col gap-0.5 px-3 flex-1">
        {links.map(({ href, label, icon: Icon, desc }) => {
          const active = path === href;
          return (
            <Link key={href} href={href}
              className={`group relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] transition-all duration-200 ${
                active
                  ? 'bg-gradient-to-r from-indigo-600/15 to-indigo-600/5 text-white'
                  : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200'
              }`}
            >
              {active && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-r-full bg-indigo-400" />
              )}
              <div className={`w-8 h-8 rounded-lg flex items-center justify-center transition-colors ${
                active
                  ? 'bg-indigo-500/20 text-indigo-400'
                  : 'bg-white/[0.03] text-slate-500 group-hover:text-slate-300 group-hover:bg-white/[0.06]'
              }`}>
                <Icon size={15} />
              </div>
              <div className="flex flex-col min-w-0">
                <span className="font-semibold leading-tight">{label}</span>
                <span className={`text-[10px] leading-tight truncate mt-0.5 ${active ? 'text-indigo-300/50' : 'text-slate-600 group-hover:text-slate-500'}`}>{desc}</span>
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-5 border-t border-white/[0.04]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[11px] text-slate-500">System Online</span>
        </div>
        <p className="text-[10px] text-slate-700 mt-1.5">CSL GraphBuilder v2 · © 2026</p>
      </div>
    </aside>
  );
}
