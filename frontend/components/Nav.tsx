'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard, Network, FileText, Database,
  ClipboardCheck, ShieldCheck, Download,
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
    <aside className="w-64 min-h-screen bg-[#ffffff] border-r border-[#d0d7de] flex flex-col">
      <div className="px-5 py-6 border-b border-[#d0d7de]">
        <p className="text-[#24292f] font-semibold tracking-tight">GraphBuilder</p>
        <p className="text-xs text-[#57606a] mt-1">Knowledge Operations</p>
      </div>

      <nav className="px-3 py-4 flex flex-col gap-1">
        {links.map(({ href, label, icon: Icon }) => {
          const active = path === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm border transition-colors ${
                active
                  ? 'bg-[#ddf4ff] text-[#0969da] border-[#54aeff] font-medium'
                  : 'text-[#57606a] border-transparent hover:bg-[#f6f8fa] hover:text-[#24292f]'
              }`}
            >
              <Icon size={16} className={active ? 'text-[#0969da]' : 'text-[#6e7781]'} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto px-5 py-4 border-t border-[#d0d7de] text-[11px] text-[#57606a]">
        CSL GraphBuilder v2
      </div>
    </aside>
  );
}
