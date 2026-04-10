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
    <nav className="w-52 min-h-screen bg-slate-900 border-r border-slate-700 flex flex-col py-6 px-3 gap-1">
      <span className="text-sky-400 font-bold text-lg px-3 mb-4">GraphBuilder</span>
      {links.map(({ href, label, icon: Icon }) => (
        <Link
          key={href}
          href={href}
          className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
            path === href
              ? 'bg-sky-600 text-white'
              : 'text-slate-300 hover:bg-slate-800 hover:text-white'
          }`}
        >
          <Icon size={16} />
          {label}
        </Link>
      ))}
    </nav>
  );
}
