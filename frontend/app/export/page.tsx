'use client';
import { getExportUrl } from '@/lib/api';
import { Download, FileJson, Network, Code2, Globe } from 'lucide-react';

const FORMATS = [
  { key: 'json', label: 'JSON', icon: FileJson, desc: 'Raw graph nodes and edges.' },
  { key: 'cytoscape', label: 'Cytoscape', icon: Network, desc: 'Compatible with Cytoscape tooling.' },
  { key: 'graphml', label: 'GraphML', icon: Code2, desc: 'Structured XML graph format.' },
  { key: 'html', label: 'Interactive HTML', icon: Globe, desc: 'Standalone interactive visualization.' },
];

export default function ExportPage() {
  const download = (fmt: string) => window.open(getExportUrl(fmt), '_blank');

  return (
    <div className="space-y-8 max-w-4xl">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Export Graph</h1>
        <p className="text-slate-600">Download graph data in the format your downstream tools expect.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {FORMATS.map(({ key, label, icon: Icon, desc }) => (
          <button key={key} onClick={() => download(key)} className="surface p-5 text-left hover:border-[#afb8c1] transition-all">
            <div className="flex items-center gap-3 mb-3">
              <div className="h-9 w-9 rounded-md bg-slate-100 text-slate-700 flex items-center justify-center border border-slate-200">
                <Icon size={16} />
              </div>
              <h2 className="text-slate-900 font-medium">{label}</h2>
            </div>
            <p className="text-sm text-slate-600 mb-4">{desc}</p>
            <span className="inline-flex items-center gap-2 text-[#0969da] text-sm font-medium">
              <Download size={14} /> Download
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
