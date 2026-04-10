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
      <header>
        <h1 className="page-title">Export Graph</h1>
        <p className="page-desc">Download graph data in the format your downstream tools expect.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {FORMATS.map(({ key, label, icon: Icon, desc }) => (
          <button key={key} onClick={() => download(key)} className="card p-5 text-left hover:border-[var(--accent)] transition-all group">
            <div className="flex items-center gap-3 mb-3">
              <div className="h-9 w-9 rounded-lg flex items-center justify-center" style={{ background: 'var(--accent-muted)', color: 'var(--accent)' }}>
                <Icon size={16} />
              </div>
              <h2 className="font-medium" style={{ color: 'var(--text-primary)' }}>{label}</h2>
            </div>
            <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>{desc}</p>
            <span className="inline-flex items-center gap-2 text-sm font-medium" style={{ color: 'var(--accent)' }}>
              <Download size={14} /> Download
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
