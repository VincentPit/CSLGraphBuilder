'use client';
import { getExportUrl } from '@/lib/api';
import { Download } from 'lucide-react';

const FORMATS = [
  { key: 'json',       label: 'JSON',       desc: 'Raw graph data as JSON'          },
  { key: 'cytoscape',  label: 'Cytoscape',  desc: 'Cytoscape.js compatible JSON'    },
  { key: 'graphml',    label: 'GraphML',    desc: 'XML-based graph interchange'     },
  { key: 'html',       label: 'HTML',       desc: 'Interactive HTML visualisation'  },
];

export default function ExportPage() {
  function download(fmt: string) {
    window.open(getExportUrl(fmt), '_blank');
  }

  return (
    <div className="max-w-xl space-y-6">
      <h1 className="text-2xl font-bold">Export Graph</h1>
      <p className="text-slate-400 text-sm">Download the current in-memory knowledge graph in your preferred format.</p>
      <div className="grid grid-cols-2 gap-4">
        {FORMATS.map(({ key, label, desc }) => (
          <button key={key} onClick={() => download(key)}
            className="flex flex-col items-start gap-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl p-5 text-left transition-colors group">
            <div className="flex items-center gap-2 text-sky-400 group-hover:text-sky-300">
              <Download size={16} />
              <span className="font-semibold">{label}</span>
            </div>
            <p className="text-xs text-slate-500">{desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
