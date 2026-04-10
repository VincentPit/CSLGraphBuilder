'use client';
import { getExportUrl } from '@/lib/api';
import { Download, FileJson, Network, Code2, Globe, ArrowDownToLine } from 'lucide-react';

const FORMATS = [
  {
    key: 'json',
    label: 'JSON',
    icon: FileJson,
    desc: 'Raw graph data as a JSON array of nodes and edges. Best for custom Python/data pipelines.',
    gradient: 'from-sky-500/15 to-sky-500/5',
    border: 'border-sky-500/10 hover:border-sky-500/30',
    color: 'text-sky-400',
    iconBg: 'bg-sky-500/10',
  },
  {
    key: 'cytoscape',
    label: 'Cytoscape',
    icon: Network,
    desc: 'Cytoscape.js JSON format. Use with Cytoscape desktop or neo4j browser tooling.',
    gradient: 'from-indigo-500/15 to-indigo-500/5',
    border: 'border-indigo-500/10 hover:border-indigo-500/30',
    color: 'text-indigo-400',
    iconBg: 'bg-indigo-500/10',
  },
  {
    key: 'graphml',
    label: 'GraphML',
    icon: Code2,
    desc: 'XML-based graph interchange format. Compatible with Gephi, yEd, and academic tooling.',
    gradient: 'from-violet-500/15 to-violet-500/5',
    border: 'border-violet-500/10 hover:border-violet-500/30',
    color: 'text-violet-400',
    iconBg: 'bg-violet-500/10',
  },
  {
    key: 'html',
    label: 'Interactive HTML',
    icon: Globe,
    desc: 'Standalone HTML file with an interactive force-directed graph visualization.',
    gradient: 'from-emerald-500/15 to-emerald-500/5',
    border: 'border-emerald-500/10 hover:border-emerald-500/30',
    color: 'text-emerald-400',
    iconBg: 'bg-emerald-500/10',
  },
];

export default function ExportPage() {
  function download(fmt: string) {
    window.open(getExportUrl(fmt), '_blank');
  }

  return (
    <div className="max-w-3xl space-y-10">
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-emerald-500/20 to-teal-500/10 border border-emerald-500/10 flex items-center justify-center">
            <ArrowDownToLine size={18} className="text-emerald-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Export Graph</h1>
            <p className="text-xs text-slate-500 font-medium">Download in multiple formats</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
          Download the current knowledge graph as a static file. Each format serves a different ecosystem. Files are generated on-demand from the live database.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {FORMATS.map(({ key, label, icon: Icon, desc, gradient, border, color, iconBg }) => (
          <button key={key} onClick={() => download(key)}
            className={`group flex flex-col items-start gap-4 p-6 rounded-2xl border bg-gradient-to-b ${gradient} ${border} text-left transition-all duration-300 hover:shadow-lg hover:shadow-black/10`}>
            <div className="flex items-center gap-3 w-full">
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${iconBg}`}>
                <Icon size={18} className={color} />
              </div>
              <span className="font-bold text-[15px] text-white">{label}</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed flex-1">{desc}</p>
            <div className={`flex items-center gap-1.5 text-xs font-semibold ${color} opacity-60 group-hover:opacity-100 transition-opacity duration-200`}>
              <Download size={12} /> Download file
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
