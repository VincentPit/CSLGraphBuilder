'use client';
import { useState, useRef } from 'react';
import { processDocument, getJobStreamUrl } from '@/lib/api';
import { FileText, Link2, Tag, Loader2, CheckCircle2, XCircle, Zap, ArrowRight } from 'lucide-react';

export default function ProcessPage() {
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  const [label, setLabel] = useState('');
  const [log, setLog] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'error'>('idle');
  const esRef = useRef<EventSource | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLog([]);
    setStatus('running');
    try {
      const job = await processDocument({ url: url || undefined, text: text || undefined, source_label: label || undefined });
      const es = new EventSource(getJobStreamUrl(job.job_id));
      esRef.current = es;
      es.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          setLog((prev) => [...prev, `[${data.status}] ${data.message ?? ''} (${Math.round((data.progress ?? 0) * 100)}%)`]);
          if (data.status === 'completed' || data.status === 'failed') {
            setStatus(data.status === 'completed' ? 'done' : 'error');
            es.close();
          }
        } catch {}
      };
      es.onerror = () => { setStatus('error'); es.close(); };
    } catch (err: any) {
      setLog([err.message]);
      setStatus('error');
    }
  }

  const inputBase = "w-full bg-[#080d1a] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 focus:border-indigo-500/40 transition-all duration-200";

  return (
    <div className="max-w-2xl space-y-10">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-violet-500/10 border border-indigo-500/10 flex items-center justify-center">
            <Zap size={18} className="text-indigo-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Process Document</h1>
            <p className="text-xs text-slate-500 font-medium">LLM-powered extraction pipeline</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
          Extract entities and relationships from any URL or raw text. The LLM will chunk the content, identify named entities (genes, diseases, pathways, etc.), and create relationship edges automatically.
        </p>
      </div>

      {/* Form Card */}
      <div className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl p-8 shadow-xl shadow-black/20">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* URL Field */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 tracking-wide">
              <Link2 size={13} className="text-slate-500" />
              Source URL
            </label>
            <input
              className={inputBase}
              value={url} onChange={(e) => setUrl(e.target.value)}
              placeholder="https://pubmed.ncbi.nlm.nih.gov/12345"
            />
          </div>

          {/* Divider */}
          <div className="flex items-center gap-4">
            <div className="flex-1 h-px bg-white/[0.04]" />
            <span className="text-[10px] font-bold text-slate-600 uppercase tracking-widest">or</span>
            <div className="flex-1 h-px bg-white/[0.04]" />
          </div>

          {/* Text Area */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 tracking-wide">
              <FileText size={13} className="text-slate-500" />
              Paste raw text
            </label>
            <textarea
              className={`${inputBase} resize-none`}
              rows={6} value={text} onChange={(e) => setText(e.target.value)}
              placeholder="Paste the full text of a research paper, article, or any document…"
            />
          </div>

          {/* Label Field */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs font-semibold text-slate-300 tracking-wide">
              <Tag size={13} className="text-slate-500" />
              Source Label
              <span className="text-slate-600 font-normal">· optional</span>
            </label>
            <input
              className={inputBase}
              value={label} onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g. PubMed:12345"
            />
          </div>

          {/* Submit */}
          <button
            type="submit"
            disabled={status === 'running' || (!url && !text)}
            className="group w-full flex items-center justify-center gap-2.5 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-semibold px-6 py-3.5 rounded-xl transition-all duration-200 shadow-lg shadow-indigo-600/20 hover:shadow-indigo-500/30"
          >
            {status === 'running'
              ? <><Loader2 size={15} className="animate-spin" /> Processing…</>
              : <>Start Extraction <ArrowRight size={14} className="group-hover:translate-x-0.5 transition-transform" /></>
            }
          </button>
        </form>
      </div>

      {/* Live log */}
      {log.length > 0 && (
        <div className="space-y-2.5">
          <p className="text-[11px] font-semibold text-slate-500 uppercase tracking-widest">Live Progress</p>
          <div className="bg-[#060a14] border border-white/[0.04] rounded-xl p-5 font-mono text-xs space-y-1.5 max-h-64 overflow-y-auto">
            {log.map((l, i) => (
              <div key={i} className={`leading-relaxed ${l.includes('failed') ? 'text-red-400' : l.includes('completed') ? 'text-emerald-400' : 'text-slate-500'}`}>
                <span className="text-slate-700 mr-2">{String(i + 1).padStart(2, '0')}</span>{l}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status banners */}
      {status === 'done' && (
        <div className="flex items-center gap-3 text-emerald-300 text-sm bg-emerald-500/[0.08] border border-emerald-500/15 rounded-xl px-5 py-4">
          <CheckCircle2 size={18} />
          <div>
            <p className="font-semibold">Extraction complete</p>
            <p className="text-emerald-400/60 text-xs mt-0.5">Entities and relationships have been saved to the graph.</p>
          </div>
        </div>
      )}
      {status === 'error' && (
        <div className="flex items-center gap-3 text-red-300 text-sm bg-red-500/[0.08] border border-red-500/15 rounded-xl px-5 py-4">
          <XCircle size={18} />
          <div>
            <p className="font-semibold">Processing failed</p>
            <p className="text-red-400/60 text-xs mt-0.5">Check the log above for details.</p>
          </div>
        </div>
      )}
    </div>
  );
}
