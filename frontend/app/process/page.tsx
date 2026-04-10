'use client';
import { useState, useRef } from 'react';
import { processDocument, getJobStreamUrl } from '@/lib/api';
import { FileText, Link2, Tag, Loader2, CheckCircle2, XCircle } from 'lucide-react';

function FieldLabel({ icon: Icon, label, optional }: { icon: any; label: string; optional?: boolean }) {
  return (
    <label className="flex items-center gap-2 text-xs uppercase tracking-wide text-slate-500 font-semibold mb-2">
      <Icon size={13} />
      {label}
      {optional ? <span className="text-slate-400 normal-case font-medium">optional</span> : null}
    </label>
  );
}

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
      es.onerror = () => {
        setStatus('error');
        es.close();
      };
    } catch (err: any) {
      setLog([err.message]);
      setStatus('error');
    }
  }

  const input = 'w-full rounded-md bg-white border border-[#d0d7de] px-3 py-2.5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0969da]/20 focus:border-[#0969da]';

  return (
    <div className="space-y-8 max-w-3xl">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Process Document</h1>
        <p className="text-slate-600 max-w-2xl">Extract entities and relationships from URLs or pasted text using your configured LLM pipeline.</p>
      </header>

      <div className="surface p-6 md:p-7">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <FieldLabel icon={Link2} label="Source URL" />
            <input className={input} value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://pubmed.ncbi.nlm.nih.gov/12345" />
          </div>

          <div>
            <FieldLabel icon={FileText} label="Raw Text" />
            <textarea className={`${input} resize-none`} rows={7} value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste article text or report content here" />
          </div>

          <div>
            <FieldLabel icon={Tag} label="Source Label" optional />
            <input className={input} value={label} onChange={(e) => setLabel(e.target.value)} placeholder="PubMed:12345" />
          </div>

          <button
            type="submit"
            disabled={status === 'running' || (!url && !text)}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-[#2da44e] border border-[#2c974b] text-white font-medium hover:bg-[#2c974b] disabled:opacity-40 disabled:cursor-not-allowed transition"
          >
            {status === 'running' ? <><Loader2 size={15} className="animate-spin" /> Processing</> : 'Start Extraction'}
          </button>
        </form>
      </div>

      {log.length > 0 && (
        <div className="surface p-5">
          <p className="text-xs uppercase tracking-wide text-slate-500 font-semibold mb-3">Live Progress</p>
          <div className="rounded-lg bg-slate-50 border border-slate-200 p-4 max-h-72 overflow-y-auto font-mono text-xs space-y-1.5">
            {log.map((line, i) => (
              <div key={i} className={line.includes('failed') ? 'text-rose-700' : line.includes('completed') ? 'text-emerald-700' : 'text-slate-600'}>{line}</div>
            ))}
          </div>
        </div>
      )}

      {status === 'done' && (
        <div className="surface p-4 text-emerald-700 text-sm flex items-center gap-2">
          <CheckCircle2 size={16} /> Processing complete. Entities and relationships were saved.
        </div>
      )}

      {status === 'error' && (
        <div className="surface p-4 text-rose-700 text-sm flex items-center gap-2">
          <XCircle size={16} /> Processing failed. Check the live log for details.
        </div>
      )}
    </div>
  );
}
