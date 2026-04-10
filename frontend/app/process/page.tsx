'use client';
import { useState, useRef } from 'react';
import { processDocument, getJobStreamUrl } from '@/lib/api';
import { FileText, Link2, Tag, Loader2, CheckCircle2, XCircle } from 'lucide-react';

function FieldLabel({ icon: Icon, label, optional }: { icon: any; label: string; optional?: boolean }) {
  return (
    <label className="field-label flex items-center gap-2">
      <Icon size={13} />
      {label}
      {optional ? <span className="normal-case font-medium" style={{ color: 'var(--text-muted)' }}>optional</span> : null}
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

  const input = 'input';

  return (
    <div className="space-y-8 max-w-3xl">
      <header>
        <h1 className="page-title">Process Document</h1>
        <p className="page-desc">Extract entities and relationships from URLs or pasted text using your configured LLM pipeline.</p>
      </header>

      <div className="card p-6 md:p-7">
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
            className="btn-primary"
          >
            {status === 'running' ? <><Loader2 size={15} className="animate-spin" /> Processing</> : 'Start Extraction'}
          </button>
        </form>
      </div>

      {log.length > 0 && (
        <div className="card p-5">
          <p className="field-label">Live Progress</p>
          <div className="rounded-lg p-4 max-h-72 overflow-y-auto font-mono text-xs space-y-1.5" style={{ background: 'var(--bg-muted)', border: '1px solid var(--border)' }}>
            {log.map((line, i) => (
              <div key={i} className={line.includes('failed') ? 'text-rose-700' : line.includes('completed') ? 'text-emerald-700' : 'text-slate-600'}>{line}</div>
            ))}
          </div>
        </div>
      )}

      {status === 'done' && (
        <div className="card p-4 text-sm flex items-center gap-2" style={{ color: 'var(--success)' }}>
          <CheckCircle2 size={16} /> Processing complete. Entities and relationships were saved.
        </div>
      )}

      {status === 'error' && (
        <div className="card p-4 text-sm flex items-center gap-2" style={{ color: 'var(--danger)' }}>
          <XCircle size={16} /> Processing failed. Check the live log for details.
        </div>
      )}
    </div>
  );
}
