'use client';
import { useState, useRef } from 'react';
import { processDocument, getJobStreamUrl } from '@/lib/api';

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
      const job = await processDocument({
        url: url || undefined,
        text: text || undefined,
        source_label: label || undefined,
      });

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

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-bold">Process Document</h1>
      <form onSubmit={handleSubmit} className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
        <Field label="URL" value={url} onChange={setUrl} placeholder="https://…" />
        <div>
          <label className="block text-sm text-slate-400 mb-1">OR Paste Text</label>
          <textarea
            className="w-full bg-slate-900 border border-slate-600 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-500"
            rows={5}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste raw document text…"
          />
        </div>
        <Field label="Source Label (optional)" value={label} onChange={setLabel} placeholder="e.g. PubMed:12345" />
        <button
          type="submit"
          disabled={status === 'running' || (!url && !text)}
          className="bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
        >
          {status === 'running' ? 'Processing…' : 'Process'}
        </button>
      </form>

      {log.length > 0 && (
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 font-mono text-xs space-y-1 max-h-64 overflow-y-auto">
          {log.map((l, i) => (
            <div key={i} className={l.includes('failed') ? 'text-red-400' : 'text-slate-300'}>{l}</div>
          ))}
        </div>
      )}
      {status === 'done' && <p className="text-green-400 text-sm">Processing complete ✓</p>}
    </div>
  );
}

function Field({ label, value, onChange, placeholder }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div>
      <label className="block text-sm text-slate-400 mb-1">{label}</label>
      <input
        className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-500"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}
