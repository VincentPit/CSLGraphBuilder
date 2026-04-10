'use client';
import { useState } from 'react';
import { ingestOpenTargets, ingestPubMed, getJob } from '@/lib/api';

type Tab = 'open-targets' | 'pubmed';

export default function IngestPage() {
  const [tab, setTab] = useState<Tab>('open-targets');
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Open Targets form
  const [diseaseId, setDiseaseId] = useState('');
  const [maxAssoc, setMaxAssoc] = useState('500');
  const [minScore, setMinScore] = useState('0.0');
  const [otTag, setOtTag] = useState('');

  // PubMed form
  const [query, setQuery] = useState('');
  const [maxArticles, setMaxArticles] = useState('50');
  const [email, setEmail] = useState('');
  const [pmTag, setPmTag] = useState('');

  const [loading, setLoading] = useState(false);

  async function pollJob(jobId: string) {
    setJobStatus('pending');
    const interval = setInterval(async () => {
      const job = await getJob(jobId);
      setJobStatus(`${job.status} — ${job.message ?? ''}`);
      if (job.status === 'completed' || job.status === 'failed') clearInterval(interval);
    }, 2000);
  }

  async function submitOT(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      const res = await ingestOpenTargets({ disease_id: diseaseId, max_associations: Number(maxAssoc), min_association_score: Number(minScore), tag: otTag || undefined });
      pollJob(res.job_id);
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function submitPM(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true); setError(null);
    try {
      const res = await ingestPubMed({ query, max_articles: Number(maxArticles), email: email || undefined, tag: pmTag || undefined });
      pollJob(res.job_id);
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <h1 className="text-2xl font-bold">Ingest Data</h1>
      <div className="flex gap-2">
        {(['open-targets', 'pubmed'] as Tab[]).map((t) => (
          <button key={t} onClick={() => setTab(t)} className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${tab === t ? 'bg-sky-600 text-white' : 'bg-slate-700 text-slate-300 hover:bg-slate-600'}`}>
            {t === 'open-targets' ? 'Open Targets' : 'PubMed'}
          </button>
        ))}
      </div>

      {tab === 'open-targets' && (
        <form onSubmit={submitOT} className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
          <Field label="Disease ID *" value={diseaseId} onChange={setDiseaseId} placeholder="EFO_0000275" required />
          <div className="grid grid-cols-2 gap-4">
            <Field label="Max Associations" value={maxAssoc} onChange={setMaxAssoc} type="number" />
            <Field label="Min Score (0–1)" value={minScore} onChange={setMinScore} type="number" />
          </div>
          <Field label="Tag (optional)" value={otTag} onChange={setOtTag} />
          <SubmitBtn loading={loading}>Run Open Targets Ingest</SubmitBtn>
        </form>
      )}

      {tab === 'pubmed' && (
        <form onSubmit={submitPM} className="bg-slate-800 rounded-xl p-5 border border-slate-700 space-y-4">
          <Field label="Search Query *" value={query} onChange={setQuery} placeholder="BRCA1 AND cancer" required />
          <div className="grid grid-cols-2 gap-4">
            <Field label="Max Articles" value={maxArticles} onChange={setMaxArticles} type="number" />
            <Field label="Email (required by NCBI)" value={email} onChange={setEmail} placeholder="you@example.com" />
          </div>
          <Field label="Tag (optional)" value={pmTag} onChange={setPmTag} />
          <SubmitBtn loading={loading}>Run PubMed Ingest</SubmitBtn>
        </form>
      )}

      {error && <p className="text-red-400 text-sm">{error}</p>}
      {jobStatus && <p className="text-slate-300 text-sm font-mono">Job: {jobStatus}</p>}
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type = 'text', required = false }: { label: string; value: string; onChange: (v: string) => void; placeholder?: string; type?: string; required?: boolean }) {
  return (
    <div>
      <label className="block text-sm text-slate-400 mb-1">{label}</label>
      <input required={required} type={type} className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-500" value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} />
    </div>
  );
}

function SubmitBtn({ loading, children }: { loading: boolean; children: React.ReactNode }) {
  return (
    <button type="submit" disabled={loading} className="bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors">
      {loading ? 'Submitting…' : children}
    </button>
  );
}
