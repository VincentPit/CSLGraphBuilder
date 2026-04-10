'use client';
import { useState } from 'react';
import { ingestOpenTargets, ingestPubMed, getJob } from '@/lib/api';
import { Database, Search, Loader2, CheckCircle2, XCircle } from 'lucide-react';

type Tab = 'open-targets' | 'pubmed';

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs uppercase tracking-wide text-slate-500 font-semibold">{label}</label>
      {hint ? <p className="text-[11px] text-slate-500">{hint}</p> : null}
      {children}
    </div>
  );
}

export default function IngestPage() {
  const [tab, setTab] = useState<Tab>('open-targets');
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const [jobState, setJobState] = useState<'pending' | 'completed' | 'failed' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [diseaseId, setDiseaseId] = useState('');
  const [maxAssoc, setMaxAssoc] = useState('500');
  const [minScore, setMinScore] = useState('0.0');
  const [otTag, setOtTag] = useState('');
  const [query, setQuery] = useState('');
  const [maxArticles, setMaxArticles] = useState('50');
  const [email, setEmail] = useState('');
  const [pmTag, setPmTag] = useState('');
  const [loading, setLoading] = useState(false);

  const inputClass = 'w-full rounded-md bg-white border border-[#d0d7de] px-3 py-2.5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0969da]/20 focus:border-[#0969da]';

  async function pollJob(jobId: string) {
    setJobStatus('Job queued, waiting to start...');
    setJobState('pending');
    const interval = setInterval(async () => {
      const job = await getJob(jobId);
      setJobStatus(job.message ?? job.status);
      if (job.status === 'completed' || job.status === 'failed') {
        setJobState(job.status);
        clearInterval(interval);
      }
    }, 2000);
  }

  async function submitOT(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setJobStatus(null);
    setJobState(null);
    try {
      const res = await ingestOpenTargets({
        disease_id: diseaseId,
        max_associations: Number(maxAssoc),
        min_association_score: Number(minScore),
        tag: otTag || undefined,
      });
      pollJob(res.job_id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitPM(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setJobStatus(null);
    setJobState(null);
    try {
      const res = await ingestPubMed({
        query,
        max_articles: Number(maxArticles),
        email: email || undefined,
        tag: pmTag || undefined,
      });
      pollJob(res.job_id);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8 max-w-3xl">
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Ingest Sources</h1>
        <p className="text-slate-600">Import structured data from Open Targets or PubMed.</p>
      </header>

      <div className="inline-flex rounded-md border border-[#d0d7de] p-1 bg-white">
        {(['open-targets', 'pubmed'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition flex items-center gap-2 ${tab === t ? 'bg-[#ddf4ff] text-[#0969da] border border-[#54aeff]' : 'text-slate-600 hover:text-slate-900'}`}
          >
            {t === 'open-targets' ? <Database size={14} /> : <Search size={14} />}
            {t === 'open-targets' ? 'Open Targets' : 'PubMed'}
          </button>
        ))}
      </div>

      {tab === 'open-targets' ? (
        <form onSubmit={submitOT} className="surface p-6 space-y-5">
          <Field label="Disease ID" hint="Use EFO or MONDO IDs">
            <input required className={inputClass} value={diseaseId} onChange={(e) => setDiseaseId(e.target.value)} placeholder="EFO_0000275" />
          </Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Associations"><input className={inputClass} type="number" value={maxAssoc} onChange={(e) => setMaxAssoc(e.target.value)} /></Field>
            <Field label="Min Score (0-1)"><input className={inputClass} type="number" value={minScore} onChange={(e) => setMinScore(e.target.value)} /></Field>
          </div>
          <Field label="Tag" hint="Optional batch label"><input className={inputClass} value={otTag} onChange={(e) => setOtTag(e.target.value)} placeholder="cancer-2026" /></Field>
          <button className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-[#2da44e] border border-[#2c974b] text-white font-medium hover:bg-[#2c974b] transition disabled:opacity-50" disabled={loading}>
            {loading ? <><Loader2 size={14} className="animate-spin" /> Submitting</> : 'Run Open Targets Ingest'}
          </button>
        </form>
      ) : (
        <form onSubmit={submitPM} className="surface p-6 space-y-5">
          <Field label="Query" hint="PubMed syntax supported"><input required className={inputClass} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="BRCA1 AND cancer" /></Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Articles"><input className={inputClass} type="number" value={maxArticles} onChange={(e) => setMaxArticles(e.target.value)} /></Field>
            <Field label="Email"><input className={inputClass} value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" /></Field>
          </div>
          <Field label="Tag" hint="Optional batch label"><input className={inputClass} value={pmTag} onChange={(e) => setPmTag(e.target.value)} placeholder="pubmed-brca" /></Field>
          <button className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-[#2da44e] border border-[#2c974b] text-white font-medium hover:bg-[#2c974b] transition disabled:opacity-50" disabled={loading}>
            {loading ? <><Loader2 size={14} className="animate-spin" /> Submitting</> : 'Run PubMed Ingest'}
          </button>
        </form>
      )}

      {error && <div className="surface p-4 text-rose-700 text-sm flex items-center gap-2"><XCircle size={15} />{error}</div>}
      {jobStatus && (
        <div className="surface p-4 text-sm flex items-center gap-2 text-slate-700">
          {jobState === 'completed' ? <CheckCircle2 size={15} className="text-emerald-600" /> : jobState === 'failed' ? <XCircle size={15} className="text-rose-600" /> : <Loader2 size={15} className="animate-spin text-slate-500" />}
          {jobStatus}
        </div>
      )}
    </div>
  );
}
