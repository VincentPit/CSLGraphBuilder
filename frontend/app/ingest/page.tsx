'use client';
import { useState } from 'react';
import { ingestOpenTargets, ingestPubMed, getJob } from '@/lib/api';
import { Database, Search, Loader2, CheckCircle2, XCircle, ArrowRight, Beaker } from 'lucide-react';

type Tab = 'open-targets' | 'pubmed';

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

  const inputBase = "w-full bg-[#080d1a] border border-white/[0.06] rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500/30 focus:border-indigo-500/40 transition-all duration-200";

  async function pollJob(jobId: string) {
    setJobStatus('Job queued, waiting to start…');
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
    e.preventDefault(); setLoading(true); setError(null); setJobStatus(null); setJobState(null);
    try {
      const res = await ingestOpenTargets({ disease_id: diseaseId, max_associations: Number(maxAssoc), min_association_score: Number(minScore), tag: otTag || undefined });
      pollJob(res.job_id);
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  }

  async function submitPM(e: React.FormEvent) {
    e.preventDefault(); setLoading(true); setError(null); setJobStatus(null); setJobState(null);
    try {
      const res = await ingestPubMed({ query, max_articles: Number(maxArticles), email: email || undefined, tag: pmTag || undefined });
      pollJob(res.job_id);
    } catch (err: any) { setError(err.message); }
    finally { setLoading(false); }
  }

  return (
    <div className="max-w-2xl space-y-10">
      {/* Header */}
      <div className="space-y-3">
        <div className="flex items-center gap-2.5">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-sky-500/20 to-cyan-500/10 border border-sky-500/10 flex items-center justify-center">
            <Beaker size={18} className="text-sky-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Ingest Data Sources</h1>
            <p className="text-xs text-slate-500 font-medium">Structured third-party imports</p>
          </div>
        </div>
        <p className="text-sm text-slate-400 leading-relaxed max-w-xl">
          Populate the knowledge graph from curated databases. <strong className="text-slate-300">Open Targets</strong> provides gene-disease associations. <strong className="text-slate-300">PubMed</strong> pulls abstracts from NCBI.
        </p>
      </div>

      {/* Tab Switcher */}
      <div className="flex gap-1 bg-[#080d1a] border border-white/[0.04] rounded-xl p-1">
        {(['open-targets', 'pubmed'] as Tab[]).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${
              tab === t ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-lg shadow-indigo-600/15' : 'text-slate-500 hover:text-slate-300'
            }`}>
            {t === 'open-targets' ? <><Database size={14} /> Open Targets</> : <><Search size={14} /> PubMed</>}
          </button>
        ))}
      </div>

      {/* Open Targets form */}
      {tab === 'open-targets' && (
        <form onSubmit={submitOT} className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl p-8 space-y-6 shadow-xl shadow-black/20">
          <Field label="Disease ID" required hint="Use EFO or MONDO identifiers from Open Targets">
            <input required className={inputBase} value={diseaseId} onChange={(e) => setDiseaseId(e.target.value)} placeholder="EFO_0000275" />
          </Field>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Max Associations" hint="Upper limit on results">
              <input type="number" className={inputBase} value={maxAssoc} onChange={(e) => setMaxAssoc(e.target.value)} />
            </Field>
            <Field label="Min Score (0–1)" hint="Filter low-confidence links">
              <input type="number" className={inputBase} value={minScore} onChange={(e) => setMinScore(e.target.value)} />
            </Field>
          </div>
          <Field label="Tag" hint="Label this batch for tracking">
            <input className={inputBase} value={otTag} onChange={(e) => setOtTag(e.target.value)} placeholder="e.g. cancer-2026" />
          </Field>
          <SubmitBtn loading={loading}>Run Open Targets Ingest</SubmitBtn>
        </form>
      )}

      {/* PubMed form */}
      {tab === 'pubmed' && (
        <form onSubmit={submitPM} className="bg-gradient-to-b from-[#0f1829] to-[#0d1526] border border-white/[0.06] rounded-2xl p-8 space-y-6 shadow-xl shadow-black/20">
          <Field label="Search Query" required hint="Standard PubMed syntax: AND/OR/NOT">
            <input required className={inputBase} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="BRCA1 AND cancer" />
          </Field>
          <div className="grid grid-cols-2 gap-4">
            <Field label="Max Articles" hint="Number of articles to download">
              <input type="number" className={inputBase} value={maxArticles} onChange={(e) => setMaxArticles(e.target.value)} />
            </Field>
            <Field label="Email (NCBI)" hint="Required by NCBI for API access">
              <input className={inputBase} value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" />
            </Field>
          </div>
          <Field label="Tag" hint="Label this batch for tracking">
            <input className={inputBase} value={pmTag} onChange={(e) => setPmTag(e.target.value)} placeholder="e.g. pubmed-brca1" />
          </Field>
          <SubmitBtn loading={loading}>Run PubMed Ingest</SubmitBtn>
        </form>
      )}

      {/* Status */}
      {error && (
        <div className="flex items-center gap-3 text-red-300 text-sm bg-red-500/[0.08] border border-red-500/15 rounded-xl px-5 py-4">
          <XCircle size={16} />{error}
        </div>
      )}
      {jobStatus && (
        <div className={`flex items-center gap-3 text-sm rounded-xl px-5 py-4 border ${
          jobState === 'completed' ? 'bg-emerald-500/[0.08] border-emerald-500/15 text-emerald-300' :
          jobState === 'failed'    ? 'bg-red-500/[0.08] border-red-500/15 text-red-300' :
          'bg-indigo-500/[0.08] border-indigo-500/15 text-indigo-300'
        }`}>
          {jobState === 'completed' ? <CheckCircle2 size={16}/> : jobState === 'failed' ? <XCircle size={16}/> : <Loader2 size={16} className="animate-spin"/>}
          {jobStatus}
        </div>
      )}
    </div>
  );
}

function Field({ label, required, hint, children }: { label: string; required?: boolean; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <label className="block text-xs font-semibold text-slate-300 tracking-wide">
        {label}
        {required && <span className="text-indigo-400 ml-0.5">*</span>}
        {!required && <span className="text-slate-600 font-normal ml-1">· optional</span>}
      </label>
      {hint && <p className="text-[11px] text-slate-600 -mt-1">{hint}</p>}
      {children}
    </div>
  );
}

function SubmitBtn({ loading, children }: { loading: boolean; children: React.ReactNode }) {
  return (
    <button type="submit" disabled={loading}
      className="group w-full flex items-center justify-center gap-2.5 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-semibold px-6 py-3.5 rounded-xl transition-all duration-200 shadow-lg shadow-indigo-600/20">
      {loading ? <><Loader2 size={15} className="animate-spin"/>Submitting…</> : children}
    </button>
  );
}
