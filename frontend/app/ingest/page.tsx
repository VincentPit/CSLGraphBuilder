'use client';
import { useState } from 'react';
import { ingestOpenTargets, ingestPubMed, ingestCrawl, getJob } from '@/lib/api';
import { Database, Search, Globe, Loader2, CheckCircle2, XCircle } from 'lucide-react';

type Tab = 'open-targets' | 'pubmed' | 'crawl';

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label className="field-label">{label}</label>
      {hint ? <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{hint}</p> : null}
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
  const [crawlUrls, setCrawlUrls] = useState('');
  const [crawlMaxPages, setCrawlMaxPages] = useState('10');
  const [crawlDomains, setCrawlDomains] = useState('');
  const [crawlTag, setCrawlTag] = useState('');
  const [loading, setLoading] = useState(false);

  const inputClass = 'input';

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

  async function submitCrawl(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setJobStatus(null);
    setJobState(null);
    try {
      const urls = crawlUrls.split('\n').map(u => u.trim()).filter(Boolean);
      if (urls.length === 0) throw new Error('Enter at least one URL');
      const domains = crawlDomains.split(',').map(d => d.trim()).filter(Boolean);
      const res = await ingestCrawl({
        urls,
        max_pages: Number(crawlMaxPages),
        allowed_domains: domains.length > 0 ? domains : undefined,
        tag: crawlTag || undefined,
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
      <header>
        <h1 className="page-title">Ingest Sources</h1>
        <p className="page-desc">Import structured data from Open Targets, PubMed, or the web.</p>
      </header>

      <div className="inline-flex rounded-lg p-1" style={{ background: 'var(--bg-muted)', border: '1px solid var(--border)' }}>
        {(['open-targets', 'pubmed', 'crawl'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition flex items-center gap-2 ${tab === t ? 'btn-ghost' : ''}`}
            style={tab === t ? { background: 'white', color: 'var(--accent)', boxShadow: '0 1px 3px rgba(0,0,0,.08)' } : { color: 'var(--text-muted)' }}
          >
            {t === 'open-targets' ? <Database size={14} /> : t === 'pubmed' ? <Search size={14} /> : <Globe size={14} />}
            {t === 'open-targets' ? 'Open Targets' : t === 'pubmed' ? 'PubMed' : 'Web Crawl'}
          </button>
        ))}
      </div>

      {tab === 'crawl' ? (
        <form onSubmit={submitCrawl} className="card p-6 space-y-5">
          <Field label="URLs" hint="One URL per line — starting points for the crawler">
            <textarea required className={inputClass} rows={4} value={crawlUrls} onChange={(e) => setCrawlUrls(e.target.value)} placeholder={"https://example.com\nhttps://example.com/docs"} />
          </Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Pages" hint="Maximum pages to crawl per seed URL"><input className={inputClass} type="number" value={crawlMaxPages} onChange={(e) => setCrawlMaxPages(e.target.value)} /></Field>
            <Field label="Allowed Domains" hint="Comma-separated, leave blank for seed domains only"><input className={inputClass} value={crawlDomains} onChange={(e) => setCrawlDomains(e.target.value)} placeholder="example.com, docs.example.com" /></Field>
          </div>
          <Field label="Tag" hint="Optional batch label"><input className={inputClass} value={crawlTag} onChange={(e) => setCrawlTag(e.target.value)} placeholder="web-docs-2026" /></Field>
          <button className="btn-primary" disabled={loading}>
            {loading ? <><Loader2 size={14} className="animate-spin" /> Crawling</> : 'Start Web Crawl'}
          </button>
        </form>
      ) : tab === 'open-targets' ? (
        <form onSubmit={submitOT} className="card p-6 space-y-5">
          <Field label="Disease ID" hint="Use EFO or MONDO IDs">
            <input required className={inputClass} value={diseaseId} onChange={(e) => setDiseaseId(e.target.value)} placeholder="EFO_0000275" />
          </Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Associations"><input className={inputClass} type="number" value={maxAssoc} onChange={(e) => setMaxAssoc(e.target.value)} /></Field>
            <Field label="Min Score (0-1)"><input className={inputClass} type="number" value={minScore} onChange={(e) => setMinScore(e.target.value)} /></Field>
          </div>
          <Field label="Tag" hint="Optional batch label"><input className={inputClass} value={otTag} onChange={(e) => setOtTag(e.target.value)} placeholder="cancer-2026" /></Field>
          <button className="btn-primary" disabled={loading}>
            {loading ? <><Loader2 size={14} className="animate-spin" /> Submitting</> : 'Run Open Targets Ingest'}
          </button>
        </form>
      ) : (
        <form onSubmit={submitPM} className="card p-6 space-y-5">
          <Field label="Query" hint="PubMed syntax supported"><input required className={inputClass} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="BRCA1 AND cancer" /></Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Articles"><input className={inputClass} type="number" value={maxArticles} onChange={(e) => setMaxArticles(e.target.value)} /></Field>
            <Field label="Email"><input className={inputClass} value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" /></Field>
          </div>
          <Field label="Tag" hint="Optional batch label"><input className={inputClass} value={pmTag} onChange={(e) => setPmTag(e.target.value)} placeholder="pubmed-brca" /></Field>
          <button className="btn-primary" disabled={loading}>
            {loading ? <><Loader2 size={14} className="animate-spin" /> Submitting</> : 'Run PubMed Ingest'}
          </button>
        </form>
      )}

      {error && <div className="card p-4 text-sm flex items-center gap-2" style={{ color: 'var(--danger)' }}><XCircle size={15} />{error}</div>}
      {jobStatus && (
        <div className="card p-4 text-sm flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
          {jobState === 'completed' ? <CheckCircle2 size={15} className="text-emerald-600" /> : jobState === 'failed' ? <XCircle size={15} className="text-rose-600" /> : <Loader2 size={15} className="animate-spin text-slate-500" />}
          {jobStatus}
        </div>
      )}
    </div>
  );
}
