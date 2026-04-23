'use client';

import { useState } from 'react';
import { Database, Globe, Loader2, Search, XCircle } from 'lucide-react';
import {
  cancelJob,
  formatApiError,
  ingestCrawl,
  ingestOpenTargets,
  ingestPubMed,
} from '@/lib/api';
import JobTimeline from '@/components/JobTimeline';
import { useJobStream } from '@/lib/useJobStream';

type Tab = 'open-targets' | 'pubmed' | 'crawl';

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label className="field-label">{label}</label>
      {hint && <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>{hint}</p>}
      {children}
    </div>
  );
}

export default function IngestPage() {
  const [tab, setTab] = useState<Tab>('open-targets');
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const job = useJobStream(jobId);

  // Form state
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

  const isRunning = job
    ? job.status === 'pending' || job.status === 'running'
    : submitting;

  async function startJob<T>(fn: () => Promise<{ job_id: string }>) {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const { job_id } = await fn();
      setJobId(job_id);
    } catch (err: any) {
      setSubmitError(formatApiError(err, 'Submit failed'));
    } finally {
      setSubmitting(false);
    }
  }

  function submitOT(e: React.FormEvent) {
    e.preventDefault();
    return startJob(() =>
      ingestOpenTargets({
        disease_id: diseaseId,
        max_associations: Number(maxAssoc),
        min_association_score: Number(minScore),
        tag: otTag || undefined,
      })
    );
  }

  function submitPM(e: React.FormEvent) {
    e.preventDefault();
    return startJob(() =>
      ingestPubMed({
        query,
        max_articles: Number(maxArticles),
        email: email || undefined,
        tag: pmTag || undefined,
      })
    );
  }

  function submitCrawl(e: React.FormEvent) {
    e.preventDefault();
    return startJob(() => {
      const urls = crawlUrls.split('\n').map((u) => u.trim()).filter(Boolean);
      if (!urls.length) throw new Error('Enter at least one URL');
      const domains = crawlDomains.split(',').map((d) => d.trim()).filter(Boolean);
      return ingestCrawl({
        urls,
        max_pages: Number(crawlMaxPages),
        allowed_domains: domains.length ? domains : undefined,
        tag: crawlTag || undefined,
      });
    });
  }

  async function handleCancel() {
    if (!jobId) return;
    try {
      await cancelJob(jobId);
    } catch {
      /* surfaced in job state */
    }
  }

  return (
    <div className="space-y-8 max-w-3xl">
      <header className="space-y-2">
        <span
          className="badge badge-brand inline-flex"
          style={{ background: 'var(--accent-soft)', border: '1px solid rgba(99,102,241,0.25)' }}
        >
          <Database size={10} className="opacity-70" />
          3 sources · same timeline UI
        </span>
        <h1 className="page-title">Ingest Sources</h1>
        <p className="page-desc">
          Pull from <span className="text-gradient font-semibold">Open Targets</span>,
          {' '}<span className="text-gradient font-semibold">PubMed</span>, or
          {' '}<span className="text-gradient font-semibold">crawl the web</span>.
          Crawls feed pages through the same extraction pipeline as the Process page.
        </p>
      </header>

      <div
        className="inline-flex rounded-lg p-1"
        style={{ background: 'var(--bg-muted)', border: '1px solid var(--border-default)' }}
      >
        {(['open-targets', 'pubmed', 'crawl'] as Tab[]).map((t) => {
          const active = tab === t;
          return (
            <button
              key={t}
              onClick={() => setTab(t)}
              disabled={isRunning}
              className="px-4 py-2 rounded-md text-sm font-medium transition flex items-center gap-2"
              style={
                active
                  ? { background: 'white', color: 'var(--accent)', boxShadow: '0 1px 3px rgba(0,0,0,.08)' }
                  : { color: 'var(--text-muted)' }
              }
            >
              {t === 'open-targets' ? (
                <Database size={14} />
              ) : t === 'pubmed' ? (
                <Search size={14} />
              ) : (
                <Globe size={14} />
              )}
              {t === 'open-targets' ? 'Open Targets' : t === 'pubmed' ? 'PubMed' : 'Web Crawl'}
            </button>
          );
        })}
      </div>

      {tab === 'crawl' ? (
        <form onSubmit={submitCrawl} className="card p-6 space-y-5">
          <Field label="URLs" hint="One URL per line — starting points for the crawler">
            <textarea
              required
              className="input"
              rows={4}
              value={crawlUrls}
              onChange={(e) => setCrawlUrls(e.target.value)}
              placeholder={'https://example.com\nhttps://example.com/docs'}
              disabled={isRunning}
            />
          </Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Pages" hint="Maximum pages to crawl per seed URL">
              <input
                className="input"
                type="number"
                value={crawlMaxPages}
                onChange={(e) => setCrawlMaxPages(e.target.value)}
                disabled={isRunning}
              />
            </Field>
            <Field label="Allowed Domains" hint="Comma-separated, leave blank for seed domains only">
              <input
                className="input"
                value={crawlDomains}
                onChange={(e) => setCrawlDomains(e.target.value)}
                placeholder="example.com, docs.example.com"
                disabled={isRunning}
              />
            </Field>
          </div>
          <Field label="Tag" hint="Optional batch label">
            <input
              className="input"
              value={crawlTag}
              onChange={(e) => setCrawlTag(e.target.value)}
              placeholder="web-docs-2026"
              disabled={isRunning}
            />
          </Field>
          <SubmitButton loading={isRunning} label="Start Web Crawl" />
        </form>
      ) : tab === 'open-targets' ? (
        <form onSubmit={submitOT} className="card p-6 space-y-5">
          <Field label="Disease ID" hint="Use EFO or MONDO IDs">
            <input
              required
              className="input"
              value={diseaseId}
              onChange={(e) => setDiseaseId(e.target.value)}
              placeholder="EFO_0000275"
              disabled={isRunning}
            />
          </Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Associations">
              <input
                className="input"
                type="number"
                value={maxAssoc}
                onChange={(e) => setMaxAssoc(e.target.value)}
                disabled={isRunning}
              />
            </Field>
            <Field label="Min Score (0-1)">
              <input
                className="input"
                type="number"
                value={minScore}
                onChange={(e) => setMinScore(e.target.value)}
                disabled={isRunning}
              />
            </Field>
          </div>
          <Field label="Tag" hint="Optional batch label">
            <input
              className="input"
              value={otTag}
              onChange={(e) => setOtTag(e.target.value)}
              placeholder="cancer-2026"
              disabled={isRunning}
            />
          </Field>
          <SubmitButton loading={isRunning} label="Run Open Targets Ingest" />
        </form>
      ) : (
        <form onSubmit={submitPM} className="card p-6 space-y-5">
          <Field label="Query" hint="PubMed syntax supported">
            <input
              required
              className="input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="BRCA1 AND cancer"
              disabled={isRunning}
            />
          </Field>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Field label="Max Articles">
              <input
                className="input"
                type="number"
                value={maxArticles}
                onChange={(e) => setMaxArticles(e.target.value)}
                disabled={isRunning}
              />
            </Field>
            <Field label="Email" hint="Required by NCBI">
              <input
                required
                className="input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                disabled={isRunning}
              />
            </Field>
          </div>
          <Field label="Tag" hint="Optional batch label">
            <input
              className="input"
              value={pmTag}
              onChange={(e) => setPmTag(e.target.value)}
              placeholder="pubmed-brca"
              disabled={isRunning}
            />
          </Field>
          <SubmitButton loading={isRunning} label="Run PubMed Ingest" />
        </form>
      )}

      {submitError && (
        <div
          className="card p-4 text-sm flex items-center gap-2"
          style={{ color: 'var(--danger)' }}
        >
          <XCircle size={15} />
          {submitError}
        </div>
      )}

      {job && <JobTimeline job={job} onCancel={handleCancel} />}
    </div>
  );
}

function SubmitButton({ loading, label }: { loading: boolean; label: string }) {
  return (
    <button className="btn-primary" disabled={loading}>
      {loading ? (
        <>
          <Loader2 size={14} className="animate-spin" /> Working
        </>
      ) : (
        label
      )}
    </button>
  );
}
