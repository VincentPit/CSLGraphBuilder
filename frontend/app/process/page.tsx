'use client';

import { useState } from 'react';
import {
  CheckCircle2,
  FileText,
  Link2,
  Loader2,
  Sliders,
  Tag,
  XCircle,
} from 'lucide-react';
import {
  cancelJob,
  formatApiError,
  processDocument,
} from '@/lib/api';
import JobTimeline from '@/components/JobTimeline';
import { useJobStream } from '@/lib/useJobStream';

function FieldLabel({
  icon: Icon,
  label,
  optional,
}: {
  icon: any;
  label: string;
  optional?: boolean;
}) {
  return (
    <label className="field-label flex items-center gap-2">
      <Icon size={13} />
      {label}
      {optional && (
        <span className="normal-case font-medium" style={{ color: 'var(--text-muted)' }}>
          optional
        </span>
      )}
    </label>
  );
}

export default function ProcessPage() {
  const [url, setUrl] = useState('');
  const [text, setText] = useState('');
  const [label, setLabel] = useState('');
  const [chunkSize, setChunkSize] = useState<number | ''>('');
  const [chunkOverlap, setChunkOverlap] = useState<number | ''>('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const job = useJobStream(jobId);

  const isRunning = job
    ? job.status === 'pending' || job.status === 'running'
    : submitting;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitError(null);
    setSubmitting(true);
    try {
      const created = await processDocument({
        url: url || undefined,
        text: text || undefined,
        source_label: label || undefined,
        chunk_size: chunkSize === '' ? undefined : Number(chunkSize),
        chunk_overlap: chunkOverlap === '' ? undefined : Number(chunkOverlap),
      });
      setJobId(created.job_id);
    } catch (err: any) {
      setSubmitError(formatApiError(err, 'Request failed'));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCancel() {
    if (!jobId) return;
    try {
      await cancelJob(jobId);
    } catch {
      /* surfaced via job state */
    }
  }

  function reset() {
    setJobId(null);
    setSubmitError(null);
  }

  return (
    <div className="space-y-8 max-w-3xl">
      <header className="space-y-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className="badge badge-brand inline-flex"
            style={{ background: 'var(--accent-soft)', border: '2px solid rgba(213,33,44,0.30)' }}
          >
            <Loader2 size={10} className="opacity-80" />
            5 stages · cancel any time
          </span>
          <span className="badge badge-xp"><Sliders size={11} /> Earns XP per chunk</span>
        </div>
        <h1 className="page-title">Start a Quest</h1>
        <p className="page-desc">
          Paste a URL or some text and watch the LLM pipeline run live. Five
          stages —
          <span className="text-gradient font-bold"> fetch · chunk · entities · relationships · finalize</span>
          {' '}— stream progress in real time. Cancel any time, no penalty.
          Every entity you extract levels up your knowledge graph!
        </p>
      </header>

      <div className="card p-6 md:p-7">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <div className="flex items-center justify-between">
              <FieldLabel icon={Link2} label="Source URL" />
              <span
                className="help-icon"
                title="The page is fetched, scrubbed (scripts/nav/footer removed), then chunked and sent to the LLM."
              >
                ?
              </span>
            </div>
            <input
              className="input"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://pubmed.ncbi.nlm.nih.gov/12345"
              disabled={isRunning}
            />
            <p className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}>
              Provide either a URL or paste raw text below.
            </p>
          </div>

          <div>
            <FieldLabel icon={FileText} label="Raw Text" />
            <textarea
              className="input resize-none"
              rows={6}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste article text, abstract, or report content here…"
              disabled={isRunning}
            />
          </div>

          <div>
            <FieldLabel icon={Tag} label="Source Label" optional />
            <input
              className="input"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g. PubMed:12345"
              disabled={isRunning}
            />
            <p className="text-[11px] mt-1.5" style={{ color: 'var(--text-muted)' }}>
              Used as the document's display name on the Job History page.
            </p>
          </div>

          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="text-xs font-medium flex items-center gap-1.5"
              style={{ color: 'var(--text-secondary)' }}
            >
              <Sliders size={12} />
              {showAdvanced ? 'Hide' : 'Show'} chunking options
            </button>
            {showAdvanced && (
              <div className="grid grid-cols-2 gap-3 mt-3">
                <div>
                  <FieldLabel icon={Sliders} label="Chunk size (tokens)" optional />
                  <input
                    type="number"
                    className="input"
                    value={chunkSize}
                    onChange={(e) =>
                      setChunkSize(e.target.value === '' ? '' : Number(e.target.value))
                    }
                    placeholder="512"
                    disabled={isRunning}
                  />
                </div>
                <div>
                  <FieldLabel icon={Sliders} label="Chunk overlap" optional />
                  <input
                    type="number"
                    className="input"
                    value={chunkOverlap}
                    onChange={(e) =>
                      setChunkOverlap(e.target.value === '' ? '' : Number(e.target.value))
                    }
                    placeholder="50"
                    disabled={isRunning}
                  />
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={isRunning || (!url && !text)}
              className="btn-primary"
            >
              {isRunning ? (
                <>
                  <Loader2 size={16} className="animate-spin" /> Processing…
                </>
              ) : (
                <>Begin Quest →</>
              )}
            </button>
            {job && !isRunning && (
              <button type="button" onClick={reset} className="btn-ghost">
                New quest
              </button>
            )}
          </div>

          {submitError && (
            <p className="text-sm flex items-center gap-2" style={{ color: 'var(--danger)' }}>
              <XCircle size={14} /> {submitError}
            </p>
          )}
        </form>
      </div>

      {job && <JobTimeline job={job} onCancel={handleCancel} />}

      {job?.status === 'completed' && job.result && (
        <ResultSummary result={job.result} />
      )}
    </div>
  );
}

function ResultSummary({ result }: { result: Record<string, unknown> }) {
  const stat = (key: string) => Number(result[key] ?? 0);
  const cards = [
    { label: 'Chunks', value: stat('chunks_created') },
    { label: 'Entities', value: stat('entities_extracted'), sub: `${stat('entities_merged')} merged` },
    {
      label: 'Relationships',
      value: stat('relationships_extracted'),
      sub: `${stat('relationships_merged')} merged`,
    },
    { label: 'Duration (s)', value: Number(result.duration_seconds ?? 0) },
  ];
  return (
    <div className="card p-5">
      <p className="field-label flex items-center gap-2">
        <CheckCircle2 size={13} style={{ color: 'var(--success)' }} />
        Extraction summary
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {cards.map((c) => (
          <div key={c.label} className="rounded-lg p-3" style={{ background: 'var(--bg-muted)' }}>
            <p className="text-[11px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              {c.label}
            </p>
            <p className="text-2xl font-semibold tabular-nums" style={{ color: 'var(--text-primary)' }}>
              {Number(c.value).toLocaleString()}
            </p>
            {c.sub && (
              <p className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>{c.sub}</p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
