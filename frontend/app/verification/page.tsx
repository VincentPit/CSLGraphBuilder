'use client';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { runVerification, getRelationships, VerificationReport, Relationship } from '@/lib/api';
import { ShieldCheck, ChevronDown, ChevronUp, Loader2, CheckSquare, Square, AlertCircle } from 'lucide-react';

function Toggle({ label, desc, checked, onChange }: { label: string; desc: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="card p-4 cursor-pointer transition" style={checked ? { borderColor: 'var(--accent)', background: 'var(--accent-muted)' } : {}}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{label}</span>
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
      </div>
      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{desc}</p>
    </label>
  );
}

export default function VerificationPage() {
  const [useEmbed, setUseEmbed] = useState(false);
  const [useLLM, setUseLLM] = useState(false);
  const [threshold, setThreshold] = useState('0.5');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data: relData, isLoading: relsLoading } = useQuery({
    queryKey: ['relationships-for-verify'],
    queryFn: () => getRelationships({ limit: 500 }),
  });

  const relationships = relData?.items ?? [];

  function toggleRel(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  function toggleAll() {
    if (selected.size === relationships.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(relationships.map((r) => r.id)));
    }
  }

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await runVerification({
        relationship_ids: [...selected],
        enable_embedding: useEmbed,
        enable_llm: useLLM,
        embedding_threshold: Number(threshold),
      });
      setReport(result);
    } catch (err: any) {
      setError(err.response?.data?.detail ?? err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8 max-w-4xl">
      <header>
        <h1 className="page-title">Verification</h1>
        <p className="page-desc">Select relationships to verify, then choose which verification stages to run.</p>
      </header>

      {/* ── Relationship Picker ─────────────────────────────── */}
      <div className="card p-5 space-y-3">
        <div className="flex items-center justify-between">
          <label className="field-label !mb-0">Select Relationships</label>
          {relationships.length > 0 && (
            <button type="button" onClick={toggleAll} className="btn-ghost text-xs">
              {selected.size === relationships.length ? 'Deselect All' : 'Select All'}
            </button>
          )}
        </div>

        {relsLoading ? (
          <div className="flex items-center gap-2 py-4 text-sm" style={{ color: 'var(--text-muted)' }}>
            <Loader2 size={14} className="animate-spin" /> Loading relationships…
          </div>
        ) : relationships.length === 0 ? (
          <div className="flex items-center gap-2 py-6 text-sm" style={{ color: 'var(--text-muted)' }}>
            <AlertCircle size={14} />
            No relationships in graph. Ingest data first, then come back to verify.
          </div>
        ) : (
          <div className="max-h-64 overflow-y-auto rounded-lg" style={{ border: '1px solid var(--border)' }}>
            {relationships.map((rel) => {
              const on = selected.has(rel.id);
              return (
                <button
                  type="button"
                  key={rel.id}
                  onClick={() => toggleRel(rel.id)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 text-left text-sm transition hover:bg-[var(--bg-muted)]"
                  style={{ borderBottom: '1px solid var(--border)' }}
                >
                  {on
                    ? <CheckSquare size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
                    : <Square size={15} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />}
                  <span className="truncate" style={{ color: 'var(--text-primary)' }}>
                    <span className="badge badge-neutral mr-2">{rel.relationship_type}</span>
                    {rel.source_entity_id} → {rel.target_entity_id}
                  </span>
                  {rel.description && (
                    <span className="ml-auto text-xs truncate max-w-[200px]" style={{ color: 'var(--text-muted)' }}>
                      {rel.description}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        )}

        {relationships.length > 0 && (
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {selected.size} of {relationships.length} selected
          </p>
        )}
      </div>

      {/* ── Verification Settings ─────────────────────────── */}
      <form onSubmit={handleRun} className="card p-6 space-y-6">
        <label className="field-label !mb-0">Verification Stages</label>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Toggle label="Text Match" desc="Literal mention checks (always on)" checked={true} onChange={() => {}} />
          <Toggle label="Embedding" desc="Semantic similarity" checked={useEmbed} onChange={setUseEmbed} />
          <Toggle label="LLM Review" desc="Reasoning pass" checked={useLLM} onChange={setUseLLM} />
        </div>

        {useEmbed && (
          <div className="max-w-sm">
            <label className="field-label">Embedding Threshold</label>
            <div className="flex items-center gap-3">
              <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(e.target.value)} className="flex-1 accent-slate-700" />
              <span className="font-mono text-sm w-10 text-right" style={{ color: 'var(--text-primary)' }}>{threshold}</span>
            </div>
          </div>
        )}

        <button type="submit" disabled={loading || selected.size === 0} className="btn-primary">
          {loading
            ? <><Loader2 size={14} className="animate-spin" /> Verifying {selected.size}…</>
            : <><ShieldCheck size={14} /> Verify {selected.size} Relationship{selected.size !== 1 ? 's' : ''}</>}
        </button>
      </form>

      {error && <div className="card p-4 text-sm" style={{ color: 'var(--danger)' }}>{error}</div>}

      {report && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              ['Total', report.total],
              ['Passed', report.passed],
              ['Failed', report.failed],
              ['Skipped', report.skipped],
            ].map(([label, value]) => (
              <div key={String(label)} className="card p-4 text-center">
                <p className="field-label !mb-1 text-center">{label}</p>
                <p className="text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>{value}</p>
              </div>
            ))}
          </div>

          <div className="card overflow-hidden">
            {report.report.map((entry) => {
              const isOpen = expanded === entry.relationship_id;
              return (
                <div key={entry.relationship_id} style={{ borderBottom: '1px solid var(--border)' }}>
                  <button className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-[var(--bg-muted)]" onClick={() => setExpanded(isOpen ? null : entry.relationship_id)}>
                    <div className="flex items-center gap-2 truncate">
                      <span className={`badge ${entry.status === 'passed' ? 'badge-success' : entry.status === 'failed' ? 'badge-danger' : 'badge-neutral'}`}>
                        {entry.status}
                      </span>
                      <span className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                        {entry.relationship_type}: {entry.source_entity_id} → {entry.target_entity_id}
                      </span>
                    </div>
                    <span className="inline-flex items-center gap-2 text-xs flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                      {(entry.confidence * 100).toFixed(0)}%
                      {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </span>
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-3 space-y-1.5">
                      {entry.reasoning && (
                        <p className="text-xs mb-2" style={{ color: 'var(--text-secondary)' }}>{entry.reasoning}</p>
                      )}
                      {entry.stage_results.map((stage, i) => (
                        <div key={i} className="text-xs flex items-center gap-3">
                          <span className="w-24" style={{ color: 'var(--text-muted)' }}>{stage.stage}</span>
                          <span className={stage.status === 'passed' ? 'text-emerald-700' : 'text-rose-700'}>{stage.status}</span>
                          <span style={{ color: 'var(--text-muted)' }}>{(stage.confidence * 100).toFixed(0)}%</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
