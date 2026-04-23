'use client';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { runVerification, getRelationships, VerificationReport, Relationship, verifyText, TextVerificationResponse, checkConflicts, ConflictCheckResponse, getPendingReviews, decideReview, PendingReviewItem, formatApiError } from '@/lib/api';
import { ShieldCheck, ChevronDown, ChevronUp, Loader2, CheckSquare, Square, AlertCircle, FileText, GitBranch, AlertTriangle, ClipboardList } from 'lucide-react';

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
  const [tab, setTab] = useState<'text' | 'relationships' | 'conflicts' | 'reviews'>('text');

  // ── Relationship verification state ──
  const [useEmbed, setUseEmbed] = useState(false);
  const [useLLM, setUseLLM] = useState(false);
  const [threshold, setThreshold] = useState('0.5');
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  // ── Text verification state ──
  const [textQuery, setTextQuery] = useState('');
  const [textEmbed, setTextEmbed] = useState(false);
  const [textLLM, setTextLLM] = useState(false);
  const [textThreshold, setTextThreshold] = useState('0.5');
  const [textReport, setTextReport] = useState<TextVerificationResponse | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const [textError, setTextError] = useState<string | null>(null);
  const [textExpanded, setTextExpanded] = useState<string | null>(null);

  // ── Conflict detection state ──
  const [conflictQuery, setConflictQuery] = useState('');
  const [conflictLLM, setConflictLLM] = useState(false);
  const [conflictReport, setConflictReport] = useState<ConflictCheckResponse | null>(null);
  const [conflictLoading, setConflictLoading] = useState(false);
  const [conflictError, setConflictError] = useState<string | null>(null);

  // ── Review queue state ──
  const { data: reviewData, isLoading: reviewsLoading, refetch: refetchReviews } = useQuery({
    queryKey: ['pending-reviews'],
    queryFn: () => getPendingReviews('pending'),
    enabled: tab === 'reviews',
  });
  const [decidingId, setDecidingId] = useState<string | null>(null);

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
      setError(formatApiError(err, 'Verification failed'));
    } finally {
      setLoading(false);
    }
  }

  async function handleTextVerify(e: React.FormEvent) {
    e.preventDefault();
    setTextLoading(true);
    setTextError(null);
    setTextReport(null);
    try {
      const result = await verifyText({
        text: textQuery,
        enable_embedding: textEmbed,
        enable_llm: textLLM,
        embedding_threshold: Number(textThreshold),
      });
      setTextReport(result);
    } catch (err: any) {
      setTextError(formatApiError(err, 'Text verification failed'));
    } finally {
      setTextLoading(false);
    }
  }

  async function handleConflictCheck(e: React.FormEvent) {
    e.preventDefault();
    setConflictLoading(true);
    setConflictError(null);
    setConflictReport(null);
    try {
      const result = await checkConflicts({ text: conflictQuery, use_llm: conflictLLM });
      setConflictReport(result);
    } catch (err: any) {
      setConflictError(formatApiError(err, 'Conflict check failed'));
    } finally {
      setConflictLoading(false);
    }
  }

  return (
    <div className="space-y-8 max-w-4xl">
      <header>
        <h1 className="page-title">Verification</h1>
        <p className="page-desc">Verify claims or relationships against the knowledge graph using cascading verification.</p>
      </header>

      {/* ── Tab Switcher ──────────────────────────────────── */}
      <div className="flex gap-1 p-1 rounded-lg" style={{ background: 'var(--bg-muted)' }}>
        <button
          onClick={() => setTab('text')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${tab === 'text' ? 'shadow-sm' : ''}`}
          style={tab === 'text' ? { background: 'var(--bg-primary)', color: 'var(--text-primary)' } : { color: 'var(--text-muted)' }}
        >
          <FileText size={14} /> Verify Text
        </button>
        <button
          onClick={() => setTab('relationships')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${tab === 'relationships' ? 'shadow-sm' : ''}`}
          style={tab === 'relationships' ? { background: 'var(--bg-primary)', color: 'var(--text-primary)' } : { color: 'var(--text-muted)' }}
        >
          <GitBranch size={14} /> Verify Relationships
        </button>
        <button
          onClick={() => setTab('conflicts')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${tab === 'conflicts' ? 'shadow-sm' : ''}`}
          style={tab === 'conflicts' ? { background: 'var(--bg-primary)', color: 'var(--text-primary)' } : { color: 'var(--text-muted)' }}
        >
          <AlertTriangle size={14} /> Conflict Check
        </button>
        <button
          onClick={() => setTab('reviews')}
          className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${tab === 'reviews' ? 'shadow-sm' : ''}`}
          style={tab === 'reviews' ? { background: 'var(--bg-primary)', color: 'var(--text-primary)' } : { color: 'var(--text-muted)' }}
        >
          <ClipboardList size={14} /> Review Queue
        </button>
      </div>

      {/* ── Text Verification Tab ────────────────────────── */}
      {tab === 'text' && (
        <>
          <form onSubmit={handleTextVerify} className="card p-6 space-y-6">
            <div>
              <label className="field-label">Description to Verify</label>
              <textarea
                value={textQuery}
                onChange={(e) => setTextQuery(e.target.value)}
                placeholder="e.g. TNF-alpha activates NF-kB signaling pathway in inflammatory response"
                rows={3}
                className="input w-full"
                style={{ resize: 'vertical' }}
              />
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Enter a claim or description. The system will search for matching entities and relationships in the knowledge graph, then verify using the three-stage pipeline.
              </p>
            </div>

            <label className="field-label !mb-0">Verification Stages</label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <Toggle label="Text Match" desc="Literal mention checks (always on)" checked={true} onChange={() => {}} />
              <Toggle label="Embedding" desc="Semantic similarity" checked={textEmbed} onChange={setTextEmbed} />
              <Toggle label="LLM Review" desc="Reasoning pass" checked={textLLM} onChange={setTextLLM} />
            </div>

            {textEmbed && (
              <div className="max-w-sm">
                <label className="field-label">Embedding Threshold</label>
                <div className="flex items-center gap-3">
                  <input type="range" min="0" max="1" step="0.05" value={textThreshold} onChange={(e) => setTextThreshold(e.target.value)} className="flex-1 accent-slate-700" />
                  <span className="font-mono text-sm w-10 text-right" style={{ color: 'var(--text-primary)' }}>{textThreshold}</span>
                </div>
              </div>
            )}

            <button type="submit" disabled={textLoading || !textQuery.trim()} className="btn-primary">
              {textLoading
                ? <><Loader2 size={14} className="animate-spin" /> Verifying…</>
                : <><ShieldCheck size={14} /> Verify Description</>}
            </button>
          </form>

          {textError && <div className="card p-4 text-sm" style={{ color: 'var(--danger)' }}>{textError}</div>}

          {textReport && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {[
                  ['Candidates', textReport.total_candidates],
                  ['Verified', textReport.verified],
                  ['Not Verified', textReport.not_verified],
                  ['Skipped', textReport.skipped],
                  ['Best Confidence', `${(textReport.best_confidence * 100).toFixed(0)}%`],
                ].map(([label, value]) => (
                  <div key={String(label)} className="card p-4 text-center">
                    <p className="field-label !mb-1 text-center">{label}</p>
                    <p className="text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>{value}</p>
                  </div>
                ))}
              </div>

              {textReport.entries.length === 0 ? (
                <div className="card p-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
                  <AlertCircle size={16} className="inline mr-2" />
                  No matching entities or relationships found in the knowledge graph for this description.
                </div>
              ) : (
                <div className="card overflow-hidden">
                  {textReport.entries.map((entry) => {
                    const isOpen = textExpanded === entry.relationship_id;
                    return (
                      <div key={entry.relationship_id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <button className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-[var(--bg-muted)]" onClick={() => setTextExpanded(isOpen ? null : entry.relationship_id)}>
                          <div className="flex items-center gap-2 truncate">
                            <span className={`badge ${entry.status === 'passed' ? 'badge-success' : entry.status === 'failed' ? 'badge-danger' : 'badge-neutral'}`}>
                              {entry.status}
                            </span>
                            <span className="text-xs truncate" style={{ color: 'var(--text-secondary)' }}>
                              <span className="badge badge-neutral mr-1">{entry.relationship_type}</span>
                              {entry.source_entity_name} → {entry.target_entity_name}
                            </span>
                          </div>
                          <span className="inline-flex items-center gap-2 text-xs flex-shrink-0" style={{ color: 'var(--text-muted)' }}>
                            {(entry.confidence * 100).toFixed(0)}%
                            {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                          </span>
                        </button>
                        {isOpen && (
                          <div className="px-4 pb-3 space-y-1.5">
                            {entry.relationship_description && (
                              <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
                                <strong>Description:</strong> {entry.relationship_description}
                              </p>
                            )}
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
              )}
            </div>
          )}
        </>
      )}

      {/* ── Relationship Verification Tab ─────────────────── */}
      {tab === 'relationships' && (
        <>
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
        </>
      )}

      {/* ── Conflict Check Tab ────────────────────────────── */}
      {tab === 'conflicts' && (
        <>
          <form onSubmit={handleConflictCheck} className="card p-6 space-y-6">
            <div>
              <label className="field-label">Claim to Check for Conflicts</label>
              <textarea
                value={conflictQuery}
                onChange={(e) => setConflictQuery(e.target.value)}
                placeholder="e.g. Drug X has no effect on Disease Y"
                rows={3}
                className="input w-full"
                style={{ resize: 'vertical' }}
              />
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
                Enter a claim. The system will check if it contradicts or conflicts with existing knowledge in the graph.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <Toggle label="Structural Check" desc="Type & negation pattern analysis (always on)" checked={true} onChange={() => {}} />
              <Toggle label="LLM Analysis" desc="Semantic conflict judgement" checked={conflictLLM} onChange={setConflictLLM} />
            </div>

            <button type="submit" disabled={conflictLoading || !conflictQuery.trim()} className="btn-primary">
              {conflictLoading
                ? <><Loader2 size={14} className="animate-spin" /> Checking…</>
                : <><AlertTriangle size={14} /> Check for Conflicts</>}
            </button>
          </form>

          {conflictError && <div className="card p-4 text-sm" style={{ color: 'var(--danger)' }}>{conflictError}</div>}

          {conflictReport && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="card p-4 text-center">
                  <p className="field-label !mb-1 text-center">Pairs Checked</p>
                  <p className="text-2xl font-semibold" style={{ color: 'var(--text-primary)' }}>{conflictReport.total_checked}</p>
                </div>
                <div className="card p-4 text-center">
                  <p className="field-label !mb-1 text-center">Conflicts Found</p>
                  <p className="text-2xl font-semibold" style={{ color: conflictReport.conflicts_found > 0 ? 'var(--danger)' : 'var(--text-primary)' }}>
                    {conflictReport.conflicts_found}
                  </p>
                </div>
              </div>

              {conflictReport.conflicts.length === 0 ? (
                <div className="card p-6 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
                  <ShieldCheck size={16} className="inline mr-2" />
                  No conflicts detected. The claim appears consistent with existing knowledge.
                </div>
              ) : (
                <div className="space-y-3">
                  {conflictReport.conflicts.map((c, i) => (
                    <div key={i} className="card p-4 space-y-2" style={c.severity === 'high' ? { borderColor: 'var(--danger)' } : c.severity === 'medium' ? { borderColor: '#d97706' } : {}}>
                      <div className="flex items-center gap-2">
                        <span className={`badge ${c.severity === 'high' ? 'badge-danger' : c.severity === 'medium' ? 'badge-warning' : 'badge-neutral'}`}>
                          {c.severity}
                        </span>
                        <span className="badge badge-neutral">{c.conflict_type}</span>
                        {c.requires_review && <span className="badge badge-danger text-[10px]">Needs Review</span>}
                        <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                          {c.source_entity_name} → {c.target_entity_name}
                        </span>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        <div className="p-2 rounded" style={{ background: 'var(--bg-muted)' }}>
                          <div className="flex items-center gap-2 mb-1">
                            <p className="font-medium" style={{ color: 'var(--text-muted)' }}>Existing Knowledge</p>
                            {c.existing_source_trust && <span className="badge badge-success text-[10px]">{c.existing_source_trust}</span>}
                          </div>
                          <p style={{ color: 'var(--text-primary)' }}>
                            <span className="badge badge-neutral mr-1">{c.existing_relationship_type}</span>
                            {c.existing_description || '(no description)'}
                          </p>
                          {c.existing_source_chunk_ids.length > 0 && (
                            <p className="mt-1" style={{ color: 'var(--text-muted)' }}>
                              Sources: {c.existing_source_chunk_ids.length} chunk(s)
                            </p>
                          )}
                        </div>
                        <div className="p-2 rounded" style={{ background: 'var(--bg-muted)' }}>
                          <p className="font-medium mb-1" style={{ color: 'var(--text-muted)' }}>New Claim</p>
                          <p style={{ color: 'var(--text-primary)' }}>
                            <span className="badge badge-neutral mr-1">{c.new_relationship_type}</span>
                            {c.new_description || '(no description)'}
                          </p>
                        </div>
                      </div>
                      <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{c.reasoning}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* ── Review Queue Tab ──────────────────────────────── */}
      {tab === 'reviews' && (
        <>
          <div className="card p-6 space-y-2">
            <h2 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>Trust Conflict Review Queue</h2>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              When document-extracted knowledge conflicts with trusted sources (PubMed, Open Targets), it is held here for review.
              Approve to inject into the graph, or reject to keep existing trusted knowledge.
            </p>
          </div>

          {reviewsLoading ? (
            <div className="flex items-center gap-2 py-8 justify-center text-sm" style={{ color: 'var(--text-muted)' }}>
              <Loader2 size={14} className="animate-spin" /> Loading reviews…
            </div>
          ) : !reviewData || reviewData.items.length === 0 ? (
            <div className="card p-8 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
              <ShieldCheck size={20} className="inline mr-2" />
              No pending reviews. All knowledge is consistent with trusted sources.
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                {reviewData.total} pending review{reviewData.total !== 1 ? 's' : ''}
              </p>
              {reviewData.items.map((item) => {
                const c = item.conflict;
                return (
                  <div key={item.review_id} className="card p-4 space-y-3" style={{ borderColor: 'var(--danger)' }}>
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`badge ${c.severity === 'high' ? 'badge-danger' : c.severity === 'medium' ? 'badge-warning' : 'badge-neutral'}`}>
                        {c.severity}
                      </span>
                      <span className="badge badge-neutral">{c.conflict_type}</span>
                      <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                        {c.source_entity_name} → {c.target_entity_name}
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                      <div className="p-3 rounded" style={{ background: 'var(--bg-muted)' }}>
                        <div className="flex items-center gap-2 mb-1">
                          <p className="font-medium" style={{ color: 'var(--text-muted)' }}>Existing (Trusted)</p>
                          {c.existing_source_trust && (
                            <span className="badge badge-success text-[10px]">{c.existing_source_trust}</span>
                          )}
                        </div>
                        <p style={{ color: 'var(--text-primary)' }}>
                          <span className="badge badge-neutral mr-1">{c.existing_relationship_type}</span>
                          {c.existing_description || '(no description)'}
                        </p>
                      </div>
                      <div className="p-3 rounded" style={{ background: 'var(--bg-muted)' }}>
                        <div className="flex items-center gap-2 mb-1">
                          <p className="font-medium" style={{ color: 'var(--text-muted)' }}>New Claim</p>
                          {c.new_source_trust && (
                            <span className="badge badge-warning text-[10px]">{c.new_source_trust}</span>
                          )}
                        </div>
                        <p style={{ color: 'var(--text-primary)' }}>
                          <span className="badge badge-neutral mr-1">{c.new_relationship_type}</span>
                          {c.new_description || '(no description)'}
                        </p>
                      </div>
                    </div>

                    <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{c.reasoning}</p>

                    <div className="flex gap-2 pt-1">
                      <button
                        disabled={decidingId === item.review_id}
                        onClick={async () => {
                          setDecidingId(item.review_id);
                          try {
                            await decideReview({ review_id: item.review_id, decision: 'approve' });
                            refetchReviews();
                          } finally { setDecidingId(null); }
                        }}
                        className="btn-primary text-xs px-3 py-1.5"
                      >
                        {decidingId === item.review_id ? <Loader2 size={12} className="animate-spin" /> : 'Approve & Inject'}
                      </button>
                      <button
                        disabled={decidingId === item.review_id}
                        onClick={async () => {
                          setDecidingId(item.review_id);
                          try {
                            await decideReview({ review_id: item.review_id, decision: 'reject' });
                            refetchReviews();
                          } finally { setDecidingId(null); }
                        }}
                        className="btn-ghost text-xs px-3 py-1.5"
                        style={{ color: 'var(--danger)' }}
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}
    </div>
  );
}
