'use client';
import { useState } from 'react';
import { runVerification, VerificationReport } from '@/lib/api';
import { ShieldCheck, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

function Toggle({ label, desc, checked, onChange }: { label: string; desc: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className={`rounded-md border p-4 cursor-pointer transition ${checked ? 'bg-[#ddf4ff] border-[#54aeff]' : 'bg-white border-[#d0d7de] hover:border-[#afb8c1]'}`}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-medium text-slate-900">{label}</span>
        <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-[#0969da]" />
      </div>
      <p className="text-xs text-slate-600">{desc}</p>
    </label>
  );
}

export default function VerificationPage() {
  const [useText, setUseText] = useState(true);
  const [useEmbed, setUseEmbed] = useState(false);
  const [useLLM, setUseLLM] = useState(false);
  const [threshold, setThreshold] = useState('0.6');
  const [report, setReport] = useState<VerificationReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await runVerification({
        use_text_match: useText,
        use_embedding: useEmbed,
        use_llm: useLLM,
        confidence_threshold: Number(threshold),
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
      <header className="space-y-2">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">Verification</h1>
        <p className="text-slate-600">Run confidence checks before finalizing relationship assertions.</p>
      </header>

      <form onSubmit={handleRun} className="surface p-6 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Toggle label="Text Match" desc="Literal mention checks" checked={useText} onChange={setUseText} />
          <Toggle label="Embedding" desc="Semantic similarity" checked={useEmbed} onChange={setUseEmbed} />
          <Toggle label="LLM Review" desc="Reasoning pass" checked={useLLM} onChange={setUseLLM} />
        </div>

        <div className="max-w-sm">
          <label className="block text-xs uppercase tracking-wide text-slate-500 font-semibold mb-2">Confidence Threshold</label>
          <div className="flex items-center gap-3">
            <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(e.target.value)} className="flex-1 accent-slate-700" />
            <span className="text-slate-700 font-mono text-sm w-10 text-right">{threshold}</span>
          </div>
        </div>

        <button type="submit" disabled={loading || (!useText && !useEmbed && !useLLM)} className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md bg-[#2da44e] border border-[#2c974b] text-white font-medium hover:bg-[#2c974b] disabled:opacity-50">
          {loading ? <><Loader2 size={14} className="animate-spin" /> Running</> : <><ShieldCheck size={14} /> Run Verification</>}
        </button>
      </form>

      {error && <div className="surface p-4 text-rose-700 text-sm">{error}</div>}

      {report && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              ['Total', report.total],
              ['Verified', report.verified],
              ['Rejected', report.rejected],
              ['Unverified', report.unverified],
            ].map(([label, value]) => (
              <div key={String(label)} className="surface p-4 text-center">
                <p className="text-xs uppercase tracking-wide text-slate-500 mb-1">{label}</p>
                <p className="text-2xl font-semibold text-slate-900">{value}</p>
              </div>
            ))}
          </div>

          <div className="surface overflow-hidden">
            {report.entries.map((entry) => {
              const isOpen = expanded === entry.relationship_id;
              return (
                <div key={entry.relationship_id} className="border-b border-slate-200 last:border-b-0">
                  <button className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-slate-50" onClick={() => setExpanded(isOpen ? null : entry.relationship_id)}>
                    <span className="text-xs font-mono text-slate-600 truncate pr-3">{entry.relationship_id}</span>
                    <span className="inline-flex items-center gap-2 text-xs text-slate-600">
                      {(entry.overall_confidence * 100).toFixed(0)}%
                      {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </span>
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-3 space-y-1.5">
                      {entry.stages.map((stage, i) => (
                        <div key={i} className="text-xs flex items-center gap-3">
                          <span className="w-24 text-slate-500">{stage.stage}</span>
                          <span className={stage.status === 'verified' ? 'text-emerald-700' : 'text-rose-700'}>{stage.status}</span>
                          {stage.confidence != null && <span className="text-slate-500">{(stage.confidence * 100).toFixed(0)}%</span>}
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
