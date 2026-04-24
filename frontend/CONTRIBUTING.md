# Frontend Contributor Guide

Welcome. This is the playbook for everyone working on the **frontend** of
CSL GraphBuilder. Read this once before you make your first change.

---

## TL;DR

1. Clone, install, and turn on the scope-guard hook:
   ```bash
   git clone https://github.com/VincentPit/CSLGraphBuilder.git
   cd CSLGraphBuilder
   git checkout frontend-only
   git config --local core.hooksPath .githooks   # one-time per clone
   cd frontend
   npm ci
   ```
2. Run the dev server pointing at the shared backend on port `8001`
   (already running, or have a backend-side dev start it):
   ```bash
   npm run dev          # → http://localhost:3000
   ```
3. Make changes only under `frontend/`. The hook will reject anything else.
4. Before pushing, prove your changes don't break the API contract:
   ```bash
   npm test             # typecheck + production build + contract-check
   ```
5. Open a PR `frontend-only` → `main`. CI will rerun the same checks plus
   the server-side scope guard. Merge after review + green CI.

---

## Branch model

```
main                ← stable; backend + frontend together; protected
└── frontend-only   ← all frontend work happens here, then PRs into main
```

| Branch | Who can push | What changes are allowed |
|---|---|---|
| `main` | maintainers via PR | anything |
| `frontend-only` | frontend contributors | `frontend/`, `README.md`, `.github/`, `.githooks/`, `.gitignore` only |

Why a separate branch: backend contracts shift slowly and need their own
review path. Isolating frontend churn means UI iteration is fast, and any
attempt to "just tweak one backend field" is caught locally and in CI.

### The scope guard

Two layers, both enforced automatically once you've run the `git config`
line above:

1. **Local** — `.githooks/pre-commit` rejects commits on `frontend-only`
   that touch anything outside the allowed paths. The error message lists
   the offending files.
2. **Server** — `.github/workflows/frontend-check.yml`'s `scope-guard`
   job runs the same check against every PR from `frontend-only`. This
   catches anyone who skipped the hook or used `git commit --no-verify`.

**Need a backend change to land your feature?** Stop. Open an issue
describing the contract you need (endpoint, request shape, response
shape). A maintainer will land the backend change in a separate PR
into `main`; rebase your branch onto it; carry on.

---

## Local development

### Without running the backend

The **contract check** lets you exercise the most fragile bit of the
frontend (error normalisation, payload shapes) without spinning up
Python or Neo4j:

```bash
npm run contract-check
```

This is enough for most styling, layout, and component-level changes.

### With the real backend

If your change involves real data flow (jobs, streaming, graph
rendering), point at the running backend:

```bash
# Default — backend on :8001
npm run dev

# Custom backend host
NEXT_PUBLIC_API_URL=http://localhost:9000 npm run dev
```

If the backend isn't running, ask a maintainer to spin it up — or, in a
pinch, you'll see graceful empty states everywhere because every page
uses TanStack Query (errors surface in the dashboard's "API unreachable"
card and the sidebar `HealthDot` goes red).

---

## Project layout

```
frontend/
├── app/                       # Next.js 14 app-router pages
│   ├── layout.tsx             # Root layout: sidebar + main content
│   ├── page.tsx               # Dashboard (/)
│   ├── graph/page.tsx         # Force-directed graph + floating inspector
│   ├── process/page.tsx       # Document processing form
│   ├── ingest/page.tsx        # Open Targets / PubMed / Web Crawl tabs
│   ├── documents/page.tsx     # Job History split-pane
│   ├── curation/page.tsx      # Approve / reject queue
│   ├── verification/page.tsx  # Verifier + conflict-detection
│   ├── export/page.tsx        # JSON / Cytoscape / GraphML / HTML download
│   └── globals.css            # Design tokens + utilities
├── components/                # Reusable UI bits
│   ├── Nav.tsx                # Sidebar
│   ├── Providers.tsx          # React Query provider
│   ├── JobTimeline.tsx        # Stage rail + event log + cancel
│   ├── AnimatedNumber.tsx     # RAF count-up
│   └── HealthDot.tsx          # Live API status indicator
├── lib/
│   ├── api.ts                 # Typed API client + formatApiError
│   ├── useJobStream.ts        # SSE hook with polling fallback
│   └── utils.ts               # cn() and similar helpers
├── __mocks__/
│   └── api-fixtures.ts        # Mock backend payloads for the contract check
└── scripts/
    └── contract-check.mjs     # Runs the assertions in CI and locally
```

---

## Conventions

### Design tokens — use them, don't hardcode

Colours, spacings, shadows live in `app/globals.css` as CSS custom
properties. Reach for them via `var(--name)` (or the Tailwind utility
that uses them). The full set:

| Token | Purpose |
|---|---|
| `--accent` `--accent-hover` `--accent-soft` `--accent-muted` | CSL Behring red. Default brand colour. |
| `--accent-2` `--accent-3` | Coral, amber — gradient companions |
| `--success` `--danger` `--warning` `--info` | Status, plus `-soft` / `-muted` backgrounds |
| `--bg-app` `--bg-card` `--bg-muted` `--bg-input` | Surfaces |
| `--text-primary` `--text-secondary` `--text-muted` | Typography hierarchy |
| `--border-default` `--border-subtle` `--border-input` | Separators |
| `--shadow-sm` `--shadow-card` `--shadow-lift` `--shadow-glow` | Depth |
| `--radius-sm/md/lg/xl` | Corner roundness |
| `--grad-brand` `--grad-success` `--grad-warning` | Pre-built gradients |

Utility classes you'll reuse:

```html
<div class="card card-hover">…</div>     <!-- standard panel + hover lift -->
<button class="btn-primary">…</button>   <!-- gradient red CTA -->
<button class="btn-ghost">…</button>     <!-- secondary action -->
<span class="badge badge-success">…</span>  <!-- success, danger, warning, neutral, brand -->
<input class="input" />                  <!-- text input + focus ring -->
<p class="page-title">…</p>              <!-- gradient page heading -->
<p class="field-label">…</p>             <!-- input label -->
<span class="help-icon">?</span>         <!-- inline ? for tooltips -->
<div class="empty-state">…</div>         <!-- consistent empty UI -->
```

Animations available out of the box:

```html
<div class="fade-up">…</div>              <!-- fade in + small upward shift -->
<div class="stagger">{children…}</div>    <!-- staggered fade-up for lists -->
<span class="pulse-soft">●</span>         <!-- gentle opacity pulse -->
<div class="skeleton h-20" />             <!-- shimmer placeholder -->
<div class="bg-animated" />               <!-- slow gradient drift -->
```

### Background gradients — use longhand, not shorthand

If you set `backgroundSize`, set the gradient with `backgroundImage`,
not the `background` shorthand. The shorthand resets size on every
re-render and React will warn:

```tsx
// ❌ warns
<div style={{ background: 'linear-gradient(...)', backgroundSize: '200% 200%' }} />

// ✅ correct
<div style={{ backgroundImage: 'linear-gradient(...)', backgroundSize: '200% 200%' }} />
```

### Always coerce errors with `formatApiError`

FastAPI returns 422 with `detail` as an array of objects. Rendering
that directly crashes React. Use the helper:

```tsx
import { formatApiError } from '@/lib/api';

try {
  await processDocument(...);
} catch (err) {
  setError(formatApiError(err, 'Could not start processing'));
}
```

Never write `setError(err.response.data.detail)` — the contract check
will still pass (it tests `formatApiError`, not your callsite), but you
can blow up at runtime on a 422.

### Refs through `next/dynamic`

`next/dynamic` returns a `LoadableComponent` that **doesn't forward
refs**. If you need a ref into a dynamically-imported component, pass a
plain prop instead. The graph page does this with `fgRef`:

```tsx
const ForceGraph2D = dynamic<{ fgRef?: React.MutableRefObject<any> } & any>(
  () => import('react-force-graph-2d').then((mod) => {
    const RFG = mod.default;
    return ({ fgRef, ...rest }) => <RFG ref={fgRef} {...rest} />;
  }),
  { ssr: false }
);
```

### Streaming jobs — use `useJobStream`

For any long-running job (Process, Ingest, Web Crawl), don't write your
own EventSource. Use the hook — it handles SSE with a polling fallback:

```tsx
import { useJobStream } from '@/lib/useJobStream';

const job = useJobStream(jobId);    // null until first event
if (job) return <JobTimeline job={job} onCancel={() => cancelJob(jobId)} />;
```

### Data fetching — use TanStack Query

Every page that reads from the backend uses React Query. Refetch
intervals are explicit:

```tsx
const { data, isLoading, error } = useQuery({
  queryKey: ['my-resource'],
  queryFn: getMyResource,
  refetchInterval: 5000,   // for live data; omit for static
});
```

---

## Adding a new page

1. Create `frontend/app/<your-page>/page.tsx`. Mark it `'use client'`
   if you need state, effects, or browser APIs.
2. Add a nav entry in `frontend/components/Nav.tsx` — pick an icon from
   `lucide-react`, write a one-line `tip` for the hover tooltip.
3. If you need new API endpoints, **stop** — open a backend issue first.
   If reusing existing ones, add helpers in `frontend/lib/api.ts` and a
   matching mock in `frontend/__mocks__/api-fixtures.ts`.
4. Run `npm test` — typecheck, build, contract-check should all pass.

---

## Testing your changes

### Two test scripts, two contexts

```bash
npm test          # typecheck + contract-check     (safe to run while `npm run dev` is up)
npm run test:full # typecheck + build + contract-check  (matches CI, but clobbers .next)
```

**Why there are two:** `next build` writes production-mode chunks to
`.next/`. If `next dev` is also running against that same directory,
the dev server's HTML keeps referencing dev chunks that the build just
overwrote, and you get `Cannot find module './682.js'` 500s. Use
`npm test` while iterating; let CI run `test:full`.

If you do clobber `.next/` by accident:

```bash
pkill -f next-server   # or close the dev terminal
rm -rf .next
npm run dev
```

Or each piece separately:

```bash
npm run typecheck         # tsc --noEmit, strict mode
npm run build             # next build, production  ⚠ don't run while dev is up
npm run contract-check    # < 2s — feeds mock payloads to api helpers
npm run lint              # next lint
npm run dev               # interactive dev server
```

CI on every push to `frontend-only` and every PR into `main` runs the
**full** chain (`typecheck`, `build`, `contract-check`) as separate
GitHub Action steps — the runner has no dev server to clobber.

### Adding a new contract assertion

When you find a backend payload shape that's tricky to handle, add it
to `frontend/__mocks__/api-fixtures.ts` and write an assertion in
`frontend/scripts/contract-check.mjs`. Future contributors won't
re-break it.

---

## Committing and PRs

1. Branch off `frontend-only`:
   ```bash
   git checkout frontend-only
   git pull
   git checkout -b feat/your-thing
   ```
2. Commit small, focused changes. Conventional Commits style:
   `feat(graph): …`, `fix(curation): …`, `chore(ci): …`, `docs: …`.
3. Run `npm test` locally. Push your branch.
4. Open a PR **into `frontend-only`** for an internal review
   *(optional — small fixes can go straight to a PR into `main`)*.
5. Open a PR **into `main`**. Two CI checks will run:
   - **Build + typecheck + contract** must be green.
   - **Branch scope guard** must be green (no backend files changed).
6. After approval, squash-merge into `main`. The merge commit retains
   the full history under "Show all commits" if you need it later.

### What the hook checks

```text
Allowed paths on frontend-only:
  frontend/**
  README.md
  .github/**
  .githooks/**
  .gitignore
```

Anything else is rejected with the offending file list. Override for a
genuine emergency: `git commit --no-verify` — but the server-side
guard will still catch it on the PR, so this is rarely useful.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Hook didn't fire on commit | You skipped `git config --local core.hooksPath .githooks` | Run it once. |
| `Failed to start server: EADDRINUSE :3000` | Old `next dev` is still running | `pkill -f next-server`, then retry. |
| 404s on `_next/static/*.js` after a build | `next build` clobbered the dev `.next/` dir | `rm -rf .next && npm run dev`. Don't run `next build` while `next dev` is up. |
| `Objects are not valid as a React child` | You rendered an error object directly | Use `formatApiError(err, 'fallback')`. |
| `Function components cannot be given refs` | Passing `ref` to a `next/dynamic` component | Use a custom prop name (`fgRef`) — see the graph page. |
| `Updating a style property during rerender (background) when a conflicting property is set (backgroundSize)` | Mixing the `background` shorthand with `backgroundSize` | Use `backgroundImage` longhand. |
| `npm run contract-check` errors on transpile | Bad import in `lib/api.ts` or `__mocks__/api-fixtures.ts` | Run `npm run typecheck` for the real error. |
| Dashboard says "API unreachable" | Backend isn't running on `:8001` | Ask a maintainer; or set `NEXT_PUBLIC_API_URL` to point elsewhere. |

---

## Where to ask

- **Bugs you found**: open a GitHub issue with steps to reproduce.
- **Backend contract changes you need**: open a GitHub issue tagged
  `backend-contract` and link your draft PR.
- **Design tokens or component decisions**: open a discussion before
  spending hours on a refactor — easier to align early.

---

## Appendix — useful npm scripts

| Script | Purpose |
|---|---|
| `npm run dev` | Dev server on `:3000` against `:8001` backend |
| `npm run dev:3002` | Dev server on `:3002` (handy when `:3000` is taken) |
| `npm run clean-dev` | `rm -rf .next` + dev on `:3002` (post-build recovery) |
| `npm run build` | Production build — **don't run while `npm run dev` is up** |
| `npm run start` | Serve the production build |
| `npm run lint` | `next lint` |
| `npm run typecheck` | `tsc --noEmit` strict |
| `npm run contract-check` | < 2s smoke test against mock backend payloads |
| `npm test` | Safe loop: typecheck + contract-check (no build) |
| `npm run test:full` | What CI runs: typecheck + build + contract-check |
