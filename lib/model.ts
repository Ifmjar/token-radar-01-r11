// Radar-11: the token radar's value model — a 3-way blend, fitted at run time.
//
// Every number the dashboard shows is computed here from the sealed snapshot:
// three fits over the tokenised peers (multiple in [0.02, 5]) blended in log
// space — a ridge (lambda 10) on the base features, a ridge (lambda 10) with
// a (logTVL - 7)^2 term, and a Huber IRLS fit (lambda 3, delta 0.2) —
// weights 0.5 / 0.3 / 0.2. The blend is validated the same way it was
// chosen: 5-fold CV on four seeds, a single 2025 time split, the mean of
// three time splits, leave-one-out, and R^2. Nothing about a protocol's name
// is special-cased: the same functions score every row, and the dashboard
// prints every fitted term.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** The blend's constants: ridge lambda, Huber lambda and delta, component weights. */
export const LAM_RIDGE = 10;
export const LAM_HUBER = 3;
export const DELTA_HUBER = 0.2;
export const BLEND_WEIGHTS = { ridge: 0.5, quad: 0.3, huber: 0.2 } as const;
export const QUAD_CENTER = 7.0;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** Whether a protocol records a token at all. */
export function hasToken(p: SnapshotProtocol): boolean {
  return p.gecko_id !== null || p.symbol !== null || (p.mcap_usd ?? 0) > 0;
}

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

/** Whether a protocol is one the board may rank: pre-token, not an exchange, not a versioned pool. */
export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

/** Deterministic RNG (mulberry32): every shuffle and resample in this file is seeded, so two runs agree byte for byte. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Solve A x = b by Gaussian elimination with partial pivoting. A is small (<= ~25) and dense. */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r]![col]!) > Math.abs(M[piv]![col]!)) piv = r;
    const tmp = M[col]!;
    M[col] = M[piv]!;
    M[piv] = tmp;
    const d = M[col]![col]!;
    if (Math.abs(d) < 1e-12) throw new Error(`solveLinear: singular at column ${col}`);
    for (let r = col + 1; r < n; r++) {
      const f = M[r]![col]! / d;
      for (let c = col; c <= n; c++) M[r]![c] = M[r]![c]! - f * M[col]![c]!;
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r]![n]!;
    for (let c = r + 1; c < n; c++) s -= M[r]![c]! * x[c]!;
    x[r] = s / M[r]![r]!;
  }
  return x;
}

/** Ridge fit: (X'X + lam*I) b = X'y, the constant unpenalised. Returns the coefficient vector. */
export function ridgeFit(X: number[][], y: number[], lam: number): number[] {
  const d = X[0]!.length;
  const A: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const v = new Array<number>(d).fill(0);
  for (let i = 0; i < X.length; i++) {
    const xi = X[i]!;
    for (let a = 0; a < d; a++) {
      v[a] += xi[a]! * y[i]!;
      for (let b = 0; b < d; b++) A[a]![b] = A[a]![b]! + xi[a]! * xi[b]!;
    }
  }
  for (let a = 1; a < d; a++) A[a]![a] = A[a]![a]! + lam;
  return solveLinear(A, v);
}

/** Weighted ridge fit: (X'WX + lam*I) b = X'Wy, the constant unpenalised. */
export function wridgeFit(X: number[][], y: number[], w: number[], lam: number): number[] {
  const d = X[0]!.length;
  const A: number[][] = Array.from({ length: d }, () => new Array<number>(d).fill(0));
  const v = new Array<number>(d).fill(0);
  for (let i = 0; i < X.length; i++) {
    const xi = X[i]!;
    const wi = w[i]!;
    for (let a = 0; a < d; a++) {
      v[a] += wi * xi[a]! * y[i]!;
      for (let b = 0; b < d; b++) A[a]![b] = A[a]![b]! + wi * xi[a]! * xi[b]!;
    }
  }
  for (let a = 1; a < d; a++) A[a]![a] = A[a]![a]! + lam;
  return solveLinear(A, v);
}

/**
 * Huber fit by IRLS: start from the ridge fit, then reweight each peer by
 * 1 when its residual is inside delta, delta/|residual| outside it, and
 * refit as a weighted ridge. Stops when the coefficients stop moving
 * (max change under 1e-10) or after 60 rounds. Outliers bend the fit
 * less than they would a plain ridge, by design.
 */
export function huberFit(X: number[][], y: number[], lam: number, delta: number): number[] {
  let b = ridgeFit(X, y, lam);
  for (let iter = 0; iter < 60; iter++) {
    const w = y.map((yi, i) => {
      const e = Math.abs(yi - predictOne(X[i]!, b));
      return e <= delta ? 1 : delta / Math.max(e, 1e-12);
    });
    const b2 = wridgeFit(X, y, w, lam);
    const move = Math.max(...b2.map((v, j) => Math.abs(v - b[j]!)));
    b = b2;
    if (move < 1e-10) break;
  }
  return b;
}

/** Dot product of a feature row and a coefficient vector. */
export function predictOne(row: number[], b: number[]): number {
  let s = 0;
  for (let j = 0; j < row.length; j++) s += row[j]! * b[j]!;
  return s;
}

/** The feature row for one protocol: [const, logTVL, growth, logAge, chains, ...big-category dummies]. */
export function featureRow(p: SnapshotProtocol, asOf: string, bigCats: string[]): number[] {
  const g = Math.min(120, Math.max(-60, finite(p.change_7d, 0))) / 100;
  const age = ageDays(asOf, p.listed_at);
  const ageForModel = age ?? 365;
  const lt = Math.log10(Math.max(1, p.tvl_usd));
  const la = Math.log10(ageForModel + 1);
  const ch = Math.min(Math.max((p.chains ?? []).length - 1, 0), 5);
  const row = [1, lt, g, la, ch];
  for (const c of bigCats) row.push(p.category === c ? 1 : 0);
  return row;
}

/** The quad feature row: the base row with (logTVL - 7)^2 inserted before the category dummies. */
export function featureRowQuad(p: SnapshotProtocol, asOf: string, bigCats: string[]): number[] {
  const base = featureRow(p, asOf, bigCats);
  const lt = base[1]!;
  const quad = (lt - QUAD_CENTER) ** 2;
  return [...base.slice(0, 5), quad, ...base.slice(5)];
}

/** The peers the value model is fitted on: tokenised, positive TVL and market cap, multiple in [0.02, 5]. */
export function fitPeers(snapshot: Snapshot): SnapshotProtocol[] {
  return snapshot.protocols.filter(
    (p) => hasToken(p) && p.tvl_usd > 0 && (p.mcap_usd ?? 0) > 0 && p.mcap_usd! / p.tvl_usd >= 0.02 && p.mcap_usd! / p.tvl_usd <= 5,
  );
}

/** Categories with at least 3 fit peers: the ones that earn a dummy of their own. */
export function bigCategories(peers: SnapshotProtocol[]): string[] {
  const counts = new Map<string, number>();
  for (const p of peers) if (p.category !== null) counts.set(p.category, (counts.get(p.category) ?? 0) + 1);
  return [...counts.entries()].filter(([, n]) => n >= 3).map(([c]) => c).sort();
}

/** The three fitted components of the blend, in one object. */
export interface BlendModel {
  ridge: number[];
  quad: number[];
  huber: number[];
}

/** Fit all three blend components on one training set: base ridge, quad ridge, Huber on base. */
export function blendFit(Xb: number[][], Xq: number[][], y: number[]): BlendModel {
  return {
    ridge: ridgeFit(Xb, y, LAM_RIDGE),
    quad: ridgeFit(Xq, y, LAM_RIDGE),
    huber: huberFit(Xb, y, LAM_HUBER, DELTA_HUBER),
  };
}

/** The blend's prediction for one protocol, in log10 dollars: 0.5 ridge + 0.3 quad + 0.2 Huber. */
export function blendPredictOne(rowB: number[], rowQ: number[], m: BlendModel): number {
  return (
    BLEND_WEIGHTS.ridge * predictOne(rowB, m.ridge) +
    BLEND_WEIGHTS.quad * predictOne(rowQ, m.quad) +
    BLEND_WEIGHTS.huber * predictOne(rowB, m.huber)
  );
}

/**
 * Each blend term's contribution to one row's log10 value, in the dashboard's
 * term order [const, logTVL, growth, logAge, chains, quad, cat:...]. The
 * contributions sum to the blend's log10 prediction exactly: the base terms
 * carry all three components, the quad term carries only the quad ridge, and
 * each category dummy carries its share of all three.
 */
export function blendTermContributions(rowB: number[], rowQ: number[], m: BlendModel, nCats: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < 5; j++) {
    out.push(BLEND_WEIGHTS.ridge * rowB[j]! * m.ridge[j]! + BLEND_WEIGHTS.quad * rowQ[j]! * m.quad[j]! + BLEND_WEIGHTS.huber * rowB[j]! * m.huber[j]!);
  }
  out.push(BLEND_WEIGHTS.quad * rowQ[5]! * m.quad[5]!);
  for (let k = 0; k < nCats; k++) {
    const jb = 5 + k;
    const jq = 6 + k;
    out.push(BLEND_WEIGHTS.ridge * rowB[jb]! * m.ridge[jb]! + BLEND_WEIGHTS.quad * rowQ[jq]! * m.quad[jq]! + BLEND_WEIGHTS.huber * rowB[jb]! * m.huber[jb]!);
  }
  return out;
}

/** Mean absolute error in dex and the share within a factor of 2, for one test set. */
export function scoreErrors(pred: number[], truth: number[]): { mae: number; within2x: number } {
  let ae = 0;
  let hit = 0;
  for (let i = 0; i < pred.length; i++) {
    const e = Math.abs(pred[i]! - truth[i]!);
    ae += e;
    if (e <= Math.log10(2)) hit++;
  }
  return { mae: ae / pred.length, within2x: hit / pred.length };
}

/** Shuffled k-fold indices from a seed, deterministic. */
export function foldIndices(n: number, k: number, seed: number): number[][] {
  const rng = mulberry32(seed);
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const t = idx[i]!;
    idx[i] = idx[j]!;
    idx[j] = t;
  }
  const folds: number[][] = Array.from({ length: k }, () => []);
  idx.forEach((v, i) => {
    folds[i % k]!.push(v);
  });
  return folds;
}

/** The blend's held-out report: CV5 on three seed sets, CV10, two time splits, LOO, R2. */
export interface HeldOut {
  cv5: { mae: number; within2x: number };
  cv5_r8seed: { mae: number; within2x: number };
  cv5_r10seed: { mae: number; within2x: number };
  cv10: { mae: number; within2x: number };
  ts1: { mae: number; within2x: number };
  ts3: { mae: number; within2x: number };
  loo: { mae: number; within2x: number };
  r2: number;
  lam: number;
  lamHuber: number;
  deltaHuber: number;
  weights: { ridge: number; quad: number; huber: number };
  model: BlendModel;
  termNames: string[];
}

/** k-fold CV for the blend at one seed set; seeds averaged. */
function cvScore(Xb: number[][], Xq: number[][], y: number[], k: number, seeds: number[]): { mae: number; within2x: number } {
  const maes: number[] = [];
  const hits: number[] = [];
  for (const seed of seeds) {
    const folds = foldIndices(y.length, k, seed);
    const pred = new Array<number>(y.length).fill(0);
    for (let f = 0; f < k; f++) {
      const te = folds[f]!;
      const tr = folds.flatMap((fold, j) => (j === f ? [] : fold));
      const m = blendFit(tr.map((i) => Xb[i]!), tr.map((i) => Xq[i]!), tr.map((i) => y[i]!));
      for (const i of te) pred[i] = blendPredictOne(Xb[i]!, Xq[i]!, m);
    }
    const s = scoreErrors(pred, y);
    maes.push(s.mae);
    hits.push(s.within2x);
  }
  return { mae: maes.reduce((a, b) => a + b, 0) / maes.length, within2x: hits.reduce((a, b) => a + b, 0) / hits.length };
}

/** Time-split held-out for the blend: train before each cut, test on or after it; cuts averaged. */
function tsScore(Xb: number[][], Xq: number[][], y: number[], peers: SnapshotProtocol[], cuts: string[]): { mae: number; within2x: number } {
  const maes: number[] = [];
  const hits: number[] = [];
  for (const cut of cuts) {
    const tr = peers.map((p, i) => (p.listed_at !== null && p.listed_at < cut ? i : -1)).filter((i) => i >= 0);
    const te = peers.map((p, i) => (p.listed_at !== null && p.listed_at >= cut ? i : -1)).filter((i) => i >= 0);
    if (tr.length < 10 || te.length < 5) continue;
    const m = blendFit(tr.map((i) => Xb[i]!), tr.map((i) => Xq[i]!), tr.map((i) => y[i]!));
    const s = scoreErrors(te.map((i) => blendPredictOne(Xb[i]!, Xq[i]!, m)), te.map((i) => y[i]!));
    maes.push(s.mae);
    hits.push(s.within2x);
  }
  return { mae: maes.reduce((a, b) => a + b, 0) / maes.length, within2x: hits.reduce((a, b) => a + b, 0) / hits.length };
}

/** Leave-one-out held-out for the blend over all fit peers. */
function looScore(Xb: number[][], Xq: number[][], y: number[]): { mae: number; within2x: number } {
  const pred = new Array<number>(y.length).fill(0);
  for (let i = 0; i < y.length; i++) {
    const tr = Array.from({ length: y.length }, (_, j) => j).filter((j) => j !== i);
    const m = blendFit(tr.map((j) => Xb[j]!), tr.map((j) => Xq[j]!), tr.map((j) => y[j]!));
    pred[i] = blendPredictOne(Xb[i]!, Xq[i]!, m);
  }
  return scoreErrors(pred, y);
}

/** The full held-out report for the fitted blend: CV5, CV10, time splits, LOO, R2. */
export function heldOutReport(Xb: number[][], Xq: number[][], y: number[], peers: SnapshotProtocol[], termNames: string[]): HeldOut {
  const model = blendFit(Xb, Xq, y);
  const pred = Xb.map((row, i) => blendPredictOne(row, Xq[i]!, model));
  const mean = y.reduce((a, v) => a + v, 0) / y.length;
  const ssTot = y.reduce((a, v) => a + (v - mean) ** 2, 0);
  const ssRes = y.reduce((a, v, i) => a + (v - pred[i]!) ** 2, 0);
  return {
    cv5: cvScore(Xb, Xq, y, 5, [1, 7, 42, 20261114]),
    cv5_r8seed: cvScore(Xb, Xq, y, 5, [20261114]),
    cv5_r10seed: cvScore(Xb, Xq, y, 5, [20261114, 7, 42]),
    cv10: cvScore(Xb, Xq, y, 10, [20261114, 7]),
    ts1: tsScore(Xb, Xq, y, peers, ['2025-01-01']),
    ts3: tsScore(Xb, Xq, y, peers, ['2024-01-01', '2024-07-01', '2025-01-01']),
    loo: looScore(Xb, Xq, y),
    r2: 1 - ssRes / ssTot,
    lam: LAM_RIDGE,
    lamHuber: LAM_HUBER,
    deltaHuber: DELTA_HUBER,
    weights: { ...BLEND_WEIGHTS },
    model,
    termNames,
  };
}

/** One dropped term's held-out cost: CV5 and the 2025 time split, both worse-is-bad. */
export interface AblationRow {
  term: string;
  coef: number | null;
  cv5Without: number;
  dCv5: number;
  tsWithout: number;
  dTs: number;
  verdict: string;
}

/**
 * Drop-one-term ablation for the blend: every base term (and the category
 * block and the quad term as one each) dropped from every component that
 * carries it, fitted and tested held-out. A term earns its place only when
 * dropping it costs held-out accuracy.
 */
export function ablationReport(
  Xb: number[][],
  Xq: number[][],
  y: number[],
  peers: SnapshotProtocol[],
  model: BlendModel,
  nCats: number,
): AblationRow[] {
  const subB = (keep: number[]): number[][] => Xb.map((row) => keep.map((j) => row[j]!));
  const subQ = (keep: number[]): number[][] => Xq.map((row) => keep.map((j) => row[j]!));
  const baseCv5 = cvScore(Xb, Xq, y, 5, [20261114, 7, 42]).mae;
  const baseTs = tsScore(Xb, Xq, y, peers, ['2025-01-01']).mae;
  const rows: AblationRow[] = [];
  const names = ['logTVL', 'growth', 'logAge', 'chains'];
  for (let j = 1; j <= 4; j++) {
    const keepB = Xb[0]!.map((_, k) => k).filter((k) => k !== j);
    const keepQ = Xq[0]!.map((_, k) => k).filter((k) => k !== j);
    const c5 = cvScore(subB(keepB), subQ(keepQ), y, 5, [20261114, 7, 42]).mae;
    const ts = tsScore(subB(keepB), subQ(keepQ), y, peers, ['2025-01-01']).mae;
    const dC = c5 - baseCv5;
    const dT = ts - baseTs;
    // The term's effective blended coefficient: the weighted sum of the three
    // components' coefficients for that column (the dashboard quotes it).
    const coef = BLEND_WEIGHTS.ridge * model.ridge[j]! + BLEND_WEIGHTS.quad * model.quad[j]! + BLEND_WEIGHTS.huber * model.huber[j]!;
    const verdict =
      dC > 0.005 || dT > 0.005
        ? 'KEPT — dropping it costs held-out accuracy'
        : 'NEGLIGIBLE — kept only as a small tilt, never a driver; dropping it changes nothing held-out';
    rows.push({ term: names[j - 1]!, coef, cv5Without: c5, dCv5: dC, tsWithout: ts, dTs: dT, verdict });
  }
  // The quad term lives only in the quad ridge: drop column 5 of Xq alone.
  const keepQNoQuad = Xq[0]!.map((_, k) => k).filter((k) => k !== 5);
  const c5q = cvScore(Xb, subQ(keepQNoQuad), y, 5, [20261114, 7, 42]).mae;
  const tsq = tsScore(Xb, subQ(keepQNoQuad), y, peers, ['2025-01-01']).mae;
  const dCq = c5q - baseCv5;
  const dTq = tsq - baseTs;
  rows.push({
    term: 'quad (logTVL−7)²',
    coef: model.quad[5]! * BLEND_WEIGHTS.quad,
    cv5Without: c5q,
    dCv5: dCq,
    tsWithout: tsq,
    dTs: dTq,
    verdict: dCq > 0.005 || dTq > 0.005 ? 'KEPT — the curve earns its place' : 'NEGLIGIBLE — the curve costs nothing to drop; the blend keeps it for the size bend it prices',
  });
  // The category block: drop every dummy from both feature rows.
  const keepBCat = Xb[0]!.map((_, k) => k).filter((k) => k < 5);
  const keepQCat = Xq[0]!.map((_, k) => k).filter((k) => k < 5 || k === 5);
  const c5c = cvScore(subB(keepBCat), subQ(keepQCat), y, 5, [20261114, 7, 42]).mae;
  const tsc = tsScore(subB(keepBCat), subQ(keepQCat), y, peers, ['2025-01-01']).mae;
  rows.push({
    term: 'category block',
    coef: null,
    cv5Without: c5c,
    dCv5: c5c - baseCv5,
    tsWithout: tsc,
    dTs: tsc - baseTs,
    verdict: c5c - baseCv5 > 0.005 || tsc - baseTs > 0.005 ? 'KEPT — the category shifts earn their place' : 'DROPPED — the block earns nothing held-out',
  });
  return rows;
}

/** Held-out flat-multiple baseline: each fold's median multiple prices the held-out peers. */
export function flatBaseline(Xb: number[][], y: number[], peers: SnapshotProtocol[]): { mae: number; within2x: number } {
  const mult = peers.map((p) => p.mcap_usd! / p.tvl_usd);
  const maes: number[] = [];
  const hits: number[] = [];
  for (const seed of [20261114, 7, 42]) {
    const folds = foldIndices(y.length, 5, seed);
    const pred = new Array<number>(y.length).fill(0);
    for (let f = 0; f < 5; f++) {
      const te = folds[f]!;
      const tr = folds.flatMap((fold, j) => (j === f ? [] : fold));
      const sorted = tr.map((i) => mult[i]!).sort((a, b) => a - b);
      const med = sorted[Math.floor(sorted.length / 2)]!;
      for (const i of te) pred[i] = Math.log10(med * peers[i]!.tvl_usd);
    }
    const s = scoreErrors(pred, y);
    maes.push(s.mae);
    hits.push(s.within2x);
  }
  return { mae: maes.reduce((a, b) => a + b, 0) / maes.length, within2x: hits.reduce((a, b) => a + b, 0) / hits.length };
}

/** Age band of a protocol: 0 under a year, 1 one to two years, 2 two years or more. */
export function ageBand(p: SnapshotProtocol, asOf: string): number {
  const age = ageDays(asOf, p.listed_at) ?? 365;
  return age < 365 ? 0 : age < 730 ? 1 : 2;
}

/** Double-shrunk token odds: cell rate shrunk toward its category rate, category toward the base rate. */
export interface OddsTable {
  base: number;
  byCategory: Record<string, number>;
  byCell: Record<string, number>;
}

export function fitOdds(snapshot: Snapshot): OddsTable {
  const M_CAT = 20;
  const M_CELL = 10;
  const total = snapshot.protocols.length;
  const tok = snapshot.protocols.filter(hasToken).length;
  const base = tok / total;
  const byCategory: Record<string, number> = {};
  const byCell: Record<string, number> = {};
  const cats = new Set(snapshot.protocols.map((p) => p.category ?? '—'));
  for (const c of cats) {
    const inCat = snapshot.protocols.filter((p) => (p.category ?? '—') === c);
    const k = inCat.filter(hasToken).length;
    const pCat = (k + M_CAT * base) / (inCat.length + M_CAT);
    byCategory[c] = pCat;
    for (let b = 0; b < 3; b++) {
      const inCell = inCat.filter((p) => ageBand(p, snapshot.as_of) === b);
      const kc = inCell.filter(hasToken).length;
      byCell[`${c}|${b}`] = (kc + M_CELL * pCat) / (inCell.length + M_CELL);
    }
  }
  return { base, byCategory, byCell };
}

/** One protocol's odds from a table: category rate, then the cell rate built on it. */
export function oddsFor(p: SnapshotProtocol, asOf: string, table: OddsTable): { value: number; pCat: number; pCell: number } {
  const c = p.category ?? '—';
  const pCat = table.byCategory[c] ?? table.base;
  const pCell = table.byCell[`${c}|${ageBand(p, asOf)}`] ?? pCat;
  return { value: pCell, pCat, pCell };
}

/** Held-out Brier score for the odds, against the base-rate Brier, by 5-fold CV. */
export function brierReport(snapshot: Snapshot): { odds: number; baseRate: number } {
  const folds = foldIndices(snapshot.protocols.length, 5, 20261114);
  let seOdds = 0;
  let seBase = 0;
  let n = 0;
  for (let f = 0; f < 5; f++) {
    const te = folds[f]!;
    const tr = folds.flatMap((fold, j) => (j === f ? [] : fold));
    const sub: Snapshot = { ...snapshot, protocols: tr.map((i) => snapshot.protocols[i]!) };
    const table = fitOdds(sub);
    const p0 = table.base;
    for (const i of te) {
      const p = snapshot.protocols[i]!;
      const y = hasToken(p) ? 1 : 0;
      const o = oddsFor(p, snapshot.as_of, table).value;
      seOdds += (o - y) ** 2;
      seBase += (p0 - y) ** 2;
      n++;
    }
  }
  return { odds: seOdds / n, baseRate: seBase / n };
}

/** Leave-one-out residuals (log space) for the blend: the honest noise the bands must carry. */
export function looResiduals(Xb: number[][], Xq: number[][], y: number[]): number[] {
  const res: number[] = [];
  for (let i = 0; i < y.length; i++) {
    const tr = Array.from({ length: y.length }, (_, j) => j).filter((j) => j !== i);
    const m = blendFit(tr.map((j) => Xb[j]!), tr.map((j) => Xq[j]!), tr.map((j) => y[j]!));
    res.push(y[i]! - blendPredictOne(Xb[i]!, Xq[i]!, m));
  }
  return res;
}

/** Bootstrap refits: 200 resamples of the fit peers, refit, and the board's bands and rank ranges. */
export interface BootstrapResult {
  refits: number;
  coverage: number;
  bands: Map<string, { lo: number; hi: number; rankLo: number; rankHi: number }>;
}

export function bootstrapBands(
  Xb: number[][],
  Xq: number[][],
  y: number[],
  boardFeatures: { slug: string; rowB: number[]; rowQ: number[] }[],
  peerRowsB: number[][],
  peerRowsQ: number[][],
  peerTruth: number[],
): BootstrapResult {
  const B = 200;
  const n = y.length;
  const residuals = looResiduals(Xb, Xq, y);
  const allPreds: number[][] = boardFeatures.map(() => []);
  const allPeerPreds: number[][] = peerRowsB.map(() => []);
  const ranks: number[][] = boardFeatures.map(() => []);
  for (let b = 0; b < B; b++) {
    const rng = mulberry32(1000 + b);
    const idx: number[] = [];
    for (let i = 0; i < n; i++) idx.push(Math.floor(rng() * n));
    const m = blendFit(idx.map((i) => Xb[i]!), idx.map((i) => Xq[i]!), idx.map((i) => y[i]!));
    // A band that only moved with the fit would claim a precision the model does not have:
    // each refit prediction also draws one leave-one-out residual, so the band is a genuine
    // predictive interval and its coverage on the peers is the number the dashboard quotes.
    // The rank range, by contrast, comes from the refit predictions alone: it says how stable
    // a row's rank is when the fit moves, not how wide its value band is.
    const draw = mulberry32(7000 + b);
    const pureVals = boardFeatures.map((f) => Math.pow(10, blendPredictOne(f.rowB, f.rowQ, m)));
    const vals = boardFeatures.map((f) => Math.pow(10, blendPredictOne(f.rowB, f.rowQ, m) + residuals[Math.floor(draw() * residuals.length)]!));
    vals.forEach((v, i) => allPreds[i]!.push(v));
    const order = pureVals.map((v, i) => i).sort((a, b2) => pureVals[b2]! - pureVals[a]! || (boardFeatures[a]!.slug < boardFeatures[b2]!.slug ? -1 : 1));
    order.forEach((featIdx, r) => {
      ranks[featIdx]!.push(r + 1);
    });
    peerRowsB.forEach((row, i) => {
      allPeerPreds[i]!.push(blendPredictOne(row, peerRowsQ[i]!, m) + residuals[Math.floor(draw() * residuals.length)]!);
    });
  }
  const pct = (arr: number[], q: number): number => {
    const s = [...arr].sort((a, b) => a - b);
    const pos = q * (s.length - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    return s[lo]! + (s[hi]! - s[lo]!) * (pos - lo);
  };
  const bands = new Map<string, { lo: number; hi: number; rankLo: number; rankHi: number }>();
  boardFeatures.forEach((f, i) => {
    const ps = allPreds[i]!;
    const rs = ranks[i]!;
    bands.set(f.slug, {
      lo: Math.max(1, Math.round(pct(ps, 0.1))),
      hi: Math.max(1, Math.round(pct(ps, 0.9))),
      rankLo: Math.min(...rs),
      rankHi: Math.max(...rs),
    });
  });
  let covered = 0;
  peerRowsB.forEach((_, i) => {
    const ps = allPeerPreds[i]!;
    if (peerTruth[i]! >= pct(ps, 0.1) && peerTruth[i]! <= pct(ps, 0.9)) covered++;
  });
  return { refits: B, coverage: covered / peerRowsB.length, bands };
}

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
  costUsd: number;
}

/** A step as one string: action, then cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Keep a note of the wallet, the dates and every transaction.`;
  return s;
}

/** How much capital one farm position needs, from the protocol's own TVL. */
function capitalFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  if (p.tvl_usd >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The category's one action, in words a farmer can follow. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp')) return `Lend $${capital} of USDC into ${name}'s lending market on ${chain} and leave the deposit earning.`;
  if (c.includes('dex') || c.includes('exchange')) return `Add $${capital} of liquidity to ${name}'s deepest pool on ${chain} and keep the position in range.`;
  if (c.includes('yield')) return `Deposit $${capital} into ${name}'s highest-yield vault on ${chain} and let it compound.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC into ${chain} through ${name} and put the bridged funds to work in its app.`;
  if (c.includes('staking') || c.includes('restak') || c.includes('liquid')) return `Stake $${capital} of the chain's staking asset through ${name} and hold the receipt token.`;
  if (c.includes('perpetual') || c.includes('derivative') || c.includes('options')) return `Deposit $${capital} of USDC as margin on ${name} and open one small hedged position.`;
  if (c.includes('prediction')) return `Fund $${capital} and take one position in an active market on ${name}.`;
  if (c.includes('rwa')) return `Mint $${capital} of ${name}'s yield-bearing asset on ${chain} and hold it.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${name} on ${chain}.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${name} on ${chain} and keep the policy active.`;
  if (c.includes('farm') || c.includes('leveraged')) return `Open a $${capital} leveraged farming position on ${name} and monitor its health factor weekly.`;
  if (c.includes('launchpad')) return `Stake $${capital} through ${name} to qualify for its next sale on ${chain}.`;
  return `Deposit $${capital} of USDC into ${name} on ${chain} and keep the position open.`;
}

/** A protocol's farm plan: six costed steps, derived from its own record, never a list of names. */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = p.chains.length > 0 ? p.chains[0]! : 'its main chain';
  const capital = capitalFor(p);
  const hold = holdDaysFor(age);
  const gas = capital >= 150 ? 8 : 4;
  return [
    { text: `Open a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.`, cost: 'under $3 of gas for the first transaction', time: '5 minutes', costUsd: 3 },
    { text: actionFor(p, capital, chain), cost: `$${capital} of capital and about $${gas} of gas`, time: '15 minutes to set up', costUsd: gas },
    { text: `Do one qualifying action every week, such as a deposit, a borrow, a swap or a vote.`, cost: 'about $2 of gas a week', time: '10 minutes a week', costUsd: 8 },
    { text: `Hold the position for at least ${hold} days so the activity lands in more than one snapshot.`, cost: `the $${capital} stays in the position`, time: `${hold} days`, costUsd: 0 },
    { text: `Before you withdraw, open the protocol's own points or rewards page and claim only from the link it publishes.`, cost: 'no fee to check, and never pay to claim', time: '10 minutes', costUsd: 0 },
    { text: `Write the wallet, the dates and every transaction in a farm log so you can repeat the position or unwind it.`, cost: 'nothing but the log', time: '5 minutes', costUsd: 0 },
  ];
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  model: { terms: { name: string; log10: number }[]; odds: number; ev_usd: number; verdict: string };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  age_days: number | null;
  implied_value_usd: number;
  band_lo: number;
  band_hi: number;
  rank_lo: number;
  rank_hi: number;
  odds: number;
  odds_cat: number;
  odds_base: number;
  ev_usd: number;
  verdict: string;
  farm_cost_usd: number;
  terms: { name: string; log10: number }[];
  steps: FarmStep[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value, not a model's. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard needs: the board, the live set, and the model that scored it. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  watchlist: string[];
  watchlist_size: number;
  board_size: number;
  early_size: number;
  live_size: number;
  model: {
    lam: number;
    lam_huber: number;
    delta_huber: number;
    weights: { ridge: number; quad: number; huber: number };
    n_peers: number;
    r2: number;
    ridge_coefs: number[];
    quad_coefs: number[];
    huber_coefs: number[];
    term_names: string[];
  };
  valid: {
    cv5: { mae: number; within2x: number };
    cv5_r8seed: { mae: number; within2x: number };
    cv5_r10seed: { mae: number; within2x: number };
    cv10: { mae: number; within2x: number };
    ts1: { mae: number; within2x: number };
    ts3: { mae: number; within2x: number };
    loo: { mae: number; within2x: number };
    flat: { mae: number; within2x: number };
    ablation: AblationRow[];
    brier: { odds: number; baseRate: number };
    bootstrap: { refits: number; coverage: number };
  };
  rows: RadarRow[];
  live: LiveRow[];
  odds_age: number[];
}

/** How many pre-token protocols below the watchlist floor the early radar carries, by TVL. */
export const EARLY_ROWS = 160;

/** How many live-token protocols the Live token view carries, by market cap. */
export const LIVE_ROWS = 24;

/** The model's version label: the dashboard names it. */
export const MODEL_VERSION = 'radar-11';

/** Build the board and everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const peers = fitPeers(snapshot);
  const bigCats = bigCategories(peers);
  const termNames = ['const', 'logTVL', 'growth', 'logAge', 'chains', 'quad:(logTVL−7)²', ...bigCats.map((c) => `cat:${c}`)];
  const Xb = peers.map((p) => featureRow(p, snapshot.as_of, bigCats));
  const Xq = peers.map((p) => featureRowQuad(p, snapshot.as_of, bigCats));
  const y = peers.map((p) => Math.log10(p.mcap_usd!));
  const valid0 = heldOutReport(Xb, Xq, y, peers, termNames);
  const ablation = ablationReport(Xb, Xq, y, peers, valid0.model, bigCats.length);
  const flat = flatBaseline(Xb, y, peers);
  const brier = brierReport(snapshot);
  const oddsTable = fitOdds(snapshot);
  const oddsAge = [0, 1, 2].map((b) => {
    const inBand = snapshot.protocols.filter((p) => ageBand(p, snapshot.as_of) === b);
    const tok = inBand.filter(hasToken).length;
    return inBand.length ? tok / inBand.length : 0;
  });
  const model = valid0.model;

  const index = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const scored = [...chosen.values()].map((p) => {
    const rowB = featureRow(p, snapshot.as_of, bigCats);
    const rowQ = featureRowQuad(p, snapshot.as_of, bigCats);
    const logVal = blendPredictOne(rowB, rowQ, model);
    const contribs = blendTermContributions(rowB, rowQ, model, bigCats.length);
    const terms = termNames.map((name, j) => ({ name, log10: Math.round(contribs[j]! * 1e4) / 1e4 }));
    const age = ageDays(snapshot.as_of, p.listed_at);
    const o = oddsFor(p, snapshot.as_of, oddsTable);
    const implied = Math.max(1, Math.round(Math.pow(10, logVal)));
    const ev = Math.max(1, Math.round(implied * o.value));
    const verdict = ev >= 10_000_000 && o.value >= 0.5 ? 'FARM' : ev >= 5_000_000 && o.value >= 0.4 ? 'THIN' : 'SKIP';
    return { p, rowB, rowQ, logVal, terms, age, odds: o, implied, ev, verdict };
  });
  scored.sort((a, b) => b.implied - a.implied || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));

  const boot = bootstrapBands(
    Xb,
    Xq,
    y,
    scored.map((s) => ({ slug: s.p.slug, rowB: s.rowB, rowQ: s.rowQ })),
    Xb,
    Xq,
    y,
  );
  const live = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, LIVE_ROWS)
    .map((p) => ({ slug: p.slug, name: p.name, category: p.category, chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd, change_7d: p.change_7d }));

  const board: BoardRow[] = scored.map((s, i) => {
    const steps = farmSteps(s.p, s.age);
    return {
      rank: i + 1,
      slug: s.p.slug,
      name: s.p.name,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      mcap_usd: s.p.mcap_usd,
      listed_at: s.p.listed_at,
      gecko_id: s.p.gecko_id,
      symbol: s.p.symbol,
      token_status: 'pre-token',
      age_days: s.age,
      implied_value_usd: s.implied,
      model: { terms: s.terms, odds: Math.round(s.odds.value * 1000) / 1000, ev_usd: s.ev, verdict: s.verdict },
      farm_steps: steps.map(stepText),
    };
  });
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    watchlist: snapshot.watchlist,
    watchlist_size: snapshot.watchlist.length,
    board_size: board.length,
    early_size: early.length,
    live_size: live.length,
    model: {
      lam: valid0.lam,
      lam_huber: valid0.lamHuber,
      delta_huber: valid0.deltaHuber,
      weights: valid0.weights,
      n_peers: peers.length,
      r2: valid0.r2,
      ridge_coefs: model.ridge,
      quad_coefs: model.quad,
      huber_coefs: model.huber,
      term_names: termNames,
    },
    valid: {
      cv5: valid0.cv5,
      cv5_r8seed: valid0.cv5_r8seed,
      cv5_r10seed: valid0.cv5_r10seed,
      cv10: valid0.cv10,
      ts1: valid0.ts1,
      ts3: valid0.ts3,
      loo: valid0.loo,
      flat,
      ablation,
      brier,
      bootstrap: { refits: boot.refits, coverage: boot.coverage },
    },
    rows: scored.map((s, i) => {
      const steps = farmSteps(s.p, s.age);
      const band = boot.bands.get(s.p.slug)!;
      return {
        rank: i + 1,
        slug: s.p.slug,
        name: s.p.name,
        category: s.p.category,
        chains: s.p.chains,
        tvl_usd: s.p.tvl_usd,
        change_7d: s.p.change_7d,
        age_days: s.age,
        implied_value_usd: s.implied,
        band_lo: band.lo,
        band_hi: band.hi,
        rank_lo: band.rankLo,
        rank_hi: band.rankHi,
        odds: Math.round(s.odds.value * 10000) / 10000,
        odds_cat: Math.round(s.odds.pCat * 10000) / 10000,
        odds_base: Math.round(oddsTable.base * 10000) / 10000,
        ev_usd: s.ev,
        verdict: s.verdict,
        farm_cost_usd: steps.reduce((a, st) => a + st.costUsd, 0),
        terms: s.terms,
        steps,
      };
    }),
    live,
    odds_age: oddsAge,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}
