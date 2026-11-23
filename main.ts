// Radar-11: the token radar's program (Node 22, standard library only).
//
// Reads the sealed snapshot from /input/snapshot.json (or --input <file|dir>),
// scores every pre-token protocol with the 3-way blend in lib/model.ts,
// writes the ranked board to /output/board.json, and copies this repository's
// dashboard/ and screenshots/ to /output. Last, it writes /output/dossier.json,
// the summary the judge reads, with the vendored make.fun library.
//
// Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
//   mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './lib/model.ts';
import { buildDossier } from './vendor/makefun-board.ts';

const here = fileURLToPath(new URL('.', import.meta.url));

/** The scorer's own caps: the text cap shapes the bytes, the rest refuse. */
const TEXT_MAX_BYTES = 65_536;
const SCREENSHOT_CAPS = { max: 4, max_bytes: 524_288 };
const IMAGE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

/** --name <value> out of the arguments, or the fallback. */
function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const withEquals = process.argv.find((a) => a.startsWith(`--${name}=`));
  return withEquals === undefined ? fallback : withEquals.slice(name.length + 3);
}

const inputArg = arg('input', '/input');
const inputFile = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputFile)) {
  process.stderr.write(`token radar: no snapshot at ${inputFile}: pass --input <snapshot.json|directory> or mount the sealed /input\n`);
  process.exit(2);
}
const out = arg('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');

const snapshot = parseSnapshot(readFileSync(inputFile, 'utf8'));
const { board, radar } = buildRadar(snapshot);
const issue = boardIssue(board, snapshot);
if (issue !== null) {
  process.stderr.write(`token radar: the model wrote a board the gate refuses: ${issue}\n`);
  process.exit(1);
}

mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'board.json'), `${JSON.stringify({ rows: board })}\n`);
writeFileSync(join(out, 'data.js'), dataScript(radar));

/** Copy one directory's regular files into `to`, skipping names that start with a dot. */
function copyFiles(from: string, to: string): string[] {
  if (!existsSync(from)) return [];
  mkdirSync(to, { recursive: true });
  const copied: string[] = [];
  for (const name of readdirSync(from)) {
    if (name.startsWith('.')) continue;
    const path = join(from, name);
    if (!statSync(path).isFile()) continue;
    copyFileSync(path, join(to, name));
    copied.push(name);
  }
  return copied;
}

copyFiles(join(here, 'dashboard'), join(out, 'dashboard'));
const shots = copyFiles(join(here, 'screenshots'), join(out, 'screenshots'));
if (shots.length < 3) {
  process.stderr.write(`token radar: screenshots/ holds ${shots.length} image${shots.length === 1 ? '' : 's'}: hand in 3, the board, a detail view and a filtered view\n`);
  process.exit(1);
}
for (const name of shots) {
  if (!IMAGE_NAME.test(name)) {
    process.stderr.write(`token radar: screenshots/${name}: a name is letters, digits and . _ -, 1 to 128 characters\n`);
    process.exit(1);
  }
}

// The dossier is written last: the scorer rebuilds it from this /output and compares bytes.
writeFileSync(join(out, 'dossier.json'), buildDossier(out, { text_max_bytes: TEXT_MAX_BYTES, screenshots: SCREENSHOT_CAPS, dashboard: { max_files: 400, max_bytes: 4_194_304 } }));

const v = radar.valid;
const farm = board.filter((r) => r.model.verdict === 'FARM').length;
const thin = board.filter((r) => r.model.verdict === 'THIN').length;
process.stdout.write(
  `token radar: ${snapshot.protocols.length} protocols, ${snapshot.watchlist.length} on the watchlist, ${board.length} rows scored (${radar.early_size} early, ${radar.live_size} live set aside)\n` +
    `token radar: blend 0.5 ridge lam=${radar.model.lam} + 0.3 quad lam=${radar.model.lam} + 0.2 huber lam=${radar.model.lam_huber} delta=${radar.model.delta_huber} n=${radar.model.n_peers} R2=${radar.model.r2.toFixed(4)}\n` +
    `token radar: CV5 4-seed=${v.cv5.mae.toFixed(3)}/${v.cv5.within2x.toFixed(2)} CV5 r8-seed=${v.cv5_r8seed.mae.toFixed(3)}/${v.cv5_r8seed.within2x.toFixed(2)} CV5 r10-seed=${v.cv5_r10seed.mae.toFixed(3)}/${v.cv5_r10seed.within2x.toFixed(2)} CV10=${v.cv10.mae.toFixed(3)}/${v.cv10.within2x.toFixed(2)}\n` +
    `token radar: TS1=${v.ts1.mae.toFixed(3)}/${v.ts1.within2x.toFixed(2)} TS3=${v.ts3.mae.toFixed(3)}/${v.ts3.within2x.toFixed(2)} LOO=${v.loo.mae.toFixed(3)}/${v.loo.within2x.toFixed(2)} flat=${v.flat.mae.toFixed(3)}/${v.flat.within2x.toFixed(2)}\n` +
    `token radar: Brier odds=${v.brier.odds.toFixed(3)} base=${v.brier.baseRate.toFixed(3)} bootstrap coverage=${(v.bootstrap.coverage * 100).toFixed(1)}% over ${v.bootstrap.refits} refits\n` +
    `token radar: verdicts FARM ${farm} / THIN ${thin} / SKIP ${board.length - farm - thin}\n` +
    `token radar: wrote ${join(out, 'board.json')}, ${join(out, 'data.js')}, ${join(out, 'dashboard')}, ${join(out, 'screenshots')} (${shots.length} images) and ${join(out, 'dossier.json')}\n`,
);
