#!/usr/bin/env node
// build.mjs: poopdeck-osm-nyc, a 400k-node sample of every tagged node ever created in OpenStreetMap
// New York City (2007-2026), with time, kind and an anonymous contributor rank.
//
// Usage: node build.mjs [--raw <dir>] [--out <dir>] [--sample 400000] [--seed 7]
//   --raw  directory for the full export (default: <os tmp>/poopdeck-osm-nyc-raw); run once, reused.
//   --out  default: ../../../public/data/poopdeck-osm-nyc
//
// Step 1 (skipped when --raw already holds a manifest): scripts/data/poopdeck/stt-export.mjs reads
//   the poopdeck.gl archive osm-nyc-nodes at its lowest zoom (z8, one copy of each node).
// Step 2: this script drops the usernames and numeric user ids (never read), ranks contributors by
//   their node count over the FULL history (0 = most active) and keeps only that rank, takes a
//   seeded uniform sample sorted by time, and writes exact full-history aggregates (monthly counts
//   per kind, top-contributor shares) that the 400k sample cannot give exactly.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) => (a.startsWith('--') ? [[a.slice(2), all[i + 1]]] : []))
);
const raw = args.raw || path.join(os.tmpdir(), 'poopdeck-osm-nyc-raw');
const out = args.out || path.join(here, '../../../public/data/poopdeck-osm-nyc');
const sampleSize = Number(args.sample || 400000);
const seed = Number(args.seed || 7);

if (!fs.existsSync(path.join(raw, 'manifest.json'))) {
  execFileSync(
    'node',
    [
      path.join(here, '../poopdeck/stt-export.mjs'),
      'https://tiles.poopdeck.gl/data/osm-nyc-nodes/manifest.json',
      '--out', raw, '--id', 'osm-nyc-raw', '--max-features', '1000000', '--attrs', 'kind,uid'
    ],
    {stdio: 'inherit', env: {...process.env, NODE_OPTIONS: '--max-old-space-size=8192'}}
  );
}

const man = JSON.parse(fs.readFileSync(path.join(raw, 'manifest.json'), 'utf8'));
const read = (file, Type) => {
  const b = fs.readFileSync(path.join(raw, file));
  return new Type(b.buffer, b.byteOffset, b.byteLength / Type.BYTES_PER_ELEMENT);
};
const position = read('position.bin', Float32Array);
const time = read('time.bin', Uint32Array);
const kind = read('kind.bin', Uint8Array);
const uid = read('uid.bin', Float32Array);
const total = time.length;
const originMs = man.properties.timeOriginMs;
const kinds = man.columns.kind.categories;

// ---- contributor ranks over the full history (uids are used here only and never written) ---------
const perUser = new Map();
for (let i = 0; i < total; i++) perUser.set(uid[i], (perUser.get(uid[i]) || 0) + 1);
const ranked = [...perUser.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
const rankOf = new Map(ranked.map(([id], r) => [id, r]));
const contributorCount = ranked.length;
if (contributorCount >= 65535) throw new Error('contributor rank does not fit uint16');
const TOP = 200;
const topShares = [];
let running = 0;
for (let r = 0; r < TOP; r++) {
  running += ranked[r][1];
  topShares.push(running / total);
}
// Gini of nodes per contributor.
const counts = ranked.map(([, c]) => c).reverse(); // ascending
let weighted = 0;
counts.forEach((c, i) => (weighted += (i + 1) * c));
const gini = (2 * weighted) / (counts.length * total) - (counts.length + 1) / counts.length;

// ---- exact monthly aggregates over the full history ---------------------------------------------
const origin = new Date(originMs);
const monthIndexOf = ms => {
  const d = new Date(ms);
  return (d.getUTCFullYear() - 2007) * 12 + d.getUTCMonth() - 5; // month 0 = June 2007
};
const monthCount = monthIndexOf(originMs + time[total - 1] * 1000) + 1;
const monthly = kinds.map(() => new Array(monthCount).fill(0));
const monthUser = Array.from({length: monthCount}, () => new Map());
const monthTotal = new Array(monthCount).fill(0);
for (let i = 0; i < total; i++) {
  const m = monthIndexOf(originMs + time[i] * 1000);
  monthly[kind[i]][m]++;
  monthTotal[m]++;
  monthUser[m].set(uid[i], (monthUser[m].get(uid[i]) || 0) + 1);
}
const monthlyTopShare = monthUser.map((users, m) => {
  let best = 0;
  for (const c of users.values()) best = Math.max(best, c);
  return monthTotal[m] ? Number((best / monthTotal[m]).toFixed(3)) : 0;
});
const kindTotals = kinds.map((_, k) => monthly[k].reduce((a, b) => a + b, 0));

// ---- seeded uniform sample, sorted by time ------------------------------------------------------
function mulberry32(s) {
  let a = s >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(seed);
const order = Uint32Array.from({length: total}, (_, i) => i);
for (let i = total - 1; i > total - 1 - sampleSize; i--) {
  const j = Math.floor(rng() * (i + 1));
  const t = order[i]; order[i] = order[j]; order[j] = t;
}
const chosen = Array.from(order.subarray(total - sampleSize)).sort((a, b) => time[a] - time[b] || a - b);
const n = chosen.length;
const outPosition = new Float32Array(n * 2);
const outTime = new Uint32Array(n);
const outKind = new Uint8Array(n);
const outContributor = new Uint16Array(n);
let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
chosen.forEach((src, i) => {
  const x = position[src * 2], y = position[src * 2 + 1];
  outPosition[i * 2] = x; outPosition[i * 2 + 1] = y;
  outTime[i] = time[src];
  outKind[i] = kind[src];
  outContributor[i] = rankOf.get(uid[src]);
  west = Math.min(west, x); east = Math.max(east, x); south = Math.min(south, y); north = Math.max(north, y);
});

fs.mkdirSync(out, {recursive: true});
const sizes = {};
const write = (file, typed) => {
  fs.writeFileSync(path.join(out, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
  sizes[file] = typed.byteLength;
};
write('position.bin', outPosition);
write('time.bin', outTime);
write('kind.bin', outKind);
write('contributor.bin', outContributor);
const unit = `seconds since ${origin.toISOString()}`;
const manifest = {
  id: 'poopdeck-osm-nyc',
  version: 1,
  kind: 'points',
  count: n,
  bbox: [west, south, east, north],
  crs: 'EPSG:4326',
  columns: {
    position: {file: 'position.bin', dtype: 'float32', components: 2, length: n},
    timestamp: {file: 'time.bin', dtype: 'uint32', unit, length: n},
    kind: {file: 'kind.bin', dtype: 'uint8', length: n, categories: kinds},
    contributor: {
      file: 'contributor.bin',
      dtype: 'uint16',
      length: n,
      unit: 'anonymous rank by node count over the full history, 0 = most active; no user name or id is shipped'
    }
  },
  properties: {
    attribution: '© OpenStreetMap contributors (ODbL 1.0)',
    timeOriginMs: originMs,
    timeRangeMs: [originMs, originMs + time[total - 1] * 1000],
    fullCount: total,
    sampleFraction: n / total,
    sampleSeed: seed,
    contributorCount,
    giniNodesPerContributor: Number(gini.toFixed(4)),
    topShares: topShares.map(v => Number(v.toFixed(5))),
    month: {firstMonth: '2007-06', count: monthCount, total: monthTotal, byKind: Object.fromEntries(kinds.map((k, i) => [k, monthly[i]])), topContributorShare: monthlyTopShare},
    kindTotals: Object.fromEntries(kinds.map((k, i) => [k, kindTotals[i]])),
    source: {
      archive: 'https://tiles.poopdeck.gl/data/osm-nyc-nodes/manifest.json',
      note: 'Node creations in the OSM full history (Geofabrik internal extract), NYC bbox; one node per creation; z8 copy of the zoom pyramid',
      exportedFrom: 'scripts/data/poopdeck/stt-export.mjs then scripts/data/poopdeck-osm-nyc/build.mjs'
    }
  }
};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest));
console.log(JSON.stringify({n, total, sizes, bytes: Object.values(sizes).reduce((a, b) => a + b, 0), contributorCount, gini, monthCount, top1: topShares[0], top10: topShares[9], top100: topShares[99]}, null, 1));
