#!/usr/bin/env node
// build.mjs: poopdeck-animals, open-licence bird tracks from the poopdeck.gl `animals` archive.
//
// Usage: node build.mjs --out <public/data/poopdeck-animals> --cache <dir for pieces.json> [--core <path>]
//
// 1. Reads every z0 tile of https://tiles.poopdeck.gl/data/animals (366 daily tiles, about 110 MB) and keeps
//    the pieces whose `dataset` is on the CC0 allow-list below (licences checked through the GBIF dataset API).
//    The archive folds multi-year tracks onto calendar 2024; pieces are cut at every day.
// 2. Joins the pieces of one (organism, segment) pair into one track (an animal-year), thins fixes to at least
//    ~1.8 h apart, drops fixes that imply more than 30 m/s, and writes showcase columns.
import {pathToFileURL} from 'node:url';
import os from 'node:os';
import path from 'node:path';
import fs from 'node:fs';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith('--') ? [...acc, [a.slice(2), all[i + 1]]] : acc), []));
if (!args.out || !args.cache) { console.error('usage: node build.mjs --out <dir> --cache <dir>'); process.exit(2); }
const CORE = args.core || path.join(os.homedir(), 'Documents/GitHub/poopdeck.gl/packages/core/dist/index.js');

// Datasets verified CC0 1.0 with https://api.gbif.org/v1/dataset/<key> (all published by INBO).
const DATASETS = [
  {prefix: 'MH_WATERLAND ', short: 'MH_WATERLAND', key: '66e0553e-75f6-49de-b614-22efd9fbf6e9', doi: '10.5281/zenodo.10053583', species: 'Circus aeruginosus'},
  {prefix: 'H_GRONINGEN ', short: 'H_GRONINGEN', key: '5124534e-2d9c-46b7-a857-e0012821526b', doi: '10.5281/zenodo.10053658', species: 'Circus aeruginosus'},
  {prefix: 'MH_ANTWERPEN ', short: 'MH_ANTWERPEN', key: 'e347ea47-db3f-4c47-8771-ea562330382c', doi: '10.5281/zenodo.10054153', species: 'Circus aeruginosus'},
  {prefix: 'BOP_RODENT ', short: 'BOP_RODENT', key: 'e2fb42ca-e408-4aa2-a7bd-a9bb4ddcc83a', doi: '10.5281/zenodo.17310324', species: 'Circus pygargus'},
  {prefix: 'SPOONBILL_VLAANDEREN ', short: 'SPOONBILL_VLAANDEREN', key: '6850e626-46fd-4843-a391-2c06b069a940', doi: '10.5281/zenodo.23103735', species: 'Platalea leucorodia'}
];
const SPECIES = ['Circus aeruginosus', 'Circus pygargus', 'Platalea leucorodia'];
const MIN_GAP_S = 6600;
const MAX_SPEED = 30; // m/s
const MIN_FIXES = 60;
const ORIGIN_MS = Date.UTC(2024, 0, 1);

fs.mkdirSync(args.cache, {recursive: true});
const cacheFile = path.join(args.cache, 'pieces.json');
let pieces;
if (fs.existsSync(cacheFile)) {
  pieces = JSON.parse(fs.readFileSync(cacheFile, 'utf8'));
} else {
  const {STTArchive, InlineTileDecoder} = await import(pathToFileURL(CORE).href);
  const archive = new STTArchive({url: 'https://tiles.poopdeck.gl/data/animals/manifest.json', decoder: new InlineTileDecoder(), opfsCache: false});
  const meta = await archive.getMetadata();
  const world = {minLon: -180, minLat: -90, maxLon: 180, maxLat: 90};
  const ids = await archive.getTileIdsInBounds(world, 0, {start: meta.timeRange.start, end: meta.timeRange.end});
  pieces = [];
  const seen = new Set();
  for (let b = 0; b < ids.length; b += 8) {
    for (const tile of await archive.getTiles(ids.slice(b, b + 8))) {
      if (!tile) continue;
      for (const layer of tile.layers) {
        if (layer.name === 'summary') continue;
        const f = layer.features;
        const cat = f.categoricalProps;
        for (let i = 0; i < f.featureCount; i++) {
          const get = (k) => { const c = cat[k]; if (!c) return ''; const code = c.indices[i]; return code === 0xffff ? '' : c.categories[code]; };
          const dataset = DATASETS.find((d) => get('dataset').startsWith(d.prefix));
          if (!dataset || !SPECIES.includes(get('species'))) continue;
          const from = f.startIndices[i], to = f.startIndices[i + 1];
          const vt = [];
          for (let v = from; v < to; v++) vt.push(f.vertexTimestamps[v] + f.timeOffset);
          const key = `${f.featureIds64 ? f.featureIds64[i] : ''}:${vt[0]}:${to - from}`;
          if (seen.has(key)) continue;
          seen.add(key);
          pieces.push({ds: dataset.short, sp: get('species'), org: get('organism'), seg: f.numericProps.segment ? f.numericProps.segment[i] : 0, pos: Array.from(f.positions.subarray(from * 2, to * 2)), vt});
        }
      }
    }
    if ((b / 8) % 20 === 0) console.error(`tiles ${b}/${ids.length}, ${pieces.length} pieces`);
  }
  fs.writeFileSync(cacheFile, JSON.stringify(pieces));
}

// ---- species of the multi-species dataset ---------------------------------------------------
// BOP_RODENT holds five raptor species but the archive labels every track with one of them, so the species
// of each tag comes from the GBIF occurrence facets (organismID per scientificName) of that dataset.
const bopFile = path.join(args.cache, 'bop-species.json');
let bopSpecies;
if (fs.existsSync(bopFile)) {
  bopSpecies = JSON.parse(fs.readFileSync(bopFile, 'utf8'));
} else {
  bopSpecies = {};
  const key = DATASETS.find((d) => d.short === 'BOP_RODENT').key;
  for (const name of ['Circus aeruginosus', 'Circus pygargus', 'Buteo buteo', 'Circus cyaneus', 'Asio flammeus']) {
    const url = `https://api.gbif.org/v1/occurrence/search?datasetKey=${key}&scientificName=${encodeURIComponent(name)}&limit=0&facet=organismID&facetLimit=1000`;
    const json = await (await fetch(url)).json();
    for (const {name: organism} of json.facets[0].counts) bopSpecies[organism.toLowerCase()] = name;
  }
  fs.writeFileSync(bopFile, JSON.stringify(bopSpecies));
}
for (const p of pieces) if (p.ds === 'BOP_RODENT') p.sp = bopSpecies[p.org.toLowerCase()] ?? '';
pieces = pieces.filter((p) => SPECIES.includes(p.sp));

// ---- join, thin, clean ---------------------------------------------------------------------
const groups = new Map();
for (const p of pieces) {
  const key = `${p.sp}|${p.org}|${p.seg}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(p);
}
const metersBetween = (lon0, lat0, lon1, lat1) => {
  const r = Math.PI / 180;
  const dLat = (lat1 - lat0) * r, dLon = (lon1 - lon0) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat0 * r) * Math.cos(lat1 * r) * Math.sin(dLon / 2) ** 2;
  return 12742000 * Math.asin(Math.sqrt(h));
};
const tracks = [];
let rawFixes = 0;
for (const [key, list] of groups) {
  const fixes = [];
  for (const p of list) for (let i = 0; i < p.vt.length; i++) fixes.push([p.vt[i], p.pos[2 * i], p.pos[2 * i + 1]]);
  rawFixes += fixes.length;
  fixes.sort((a, b) => a[0] - b[0]);
  const kept = [];
  for (const fix of fixes) {
    const last = kept[kept.length - 1];
    if (last) {
      const dt = (fix[0] - last[0]) / 1000;
      if (dt < MIN_GAP_S) continue;
      if (metersBetween(last[1], last[2], fix[1], fix[2]) / dt > MAX_SPEED) continue;
    }
    kept.push(fix);
  }
  if (kept.length < MIN_FIXES) continue;
  const [species, organism, segment] = key.split('|');
  tracks.push({species, organism, segment: Number(segment), dataset: list[0].ds, fixes: kept});
}
tracks.sort((a, b) => SPECIES.indexOf(a.species) - SPECIES.indexOf(b.species) || a.organism.localeCompare(b.organism) || a.segment - b.segment);

// ---- write ---------------------------------------------------------------------------------
fs.mkdirSync(args.out, {recursive: true});
const total = tracks.reduce((s, t) => s + t.fixes.length, 0);
const offsets = new Uint32Array(tracks.length + 1);
const vertices = new Float32Array(total * 2);
const times = new Uint32Array(total);
let o = 0, minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
tracks.forEach((t, i) => {
  offsets[i] = o;
  for (const [ms, lon, lat] of t.fixes) {
    vertices[o * 2] = lon; vertices[o * 2 + 1] = lat;
    times[o] = Math.round((ms - ORIGIN_MS) / 1000);
    minX = Math.min(minX, lon); maxX = Math.max(maxX, lon); minY = Math.min(minY, lat); maxY = Math.max(maxY, lat);
    o++;
  }
});
offsets[tracks.length] = o;
const individuals = [...new Set(tracks.map((t) => t.organism))];
const datasetNames = DATASETS.map((d) => d.short);
const write = (file, typed) => { fs.writeFileSync(path.join(args.out, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength)); return typed.byteLength; };
write('pathOffsets.bin', offsets); write('vertices.bin', vertices); write('timestamp.bin', times);
write('species.bin', Uint8Array.from(tracks, (t) => SPECIES.indexOf(t.species)));
write('individual.bin', Uint16Array.from(tracks, (t) => individuals.indexOf(t.organism)));
write('year.bin', Uint8Array.from(tracks, (t) => t.segment));
write('source.bin', Uint8Array.from(tracks, (t) => datasetNames.indexOf(t.dataset)));
const manifest = {
  id: 'poopdeck-animals', version: 1, kind: 'trajectories', count: tracks.length, bbox: [minX, minY, maxX, maxY], crs: 'EPSG:4326',
  columns: {
    pathOffsets: {file: 'pathOffsets.bin', dtype: 'uint32', components: 1, length: tracks.length + 1},
    vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: total},
    timestamp: {file: 'timestamp.bin', dtype: 'uint32', components: 1, length: total, unit: 'seconds since 2024-01-01T00:00:00Z (multi-year tracks folded onto calendar 2024)'},
    species: {file: 'species.bin', dtype: 'uint8', components: 1, length: tracks.length, categories: SPECIES},
    individual: {file: 'individual.bin', dtype: 'uint16', components: 1, length: tracks.length, categories: individuals},
    year: {file: 'year.bin', dtype: 'uint8', components: 1, length: tracks.length, unit: 'segment: 1 = first tagged year of the animal'},
    source: {file: 'source.bin', dtype: 'uint8', components: 1, length: tracks.length, categories: datasetNames}
  },
  properties: {
    timeOriginMs: ORIGIN_MS,
    fixes: total,
    rawFixes,
    minGapSeconds: MIN_GAP_S,
    note: 'One track per animal-year (organism + segment). Years are folded onto calendar 2024 by the archive, so a track is one animal in one tagged year; all years of an animal overlap in time.',
    datasets: DATASETS.map(({short, key, doi, species}) => ({short, gbifKey: key, doi, species, license: 'CC0-1.0', publisher: 'Research Institute for Nature and Forest (INBO)'})),
    source: {archive: 'https://tiles.poopdeck.gl/data/animals/manifest.json', gbif: 'https://api.gbif.org/v1/dataset/<key>'}
  }
};
fs.writeFileSync(path.join(args.out, 'manifest.json'), JSON.stringify(manifest, null, 1));
console.log(JSON.stringify({tracks: tracks.length, individuals: individuals.length, rawFixes, fixes: total, bytes: total * 12 + tracks.length * 8, bbox: manifest.bbox}));
