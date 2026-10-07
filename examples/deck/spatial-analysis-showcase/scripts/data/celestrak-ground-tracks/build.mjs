// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Builds `public/data/celestrak-ground-tracks`: SGP4 ground tracks of about 900 satellites over a
 * few hours, propagated from current CelesTrak TLEs.
 *
 *   node build.mjs --tle-dir <dir of *.tle downloads> --satellite-js <dir containing node_modules/satellite.js>
 *                  [--start 2026-10-07T00:00:00Z] [--hours 3] [--step 30] [--starlink 800]
 *
 * The TLE files are fetched beforehand (see README.md). satellite.js is NOT an app dependency:
 * install it in a scratch directory and pass that directory with --satellite-js.
 */
import {createRequire} from 'node:module';
import {mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ''), process.argv[i + 1]);
const tleDir = resolve(args.get('tle-dir') ?? '');
const satelliteJsDir = resolve(args.get('satellite-js') ?? '');
const startIso = args.get('start') ?? '2026-10-07T00:00:00Z';
const hours = Number(args.get('hours') ?? 3);
const stepSeconds = Number(args.get('step') ?? 30);
const starlinkSample = Number(args.get('starlink') ?? 800);
const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, '../../../public/data/celestrak-ground-tracks');

const satelliteRequire = createRequire(join(satelliteJsDir, 'package.json'));
const satellite = await import(pathToFileURL(satelliteRequire.resolve('satellite.js')).href);

const GROUPS = ['Stations', 'Starlink (sample)', 'GPS', 'Weather', 'Earth observation'];
const ORBIT_TYPES = ['LEO', 'MEO', 'HEO'];

// Nominal imaging swath widths (km) of the Earth observation instruments, from agency fact sheets.
const EARTH_OBSERVATION = [
  [/^LANDSAT/, 185, 'OLI'],
  [/^SENTINEL-2/, 290, 'MSI'],
  [/^SENTINEL-1/, 250, 'C-SAR IW'],
  [/^SENTINEL-3/, 1270, 'OLCI'],
  [/^SENTINEL-5P/, 2600, 'TROPOMI'],
  [/^(TERRA|AQUA)$/, 2330, 'MODIS'],
  [/^(SUOMI NPP|NOAA 20|NOAA 21)/, 3000, 'VIIRS'],
  [/^METOP-[BC]/, 2900, 'AVHRR']
];

function parseTle(name) {
  const lines = readFileSync(join(tleDir, `${name}.tle`), 'utf8').split(/\r?\n/).filter(Boolean);
  const out = [];
  for (let i = 0; i + 2 < lines.length; i += 3) out.push({name: lines[i].trim(), l1: lines[i + 1], l2: lines[i + 2]});
  return out;
}
function seededRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let v = s;
    v = Math.imul(v ^ (v >>> 15), v | 1);
    v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
    return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- Select satellites ------------------------------------------------------------------------
const chosen = new Map(); // norad -> {name, l1, l2, group, swathKm}
const add = (entry, group, swathKm = 0, sensor = '') => {
  const norad = Number(entry.l2.slice(2, 7));
  if (!chosen.has(norad)) chosen.set(norad, {...entry, norad, group, swathKm, sensor});
};
for (const entry of parseTle('stations')) add(entry, 0);
for (const list of ['resource', 'weather', 'science']) {
  for (const entry of parseTle(list)) {
    const match = EARTH_OBSERVATION.find(([pattern]) => pattern.test(entry.name));
    if (match) add(entry, 4, match[1], match[2]);
  }
}
for (const entry of parseTle('gps-ops')) add(entry, 2);
const meanMotionOf = entry => Number(entry.l2.slice(52, 63));
for (const entry of parseTle('weather')) {
  if (meanMotionOf(entry) > 5) add(entry, 3); // drop geostationary weather satellites
}
const starlink = parseTle('starlink');
const random = seededRandom(20261007);
const order = starlink.map((entry, index) => ({entry, key: random(), index})).sort((a, b) => a.key - b.key);
for (const {entry} of order.slice(0, starlinkSample)) add(entry, 1);

const satellites = [...chosen.values()];
console.log(`satellites: ${satellites.length}`);

// ---- Propagate and split at the antimeridian ----------------------------------------------------
const start = new Date(startIso);
const startJulian = start.getTime() / 86400000 + 2440587.5;
const steps = Math.round((hours * 3600) / stepSeconds) + 1;
const degrees = 180 / Math.PI;

const vertices = [];
const timestamps = [];
const altitudes = [];
const pathOffsets = [0];
const trackColumns = {satellite: [], group: [], orbitType: [], inclination: [], meanAltitude: [], swathKm: []};
const satelliteInfo = [];
let minLatitude = 90;
let maxLatitude = -90;

satellites.forEach((sat, satelliteIndex) => {
  const satrec = satellite.twoline2satrec(sat.l1, sat.l2);
  const samples = [];
  for (let step = 0; step < steps; step++) {
    const seconds = step * stepSeconds;
    const date = new Date(start.getTime() + seconds * 1000);
    const result = satellite.propagate(satrec, date);
    if (!result || !result.position) break;
    const geodetic = satellite.eciToGeodetic(result.position, satellite.gstime(date));
    samples.push({
      t: seconds,
      lon: geodetic.longitude * degrees,
      lat: geodetic.latitude * degrees,
      alt: geodetic.height * 1000
    });
  }
  if (samples.length < steps || samples.some(s => !Number.isFinite(s.lon) || !Number.isFinite(s.alt))) {
    console.warn(`dropped ${sat.name}: propagation failed`);
    satellites[satelliteIndex] = null;
    return;
  }
  const meanAltitude = samples.reduce((sum, s) => sum + s.alt, 0) / samples.length;
  const maxAltitude = Math.max(...samples.map(s => s.alt));
  const eccentricity = Number(`0.${sat.l2.slice(26, 33).trim()}`);
  const inclination = Number(sat.l2.slice(8, 16));
  const meanMotion = Number(sat.l2.slice(52, 63));
  const periodMinutes = 1440 / meanMotion;
  const orbitType = maxAltitude > 2_000_000 ? (eccentricity > 0.25 ? 2 : 1) : 0;
  const epochAgeDays = startJulian - satrec.jdsatepoch - (satrec.jdsatepochF ?? 0);
  satelliteInfo.push({
    name: sat.name,
    norad: sat.norad,
    group: sat.group,
    orbitType,
    inclination,
    meanAltitudeKm: meanAltitude / 1000,
    periodMinutes,
    eccentricity,
    tleAgeDays: epochAgeDays,
    swathKm: sat.swathKm,
    sensor: sat.sensor
  });
  const index = satelliteInfo.length - 1;

  let segment = [];
  const flush = () => {
    if (segment.length >= 2) {
      for (const s of segment) {
        vertices.push(s.lon, s.lat);
        timestamps.push(Math.round(s.t));
        altitudes.push(s.alt);
        minLatitude = Math.min(minLatitude, s.lat);
        maxLatitude = Math.max(maxLatitude, s.lat);
      }
      pathOffsets.push(timestamps.length);
      trackColumns.satellite.push(index);
      trackColumns.group.push(sat.group);
      trackColumns.orbitType.push(orbitType);
      trackColumns.inclination.push(inclination);
      trackColumns.meanAltitude.push(meanAltitude);
      trackColumns.swathKm.push(sat.swathKm);
    }
    segment = [];
  };
  for (let i = 0; i < samples.length; i++) {
    const current = samples[i];
    if (i > 0) {
      const previous = samples[i - 1];
      const jump = current.lon - previous.lon;
      if (Math.abs(jump) > 180) {
        // Crossing: unwrap, interpolate latitude, altitude and time at +-180.
        const eastward = jump < 0; // lon went 179 -> -179
        const unwrapped = eastward ? current.lon + 360 : current.lon - 360;
        const edge = eastward ? 180 : -180;
        const fraction = (edge - previous.lon) / (unwrapped - previous.lon);
        const crossing = {
          t: previous.t + fraction * (current.t - previous.t),
          lat: previous.lat + fraction * (current.lat - previous.lat),
          alt: previous.alt + fraction * (current.alt - previous.alt)
        };
        segment.push({...crossing, lon: edge});
        flush();
        segment.push({...crossing, lon: -edge});
      }
    }
    segment.push(current);
  }
  flush();
});

const trackCount = pathOffsets.length - 1;
const vertexCount = timestamps.length;
console.log(`tracks: ${trackCount}, vertices: ${vertexCount}`);

// ---- Write ------------------------------------------------------------------------------------
mkdirSync(outDir, {recursive: true});
const files = {};
const write = (file, typed) => {
  writeFileSync(join(outDir, file), Buffer.from(typed.buffer, typed.byteOffset, typed.byteLength));
  files[file] = typed.byteLength;
};
write('pathOffsets.bin', Uint32Array.from(pathOffsets));
write('vertices.bin', Float32Array.from(vertices));
write('timestamp.bin', Uint32Array.from(timestamps));
write('altitude.bin', Float32Array.from(altitudes));
write('satellite.bin', Uint32Array.from(trackColumns.satellite));
write('group.bin', Uint8Array.from(trackColumns.group));
write('orbitType.bin', Uint8Array.from(trackColumns.orbitType));
write('inclination.bin', Float32Array.from(trackColumns.inclination));
write('meanAltitude.bin', Float32Array.from(trackColumns.meanAltitude));
write('swathKm.bin', Float32Array.from(trackColumns.swathKm));

const column = (file, dtype, length, extra = {}) => ({file, dtype, components: 1, length, ...extra});
const manifest = {
  id: 'celestrak-ground-tracks',
  version: 1,
  kind: 'trajectories',
  count: trackCount,
  bbox: [-180, Math.floor(minLatitude), 180, Math.ceil(maxLatitude)],
  crs: 'EPSG:4326',
  columns: {
    pathOffsets: column('pathOffsets.bin', 'uint32', trackCount + 1),
    vertices: {file: 'vertices.bin', dtype: 'float32', components: 2, length: vertexCount},
    timestamp: column('timestamp.bin', 'uint32', vertexCount, {unit: `seconds since ${start.toISOString()}`}),
    altitude: column('altitude.bin', 'float32', vertexCount, {unit: 'meters above the WGS84 ellipsoid'}),
    satellite: column('satellite.bin', 'uint32', trackCount, {description: 'index into properties.satellites; a satellite has several tracks because tracks are cut at the antimeridian'}),
    group: column('group.bin', 'uint8', trackCount, {categories: GROUPS}),
    orbitType: column('orbitType.bin', 'uint8', trackCount, {categories: ORBIT_TYPES}),
    inclination: column('inclination.bin', 'float32', trackCount, {unit: 'degrees (TLE)'}),
    meanAltitude: column('meanAltitude.bin', 'float32', trackCount, {unit: 'meters, mean over the window'}),
    swathKm: column('swathKm.bin', 'float32', trackCount, {description: 'nominal imaging swath, 0 when not an Earth observation satellite'})
  },
  properties: {
    timeOrigin: start.toISOString(),
    timeOriginUnixSeconds: start.getTime() / 1000,
    stepSeconds,
    durationSeconds: (steps - 1) * stepSeconds,
    satelliteCount: satelliteInfo.length,
    groups: GROUPS,
    orbitTypes: ORBIT_TYPES,
    satellites: satelliteInfo,
    projection: 'longitude/latitude degrees; split at the antimeridian with interpolated crossing vertices at exactly +-180',
    source: 'CelesTrak GP element sets (TLE), SGP4 via satellite.js'
  }
};
writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 1));
const total = Object.values(files).reduce((a, b) => a + b, 0);
console.log(`wrote ${(total / 1e6).toFixed(2)} MB to ${outDir}`);
const tleAges = satelliteInfo.map(s => s.tleAgeDays).sort((a, b) => a - b);
console.log('TLE age days min/median/max', tleAges[0].toFixed(2), tleAges[tleAges.length >> 1].toFixed(2), tleAges.at(-1).toFixed(2));
const byGroup = GROUPS.map((g, i) => `${g}: ${satelliteInfo.filter(s => s.group === i).length}`);
console.log(byGroup.join(', '));
