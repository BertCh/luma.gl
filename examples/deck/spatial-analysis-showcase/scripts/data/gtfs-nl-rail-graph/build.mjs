#!/usr/bin/env node
// Builds `gtfs-nl-rail-graph`: a directed station-to-station rail graph with scheduled in-vehicle
// times, derived from the OVapi national GTFS feed (CC0).
//
//   node build.mjs --gtfs <dir with the unzipped feed> [--out <dir>] [--date 20261009]
//
// The feed is https://gtfs.ovapi.nl/nl/gtfs-nl.zip (about 245 MB, 1.5 GB unzipped). Unzip only
// agency, routes, stops, trips, calendar_dates and stop_times (shapes.txt is not needed) into a
// fresh directory outside the app tree and pass it with --gtfs. stop_times.txt is streamed.
//
// Method: keep route_type 2 routes (replacement buses that carry route_type 2 are dropped by name),
// take the trips whose service runs on --date (a Friday), and for every pair of consecutive stops
// of a trip add 1 to the directed edge (station A, station B, train class). Stops are merged into
// their parent station (`stoparea:*`). Edge time = median over trips of
// arrival(B) - departure(A), so dwell at A is excluded and dwell at B is included only at the
// next hop. Counts per local hour come from the departure at A.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {fileURLToPath} from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, token, index, all) => {
    if (token.startsWith('--')) pairs.push([token.slice(2), all[index + 1]]);
    return pairs;
  }, [])
);
if (!args.gtfs) throw new Error('pass --gtfs <dir>');
const feed = args.gtfs;
const out = args.out || path.join(here, '../../../public/data/gtfs-nl-rail-graph');
const SERVICE_DATE = args.date || '20261009';
// NL and a margin: drops far foreign stops of international trains.
const BOX = [3.0, 50.6, 7.6, 53.7];

function splitCsv(line) {
  const cells = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quoted) {
      if (c === '"') {
        if (line[i + 1] === '"') { cell += '"'; i++; } else quoted = false;
      } else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { cells.push(cell); cell = ''; } else cell += c;
  }
  cells.push(cell);
  return cells;
}
async function* rows(file) {
  const lines = readline.createInterface({input: fs.createReadStream(path.join(feed, file)), crlfDelay: Infinity});
  let header = null;
  for await (const line of lines) {
    if (!line) continue;
    const cells = splitCsv(line);
    if (!header) { header = Object.fromEntries(cells.map((name, i) => [name.replace(/^﻿/, ''), i])); continue; }
    yield new Proxy(cells, {get: (target, key) => (key in header ? target[header[key]] : undefined)});
  }
}
const seconds = (text) => { const [h, m, s] = text.split(':').map(Number); return h * 3600 + m * 60 + s; };

const CLASS_NAMES = ['intercity and international', 'express (Sneltrein)', 'stopping (Sprinter, Stoptrein, other)'];
const classOf = (name) => {
  if (/^(Intercity|ICE|Eurostar|Eurocity|EuroCity|Nightjet|Nachttrein|European Sleeper)/i.test(name)) return 0;
  if (/^Sneltrein/i.test(name)) return 1;
  return 2;
};

const railRoute = new Map();
for await (const r of rows('routes.txt')) {
  if (r.route_type !== '2') continue;
  if (/^Drempelvrije bus/i.test(r.route_short_name)) continue;
  railRoute.set(r.route_id, classOf(r.route_short_name));
}
const activeServices = new Set();
for await (const r of rows('calendar_dates.txt')) if (r.date === SERVICE_DATE && r.exception_type === '1') activeServices.add(r.service_id);
const tripClass = new Map();
for await (const r of rows('trips.txt')) {
  if (railRoute.has(r.route_id) && activeServices.has(r.service_id)) tripClass.set(r.trip_id, railRoute.get(r.route_id));
}
console.error(`rail routes ${railRoute.size}, rail trips on ${SERVICE_DATE}: ${tripClass.size}`);

const stops = new Map();
for await (const r of rows('stops.txt')) {
  stops.set(r.stop_id, {name: r.stop_name, lat: Number(r.stop_lat), lon: Number(r.stop_lon), parent: r.parent_station});
}
const stationOf = (stopId) => {
  const stop = stops.get(stopId);
  if (!stop) return null;
  return stop.parent && stops.has(stop.parent) ? stop.parent : stopId;
};

// Stream stop_times: rows of one trip are contiguous.
const edges = new Map(); // key -> {a, b, cls, times: [], hours: Uint16Array(24)}
let currentTrip = null;
let previous = null;
let tripsSeen = 0;
for await (const r of rows('stop_times.txt')) {
  const cls = tripClass.get(r.trip_id);
  if (cls === undefined) { previous = null; currentTrip = null; continue; }
  if (r.trip_id !== currentTrip) { currentTrip = r.trip_id; previous = null; tripsSeen++; }
  const station = stationOf(r.stop_id);
  if (station === null) continue;
  const arrival = seconds(r.arrival_time);
  const departure = seconds(r.departure_time);
  if (previous && previous.station !== station) {
    const key = `${previous.station}|${station}|${cls}`;
    let edge = edges.get(key);
    if (!edge) { edge = {a: previous.station, b: station, cls, times: [], hours: new Uint16Array(24)}; edges.set(key, edge); }
    edge.times.push(arrival - previous.departure);
    edge.hours[Math.min(23, Math.floor(previous.departure / 3600))]++;
  }
  previous = {station, departure};
}
console.error(`trips streamed ${tripsSeen}, raw edges ${edges.size}`);

// Nodes: stations in the box that appear in an edge.
const inBox = (id) => { const s = stops.get(id); return s.lon >= BOX[0] && s.lon <= BOX[2] && s.lat >= BOX[1] && s.lat <= BOX[3]; };
const used = new Set();
const kept = [];
for (const edge of edges.values()) {
  if (!inBox(edge.a) || !inBox(edge.b)) continue;
  kept.push(edge);
  used.add(edge.a);
  used.add(edge.b);
}
const stationIds = [...used].sort((p, q) => stops.get(p).name.localeCompare(stops.get(q).name) || p.localeCompare(q));
const nodeIndex = new Map(stationIds.map((id, i) => [id, i]));
kept.sort((p, q) => nodeIndex.get(p.a) - nodeIndex.get(q.a) || nodeIndex.get(p.b) - nodeIndex.get(q.b) || p.cls - q.cls);

const nodeCount = stationIds.length;
const edgeCount = kept.length;
const nodes = new Float32Array(nodeCount * 2);
stationIds.forEach((id, i) => { nodes[i * 2] = stops.get(id).lon; nodes[i * 2 + 1] = stops.get(id).lat; });
const edgeSource = new Uint32Array(edgeCount);
const edgeTarget = new Uint32Array(edgeCount);
const edgeClass = new Uint8Array(edgeCount);
const edgeTravelTime = new Float32Array(edgeCount);
const edgeLength = new Float32Array(edgeCount);
const edgeTripsPerHour = new Uint16Array(edgeCount * 24);
const nodeEdgeOffsets = new Uint32Array(nodeCount + 1);
const median = (values) => { const s = [...values].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const metersPerDegreeLon = 111320 * Math.cos((52 * Math.PI) / 180);
kept.forEach((edge, i) => {
  const a = nodeIndex.get(edge.a);
  const b = nodeIndex.get(edge.b);
  edgeSource[i] = a;
  edgeTarget[i] = b;
  edgeClass[i] = edge.cls;
  edgeTravelTime[i] = Math.max(60, median(edge.times));
  edgeLength[i] = Math.hypot((nodes[a * 2] - nodes[b * 2]) * metersPerDegreeLon, (nodes[a * 2 + 1] - nodes[b * 2 + 1]) * 110574);
  edgeTripsPerHour.set(edge.hours, i * 24);
  nodeEdgeOffsets[a + 1]++;
});
for (let n = 0; n < nodeCount; n++) nodeEdgeOffsets[n + 1] += nodeEdgeOffsets[n];

fs.mkdirSync(out, {recursive: true});
const write = (file, typed) => fs.writeFileSync(path.join(out, file), new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
write('nodes.bin', nodes);
write('edgeSource.bin', edgeSource);
write('edgeTarget.bin', edgeTarget);
write('edgeClass.bin', edgeClass);
write('edgeTravelTime.bin', edgeTravelTime);
write('edgeLength.bin', edgeLength);
write('edgeTripsPerHour.bin', edgeTripsPerHour);
write('nodeEdgeOffsets.bin', nodeEdgeOffsets);

const bbox = [Math.min(...stationIds.map((id) => stops.get(id).lon)), Math.min(...stationIds.map((id) => stops.get(id).lat)), Math.max(...stationIds.map((id) => stops.get(id).lon)), Math.max(...stationIds.map((id) => stops.get(id).lat))];
const manifest = {
  id: 'gtfs-nl-rail-graph',
  version: 1,
  kind: 'network',
  count: edgeCount,
  bbox,
  crs: 'EPSG:4326',
  columns: {
    nodes: {file: 'nodes.bin', dtype: 'float32', components: 2, length: nodeCount, note: 'station lon/lat (stoparea)'},
    edgeSource: {file: 'edgeSource.bin', dtype: 'uint32', length: edgeCount, note: 'edges sorted by (source, target, class); directed'},
    edgeTarget: {file: 'edgeTarget.bin', dtype: 'uint32', length: edgeCount},
    edgeClass: {file: 'edgeClass.bin', dtype: 'uint8', length: edgeCount, categories: CLASS_NAMES},
    edgeTravelTime: {file: 'edgeTravelTime.bin', dtype: 'float32', length: edgeCount, unit: 's', note: 'median scheduled arrival(B) - departure(A) over the day, at least 60 s'},
    edgeLength: {file: 'edgeLength.bin', dtype: 'float32', length: edgeCount, unit: 'm', note: 'straight line between the two stations'},
    edgeTripsPerHour: {file: 'edgeTripsPerHour.bin', dtype: 'uint16', components: 24, length: edgeCount, note: 'trains per local departure hour 0-23 at the source station'},
    nodeEdgeOffsets: {file: 'nodeEdgeOffsets.bin', dtype: 'uint32', length: nodeCount + 1, note: 'CSR: out-edges of node i are nodeEdgeOffsets[i]..nodeEdgeOffsets[i+1]-1'}
  },
  properties: {
    description: 'Scheduled station-to-station rail graph of the Netherlands (plus border stations), one service day.',
    serviceDate: SERVICE_DATE,
    stationNames: stationIds.map((id) => stops.get(id).name),
    railTrips: tripsSeen,
    source: {upstream: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip', attribution: 'OVapi / NDOV, CC0', feedVersion: 'OVapi feed valid 2026-10-06 to 2026-12-12'}
  }
};
fs.writeFileSync(path.join(out, 'manifest.json'), JSON.stringify(manifest));
console.log(JSON.stringify({nodeCount, edgeCount, tripsSeen}));
