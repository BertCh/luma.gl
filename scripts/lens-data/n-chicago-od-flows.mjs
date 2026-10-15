// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Extracts origin-destination taxi flows between Chicago community areas from the showcase
// `chicago-taxi-od` dataset (read-only; City of Chicago Data Portal, taxi trips 2023). It keeps the
// 150 busiest full-year pairs (origin != destination) as `annual` rows (the edge-bundling input) and,
// for the same pairs, the weekday 07:00-09:59 rows as `hourly` rows (the flow-aggregation input,
// several records per pair). Zones are renumbered by community-area number.

import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';

const root = new URL('../../', import.meta.url);
const source = new URL('examples/deck/spatial-analysis-showcase/public/data/', root);
const readColumn = (file, Type) => {
  const bytes = readFileSync(new URL(`chicago-taxi-od/${file}.bin`, source));
  return new Type(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
};
const manifest = JSON.parse(readFileSync(new URL('chicago-taxi-od/manifest.json', source), 'utf8'));
const locations = readColumn('locations', Float32Array);
const origin = readColumn('origin', Uint32Array);
const destination = readColumn('destination', Uint32Array);
const count = readColumn('count', Uint32Array);
const hourlyOrigin = readColumn('hourlyOrigin', Uint32Array);
const hourlyDestination = readColumn('hourlyDestination', Uint32Array);
const hourlyHour = readColumn('hourlyHour', Uint8Array);
const hourlyDaytype = readColumn('hourlyDaytype', Uint8Array);
const hourlyCount = readColumn('hourlyCount', Uint32Array);
const rows = Array.from({length: origin.length}, (_, row) => row)
  .filter(row => origin[row] !== destination[row])
  .sort((a, b) => count[b] - count[a] || origin[a] - origin[b] || destination[a] - destination[b])
  .slice(0, 150);
const pairKey = (from, to) => from * 77 + to;
const chosen = new Set(rows.map(row => pairKey(origin[row], destination[row])));
const zoneIds = [...new Set(rows.flatMap(row => [origin[row], destination[row]]))].sort((a, b) => a - b);
const zoneIndex = new Map(zoneIds.map((id, index) => [id, index]));
const annual = rows
  .map(row => ({origin: zoneIndex.get(origin[row]), destination: zoneIndex.get(destination[row]), trips: count[row]}))
  .sort((a, b) => a.origin - b.origin || a.destination - b.destination);
const hourly = [];
for (let row = 0; row < hourlyOrigin.length; row++)
  if (
    hourlyDaytype[row] === 0 &&
    hourlyHour[row] >= 7 &&
    hourlyHour[row] <= 9 &&
    chosen.has(pairKey(hourlyOrigin[row], hourlyDestination[row]))
  )
    hourly.push({
      origin: zoneIndex.get(hourlyOrigin[row]),
      destination: zoneIndex.get(hourlyDestination[row]),
      hour: hourlyHour[row],
      trips: hourlyCount[row]
    });
hourly.sort((a, b) => a.origin - b.origin || a.destination - b.destination || a.hour - b.hour);
const data = {
  year: 2023,
  note: 'Taxi trips between Chicago community-area centroids; zone positions are centroids, not trip endpoints',
  zoneCount: zoneIds.length,
  zoneNames: zoneIds.map(id => manifest.properties.areaNames[id]),
  zonePositions: zoneIds.flatMap(id => [Number(locations[2 * id].toFixed(5)), Number(locations[2 * id + 1].toFixed(5))]),
  annual: {
    originZones: annual.map(row => row.origin),
    destinationZones: annual.map(row => row.destination),
    trips: annual.map(row => row.trips)
  },
  hourly: {
    originZones: hourly.map(row => row.origin),
    destinationZones: hourly.map(row => row.destination),
    hours: hourly.map(row => row.hour),
    trips: hourly.map(row => row.trips)
  }
};
const bytes = `${JSON.stringify(data)}\n`;
if (annual.length < 100 || hourly.length > 500 || hourly.length < 100 || Buffer.byteLength(bytes) > 150000)
  throw new Error('OD flow budget failed');
const output = new URL('modules/experimental/src/computation-lenses/data/n-chicago-od-flows.json', root);
writeFileSync(output, bytes);
// biome-ignore lint/suspicious/noConsole: Extraction reports its deterministic sizes and counts.
console.log(`${fileURLToPath(output)}: ${Buffer.byteLength(bytes)} bytes; ${data.zoneCount} zones; ${annual.length} annual rows; ${hourly.length} hourly rows`);
