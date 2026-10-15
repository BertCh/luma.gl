// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile, writeFile} from 'node:fs/promises';

const root = new URL('../../', import.meta.url);
const sourceDirectory = new URL('examples/deck/spatial-analysis-showcase/public/data/chicago-tracts/', root);
const manifest = JSON.parse(await readFile(new URL('manifest.json', sourceDirectory), 'utf8'));
const readColumn = async name => {
  const column = manifest.columns[name];
  const bytes = await readFile(new URL(column.file, sourceDirectory));
  const width = column.dtype === 'uint8' ? 1 : 4;
  return Array.from({length: bytes.length / width}, (_, index) =>
    column.dtype === 'float32' ? bytes.readFloatLE(index * width) :
      column.dtype === 'uint32' ? bytes.readUInt32LE(index * width) : bytes[index]
  );
};
const vertices = await readColumn('vertices');
const ringOffsets = await readColumn('ringOffsets');
const origin = [-88, 41];
const quantum = 1e-5;
const deltas = [];
for (let ring = 0; ring < ringOffsets.length - 1; ring++) {
  let previousLongitude = 0;
  let previousLatitude = 0;
  for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
    const longitude = Math.round((vertices[vertex * 2] - origin[0]) / quantum);
    const latitude = Math.round((vertices[vertex * 2 + 1] - origin[1]) / quantum);
    deltas.push(longitude - previousLongitude, latitude - previousLatitude);
    previousLongitude = longitude;
    previousLatitude = latitude;
  }
}
const columnPrecision = {
  population: 0, medianHouseholdIncome: 0, poverty150: 0, poverty150Pct: 1,
  asthma: 1, diabetes: 1, noVehiclePct: 1, uninsuredPct: 1, unemployedPct: 1,
  housingBurdened: 0, areaKm2: 4, communityArea: 0
};
const columns = {};
for (const [name, digits] of Object.entries(columnPrecision)) {
  columns[name] = (await readColumn(name)).map(value =>
    Number.isFinite(value) ? Number(value.toFixed(digits)) : null
  );
}
const data = {
  schema: 's1-chicago-tracts/1',
  source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022); CDC PLACES release 2025 (2023 BRFSS); ACS 2020-2024 B19013 via Census Reporter. Whole Cook County tracts whose representative point is inside Chicago, sorted by GEOID.',
  licence: ['boundaries', 'svi', 'places', 'acs'].map(name => manifest.properties.licenses[name]).join('; '),
  crs: 'EPSG:4326', count: manifest.count, quantum, origin,
  ringOffsets, polygonRingOffsets: await readColumn('polygonRingOffsets'),
  partFeature: await readColumn('partFeature'), deltas, columns
};
// Keep the file compact; formatting tools must not expand this generated payload.
const serialize = () => `${JSON.stringify(data)}\n`;
for (const name of Object.keys(columns).reverse()) {
  if (Buffer.byteLength(serialize()) <= 150_000) break;
  delete columns[name];
}
if (Buffer.byteLength(serialize()) > 150_000) throw new Error('Geometry exceeds the lens data budget');
await writeFile(new URL('modules/experimental/src/computation-lenses/data/s1-chicago-tracts.json', root), serialize());
