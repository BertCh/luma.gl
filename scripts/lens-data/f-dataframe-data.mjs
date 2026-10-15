// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile, writeFile, mkdir, appendFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import assert from 'node:assert/strict';

const root = fileURLToPath(new URL('../../', import.meta.url));
const sourceDirectory = path.join(root, 'examples/deck/spatial-analysis-showcase/public/data');
const outputDirectory = path.join(root, 'modules/experimental/src/computation-lenses/data');
const extractedBy = 'scripts/lens-data/f-dataframe-data.mjs';

async function loadDataset(name) {
  const directory = path.join(sourceDirectory, name);
  const manifest = JSON.parse(await readFile(path.join(directory, 'manifest.json'), 'utf8'));
  const columns = {};
  const readers = {
    float32: ['getFloat32', 4],
    uint32: ['getUint32', 4],
    uint16: ['getUint16', 2],
    uint8: ['getUint8', 1]
  };
  for (const [name, column] of Object.entries(manifest.columns)) {
    const bytes = await readFile(path.join(directory, column.file));
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const [reader, width] = readers[column.dtype];
    columns[name] = Array.from({length: column.length * column.components}, (_, index) =>
      view[reader](index * width, true)
    );
  }
  return {manifest, columns};
}

function makeProjection(bounds) {
  return {
    kind: 'local-equirectangular',
    unit: 'm',
    radius: 6371008.8,
    longitudeOrigin: (bounds[0] + bounds[2]) / 2,
    latitudeOrigin: (bounds[1] + bounds[3]) / 2
  };
}

function projectPosition(longitude, latitude, projection) {
  const radians = Math.PI / 180;
  return [
    (longitude - projection.longitudeOrigin) *
      radians *
      projection.radius *
      Math.cos(projection.latitudeOrigin * radians),
    (latitude - projection.latitudeOrigin) * radians * projection.radius
  ];
}

function roundValue(value) {
  return Number.isFinite(value) ? Number(value.toFixed(4)) : null;
}

async function saveDataset(dataset) {
  for (const [name, values] of Object.entries(dataset.columns))
    dataset.columns[name] = values.map(roundValue);
  const text = `${JSON.stringify({...dataset, extractedBy})}\n`;
  const size = Buffer.byteLength(text);
  assert(size <= 150000, `${dataset.id} exceeds 150 KB`);
  assert(Object.values(dataset.columns).every(column => column.length === dataset.rowCount));
  await writeFile(path.join(outputDirectory, `${dataset.id}.json`), text);
  console.log(`${dataset.id}.json: ${dataset.rowCount} rows, ${size} bytes`);
}

const tracts = await loadDataset('chicago-tracts');
const nature = await loadDataset('chicago-nature');
const stations = await loadDataset('ghcn-stations');
const projection = makeProjection(tracts.manifest.bbox);
const tractColumns = {tract: Array.from({length: tracts.manifest.count}, (_, index) => index)};
for (const name of [
  'communityArea',
  'population',
  'households',
  'noVehicle',
  'poverty150Pct',
  'unemployedPct',
  'uninsuredPct',
  'age65Pct',
  'medianHouseholdIncome',
  'perCapitaIncome',
  'areaKm2',
  'natureObs2023',
  'speciesRichness2023'
]) {
  tractColumns[name] = tracts.columns[name];
}
tractColumns.x = [];
tractColumns.y = [];
for (let tract = 0; tract < tracts.manifest.count; tract++) {
  const ring = tracts.columns.polygonRingOffsets[tract];
  const start = tracts.columns.ringOffsets[ring];
  const end = tracts.columns.ringOffsets[ring + 1];
  const points = Array.from({length: end - start}, (_, index) =>
    projectPosition(
      tracts.columns.vertices[(start + index) * 2],
      tracts.columns.vertices[(start + index) * 2 + 1],
      projection
    )
  );
  let areaTwice = 0;
  let weightedX = 0;
  let weightedY = 0;
  for (let index = 0; index < points.length; index++) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const cross = current[0] * next[1] - next[0] * current[1];
    areaTwice += cross;
    weightedX += (current[0] + next[0]) * cross;
    weightedY += (current[1] + next[1]) * cross;
  }
  assert(areaTwice !== 0, `Degenerate tract ring ${tract}`);
  tractColumns.x.push(weightedX / (3 * areaTwice));
  tractColumns.y.push(weightedY / (3 * areaTwice));
}
await mkdir(outputDirectory, {recursive: true});
const tractLicence =
  'U.S. Census ACS 5-year, Census boundaries, CDC/ATSDR SVI 2022 and CDC PLACES: public domain; nature counts: iNaturalist contributors, per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0, non-commercial';
await saveDataset({
  id: 'f-chicago-tracts',
  source:
    'Chicago showcase chicago-tracts v1: CDC/ATSDR SVI 2022 (ACS 2018–2022); ACS 2020–2024 B19301/B19013 via Census Reporter; CDC PLACES 2025; 2023 iNaturalist counts',
  licence: tractLicence,
  projection,
  rowCount: tracts.manifest.count,
  columns: tractColumns
});

const natureStride = Math.floor(nature.manifest.count / 1500);
const natureRows = Array.from({length: 1500}, (_, index) => index * natureStride);
const natureColumns = {x: [], y: []};
for (const index of natureRows) {
  const position = projectPosition(
    nature.columns.position[index * 2],
    nature.columns.position[index * 2 + 1],
    projection
  );
  natureColumns.x.push(position[0]);
  natureColumns.y.push(position[1]);
}
for (const name of [
  'timestamp',
  'category',
  'researchGrade',
  'species',
  'communityArea',
  'tract'
]) {
  natureColumns[name] = natureRows.map(index => nature.columns[name][index]);
}
await saveDataset({
  id: 'f-chicago-nature',
  source: nature.manifest.properties.source,
  licence:
    'iNaturalist contributors; per-observation CC0 / CC BY 4.0 / CC BY-NC 4.0; non-commercial',
  projection,
  rowCount: 1500,
  categories: nature.manifest.columns.category.categories,
  timestampUnit: nature.manifest.columns.timestamp.unit,
  studyWindow: [
    ...projectPosition(...tracts.manifest.bbox.slice(0, 2), projection),
    ...projectPosition(...tracts.manifest.bbox.slice(2), projection)
  ].map(roundValue),
  columns: natureColumns
});

const finiteStations = stations.columns.prcp.flatMap((value, index) =>
  Number.isFinite(value) ? [index] : []
);
assert(finiteStations.length >= 1200);
const stationStride = Math.floor(finiteStations.length / 1200);
const stationRows = Array.from({length: 1200}, (_, index) => finiteStations[index * stationStride]);
const stationProjection = makeProjection(stations.manifest.bbox);
const stationColumns = {x: [], y: []};
for (const index of stationRows) {
  const position = projectPosition(
    stations.columns.position[index * 2],
    stations.columns.position[index * 2 + 1],
    stationProjection
  );
  stationColumns.x.push(position[0]);
  stationColumns.y.push(position[1]);
}
for (const name of ['prcp', 'elevation', 'tmax'])
  stationColumns[name] = stationRows.map(index => stations.columns[name][index]);
await saveDataset({
  id: 'f-ghcn-stations',
  source: stations.manifest.properties.source,
  licence: 'NOAA NCEI GHCN-Daily, public domain',
  projection: stationProjection,
  rowCount: 1200,
  units: {prcp: 'mm', elevation: 'm', tmax: 'degC'},
  observationDate: '2024-09-27',
  columns: stationColumns
});

const passingRow = tractColumns.population.findIndex(
  (population, index) =>
    population > 1000 &&
    Number.isFinite(tractColumns.medianHouseholdIncome[index]) &&
    tractColumns.households[index] > 0
);
console.log(
  `Receipt row ${passingRow}, tract ${tracts.manifest.geoid[passingRow]}: population ${tractColumns.population[passingRow]}, households ${tractColumns.households[passingRow]}, noVehicle ${tractColumns.noVehicle[passingRow]}`
);
const readmePath = path.join(outputDirectory, 'README.md');
let readme = await readFile(readmePath, 'utf8').catch(error => {
  if (error.code !== 'ENOENT') throw error;
  return '';
});
if (!readme) {
  readme =
    '# Computation lens data\n\n| file | dataset | source | licence | rows | extraction |\n| --- | --- | --- | --- | --- | --- |\n';
  await writeFile(readmePath, readme);
}
for (const [file, dataset, source, licence, rows, extraction] of [
  [
    'f-chicago-tracts.json',
    'Chicago census tracts',
    'Showcase chicago-tracts v1; ACS 2018–2022 SVI and 2020–2024 income; PLACES 2025; 2023 nature counts',
    tractLicence,
    791,
    'All rows; first-ring area centroid; local equirectangular metres; four decimal places'
  ],
  [
    'f-chicago-nature.json',
    'Chicago wild observations 2023',
    'iNaturalist contributors; showcase chicago-nature v1',
    'CC0 / CC BY 4.0 / CC BY-NC 4.0 per observation; non-commercial',
    1500,
    `First 1500 every-${natureStride}th rows; local-clock seconds; tracts projection`
  ],
  [
    'f-ghcn-stations.json',
    'Helene station observations 2024-09-27',
    'NOAA NCEI GHCN-Daily via AWS Open Data; showcase ghcn-stations v1',
    'Public domain',
    1200,
    `First 1200 every-${stationStride}th finite-PRCP rows; local equirectangular metres`
  ]
]) {
  if (!readme.includes(`| ${file} |`)) {
    // The shared README may already use five columns; never change another stream's header.
    const row = readme.includes('| rows |')
      ? `| ${file} | ${dataset} | ${source} | ${licence} | ${rows} | ${extractedBy}: ${extraction} |\n`
      : `| ${file} | ${dataset} | ${source} | ${licence} | ${rows} rows; ${extractedBy}: ${extraction} |\n`;
    await appendFile(readmePath, row);
  }
}
