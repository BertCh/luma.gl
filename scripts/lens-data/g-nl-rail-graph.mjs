// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile, writeFile} from 'node:fs/promises';

const sourceDirectory = new URL(
  '../../examples/deck/spatial-analysis-showcase/public/data/gtfs-nl-rail-graph/',
  import.meta.url
);
const manifest = JSON.parse(await readFile(new URL('manifest.json', sourceDirectory), 'utf8'));

async function readColumn(name) {
  const column = manifest.columns[name];
  const bytes = await readFile(new URL(column.file, sourceDirectory));
  const readValue = {float32: 'readFloatLE', uint32: 'readUInt32LE'}[column.dtype];
  if (!readValue) throw new Error(`Unsupported dtype: ${column.dtype}`);
  const length = column.length * (column.components ?? 1);
  if (bytes.length !== length * 4) throw new Error(`Invalid byte length: ${name}`);
  return Array.from({length}, (_, index) => bytes[readValue](index * 4));
}

const [lonLat, source, target, travelTimeSeconds] = await Promise.all(
  ['nodes', 'edgeSource', 'edgeTarget', 'edgeTravelTime'].map(readColumn)
);
const serviceDate = manifest.properties.serviceDate.replace(/^(\d{4})(\d{2})(\d{2})$/, '$1-$2-$3');
const dataset = {
  id: 'g-nl-rail-graph',
  source: 'https://gtfs.ovapi.nl/nl/gtfs-nl.zip (OVapi / NDOV)',
  licence: 'CC0',
  serviceDate,
  stationNames: manifest.properties.stationNames,
  lonLat: lonLat.map(value => Number(value.toFixed(5))),
  edges: {source, target, travelTimeSeconds: travelTimeSeconds.map(Math.round)}
};
const text = `${JSON.stringify(dataset)}\n`;
if (Buffer.byteLength(text) > 150_000) throw new Error('Rail graph exceeds 150 KB');
await writeFile(
  new URL(
    '../../modules/experimental/src/computation-lenses/data/g-nl-rail-graph.json',
    import.meta.url
  ),
  text
);
process.stdout.write(
  `${dataset.stationNames.length} stations, ${source.length} edges, ${Buffer.byteLength(text)} bytes\n`
);
