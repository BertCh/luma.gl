// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFile, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {PNG} from 'pngjs';

const root = new URL('../../', import.meta.url);
const dataRoot = new URL('examples/deck/spatial-analysis-showcase/public/data/', root);
const output = new URL('modules/experimental/src/computation-lenses/data/k-core.json', root);
const readJSON = async path => JSON.parse(await readFile(new URL(path, dataRoot), 'utf8'));
const stationManifest = await readJSON('ghcn-stations/manifest.json');
const terrainManifest = await readJSON('alps-dem/manifest.json');
const readColumn = async name => {
  const bytes = await readFile(new URL(`ghcn-stations/${name}.bin`, dataRoot));
  return Array.from({length: bytes.length / 4}, (_, index) => bytes.readFloatLE(index * 4));
};
const [position, elevation, precipitation, previousPrecipitation, stationCSV, imageBytes] =
  await Promise.all([
    readColumn('position'), readColumn('elevation'), readColumn('prcp'), readColumn('prcpPrevDay'),
    readFile(new URL('ghcn-stations/stations.csv', dataRoot), 'utf8'),
    readFile(new URL('alps-dem/dem.png', dataRoot))
  ]);
// State is the last CSV field; station names may contain quoted commas.
const stateCodes = stationCSV.trimEnd().split(/\r?\n/).slice(1).map(line => line.slice(line.lastIndexOf(',') + 1));
const eligible = precipitation.map((_, row) => row).filter(row =>
  [precipitation[row], previousPrecipitation[row], elevation[row]].every(Number.isFinite));
if (eligible.length < 1024) throw new Error('Need at least 1,024 complete station rows');
// Take every k-th eligible source row, then the first 1,024 samples, before sorting by state.
const stride = Math.floor(eligible.length / 1024);
const selected = Array.from({length: 1024}, (_, index) => eligible[index * stride]).sort(
  (left, right) => stateCodes[left] < stateCodes[right] ? -1 : stateCodes[left] > stateCodes[right] ? 1 : left - right
);
const states = [...new Set(selected.map(row => stateCodes[row]))];
const round = (value, decimals) => Number(value.toFixed(decimals));
const stations = {
  row: selected,
  lon: selected.map(row => round(position[row * 2], 5)),
  lat: selected.map(row => round(position[row * 2 + 1], 5)),
  prcpMm: selected.map(row => round(precipitation[row], 1)),
  prcpPrevDayMm: selected.map(row => round(previousPrecipitation[row], 1)),
  elevationM: selected.map(row => round(elevation[row], 1)),
  state: selected.map(row => states.indexOf(stateCodes[row])),
  states,
  stateOffsets: [...states.map(state => selected.findIndex(row => stateCodes[row] === state)), selected.length]
};
const image = PNG.sync.read(imageBytes);
if (image.width !== 2048 || image.height !== 2048) throw new Error('Expected a 2,048-square Terrarium DEM');
const elevationM = [];
for (let blockRow = 0; blockRow < 64; blockRow++) {
  for (let blockColumn = 0; blockColumn < 64; blockColumn++) {
    let sum = 0;
    for (let row = 0; row < 32; row++) {
      for (let column = 0; column < 32; column++) {
        const offset = ((blockRow * 32 + row) * image.width + blockColumn * 32 + column) * 4;
        sum += image.data[offset] * 256 + image.data[offset + 1] + image.data[offset + 2] / 256 - 32768;
      }
    }
    elevationM.push(Math.round(sum / 1024 * 4) / 4);
  }
}
const data = {
  meta: {
    sources: {stations: stationManifest.properties.source, dem: terrainManifest.properties.source},
    licences: {stations: 'Public domain (US Government work)', dem: 'swissALTI3D © swisstopo OGD / Copernicus GLO-30; free with attribution'},
    extractedBy: 'scripts/lens-data/k-core.mjs',
    date: '2024-09-27'
  },
  stations,
  dem: {width: 64, height: 64, cellSizeM: round(32 * terrainManifest.raster.cellSizeGroundM, 1),
    bounds: terrainManifest.raster.bounds, elevationM}
};
const text = `${JSON.stringify(data)}\n`;
if (Buffer.byteLength(text) > 150 * 1024) throw new Error('k-core.json exceeds 150 KB');
await writeFile(fileURLToPath(output), text);
