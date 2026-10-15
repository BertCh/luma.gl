// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Run from any directory: node scripts/lens-data/s1-dixie-lattice.mjs.
// Cuts the central 32 x 32 cells out of the 64 x 64 Dixie DEM (120 m block means of USGS 3DEP).
import {readFile, writeFile} from 'node:fs/promises';

const root = new URL('../../', import.meta.url);
const directory = new URL('modules/experimental/src/computation-lenses/data/', root);
const dem = JSON.parse(await readFile(new URL('t-dixie-dem.json', directory), 'utf8'));
const FIRST = 16;
const SIZE = 32;
if (dem.width !== 64 || dem.height !== 64) throw new Error('Unexpected Dixie DEM size');
const values = [];
for (let row = FIRST; row < FIRST + SIZE; row++) {
  for (let column = FIRST; column < FIRST + SIZE; column++) {
    values.push(dem.values[row * dem.width + column]);
  }
}
const result = {
  id: 's1-dixie-lattice',
  description: 'Central 32 x 32 block of the Dixie DEM (rows north to south), a regular lattice for rook and queen contiguity.',
  source: dem.source,
  licence: dem.licence,
  crs: dem.crs,
  width: SIZE,
  height: SIZE,
  cellSize: dem.cellSize,
  origin: [dem.origin[0] + FIRST * dem.cellSize, dem.origin[1] - FIRST * dem.cellSize],
  unit: dem.unit,
  values
};
await writeFile(new URL('s1-dixie-lattice.json', directory), `${JSON.stringify(result)}\n`);
