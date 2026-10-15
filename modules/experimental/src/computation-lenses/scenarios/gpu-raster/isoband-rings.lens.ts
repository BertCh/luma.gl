// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_DEM} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.isoband-rings';

/** Break values every 100 m across the 1074 to 1786 m range of the window. */
const BREAKS = [1100, 1200, 1300, 1400, 1500, 1600, 1700];
/** Band boundary edges before chaining: about 2900 exist at these breaks. */
const EDGE_CAPACITY = 3600;
/** Ring capacity: about 50 rings exist over the eight bands. */
const RING_CAPACITY = 96;
/** Ring vertex capacity: one vertex per edge plus the repeated first vertex of each ring. */
const VERTEX_CAPACITY = 3800;
const UNIT = 'm (relative to localOrigin)';

/**
 * Records GPUIsobandRings on the Dixie Fire window. The DEM file runs north to south, while the
 * contributor places row 0 at the southern edge, so rows are flipped before upload.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixieDem, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./isoband-rings.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const terrain = await loadDixieDem();
  const entry = getRasterOperatorEntries()['GPUIsobandRings'];
  const {grid} = terrain;
  const southUp = new Float32Array(terrain.elevation.length);
  for (let row = 0; row < grid.height; row++) {
    const source = (grid.height - 1 - row) * grid.width;
    southUp.set(terrain.elevation.subarray(source, source + grid.width), row * grid.width);
  }
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUIsobandRings on the Dixie Fire window',
    grid,
    coordinateModel: terrain.coordinateModel,
    unit: terrain.unit,
    dataset: {id: T_DIXIE_DEM.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [{id: 'rings', label: 'band rings', size: RING_CAPACITY, indexSpace: '1d'}],
    choices: [
      {stage: 'operator', name: 'breakCount', value: BREAKS.length, sweepable: true},
      {stage: 'support', name: 'edgeCapacity', value: EDGE_CAPACITY, sweepable: false},
      {stage: 'support', name: 'localOrigin', value: 'south-west corner', sweepable: false}
    ],
    build: async session => {
      const ringGroups = session.output(
        'isobandRingBands',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'rings',
          meaning: 'band index of each ring (0 = below 1100 m)'
        },
        RING_CAPACITY
      );
      const ringOffsets = session.output(
        'isobandRingOffsets',
        'uint32',
        {role: 'intermediate', meaning: 'vertex offsets of each ring; unused rows repeat the end'},
        RING_CAPACITY + 1
      );
      const count = session.output(
        'isobandRingCount',
        'uint32',
        {role: 'intermediate', meaning: 'rings written'},
        1
      );
      const overflow = session.output(
        'isobandRingOverflow',
        'uint32',
        {
          role: 'intermediate',
          meaning: '1 when rings or vertices were dropped for lack of capacity'
        },
        1
      );
      // Unused vertex capacity stays NaN, so only the written ring vertices are drawn.
      const positionValues = session.input(
        'isobandRingPositions',
        'float32',
        new Float32Array(2 * VERTEX_CAPACITY).fill(NaN),
        {
          role: 'output',
          components: 2,
          geometry: {
            kind: 'polygons',
            coordinateModel: terrain.coordinateModel,
            unit: UNIT,
            offsetsCapture: 'isobandRingOffsets',
            valuesCapture: 'isobandRingBands'
          },
          meaning: `closed band rings, shells and holes, vertices in ${UNIT}`
        }
      );
      const positions = session.graph.createDataView(positionValues.buffer, {
        format: 'float32x2',
        length: VERTEX_CAPACITY
      });
      const dem = session.input('isobandRingsElevation', 'float32', southUp, {
        role: 'input',
        keySpace: 'cells',
        geometry: {
          kind: 'raster',
          shape: [grid.width, grid.height],
          coordinateModel: terrain.coordinateModel,
          cellSize: grid.affine[0],
          unit: UNIT,
          origin: [0, 0],
          rowDirection: 'north'
        },
        meaning: 'elevation (m), rows counted from the southern edge'
      });
      const breaks = session.input('isobandRingBreaks', 'float32', BREAKS, false);
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        dem,
        breaks,
        ringOffsets,
        positions,
        ringGroups,
        count,
        overflow
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUIsobandRings',
  title: 'IsobandRings',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_DEM,
  load: async () => ({run})
});
