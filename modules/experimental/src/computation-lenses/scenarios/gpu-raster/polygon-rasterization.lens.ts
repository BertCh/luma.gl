// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_PERIMETER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.polygon-rasterization';

/**
 * Records GPUPolygonRasterization on the Dixie Fire window.
 *
 * The perimeter has three rings (a 5-vertex island, the 39-vertex fire outline and its 20-vertex
 * hole). Cell centres inside the even-odd union are 3649 of 4096 (the data file's `coverage`
 * of 0.8909). Coordinates stay absolute: the file rounds them to 1 m, which float32 holds exactly
 * at a UTM northing of 4.4e6 m.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixiePerimeter, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./polygon-rasterization.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const data = await loadDixiePerimeter();
  const entry = getRasterOperatorEntries()['GPUPolygonRasterization'];
  const {grid} = data;
  const placement = {
    kind: 'raster' as const,
    shape: [grid.width, grid.height] as [number, number],
    coordinateModel: data.coordinateModel,
    cellSize: grid.affine[0],
    unit: data.unit,
    origin: [grid.affine[2], grid.affine[5]] as [number, number],
    rowDirection: 'south' as const
  };
  // Ring r owns vertices [ringStarts[r], ringStarts[r + 1]); rings close implicitly (the file repeats
  // the first vertex, which only adds a zero-length edge).
  const ringStarts = [0];
  for (const ring of data.rings) ringStarts.push(ringStarts.at(-1)! + ring.length / 2);
  const positions = Float32Array.from(data.rings.flatMap(ring => [...ring]));
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPUPolygonRasterization on the Dixie Fire window',
    grid,
    coordinateModel: data.coordinateModel,
    unit: data.unit,
    dataset: {id: T_DIXIE_PERIMETER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    choices: [
      {stage: 'support', name: 'crossingCapacity', value: 256, sweepable: false},
      {stage: 'operator', name: 'spatialContext', value: 'planar', sweepable: false}
    ],
    build: async session => {
      const ringOffsets = session.input('perimeterRingOffsets', 'uint32', ringStarts, {
        role: 'intermediate',
        meaning: 'ring start vertices plus the final end'
      });
      const perimeterValues = session.input('perimeterPositions', 'float32', positions, {
        role: 'input',
        components: 2,
        geometry: {
          kind: 'polygons',
          coordinateModel: data.coordinateModel,
          unit: data.unit,
          offsetsCapture: 'perimeterRingOffsets'
        },
        meaning: 'Dixie perimeter ring vertices (UTM 10N m)'
      });
      const perimeterPositions = session.graph.createDataView(perimeterValues.buffer, {
        format: 'float32x2',
        length: positions.length / 2
      });
      // One feature of two polygons: ring 0 is the island, rings 1 and 2 are the fire and its hole.
      const featureOffsets = session.input('perimeterFeatureOffsets', 'uint32', [0, 2], false);
      const polygonOffsets = session.input('perimeterPolygonOffsets', 'uint32', [0, 1, 3], false);
      const zones = session.output('polygonZones', 'uint32', {
        role: 'output',
        geometry: placement,
        keySpace: 'cells',
        meaning: 'zone raster: feature row 0, or 4294967295 where no polygon covers the cell centre'
      });
      const boundary = session.output('polygonBoundary', 'uint32', {
        role: 'intermediate',
        geometry: placement,
        keySpace: 'cells',
        meaning: '1 where a polygon edge touches the closed cell'
      });
      const overflow = session.output(
        'polygonOverflow',
        'uint32',
        {role: 'intermediate', meaning: '1 when the crossings exceeded crossingCapacity'},
        1
      );
      const crossingCount = session.output(
        'polygonCrossingCount',
        'uint32',
        {role: 'intermediate', meaning: 'edge crossings of raster row centre lines (cap 256)'},
        1
      );
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        perimeterPositions,
        featureOffsets,
        polygonOffsets,
        ringOffsets,
        zones,
        boundary,
        overflow,
        crossingCount
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPUPolygonRasterization',
  title: 'PolygonRasterization',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_PERIMETER,
  load: async () => ({run})
});
