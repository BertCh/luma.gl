// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {registerLensScenario} from '../../registry';
import type {LensScenarioResult} from '../../types';
import {T_DIXIE_PERIMETER} from '../gpu-raster/raster-lens-datasets';

const SCENARIO_ID = 'gpu-raster.raster-join';

/** USGS threshold of moderate-low burn severity; cells above it are the joined points. */
const BURN_DNBR_THRESHOLD = 0.27;

/**
 * Records GPURasterJoin on the Dixie Fire window: the centres of cells with dNBR above 0.27
 * (NBR = (NIR - SWIR) / (NIR + SWIR), before minus after; 2473 of 4096 cells) join the zone raster
 * of the Dixie perimeter. 27 of them lie outside the perimeter and take no zone.
 */
export async function run(device: Device): Promise<LensScenarioResult> {
  const [
    {runExample},
    {runRasterLens},
    {loadDixiePerimeter, loadDixieBands, T_DATA_VERSION},
    {getRasterOperatorEntries}
  ] = await Promise.all([
    import('./raster-join.example'),
    import('./raster-lens-harness'),
    import('./raster-lens-data'),
    import('../../../gpu-raster/computation-pack/index')
  ]);
  const [perimeter, bands] = await Promise.all([loadDixiePerimeter(), loadDixieBands()]);
  const entry = getRasterOperatorEntries()['GPURasterJoin'];
  const {grid} = perimeter;
  const placement = {
    kind: 'raster' as const,
    shape: [grid.width, grid.height] as [number, number],
    coordinateModel: perimeter.coordinateModel,
    cellSize: grid.affine[0],
    unit: perimeter.unit,
    origin: [grid.affine[2], grid.affine[5]] as [number, number],
    rowDirection: 'south' as const
  };
  // Cell centres (UTM 10N metres) of the burned cells, in row-major order.
  const getNormalizedBurnRatio = (nir: number, swir: number) =>
    nir > bands.noData && swir > bands.noData ? (nir - swir) / (nir + swir) : NaN;
  const burnedCells: number[] = [];
  for (let cell = 0; cell < grid.width * grid.height; cell++) {
    const dnbr =
      getNormalizedBurnRatio(bands.bands.nirBefore[cell], bands.bands.swirBefore[cell]) -
      getNormalizedBurnRatio(bands.bands.nirAfter[cell], bands.bands.swirAfter[cell]);
    if (dnbr > BURN_DNBR_THRESHOLD) {
      const column = cell % grid.width;
      const row = Math.floor(cell / grid.width);
      burnedCells.push(
        grid.affine[2] + (column + 0.5) * grid.affine[0],
        grid.affine[5] + (row + 0.5) * grid.affine[4]
      );
    }
  }
  const ringStarts = [0];
  for (const ring of perimeter.rings) ringStarts.push(ringStarts.at(-1)! + ring.length / 2);
  const ringPositions = Float32Array.from(perimeter.rings.flatMap(ring => [...ring]));
  const pointCount = burnedCells.length / 2;
  return runRasterLens({
    device,
    scenarioId: SCENARIO_ID,
    domain: 'gpu-raster',
    label: 'GPURasterJoin on the Dixie Fire window',
    grid,
    coordinateModel: perimeter.coordinateModel,
    unit: perimeter.unit,
    dataset: {id: T_DIXIE_PERIMETER.id, version: T_DATA_VERSION},
    formulas: 'tex' in entry.formula ? [entry.formula] : [],
    keySpaces: [
      {id: 'points', label: 'burned-cell points', size: pointCount, indexSpace: '1d'},
      {id: 'zones', label: 'perimeter zones', size: 1, indexSpace: '1d'}
    ],
    choices: [
      {stage: 'support', name: 'zoneCount', value: 1, sweepable: false},
      {stage: 'operator', name: 'spatialContext', value: 'planar', sweepable: false}
    ],
    build: async session => {
      // Per-point and per-zone results are registered first so the datum rung pins a point.
      const pointZones = session.output(
        'joinPointZones',
        'uint32',
        {
          role: 'intermediate',
          keySpace: 'points',
          meaning: 'zone of each point: 0 inside the perimeter, 4294967295 outside'
        },
        pointCount
      );
      const counts = session.output(
        'joinZoneCounts',
        'uint32',
        {role: 'intermediate', keySpace: 'zones', meaning: 'column points joined to each zone'},
        1
      );
      const outsideCount = session.output(
        'joinOutsideCount',
        'uint32',
        {role: 'intermediate', meaning: 'points outside the raster'},
        1
      );
      const pointValues = session.input('joinPoints', 'float32', burnedCells, {
        role: 'input',
        components: 2,
        geometry: {kind: 'points', coordinateModel: perimeter.coordinateModel, unit: 'm'},
        meaning: 'burned cell centres, dNBR above 0.27 (UTM 10N m)'
      });
      const burnedCellView = session.graph.createDataView(pointValues.buffer, {
        format: 'float32x2',
        length: pointCount
      });
      const zones = session.output('joinZones', 'uint32', {
        role: 'input',
        geometry: placement,
        keySpace: 'cells',
        meaning: 'zone raster from GPUPolygonRasterization: 0 inside the perimeter, else no zone'
      });
      const overflow = session.output('joinRasterizationOverflow', 'uint32', false, 1);
      // The perimeter polygons: one feature of two polygons (an island, and the fire with its hole).
      const ringOffsets = session.input('joinRingOffsets', 'uint32', ringStarts, false);
      const positionValues = session.input('joinRingPositions', 'float32', ringPositions, false);
      const positions = session.graph.createDataView(positionValues.buffer, {
        format: 'float32x2',
        length: ringPositions.length / 2
      });
      const featureOffsets = session.input('joinFeatureOffsets', 'uint32', [0, 2], false);
      const polygonOffsets = session.input('joinPolygonOffsets', 'uint32', [0, 1, 3], false);
      // The joined points drawn on top, coloured by the zone each one took.
      session.capture('joinedPoints', pointValues, {
        role: 'output',
        components: 2,
        geometry: {
          kind: 'points',
          coordinateModel: perimeter.coordinateModel,
          unit: 'm',
          valuesCapture: 'joinPointZones'
        },
        meaning: 'burned cell centres coloured by joined zone'
      });
      const {parameters} = await runExample(device, {
        graph: session.graph,
        grid,
        perimeter: {positions, featureOffsets, polygonOffsets, ringOffsets},
        burnedCells: burnedCellView,
        zones,
        overflow,
        counts,
        outsideCount,
        pointZones
      });
      session.track(parameters);
    }
  });
}

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-raster',
  element: 'GPURasterJoin',
  title: 'RasterJoin',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph', 'table'],
  defaultRung: 'result',
  dataset: T_DIXIE_PERIMETER,
  load: async () => ({run})
});
