// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.pycnophylactic',
  entry: 'gpu-spatial-analysis',
  element: 'GPUPycnophylactic',
  title: 'Smooth population surface from Chicago North Side tracts',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [
      {
        runS2Lens,
        createS2ChicagoFrame,
        projectS2Positions,
        createS2RasterGrid,
        estimateS2RasterCrossings,
        uploadS2JoinPolygons
      },
      {runExample},
      tracts
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./pycnophylactic.example'),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const tractVertices = projectS2Positions(frame, tracts.polygons.vertices);
        const raster = createS2RasterGrid(tractVertices.extent);
        const crossingCapacity = estimateS2RasterCrossings(
          tractVertices.positions,
          tracts.polygons.ringOffsets,
          raster.cellSize
        );
        const keySpaces: GPUComputationKeySpace[] = [
          {
            id: 'cell',
            label: 'grid cell',
            size: raster.width * raster.height,
            indexSpace: '2d',
            shape: [raster.width, raster.height]
          }
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.pycnophylactic',
          element: 'GPUPycnophylactic',
          datasets: ['s2-chicago-north-tracts'],
          retain: ['mass', 'converged', 'iterationCount', 'invalidCount', 'rasterOverflow'],
          keySpaces,
          choices: {
            kernel: 'rook',
            maximumIterations: 30,
            tolerance: 0.001,
            cellSize: raster.cellSize
          },
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            keySpaces.push(...tractGeometry.keySpaces, {
              id: 'status',
              label: 'run status',
              size: 1,
              indexSpace: '1d'
            });
            const population = uploads.view(
              'population',
              Float32Array.from(tracts.population),
              'float32'
            );
            const result = await runExample(graph, {
              tracts: tractGeometry.geometry,
              population,
              spatialContext: frame.spatialContext,
              raster: {
                width: raster.width,
                height: raster.height,
                cellSize: raster.cellSize,
                origin: [raster.extent[0], raster.extent[3]]
              },
              crossingCapacity
            });
            uploads.own(result.placement);
            uploads.own(result.tolerance);
            const captures: Record<string, S2CaptureSpec> = {...tractGeometry.captures};
            captures['tracts'].meaning = 'context';
            captures['population'] = {view: population, role: 'input', keySpace: 'tracts'};
            captures['mass'] = {
              view: result.mass,
              role: 'output',
              keySpace: 'cell',
              geometry: {
                kind: 'raster',
                shape: [raster.width, raster.height],
                coordinateModel: 'projected',
                cellSize: raster.cellSize,
                unit: 'm',
                origin: [raster.extent[0], raster.extent[3]],
                rowDirection: 'south'
              }
            };
            for (const name of [
              'converged',
              'iterationCount',
              'invalidCount',
              'rasterOverflow'
            ] as const) {
              captures[name] = {view: result[name], role: 'output', keySpace: 'status'};
            }
            return captures;
          }
        });
      }
    };
  }
});
