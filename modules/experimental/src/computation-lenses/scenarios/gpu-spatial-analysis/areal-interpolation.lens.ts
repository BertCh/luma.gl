// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';
import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';

registerLensScenario({
  id: 'gpu-spatial-analysis.areal-interpolation',
  entry: 'gpu-spatial-analysis',
  element: 'GPUArealInterpolation',
  title: 'Tract population shared out to Chicago North Side community areas',
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
      tracts,
      areas
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./areal-interpolation.example'),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default),
      import('../../data/s2-chicago-north-areas.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const tractVertices = projectS2Positions(frame, tracts.polygons.vertices);
        const areaVertices = projectS2Positions(frame, areas.polygons.vertices);
        const raster = createS2RasterGrid([
          Math.min(tractVertices.extent[0], areaVertices.extent[0]),
          Math.min(tractVertices.extent[1], areaVertices.extent[1]),
          Math.max(tractVertices.extent[2], areaVertices.extent[2]),
          Math.max(tractVertices.extent[3], areaVertices.extent[3])
        ]);
        const crossingCapacity = Math.max(
          estimateS2RasterCrossings(
            tractVertices.positions,
            tracts.polygons.ringOffsets,
            raster.cellSize
          ),
          estimateS2RasterCrossings(
            areaVertices.positions,
            areas.polygons.ringOffsets,
            raster.cellSize
          )
        );
        const keySpaces: GPUComputationKeySpace[] = [];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.areal-interpolation',
          element: 'GPUArealInterpolation',
          datasets: ['s2-chicago-north-tracts', 's2-chicago-north-areas'],
          retain: [
            'areaPopulation',
            'transferredFractions',
            'weightOffsets',
            'weightNeighbors',
            'weightValues',
            'overlapAreas',
            'pairOverflow',
            'tractRasterOverflow',
            'areaRasterOverflow'
          ],
          keySpaces,
          choices: {
            mode: 'extensive',
            denominator: 'zone',
            cellSize: raster.cellSize
          },
          async build(graph, uploads) {
            const areaGeometry = uploadS2JoinPolygons(uploads, frame, 'areas', areas.polygons);
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            const population = uploads.view(
              'population',
              Float32Array.from(tracts.population),
              'float32'
            );
            const result = await runExample(graph, {
              tracts: tractGeometry.geometry,
              communityAreas: areaGeometry.geometry,
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
            keySpaces.push(
              ...areaGeometry.keySpaces,
              ...tractGeometry.keySpaces,
              {
                id: 'area-row-offset',
                label: 'area row offset',
                size: areas.areaId.length + 1,
                indexSpace: '1d'
              },
              {
                id: 'pair',
                label: 'tract-area pair',
                size: result.weights.neighbors.length,
                indexSpace: '1d'
              },
              {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
            );
            const captures: Record<string, S2CaptureSpec> = {
              ...tractGeometry.captures,
              ...areaGeometry.captures
            };
            captures['areas'].meaning = 'context';
            captures['population'] = {view: population, role: 'input', keySpace: 'tracts'};
            captures['resultAreas'] = {
              view: areaGeometry.display,
              role: 'output',
              keySpace: 'areas',
              geometry: {
                kind: 'polygons',
                offsetsCapture: 'areasDisplayOffsets',
                valuesCapture: 'areaPopulation',
                coordinateModel: 'projected',
                unit: 'm'
              }
            };
            captures['areaPopulation'] = {
              view: result.areaPopulation,
              role: 'output',
              keySpace: 'areas'
            };
            captures['transferredFractions'] = {
              view: result.transferredFractions,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['weightOffsets'] = {
              view: result.weights.offsets,
              role: 'intermediate',
              keySpace: 'area-row-offset'
            };
            captures['weightNeighbors'] = {
              view: result.weights.neighbors,
              role: 'intermediate',
              keySpace: 'pair'
            };
            captures['weightValues'] = {
              view: result.weights.weights,
              role: 'intermediate',
              keySpace: 'pair'
            };
            captures['overlapAreas'] = {
              view: result.overlapAreas,
              role: 'intermediate',
              keySpace: 'pair'
            };
            for (const name of [
              'pairOverflow',
              'tractRasterOverflow',
              'areaRasterOverflow'
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
