// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';
import type {S2CaptureSpec} from './s2-lens-run';

/** Raster cell size in metres. */
const CELL_SIZE = 125;

registerLensScenario({
  id: 'gpu-spatial-analysis.change-of-support-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addChangeOfSupportRecipe',
  title: 'Tract counts moved onto community areas, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.tracts,
  load: async () => {
    const [
      {runS2Lens, createS2ChicagoFrame, projectS2Positions, snapS2Extent, uploadS2JoinPolygons},
      {runExample},
      tracts,
      areas
    ] = await Promise.all([
      import('./s2-lens-run'),
      import('./change-of-support-recipe.example'),
      import('../../data/s2-chicago-north-tracts.json').then(module => module.default),
      import('../../data/s2-chicago-north-areas.json').then(module => module.default)
    ]);
    return {
      run: device => {
        const frame = createS2ChicagoFrame('lambert-equal-area');
        const extent = snapS2Extent(
          projectS2Positions(frame, areas.polygons.vertices, CELL_SIZE).extent,
          CELL_SIZE
        );
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'status', label: 'run status', size: 1, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.change-of-support-recipe',
          element: 'addChangeOfSupportRecipe',
          datasets: ['s2-chicago-north-tracts', 's2-chicago-north-areas'],
          retain: [
            'areaCounts',
            'areaMeans',
            'sourceAreas',
            'sourceTransferredFractions',
            'overflow',
            'requiredCount',
            'invalidCount'
          ],
          keySpaces,
          choices: {cellSize: CELL_SIZE, denominator: 'zone'},
          async build(graph, uploads) {
            const tractGeometry = uploadS2JoinPolygons(uploads, frame, 'tracts', tracts.polygons);
            const areaGeometry = uploadS2JoinPolygons(uploads, frame, 'areas', areas.polygons);
            keySpaces.unshift(...tractGeometry.keySpaces, ...areaGeometry.keySpaces);
            const tractCounts = uploads.view(
              'tractCounts',
              Float32Array.from(tracts.natureObs2023),
              'float32'
            );
            const result = await runExample(graph, {
              tracts: tractGeometry.geometry,
              areas: areaGeometry.geometry,
              tractCounts,
              spatialContext: frame.spatialContext,
              extent,
              cellSize: CELL_SIZE
            });
            uploads.own(result.parameters);
            const captures: Record<string, S2CaptureSpec> = {
              ...tractGeometry.captures,
              ...areaGeometry.captures
            };
            captures['tractCounts'] = {view: tractCounts, role: 'input', keySpace: 'tracts'};
            const polygonGeometry = (valuesCapture: string, offsetsCapture: string) => ({
              kind: 'polygons' as const,
              offsetsCapture,
              valuesCapture,
              coordinateModel: 'projected' as const,
              unit: 'm'
            });
            captures['tracts'].geometry = polygonGeometry('tractCounts', 'tractsDisplayOffsets');
            captures['areas'].meaning = 'context';
            captures['resultAreas'] = {
              view: areaGeometry.display,
              role: 'output',
              keySpace: 'areas',
              geometry: polygonGeometry('areaCounts', 'areasDisplayOffsets')
            };
            captures['resultTracts'] = {
              view: tractGeometry.display,
              role: 'output',
              keySpace: 'tracts',
              geometry: polygonGeometry('sourceTransferredFractions', 'tractsDisplayOffsets')
            };
            captures['areaCounts'] = {view: result.areaCounts, role: 'output', keySpace: 'areas'};
            captures['areaMeans'] = {view: result.areaMeans, role: 'output', keySpace: 'areas'};
            captures['sourceAreas'] = {
              view: result.sourceAreas,
              role: 'output',
              keySpace: 'tracts'
            };
            captures['sourceTransferredFractions'] = {
              view: result.sourceTransferredFractions,
              role: 'output',
              keySpace: 'tracts'
            };
            for (const name of ['overflow', 'requiredCount', 'invalidCount'] as const) {
              captures[name] = {view: result[name], role: 'output', keySpace: 'status'};
            }
            return captures;
          }
        });
      }
    };
  }
});
