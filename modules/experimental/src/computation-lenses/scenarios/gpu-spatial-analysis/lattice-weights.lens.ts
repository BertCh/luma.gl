// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

const SCENARIO_ID = 'gpu-spatial-analysis.lattice-weights';

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPULatticeWeights',
  title: 'Queen neighbours on a 32 by 32 terrain lattice',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's1-dixie-lattice/1',
    source:
      'AWS Terrain Tiles (Terrarium, USGS 3DEP ~10 m), central block of the Dixie DEM at 120 m',
    licence: 'Public domain (USGS 3DEP); terrain tiles by Mapzen/Tilezen on AWS Open Data'
  },
  async load() {
    const [{runExample}, utilities, {WEIGHTS_OPERATORS}, data] = await Promise.all([
      import('./lattice-weights.example'),
      import('./spatial-scenario-utils'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/weights'),
      import('../../data/s1-dixie-lattice.json').then(module => module.default)
    ]);
    const {width, height} = data;
    return {
      async run(device) {
        const retained: string[] = [];
        const graph = utilities.createSpatialRecordingGraph(device, SCENARIO_ID, retained);
        const elevation = utilities.uploadSpatialView(
          graph,
          'elevation',
          Float32Array.from(data.values),
          'float32'
        );
        try {
          const execution = await runExample(device, {graph, width, height});
          const views = {elevation: elevation.view, ...execution.views};
          // Retain the example's transients by buffer id (capture names are not resource ids;
          // the uploaded elevation is imported, not transient).
          retained.push(...utilities.getRetainedViewIds(execution.views));
          const raster = {
            kind: 'raster' as const,
            shape: [width, height] as const,
            coordinateModel: 'projected' as const,
            cellSize: data.cellSize,
            unit: 'm'
          };
          const record = await utilities.runSpatialScenario({
            id: SCENARIO_ID,
            graph,
            encodeOptions: {parameters: undefined},
            csrSlotCaptures: {weightsRowOffsets: ['weightsNeighbors', 'weightsValues']},
            views,
            entry: WEIGHTS_OPERATORS['GPULatticeWeights'],
            choices: {criterion: 'queen', radius: 1},
            metadata: {coordinateModel: 'projected', weightsStandardisation: 'binary'},
            datasetVersion: data.id,
            seed: 1,
            axes: {
              axes: [
                {
                  id: 'entity',
                  label: 'Lattice cell',
                  kind: 'data',
                  keySpace: 'cell',
                  size: width * height
                }
              ],
              keySpaces: [
                {
                  id: 'cell',
                  label: '120 m cell, rows from north',
                  size: width * height,
                  indexSpace: '2d',
                  shape: [width, height]
                },
                {
                  id: 'weights',
                  label: 'CSR slot capacity',
                  size: execution.weights.neighbors.length,
                  indexSpace: '1d'
                },
                {id: 'summary', label: 'Weights moments S0, S1, S2', size: 3, indexSpace: '1d'},
                {
                  id: 'counts',
                  label: 'Slots, asymmetry, isolates, minimum, maximum, empty',
                  size: 6,
                  indexSpace: '1d'
                }
              ],
              relations: [
                {
                  id: 'weights',
                  from: 'cell',
                  to: 'cell',
                  cardinality: 'N:M',
                  kind: 'buffer',
                  timeDependent: false,
                  ref: {
                    layout: 'csr',
                    resource: execution.weights.neighbors.buffer.id,
                    offsetsResource: execution.weights.offsets.buffer.id
                  }
                }
              ]
            },
            tags: {
              elevation: {
                role: 'input',
                keySpace: 'cell',
                meaning: 'Elevation of the 120 m cell (m)'
              },
              neighborCounts: {
                role: 'output',
                keySpace: 'cell',
                meaning: 'Queen neighbours of each cell'
              },
              weightsValues: {role: 'intermediate', keySpace: 'weights'},
              weightsNeighbors: {role: 'intermediate', keySpace: 'weights'},
              weightsRowOffsets: {role: 'intermediate', keySpace: 'cell'},
              statistics: {role: 'intermediate', keySpace: 'summary'},
              counts: {role: 'intermediate', keySpace: 'counts'}
            }
          });
          return utilities.finishSpatialWeightsRecord(record, {
            elevation: {role: 'input', geometry: raster},
            neighborCounts: {role: 'output', geometry: raster}
          });
        } finally {
          elevation.destroy();
        }
      }
    };
  }
});
