// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.local-moran',
  entry: 'gpu-spatial-analysis',
  element: 'GPULocalMoran',
  title: 'Local Moran: Chicago poverty percentages',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's1-chicago-tracts/1',
    source: 'Census cb_2022_17_tract_500k; CDC/ATSDR SVI 2022 (ACS 2018-2022)',
    licence:
      'US Census Bureau cartographic boundary file cb_2022_17_tract_500k, public domain; CDC/ATSDR SVI 2022, public domain'
  },
  async load() {
    const [{runExample}, {loadChicagoTracts}, utilities, spatial, {STATISTICS_OPERATORS}] =
      await Promise.all([
        import('./local-moran.example'),
        import('./spatial-lens-data'),
        import('./spatial-scenario-utils'),
        import('@luma.gl/experimental/gpu-spatial-analysis'),
        import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
      ]);
    const tracts = await loadChicagoTracts();
    return {
      async run(device) {
        const retained = [
          'localI',
          'adjustedPValues',
          'pValues',
          'quadrants',
          'spatialLag',
          'zScores',
          'neighborCounts',
          'globalStatistics',
          'weightsRowOffsets',
          'weightsNeighbors',
          'weightsValues',
          'weightsOverflow'
        ];
        const graph = utilities.createSpatialRecordingGraph(
          device,
          'chicago-local-moran',
          retained
        );
        const uploads: {destroy(): void}[] = [];
        const upload: typeof utilities.uploadSpatialView = (...arguments_) => {
          const uploaded = utilities.uploadSpatialView(...arguments_);
          uploads.push(uploaded);
          return uploaded;
        };
        try {
          // This extract has exactly one part and one ring per feature (checked by the node spec).
          const featureOffsets = Uint32Array.from({length: tracts.count + 1}, (_, index) => index);
          const vertices = upload(graph, 'vertices', tracts.vertices, 'float32x2').view;
          const ringOffsets = upload(graph, 'ringOffsets', tracts.ringOffsets, 'uint32').view;
          const polygons = {
            kind: 'polygons' as const,
            positions: vertices,
            ringOffsets,
            featureOffsets: upload(graph, 'featureOffsets', featureOffsets, 'uint32').view,
            polygonOffsets: upload(graph, 'polygonRingOffsets', tracts.polygonRingOffsets, 'uint32')
              .view
          };
          const values = upload(graph, 'values', tracts.columns['poverty150Pct'], 'float32').view;
          const parameters = upload(
            graph,
            'parameters',
            spatial.getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05}),
            'float32'
          ).view;
          const execution = await runExample(device, {graph, tracts: polygons, values, parameters});
          const geometry = {verticesCapture: 'vertices', offsetsCapture: 'ringOffsets'};
          return await utilities.runSpatialScenario({
            ...execution,
            id: 'gpu-spatial-analysis.local-moran',
            csrSlotCaptures: {weightsRowOffsets: ['weightsNeighbors', 'weightsValues']},
            views: {...execution.views, vertices, ringOffsets},
            entry: STATISTICS_OPERATORS['GPULocalMoran'],
            choices: {
              transform: 'row',
              islands: 'exclude',
              minimumNeighbors: 4,
              multipleTesting: 'benjamini-hochberg',
              significanceLevel: 0.05,
              quadrantGating: 'analytic'
            },
            metadata: {
              pCorrection: 'benjamini-hochberg',
              significanceAvailable: true,
              weightsStandardisation: 'row',
              coordinateModel: 'geographic',
              valueColumn: 'poverty150Pct',
              mappedVariableExtent: 'intensive'
            },
            datasetVersion: tracts.dataset.id,
            seed: 1,
            axes: {
              axes: [
                {
                  id: 'entity',
                  label: 'Chicago tract',
                  kind: 'data',
                  keySpace: 'tract',
                  size: tracts.count
                }
              ],
              keySpaces: [
                {id: 'tract', label: 'Chicago tract', size: tracts.count, indexSpace: 'csr'},
                {
                  id: 'weights',
                  label: 'CSR slot capacity',
                  size: execution.weights.neighbors.length,
                  indexSpace: '1d'
                },
                {
                  id: 'vertex',
                  label: 'Ring vertex',
                  size: tracts.vertices.length / 2,
                  indexSpace: '1d'
                }
              ],
              relations: [
                {
                  id: 'weights',
                  from: 'tract',
                  to: 'tract',
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
              values: {
                role: 'input',
                keySpace: 'tract',
                polygons: geometry,
                meaning: 'Population below 150% poverty line (%)'
              },
              localI: {
                role: 'output',
                keySpace: 'tract',
                polygons: geometry,
                meaning: 'Local Moran I (not significance)'
              },
              ...Object.fromEntries(
                [
                  'adjustedPValues',
                  'pValues',
                  'quadrants',
                  'spatialLag',
                  'zScores',
                  'neighborCounts'
                ].map(name => [name, {role: 'intermediate' as const, keySpace: 'tract'}])
              ),
              weightsValues: {role: 'intermediate', keySpace: 'weights'},
              weightsNeighbors: {role: 'intermediate', keySpace: 'weights'},
              weightsRowOffsets: {role: 'intermediate', keySpace: 'tract'},
              vertices: {role: 'intermediate', keySpace: 'vertex'}
            }
          });
        } finally {
          uploads.forEach(uploaded => uploaded.destroy());
        }
      }
    };
  }
});
