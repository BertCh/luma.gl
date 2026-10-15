// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';

registerLensScenario({
  id: 'gpu-spatial-analysis.geographic-distribution',
  entry: 'gpu-spatial-analysis',
  element: 'GPUGeographicDistribution',
  title: 'Geographic distribution: where Chicago’s people are centred',
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
    const [
      {runExample, GEOGRAPHIC_DISTRIBUTION_RING_VERTICES},
      {loadChicagoTracts},
      utilities,
      runner,
      spatial,
      {STATISTICS_OPERATORS}
    ] = await Promise.all([
      import('./geographic-distribution.example'),
      import('./spatial-lens-data'),
      import('./spatial-scenario-utils'),
      import('./s2-lens-run'),
      import('@luma.gl/experimental/gpu-spatial-analysis'),
      import('../../../gpu-spatial-analysis/computation-pack/operators/statistics')
    ]);
    const tracts = await loadChicagoTracts();
    // The statistics are Euclidean: project to local metres, then take each tract's area centroid.
    const frame = runner.createS2ChicagoFrame();
    const projected = frame.projectArray(tracts.vertices);
    const centroids = runner.computeS2RingCentroids(projected, tracts.ringOffsets);
    return {
      run: async device => {
        const extra: {destroy(): void}[] = [];
        try {
          return await utilities.runTractStatisticsScenario({
            device,
            id: 'gpu-spatial-analysis.geographic-distribution',
            tracts,
            entry: STATISTICS_OPERATORS['GPUGeographicDistribution'],
            columns: {weights: 'population'},
            choices: {standardDeviations: 1, ellipseConvention: 'arcgis'},
            metadata: {
              // Centroids are metres in a local azimuthal-equidistant frame, not degrees.
              coordinateModel: 'projected',
              valueColumn: 'population'
            },
            async execute({graph, columns}) {
              const positions = utilities.uploadSpatialView(
                graph,
                'positions',
                centroids,
                'float32x2'
              );
              const parameters = utilities.uploadSpatialView(
                graph,
                'parameters',
                spatial.getGPUGeographicDistributionParameterValues({
                  standardDeviations: 1,
                  ellipseConvention: 'arcgis'
                }),
                'float32'
              );
              extra.push(positions, parameters);
              const execution = await runExample(device, {
                graph,
                positions: positions.view,
                weights: columns['weights'],
                parameters: parameters.view
              });
              return {
                ...execution,
                keySpaces: [
                  {id: 'group', label: 'Group (one)', size: 1, indexSpace: '1d' as const},
                  {
                    id: 'ellipse',
                    label: 'Ellipse angle, sigmaX, sigmaY',
                    size: 3,
                    indexSpace: '1d' as const
                  },
                  {
                    id: 'ellipseVertex',
                    label: 'Ellipse ring vertex',
                    size: GEOGRAPHIC_DISTRIBUTION_RING_VERTICES,
                    indexSpace: '1d' as const
                  }
                ],
                tags: {
                  positions: {
                    role: 'input',
                    meaning: 'Tract area centroids in local metres (azimuthal equidistant frame)'
                  },
                  weights: {
                    role: 'input',
                    polygons: true,
                    meaning: 'Population per tract; zero-population tracts are excluded'
                  },
                  counts: {role: 'intermediate', keySpace: 'group', meaning: 'Included tracts'},
                  weightSums: {
                    role: 'intermediate',
                    keySpace: 'group',
                    meaning: 'Total included population'
                  },
                  meanCenters: {
                    role: 'intermediate',
                    keySpace: 'group',
                    meaning: 'Population-weighted mean centre [x, y] (m)'
                  },
                  standardDistances: {
                    role: 'output',
                    keySpace: 'group',
                    meaning: 'Standard distance: radius of the one-sigma circle (m)'
                  },
                  ellipses: {
                    role: 'intermediate',
                    keySpace: 'ellipse',
                    meaning: 'Ellipse angle (rad), short and long semi-axes (m)'
                  },
                  ellipseVertices: {
                    role: 'intermediate',
                    keySpace: 'ellipseVertex',
                    meaning: 'Ellipse ring in local metres'
                  }
                }
              };
            }
          });
        } finally {
          extra.forEach(resource => resource.destroy());
        }
      }
    };
  }
});
