// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationKeySpace} from '@luma.gl/gpgpu';
import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.spatial-scan-statistic',
  entry: 'gpu-spatial-analysis',
  element: 'GPUSpatialScanStatistic',
  title: 'Spatial scan of county traffic deaths, Midwest 2017-2023',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.counties,
  load: async () => {
    const [{runS2Lens, uploadS2CountyShapes}, {runExample}, counties, shapes, frameModule] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./spatial-scan-statistic.example'),
        import('../../data/s2-midwest-counties.json').then(module => module.default),
        import('../../data/s2-midwest-county-shapes.json').then(module => module.default),
        import('../../../analysis-core/contracts/local-frame')
      ]);
    return {
      run: device => {
        // Azimuthal equidistant about the middle of the five states: exact distances from the origin only.
        const frame = frameModule.createGPULocalFrame({origin: [-86.5, 42]});
        const countyCount = counties.deaths.length;
        const centroidPositions = frame.projectArray(counties.centroid.flat());
        const keySpaces: GPUComputationKeySpace[] = [
          {id: 'county', label: 'Midwest county (zone)', size: countyCount, indexSpace: '1d'},
          {id: 'cluster-index', label: 'cluster record word', size: 20, indexSpace: '1d'},
          {id: 'cluster-statistic', label: 'cluster record word', size: 40, indexSpace: '1d'},
          {id: 'replicate', label: 'replicate (0: observed)', size: 200, indexSpace: '1d'},
          {id: 'summary', label: 'scan summary word', size: 4, indexSpace: '1d'}
        ];
        return runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.spatial-scan-statistic',
          element: 'GPUSpatialScanStatistic',
          datasets: ['s2-midwest-counties', 's2-midwest-county-shapes'],
          retain: [
            'clusterIndices',
            'clusterStatistics',
            'statistics',
            'summary',
            'zoneStatistics'
          ],
          seed: 1,
          keySpaces,
          choices: {
            maximumPopulationFraction: 0.5,
            windowShape: 'circle',
            rateDirection: 'high',
            permutationCount: 199,
            seed: 1
          },
          async build(graph, uploads) {
            const centroids = uploads.view('centroids', centroidPositions, 'float32x2');
            const deaths = uploads.view('deaths', Uint32Array.from(counties.deaths), 'uint32');
            const personYears = uploads.view(
              'personYears',
              Float32Array.from(counties.personYears),
              'float32'
            );
            const outline = uploadS2CountyShapes(uploads, shapes.polygons);
            keySpaces.push(...outline.keySpaces);
            const result = await runExample(graph, {
              centroids,
              spatialContext: frame.spatialContext,
              deaths,
              personYears
            });
            uploads.own(result.scanParameters);
            return {
              ...outline.captures,
              centroids: {
                view: centroids,
                role: 'input',
                keySpace: 'county',
                meaning: 'County centroids in metres, azimuthal equidistant about (-86.5, 42)'
              },
              deaths: {
                view: deaths,
                role: 'input',
                keySpace: 'county',
                meaning: 'Traffic deaths 2017-2023 (FARS)'
              },
              personYears: {
                view: personYears,
                role: 'input',
                keySpace: 'county',
                meaning: 'Person-years at risk: seven times the 2020 census population'
              },
              zoneStatistics: {view: result.zoneStatistics, role: 'output', keySpace: 'county'},
              zoneMap: outline.paint('zoneStatistics'),
              clusterIndices: {
                view: result.clusterIndices,
                role: 'output',
                keySpace: 'cluster-index'
              },
              clusterStatistics: {
                view: result.clusterStatistics,
                role: 'output',
                keySpace: 'cluster-statistic'
              },
              statistics: {view: result.statistics, role: 'output', keySpace: 'replicate'},
              summary: {view: result.summary, role: 'output', keySpace: 'summary'}
            };
          }
        });
      }
    };
  }
});
