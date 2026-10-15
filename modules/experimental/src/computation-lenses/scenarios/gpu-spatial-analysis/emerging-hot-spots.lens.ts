// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.emerging-hot-spots',
  entry: 'gpu-spatial-analysis',
  element: 'GPUEmergingHotSpots',
  title: 'Emerging hot spots of monthly iNaturalist counts, Chicago North Side',
  audience: 'analyst',
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample, EMERGING_HOT_SPOTS_SLICE_COUNT}, support, spatial] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./emerging-hot-spots.example'),
        import('./s1-clustering-support'),
        import('@luma.gl/experimental/gpu-spatial-analysis')
      ]);
    const points = await support.loadNaturePoints();
    const lattice = support.getNatureLattice(points);
    const cellCount = lattice.width * lattice.height;
    const binCount = cellCount * EMERGING_HOT_SPOTS_SLICE_COUNT;
    // The cube is the input table: counts of the real observations, binned on the CPU.
    const cube = support.getNatureCube(points, lattice);
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.emerging-hot-spots',
          element: 'GPUEmergingHotSpots',
          datasets: ['s2-chicago-north-nature'],
          retain: [
            'giZScores',
            'adjustedPValues',
            'trendZ',
            'trendP',
            'trendS',
            'category',
            'hotSliceCount',
            'coldSliceCount'
          ],
          keySpaces: [
            {
              id: 'cell',
              label: 'lattice cell',
              size: cellCount,
              indexSpace: '2d',
              shape: [lattice.width, lattice.height]
            },
            {id: 'bin', label: 'cell and month', size: binCount, indexSpace: '1d'},
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'}
          ],
          choices: {
            radius: 1.5,
            temporalWindow: 1,
            confidenceLevel: 0.95,
            pCorrection: 'benjamini-hochberg',
            finalSlice: 'require'
          },
          async build(graph, uploads) {
            const cubeView = uploads.view('cube', cube, 'uint32');
            const parameters = uploads.view(
              'parameters',
              spatial.getGPUEmergingHotSpotsParameterValues({radius: 1.5, temporalWindow: 1}),
              'float32'
            );
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const result = await runExample(graph, {
              cube: cubeView,
              gridWidth: lattice.width,
              gridHeight: lattice.height,
              parameters
            });
            return {
              observations: {
                view: observations,
                role: 'context',
                keySpace: 'observation',
                meaning: 'Observations of 2023, local metres',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              cube: {
                view: cubeView,
                role: 'intermediate',
                keySpace: 'bin',
                meaning: 'Observation counts per 800 m cell and month, binned on the CPU'
              },
              giZScores: {
                view: result.giZScores,
                role: 'intermediate',
                keySpace: 'bin',
                meaning: 'Space-time Gi* z-score per cell and month'
              },
              adjustedPValues: {
                view: result.adjustedPValues,
                role: 'intermediate',
                keySpace: 'bin',
                meaning: 'Benjamini-Hochberg adjusted p-value per cell and month'
              },
              trendZ: {view: result.trendZ, role: 'intermediate', keySpace: 'cell'},
              trendP: {view: result.trendP, role: 'intermediate', keySpace: 'cell'},
              trendS: {view: result.trendS, role: 'intermediate', keySpace: 'cell'},
              hotSliceCount: {
                view: result.hotSliceCount,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'Months in which the cell is a significant hot spot'
              },
              coldSliceCount: {
                view: result.coldSliceCount,
                role: 'intermediate',
                keySpace: 'cell',
                meaning: 'Months in which the cell is a significant cold spot'
              },
              category: {
                view: result.category,
                role: 'output',
                keySpace: 'cell',
                meaning: 'Emerging hot spot category (nominal): 0 none, 6 sporadic hot',
                geometry: {
                  kind: 'raster',
                  shape: [lattice.width, lattice.height],
                  coordinateModel: 'projected',
                  cellSize: lattice.cellSize,
                  unit: 'm',
                  origin: [lattice.bounds[0], lattice.bounds[1]],
                  rowDirection: 'north'
                }
              }
            };
          }
        })
    };
  }
});
