// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {S2_DATASETS} from './s2-datasets';

registerLensScenario({
  id: 'gpu-spatial-analysis.space-time-hot-spots-recipe',
  entry: 'gpu-spatial-analysis',
  element: 'addSpaceTimeHotSpotsRecipe',
  title: 'Emerging hot spots from timestamped iNaturalist observations',
  audience: 'analyst',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: S2_DATASETS.nature,
  load: async () => {
    const [{runS2Lens}, {runExample, SPACE_TIME_SLICE_COUNT}, support, dataframe] =
      await Promise.all([
        import('./s2-lens-run'),
        import('./space-time-hot-spots-recipe.example'),
        import('./s1-clustering-support'),
        import('@luma.gl/experimental/gpu-dataframe')
      ]);
    const points = await support.loadNaturePoints();
    const lattice = support.getNatureLattice(points);
    const cellCount = lattice.width * lattice.height;
    return {
      run: device =>
        runS2Lens(device, {
          scenarioId: 'gpu-spatial-analysis.space-time-hot-spots-recipe',
          element: 'addSpaceTimeHotSpotsRecipe',
          datasets: ['s2-chicago-north-nature'],
          retain: ['cube', 'giZScores', 'trendZ', 'category', 'hotSliceCount', 'coldSliceCount'],
          keySpaces: [
            {
              id: 'cell',
              label: 'lattice cell',
              size: cellCount,
              indexSpace: '2d',
              shape: [lattice.width, lattice.height]
            },
            {
              id: 'bin',
              label: 'cell and month',
              size: cellCount * SPACE_TIME_SLICE_COUNT,
              indexSpace: '1d'
            },
            {id: 'observation', label: 'observation', size: points.count, indexSpace: '1d'}
          ],
          choices: {
            step: 'month',
            cellSize: lattice.cellSize,
            radius: 1.5,
            temporalWindow: 1,
            confidenceLevel: 0.95,
            pCorrection: 'benjamini-hochberg',
            finalSlice: 'require'
          },
          async build(graph, uploads) {
            const observations = uploads.view('observations', points.positions, 'float32x2');
            const timestamps = uploads.view(
              'timestamps',
              support.getNatureEventTimes(points),
              'uint32x2'
            );
            const calendarParameters = uploads.view(
              'calendarParameters',
              dataframe.getGPUCalendarBucketsParameterValues(0),
              'sint32'
            );
            const result = await runExample(graph, {
              observations,
              timestamps,
              calendarParameters,
              spatialContext: points.frame.spatialContext,
              width: lattice.width,
              height: lattice.height,
              bounds: lattice.bounds
            });
            uploads.own(result.parameters);
            return {
              observations: {
                view: observations,
                role: 'context',
                keySpace: 'observation',
                meaning: 'Observations of 2023, local metres',
                geometry: {kind: 'points', coordinateModel: 'projected', unit: 'm'}
              },
              cube: {
                view: result.cube,
                role: 'intermediate',
                keySpace: 'bin',
                meaning: 'Observation counts per 800 m cell and month, binned on the GPU'
              },
              giZScores: {
                view: result.giZScores,
                role: 'intermediate',
                keySpace: 'bin',
                meaning: 'Space-time Gi* z-score per cell and month'
              },
              trendZ: {view: result.trendZ, role: 'intermediate', keySpace: 'cell'},
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
