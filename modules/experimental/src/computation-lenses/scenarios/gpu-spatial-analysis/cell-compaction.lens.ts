// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {checkDrawnPrefix} from './cell-display-check';

const SCENARIO_ID = 'gpu-spatial-analysis.cell-compaction';
/** Resolution-9 descendants of one resolution-7 cell: 7 groups of 7 siblings. */
const CHILD_COUNT = 49;
const SIBLINGS = 7;
/** Rows masked out of the input, in two different sibling groups. */
const MASKED_ROWS = [0, 8];

registerLensScenario({
  id: SCENARIO_ID,
  entry: 'gpu-spatial-analysis',
  element: 'GPUCellCompaction',
  title: 'Compacting 47 H3 cells around the central Wicker Park place',
  audience: 'analyst',
  rungs: ['result', 'datum', 'kernel', 'graph'],
  defaultRung: 'result',
  dataset: {
    id: 's3-chicago-places',
    source: 'showcase public/data/chicago-places (Overture Maps release 2026-09-23.1)',
    licence: 'CDLA-Permissive-2.0 (most sources), Apache-2.0 (Foursquare), CC0 (AllThePlaces)'
  },
  load: async () => {
    const [
      {runS3Lens, geographicGeometry, geographicMetadata, createFixedOffsets},
      {runExample},
      {
        addCellBoundaries,
        addH3Cells,
        addH3Children,
        findCentralPlaceIndex,
        CELL_DISPLAY_H3_VERTEX_COUNT,
        CELL_DISPLAY_TABLE_HEADROOM
      },
      data
    ] = await Promise.all([
      import('./s3-lens-run'),
      import('./cell-compaction.example'),
      import('./cell-display'),
      import('./s3-lens-data').then(module => module.loadS3ChicagoPlaces())
    ]);
    // One resolution-7 cell around the place nearest the mean position; its 49 resolution-9
    // descendants come in ascending key order, so each run of 7 rows is one resolution-8 parent's
    // children. Masking rows 0 and 8 keeps two of the seven groups incomplete.
    const central = findCentralPlaceIndex(data.positionsLonLat);
    const positions = data.positionsLonLat.slice(2 * central, 2 * central + 2);
    const maskRows = new Uint32Array(CHILD_COUNT).fill(1);
    for (const row of MASKED_ROWS) maskRows[row] = 0;
    let completeGroups = 0;
    for (let group = 0; group < CHILD_COUNT / SIBLINGS; group++) {
      const members = maskRows.slice(group * SIBLINGS, (group + 1) * SIBLINGS);
      if (members.every(flag => flag === 1)) completeGroups++;
    }
    // Complete groups merge into one resolution-8 cell; the rest stay at resolution 9.
    const mergedCount =
      completeGroups + (CHILD_COUNT - MASKED_ROWS.length - completeGroups * SIBLINGS);
    const capacity = mergedCount + CELL_DISPLAY_TABLE_HEADROOM;
    return {
      run: device =>
        runS3Lens(device, {
          scenarioId: SCENARIO_ID,
          element: 'GPUCellCompaction',
          datasets: [data.id],
          retain: [
            'children',
            'compactedCells',
            'compactedCount',
            'droppedCount',
            'inputBoundaries',
            'inputBoundariesCenters',
            'boundaries',
            'boundariesCenters'
          ],
          keySpaces: [
            {id: 'cell', label: 'compacted cell', size: capacity, indexSpace: '1d'},
            {id: 'inputCell', label: 'input cell', size: CHILD_COUNT, indexSpace: '1d'},
            {
              id: 'boundaryVertex',
              label: 'compacted boundary vertex',
              size: capacity * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            },
            {
              id: 'inputBoundaryVertex',
              label: 'input boundary vertex',
              size: CHILD_COUNT * CELL_DISPLAY_H3_VERTEX_COUNT,
              indexSpace: '1d'
            }
          ],
          choices: {
            family: 'h3',
            'operation.type': 'compact',
            'operation.resolution': 9,
            'operation.minimumResolution': 7
          },
          metadata: snapshots => {
            // Draw exactly the rows the GPU wrote.
            checkDrawnPrefix(snapshots, [{capture: 'compactedCount', drawn: mergedCount}]);
            return geographicMetadata();
          },
          async build(graph, uploads) {
            const place = uploads.view('place', positions, 'float32x2');
            const parents = addH3Cells(graph, place, 7, 'parentCells');
            const children = addH3Children(graph, parents, 7, 9, 'children');
            const mask = uploads.view('mask', maskRows, 'uint32');
            const output = await runExample(graph, {cells: children, mask, capacity});
            // Boundaries from GPUCellGeometry for display only; the output draws its occupied prefix.
            const inputDisplay = addCellBoundaries(graph, children, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT,
              name: 'inputBoundaries'
            });
            const outputDisplay = addCellBoundaries(graph, output.compactedCells, {
              family: 'h3',
              vertexCount: CELL_DISPLAY_H3_VERTEX_COUNT
            });
            const inputOffsets = uploads.view(
              'inputBoundaryOffsets',
              createFixedOffsets(CHILD_COUNT, CELL_DISPLAY_H3_VERTEX_COUNT),
              'uint32'
            );
            const outputOffsets = uploads.view(
              'boundaryOffsets',
              createFixedOffsets(mergedCount, CELL_DISPLAY_H3_VERTEX_COUNT),
              'uint32'
            );
            return {
              inputBoundaries: {
                view: inputDisplay.boundaries,
                role: 'input',
                keySpace: 'inputBoundaryVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'inputBoundaryOffsets'}),
                meaning:
                  '49 resolution-9 cells, two of them masked out; boundaries from GPUCellGeometry for display'
              },
              inputBoundaryOffsets: {view: inputOffsets, role: 'intermediate'},
              mask: {
                view: mask,
                role: 'intermediate',
                keySpace: 'inputCell',
                meaning: '1 keeps the input cell, 0 drops it before merging'
              },
              compactedCells: {
                view: output.compactedCells,
                role: 'intermediate',
                keySpace: 'cell',
                meaning:
                  'merged H3 cell key (mixed resolutions 8 and 9) as (low, high) uint32 words'
              },
              compactedCount: {
                view: output.status.count,
                role: 'intermediate',
                meaning: 'cells in the compacted set'
              },
              droppedCount: {
                view: output.status.droppedCount,
                role: 'intermediate',
                meaning: 'input rows dropped by the mask or another resolution'
              },
              boundaryOffsets: {view: outputOffsets, role: 'intermediate'},
              boundaries: {
                view: outputDisplay.boundaries,
                role: 'output',
                keySpace: 'boundaryVertex',
                geometry: geographicGeometry('polygons', {offsetsCapture: 'boundaryOffsets'}),
                meaning:
                  'compacted cells (resolutions 8 and 9); boundaries from GPUCellGeometry for display'
              }
            };
          }
        })
    };
  }
});
