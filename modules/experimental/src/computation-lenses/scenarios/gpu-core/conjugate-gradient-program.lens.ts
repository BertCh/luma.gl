// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerLensScenario} from '../../registry';
import {CORE_DATASET} from './k-core-data';
import type {CoreLensRunProps, CoreObservation} from './core-lens-run';

const PROGRAM_ID = 'smooth';
const INTENT_GRAPH_ID = `${PROGRAM_ID}-intent`;
const SIDE = 16;
const UNKNOWN_INDEX = 8 * SIDE + 6;
const UNKNOWN_KEY_SPACE = 'selected-unknown';
const MAXIMUM_ITERATIONS = 12;
const TOLERANCE_SQUARED = 35000;
/** The value arena that holds the solver's GPU scalars; retained so its residual can be read. */
const VALUE_ARENA_BUFFER_ID = 'gpu-value-arena-buffer';

/** Snapshot frame index and value of `residualSquared` after the last node of an intent operation. */
function getResidualAfter(observation: CoreObservation, intentNodeId: string, iteration?: number) {
  const reference = `${INTENT_GRAPH_ID}:${intentNodeId}`;
  const nodeIds = new Set(
    observation.graph.nodes
      .filter(
        node =>
          // A repeated intent name may carry a #n suffix.
          node.derivedFrom?.some(link => link === reference || link.startsWith(`${reference}#`)) &&
          (iteration === undefined || node.loop?.iteration === iteration)
      )
      .map(node => node.id)
  );
  const frames = observation.snapshots.filter(
    snapshot => snapshot.name === 'residualSquared' && nodeIds.has(snapshot.afterNode ?? '')
  );
  const last = frames.at(-1);
  return last && {afterNode: last.afterNode!, residualSquared: Number(last.values[0])};
}

/**
 * Residual squared after the initial dot product (index 0) and after each executed iteration,
 * read from the solver's own GPU scalar. Iterations gated off by the GPU after convergence leave
 * the scalar unchanged and are not listed.
 */
function getResidualTrace(observation: CoreObservation) {
  const trace = [getResidualAfter(observation, `${PROGRAM_ID}-rr0`)];
  for (let iteration = 0; iteration < MAXIMUM_ITERATIONS && trace.at(-1); iteration++) {
    if (trace.at(-1)!.residualSquared <= TOLERANCE_SQUARED) break;
    trace.push(getResidualAfter(observation, `${PROGRAM_ID}-rr-copy`, iteration));
  }
  return trace.filter(entry => entry !== undefined);
}

registerLensScenario({
  id: 'gpu-core.conjugate-gradient-program',
  entry: 'gpu-core',
  element: 'createGPUConjugateGradientProgram',
  title: 'Diffusion-smoothing a 16 × 16 Alps elevation block',
  audience: 'engineer',
  defaultRung: 'result',
  rungs: ['result', 'kernel', 'graph', 'datum', 'iterations', 'formula', 'table'],
  dataset: CORE_DATASET,

  load: async () => {
    const [{runCoreLens, describeCoreLens}, {runExample, createScreenedPoissonSystem}, data] =
      await Promise.all([
        import('./core-lens-run'),
        import('./conjugate-gradient-program.example'),
        import('../../data/k-core.json').then(module => module.default)
      ]);
    // Tile rows and columns 24–39 of the 64 × 64 DEM, row-major from the north-west, in m.
    const block = Float32Array.from(
      {length: SIDE * SIDE},
      (_, index) =>
        data.dem.elevationM[(24 + Math.floor(index / SIDE)) * data.dem.width + 24 + (index % SIDE)]
    );
    const props: CoreLensRunProps = {
      scenarioId: 'gpu-core.conjugate-gradient-program',
      element: 'createGPUConjugateGradientProgram',
      label: 'Diffusion-smoothing a 16 × 16 Alps elevation block',
      keySpaces: [
        {
          id: UNKNOWN_KEY_SPACE,
          label: 'Selected unknown x[134] (row 8, column 6), receipt',
          size: 1,
          indexSpace: '1d'
        },
        {
          id: 'block-cells',
          label: 'Block cell',
          size: SIDE * SIDE,
          shape: [SIDE, SIDE],
          indexSpace: '2d'
        }
      ],
      at: 'step',
      retain: [VALUE_ARENA_BUFFER_ID, `${PROGRAM_ID}-r`, `${PROGRAM_ID}-p`],
      iterationCount: MAXIMUM_ITERATIONS + 1,

      async buildProgram(device, recording, uploads) {
        const system = createScreenedPoissonSystem(SIDE, 4);
        const {compilation, solver} = await runExample(device, {
          device,
          recording,
          rowOffsets: uploads.data('rowOffsets', system.rowOffsets, 'uint32'),
          columnIndices: uploads.data('columnIndices', system.columnIndices, 'uint32'),
          values: uploads.data('values', system.values, 'float32'),
          elevation: uploads.data('elevation', block, 'float32'),
          smoothed: uploads.data('smoothed', new Float32Array(SIDE * SIDE), 'float32')
        });
        const graph = compilation.graph;
        const selectUnknown = (name: string) => {
          const view = compilation.vectors.get(name)!.data[0];
          return graph.createDataView(view.buffer, {
            format: view.format,
            byteOffset: view.byteOffset + UNKNOWN_INDEX * view.byteStride,
            length: 1
          });
        };
        const rowStart = system.rowOffsets[UNKNOWN_INDEX];
        const rowLength = system.rowOffsets[UNKNOWN_INDEX + 1] - rowStart;
        const selectMatrixRow = (name: string) => {
          const view = compilation.vectors.get(name)!.data[0];
          return graph.createDataView(view.buffer, {
            format: view.format,
            byteOffset: view.byteOffset + rowStart * view.byteStride,
            length: rowLength
          });
        };
        return {
          compilation,
          formula: {id: 'gpu-core.conjugate-gradient', intentNodeId: `${PROGRAM_ID}-loop`},
          specs: {
            elevation: {
              view: compilation.vectors.get(solver.rhs.id)!.data[0],
              role: 'input',
              keySpace: 'block-cells'
            },
            smoothed: {
              view: compilation.vectors.get(solver.solution.id)!.data[0],
              role: 'output',
              keySpace: 'block-cells',
              meaning: 'operator:→',
              highlight: (values: ArrayLike<number>) => {
                const changes = Array.from(values, (value, index) => value - block[index]);
                const key = changes.reduce(
                  (best, change, index) =>
                    Math.abs(change) > Math.abs(changes[best]) ? index : best,
                  0
                );
                return {
                  key,
                  label: `largest change: row ${Math.floor(key / SIDE)}, column ${key % SIDE}, ${changes[key] > 0 ? '+' : ''}${changes[key].toFixed(1)} m`
                };
              }
            },
            matrixColumns: {
              view: selectMatrixRow(solver.columnIndices.id),
              keySpace: UNKNOWN_KEY_SPACE,
              meaning: 'Column indices j of the stored entries A[134,j], in CSR order'
            },
            matrixValues: {
              view: selectMatrixRow(solver.values.id),
              keySpace: UNKNOWN_KEY_SPACE,
              meaning: 'Stored coefficients A[134,j], paired with matrixColumns'
            },
            solution: {view: selectUnknown(solver.solution.id), keySpace: UNKNOWN_KEY_SPACE},
            residual: {view: selectUnknown(`${PROGRAM_ID}-r`), keySpace: UNKNOWN_KEY_SPACE},
            searchBeforeUpdate: {
              view: selectUnknown(`${PROGRAM_ID}-p`),
              keySpace: UNKNOWN_KEY_SPACE
            },
            alpha: {
              view: compilation.scalars.get(`${PROGRAM_ID}-alpha`)!.view,
              keySpace: UNKNOWN_KEY_SPACE
            },
            residualSquared: {
              view: compilation.scalars.get(solver.residualSquared.id)!.view,
              keySpace: UNKNOWN_KEY_SPACE
            }
          }
        };
      },

      series: (_compiled, observation) => [
        {
          name: 'residual',
          axis: 'iteration',
          unit: 'm',
          values: getResidualTrace(observation).map(entry => Math.sqrt(entry.residualSquared))
        }
      ],

      keyFrames(observation) {
        const trace = getResidualTrace(observation);
        const last = trace.length - 1;
        const frames = new Map<number, number>();
        const indexOf = (name: string, afterNode?: string) =>
          afterNode === undefined
            ? observation.snapshots.map(snapshot => snapshot.name).lastIndexOf(name)
            : observation.snapshots.findIndex(
                snapshot => snapshot.name === name && snapshot.afterNode === afterNode
              );
        // The receipt keeps each completed update, and the search direction before x changes.
        // Only snapshots are selected here: no solver values are reconstructed on the CPU.
        for (const [iteration, entry] of trace.entries()) {
          for (const name of ['solution', 'residual', 'residualSquared']) {
            frames.set(indexOf(name, entry.afterNode), iteration);
          }
          if (iteration > 0) {
            const beforeUpdate = getResidualAfter(
              observation,
              `${PROGRAM_ID}-alpha`,
              iteration - 1
            );
            if (beforeUpdate) {
              for (const name of ['searchBeforeUpdate', 'alpha']) {
                frames.set(indexOf(name, beforeUpdate.afterNode), iteration);
              }
            }
          }
        }
        for (const name of ['matrixColumns', 'matrixValues']) frames.set(indexOf(name), last);
        // Log-spaced key iterations; the final state is the last raw frame.
        for (let iteration = 0; iteration < last; iteration = iteration ? iteration * 2 : 1) {
          frames.set(indexOf('smoothed', trace[iteration].afterNode), iteration);
        }
        frames.set(indexOf('smoothed'), last);
        frames.set(indexOf('elevation'), last);
        return frames;
      }
    };
    return {
      async run(device) {
        const record = await runCoreLens(device, props);
        // Each CSR row is one tuple for the selected unknown, not five independent unknowns.
        return {
          ...record,
          captures: record.captures && {
            ...record.captures,
            snapshots: record.captures.snapshots.map(capture =>
              capture.name === 'matrixColumns' || capture.name === 'matrixValues'
                ? {...capture, components: capture.values.length}
                : capture
            )
          }
        };
      },
      describeStructure: device => describeCoreLens(device, props)
    };
  }
});
