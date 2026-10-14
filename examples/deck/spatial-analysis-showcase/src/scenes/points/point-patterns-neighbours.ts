// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUNeighborSearch,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  createTransientView,
  GPUReduction,
  GPURunLengthEncode,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';

/** The exact maximum radius count and its lowest input row. */
export type DensestPoint = {row: number; neighbours: number};

/**
 * GPU radius counts over precisely the statistics' included rows: a row is included when its
 * nearest-neighbour distance from `GPUPointPatternIndices` is finite, which reuses the exact
 * window predicate (including its boundary convention) without copying internal library WGSL.
 * With fewer than two included points there is no densest point and the row word stays
 * 0xffffffff. The CSR has just one slot: its overflow is intentional; neighborCounts is always
 * unclamped. Max-count then min-row reductions inspect every row, including zero-neighbor ties.
 */
export function addDensestPoint(
  graph: GPUCommandGraph<void>,
  props: {
    positions: GraphDataView<'float32x2'>;
    nearest: GraphDataView<'float32'>;
    spatialContext: GPUSpatialContext;
    parameters: GraphDataView<'float32'>;
    output: GraphDataView<'uint32'>;
  }
): void {
  const {positions, nearest, spatialContext, parameters, output} = props;
  const count = positions.length;
  const temporary = (name: string, length = count) =>
    createTransientView(graph, name, 'uint32', length);
  const included = temporary('radius-included');
  addKernelPass(graph, {
    id: 'radius-inclusion',
    invocationCount: count,
    bindings: [
      {name: 'nearest', view: nearest, type: 'f32', access: 'read'},
      {name: 'included', view: included, type: 'u32', access: 'read_write'}
    ],
    body: `let distanceBits = bitcast<u32>(nearest[nearestOffset + index]) & 0x7fffffffu;
      included[includedOffset + index] = select(0u, 1u, distanceBits < 0x7f800000u);`
  });
  const counts = temporary('radius-counts');
  graph.add(
    new GPUNeighborSearch({
      id: 'radius-neighbors',
      mode: 'radius',
      positions,
      mask: included,
      parameters,
      // The combined point-pattern statistics measure Euclidean distances in their Lambert
      // analysis plane. Request that same native-plane metric here; the distance-only operator
      // rejects a Lambert frame declaration because it cannot promise geodesic distances.
      // Retain units/datum, without claiming a different projection or reprojecting the points.
      spatialContext: {...spatialContext, frame: undefined},
      gridSize: [96, 96],
      output: {
        neighborCounts: counts,
        offsets: temporary('radius-offsets', count + 1),
        neighbors: temporary('radius-neighbors', 1),
        weights: createTransientView(graph, 'radius-weights', 'float32', 1),
        status: {overflow: temporary('radius-overflow', 1)}
      }
    })
  );
  const maximum = temporary('radius-maximum', 1);
  graph.add(
    new GPUReduction({
      id: 'maximum-radius-count',
      input: counts,
      mask: included,
      output: maximum,
      operation: 'max'
    })
  );
  const candidates = temporary('densest-candidates');
  addKernelPass(graph, {
    id: 'densest-ties',
    invocationCount: count,
    bindings: [
      {name: 'counts', view: counts, type: 'u32', access: 'read'},
      {name: 'included', view: included, type: 'u32', access: 'read'},
      {name: 'maximum', view: maximum, type: 'u32', access: 'read'},
      {name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}
    ],
    body: `candidates[candidatesOffset + index] = select(0xffffffffu, index,
      included[includedOffset + index] != 0u && counts[countsOffset + index] == maximum[maximumOffset]);`
  });
  const row = temporary('densest-row', 1);
  graph.add(
    new GPUReduction({id: 'lowest-densest-row', input: candidates, output: row, operation: 'min'})
  );
  addKernelPass(graph, {
    id: 'densest-summary',
    invocationCount: 1,
    bindings: [
      {name: 'row', view: row, type: 'u32', access: 'read'},
      {name: 'maximum', view: maximum, type: 'u32', access: 'read'},
      {name: 'result', view: output, type: 'u32', access: 'read_write'}
    ],
    body: `result[resultOffset] = row[rowOffset]; result[resultOffset + 1u] = maximum[maximumOffset];`
  });
}

/**
 * Exact duplicate share of selected, unjittered f32 coordinates. Stable y-bit then x-bit sorts
 * order the coordinate pairs lexicographically. Pair-change flags and a scan give unique run
 * IDs for public RLE; a reduction sums lengths > 1, excluding the masked sentinel run.
 * Signed zero is canonicalized so +0 and -0 describe the same coordinate.
 */
export function addSharedCoordinates(
  graph: GPUCommandGraph<void>,
  base: GraphDataView<'float32x2'>,
  mask: GraphDataView<'uint32'>,
  output: GraphDataView<'float32'>
): void {
  const count = base.length;
  const temporary = (name: string, length = count) =>
    createTransientView(graph, `duplicates-${name}`, 'uint32', length);
  const keys = temporary('keys');
  const rows = temporary('rows');
  addKernelPass(graph, {
    id: 'duplicate-y-keys',
    invocationCount: count,
    bindings: [
      {name: 'base', view: base, type: 'f32', access: 'read'},
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
      {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
    ],
    body: `let coordinate = base[baseOffset + 2u * index + 1u];
      keys[keysOffset + index] = select(0xffffffffu, select(bitcast<u32>(coordinate), 0u, coordinate == 0.0), mask[maskOffset + index] != 0u);
      rows[rowsOffset + index] = index;`
  });
  const sortedY = temporary('sorted-y');
  const rowsByY = temporary('rows-by-y');
  graph.add(
    new GPUSort({
      id: 'duplicate-sort-y',
      keys,
      values: rows,
      outputKeys: sortedY,
      outputValues: rowsByY
    })
  );
  const keysX = temporary('keys-x');
  addKernelPass(graph, {
    id: 'duplicate-x-keys',
    invocationCount: count,
    bindings: [
      {name: 'base', view: base, type: 'f32', access: 'read'},
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'rows', view: rowsByY, type: 'u32', access: 'read'},
      {name: 'keys', view: keysX, type: 'u32', access: 'read_write'}
    ],
    body: `let row = rows[rowsOffset + index]; let coordinate = base[baseOffset + 2u * row];
      keys[keysOffset + index] = select(0xffffffffu, select(bitcast<u32>(coordinate), 0u, coordinate == 0.0), mask[maskOffset + row] != 0u);`
  });
  const sortedX = temporary('sorted-x');
  const sortedRows = temporary('sorted-rows');
  graph.add(
    new GPUSort({
      id: 'duplicate-sort-x',
      keys: keysX,
      values: rowsByY,
      outputKeys: sortedX,
      outputValues: sortedRows
    })
  );
  const flags = temporary('heads');
  addKernelPass(graph, {
    id: 'duplicate-pair-heads',
    invocationCount: count,
    bindings: [
      {name: 'base', view: base, type: 'f32', access: 'read'},
      {name: 'mask', view: mask, type: 'u32', access: 'read'},
      {name: 'rows', view: sortedRows, type: 'u32', access: 'read'},
      {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
    ],
    body: `let row = rows[rowsOffset + index]; var head = index == 0u;
      if (index > 0u) {
        let previous = rows[rowsOffset + index - 1u];
        let selected = mask[maskOffset + row] != 0u;
        head = selected != (mask[maskOffset + previous] != 0u) || (selected &&
          (base[baseOffset + 2u * row] != base[baseOffset + 2u * previous] ||
           base[baseOffset + 2u * row + 1u] != base[baseOffset + 2u * previous + 1u]));
      }
      flags[flagsOffset + index] = select(0u, 1u, head);`
  });
  const groups = temporary('groups');
  graph.add(
    new GPUScan({id: 'duplicate-group-ids', input: flags, output: groups, mode: 'inclusive'})
  );
  const runCount = temporary('run-count', 1);
  const lengths = temporary('lengths');
  graph.add(
    new GPURunLengthEncode({
      id: 'duplicate-runs',
      input: groups,
      values: temporary('values'),
      lengths,
      count: runCount
    })
  );
  const selectedCount = temporary('selected', 1);
  graph.add(
    new GPUReduction({
      id: 'duplicate-selected-count',
      input: mask,
      output: selectedCount,
      operation: 'sum'
    })
  );
  const terms = temporary('terms');
  addKernelPass(graph, {
    id: 'duplicate-shared-terms',
    invocationCount: count,
    bindings: [
      {name: 'lengths', view: lengths, type: 'u32', access: 'read'},
      {name: 'runCount', view: runCount, type: 'u32', access: 'read'},
      {name: 'selectedCount', view: selectedCount, type: 'u32', access: 'read'},
      {name: 'terms', view: terms, type: 'u32', access: 'read_write'}
    ],
    body: `var amount = 0u;
      let validRuns = runCount[runCountOffset] - select(0u, 1u, selectedCount[selectedCountOffset] < ${count}u);
      if (index < validRuns) { let runLength = lengths[lengthsOffset + index]; amount = select(0u, runLength, runLength > 1u); }
      terms[termsOffset + index] = amount;`
  });
  const sharedCount = temporary('shared', 1);
  graph.add(
    new GPUReduction({
      id: 'duplicate-shared-count',
      input: terms,
      output: sharedCount,
      operation: 'sum'
    })
  );
  addKernelPass(graph, {
    id: 'duplicate-share',
    invocationCount: 1,
    bindings: [
      {name: 'sharedCount', view: sharedCount, type: 'u32', access: 'read'},
      {name: 'selectedCount', view: selectedCount, type: 'u32', access: 'read'},
      {name: 'result', view: output, type: 'f32', access: 'read_write'}
    ],
    body: `var nanBits = 0x7fc00000u;
      let total = selectedCount[selectedCountOffset];
      result[resultOffset] = select(bitcast<f32>(nanBits), f32(sharedCount[sharedCountOffset]) / f32(total), total > 0u);`
  });
}
