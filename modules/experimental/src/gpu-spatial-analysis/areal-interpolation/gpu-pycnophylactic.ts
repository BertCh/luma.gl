// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {createSegmentSumNode, getSortKeyBits} from '../../utils/sorted-segment-sums';

const OPERATION = 'GPUPycnophylactic';

/** Maximum number of smoothing iterations of {@link GPUPycnophylactic}. */
export const GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS = 512;

/** Smoothing neighborhood of {@link GPUPycnophylactic}. */
export type GPUPycnophylacticKernel = 'rook' | 'box';

/**
 * Properties for {@link GPUPycnophylactic}.
 *
 * Compile-time: `width`, `height`, `zoneCount`, `iterations`, `kernel`. Per-frame: the contents of
 * `zones` and `totals`.
 */
export type GPUPycnophylacticProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'pycnophylactic'`. */
  id?: string;
  /** Raster width in cells. */
  width: number;
  /** Raster height in cells. */
  height: number;
  /**
   * Zone raster (`width * height` rows, row-major): the zone of each cell, or any value
   * `>= zoneCount` (such as `GPU_POLYGON_RASTERIZATION_NO_ZONE`) for a cell outside the study area.
   * Typically the `zones` output of a `GPUPolygonRasterization`.
   */
  zones: GraphDataView<'uint32'>;
  /** Number of zones `Z`. */
  zoneCount: number;
  /** Per-zone totals to preserve (`zoneCount` rows). Negative or non-finite totals count as 0. */
  totals: GraphDataView<'float32'>;
  /** Number of smoothing iterations, in `[0, GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS]`. */
  iterations: number;
  /**
   * Focal mean: `'rook'` (default, Tobler 1979) averages the 4 edge neighbors inside the study
   * area; `'box'` averages the 3x3 block including the cell itself.
   */
  kernel?: GPUPycnophylacticKernel;
  /**
   * Caller-owned `width * height` result: the mass per cell. The cells of zone `z` sum to
   * `totals[z]` (to float32 accuracy) and are non-negative. Cells outside the study area and cells
   * of zones without cells are 0.
   */
  output: GraphDataView<'float32'>;
};

/**
 * Tobler's pycnophylactic (volume-preserving) interpolation on a raster: a smooth density surface
 * whose mass in every zone equals the zone total.
 *
 * Algorithm (fixed iteration count, deterministic):
 * 1. Initialize every cell of zone `z` to `totals[z] / cells(z)`.
 * 2. Repeat `iterations` times: replace every cell by the focal mean of its in-area neighbors
 *    (`kernel`); add `(totals[z] - sum_z) / cells(z)` to every cell of each zone so the total is
 *    restored; clamp negative values to 0; multiply the cells of each zone by
 *    `totals[z] / sum_z` (a zone whose sum is 0 is filled uniformly). The last step guarantees the
 *    totals and non-negativity exactly after every iteration.
 *
 * Zone sums use a fixed-order segmented reduction over cells sorted once by zone, so results are
 * bitwise reproducible. There is no convergence test: iterate a fixed number of times, as the
 * smoothing converges geometrically with the zone size. Cost is `iterations` times a few passes
 * over the cells plus one radix sort of the cells.
 *
 * Together with `GPUArealInterpolation` this covers the tobler pycnophylactic and area
 * interpolation workloads. Unlike areal weighting, the surface is smooth across zone borders.
 * Limitation: no ancillary (dasymetric) initial surface and no barriers; the study area is the
 * union of the zone cells.
 */
export class GPUPycnophylactic implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUPycnophylacticProps;

  constructor(props: GPUPycnophylacticProps) {
    const id = props.id ?? 'pycnophylactic';
    this.id = id;
    this.props = props;
    for (const [name, value] of [
      ['width', props.width],
      ['height', props.height],
      ['zoneCount', props.zoneCount]
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 1) {
        throw new Error(`${id} ${name} must be a positive integer`);
      }
    }
    if (
      !Number.isSafeInteger(props.iterations) ||
      props.iterations < 0 ||
      props.iterations > GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} iterations must be an integer in [0, ${GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS}]`
      );
    }
    if (props.kernel !== undefined && props.kernel !== 'rook' && props.kernel !== 'box') {
      throw new Error(`${id} kernel must be 'rook' or 'box'`);
    }
    const cellCount = props.width * props.height;
    validatePackedUint32View(props.zones, `${id} zones`);
    validatePackedView(props.totals, ['float32'], `${id} totals`);
    validatePackedView(props.output, ['float32'], `${id} output`);
    if (props.zones.length !== cellCount) {
      throw new Error(`${id} zones must contain width * height rows`);
    }
    if (props.output.length !== cellCount) {
      throw new Error(`${id} output must contain width * height rows`);
    }
    if (props.totals.length !== props.zoneCount) {
      throw new Error(`${id} totals must contain zoneCount rows`);
    }
    validateGraphOutputsDisjointFromInputs(id, [props.output], [props.zones, props.totals]);
  }

  /** Returns the sort, initialization and per-iteration smoothing and rescaling nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {width, height, zones, totals, zoneCount, iterations, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [zones, totals, output]);
    const cellCount = width * height;
    const kernel = props.kernel ?? 'rook';
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Cells sorted once by zone; zone sums are then fixed-order segmented sums.
    const counts = createTransientView(graph, `${id}-zone-counts`, 'uint32', zoneCount);
    const sortKeys = createTransientView(graph, `${id}-sort-keys`, 'uint32', cellCount);
    const sortIndices = createTransientView(graph, `${id}-sort-indices`, 'uint32', cellCount);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', cellCount);
    const sortedIndices = createTransientView(graph, `${id}-sorted-indices`, 'uint32', cellCount);
    const segmentOffsets = createTransientView(
      graph,
      `${id}-segment-offsets`,
      'uint32',
      zoneCount + 1
    );
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-counts-clear`,
        operation: OPERATION,
        view: counts,
        type: 'u32',
        value: '0u'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'zones', view: zones, type: 'u32', access: 'read'},
          {name: 'counts', view: counts, type: 'atomic<u32>', access: 'read_write'},
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: `const ZONE_COUNT: u32 = ${zoneCount}u;`,
        body: `let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    atomicAdd(&counts[countsOffset + zone], 1u);
  }
  sortKeys[sortKeysOffset + index] = min(zone, ZONE_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`
      }),
      ...new GPUSort({
        id: `${id}-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(zoneCount)
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-segment-scan`,
        input: counts,
        output: segmentOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-segment-total`,
        operation: OPERATION,
        variant: 'segment-total',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'segmentOffsets', view: segmentOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LAST_ZONE: u32 = ${zoneCount - 1}u;`,
        body: `segmentOffsets[segmentOffsetsOffset + LAST_ZONE + 1u] =
    segmentOffsets[segmentOffsetsOffset + LAST_ZONE] + counts[countsOffset + LAST_ZONE];`
      })
    );

    const scratch = createTransientView(graph, `${id}-scratch`, 'float32', cellCount);
    const sortedValues = createTransientView(graph, `${id}-sorted-values`, 'float32', cellCount);
    const zoneSums = createTransientView(graph, `${id}-zone-sums`, 'float32', zoneCount);

    const sharedDeclarations = `const ZONE_COUNT: u32 = ${zoneCount}u;
const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
fn getTotal(zone: u32) -> f32 {
  let total = totals[totalsOffset + zone];
  // Negative, NaN and infinite totals count as 0.
  let isFinite = (bitcast<u32>(total) & 0x7f800000u) != 0x7f800000u;
  return select(0.0, total, isFinite && total > 0.0);
}`;
    const getSumNodes = (label: string, values: GraphDataView<'float32'>) => [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${label}-gather`,
        operation: OPERATION,
        variant: 'gather',
        bindings: [
          {name: 'sortedIndices', view: sortedIndices, type: 'u32', access: 'read'},
          {name: 'values', view: values, type: 'f32', access: 'read'},
          {name: 'sortedValues', view: sortedValues, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        body: 'sortedValues[sortedValuesOffset + index] = values[valuesOffset + sortedIndices[sortedIndicesOffset + index]];'
      }),
      createSegmentSumNode<Parameters>(graph, {
        id: `${id}-${label}-sum`,
        operation: OPERATION,
        segmentCount: zoneCount,
        input: sortedValues,
        segmentOffsets,
        output: zoneSums
      })
    ];

    // The last iteration writes `output`; earlier ones alternate with the scratch raster.
    const bufferFor = (iteration: number) =>
      (iterations - 1 - iteration) % 2 === 0 ? output : scratch;
    const initial = iterations === 0 ? output : bufferFor(0) === output ? scratch : output;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        bindings: [
          {name: 'zones', view: zones, type: 'u32', access: 'read'},
          {name: 'totals', view: totals, type: 'f32', access: 'read'},
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'destination', view: initial, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        declarations: sharedDeclarations,
        body: `let zone = zones[zonesOffset + index];
  var value = 0.0;
  if (zone < ZONE_COUNT) {
    value = getTotal(zone) / f32(counts[countsOffset + zone]);
  }
  destination[destinationOffset + index] = value;`
      })
    );

    let source = initial;
    for (let iteration = 0; iteration < iterations; iteration++) {
      const destination = bufferFor(iteration);
      const label = `iteration-${iteration}`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-smooth`,
          operation: OPERATION,
          variant: `smooth-${kernel}`,
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'source', view: source, type: 'f32', access: 'read'},
            {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: sharedDeclarations,
          body: `let zone = zones[zonesOffset + index];
  var result = 0.0;
  if (zone < ZONE_COUNT) {
    let column = i32(index % WIDTH);
    let row = i32(index / WIDTH);
    var sum = 0.0;
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let isCenter = dx == 0 && dy == 0;
        let isDiagonal = dx != 0 && dy != 0;
        if ((isCenter && ${kernel === 'rook' ? 'true' : 'false'}) || (isDiagonal && ${kernel === 'rook' ? 'true' : 'false'})) {
          continue;
        }
        let x = column + dx;
        let y = row + dy;
        if (x < 0 || y < 0 || x >= i32(WIDTH) || y >= i32(HEIGHT)) {
          continue;
        }
        let neighbor = u32(y) * WIDTH + u32(x);
        if (zones[zonesOffset + neighbor] < ZONE_COUNT) {
          sum += source[sourceOffset + neighbor];
          count += 1.0;
        }
      }
    }
    result = select(source[sourceOffset + index], sum / count, count > 0.0);
  }
  destination[destinationOffset + index] = result;`
        }),
        ...getSumNodes(`${label}-restore`, destination),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-restore-apply`,
          operation: OPERATION,
          variant: 'restore-and-clamp',
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'zoneSums', view: zoneSums, type: 'f32', access: 'read'},
            {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: sharedDeclarations,
          body: `let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    let shift = (getTotal(zone) - zoneSums[zoneSumsOffset + zone]) / f32(counts[countsOffset + zone]);
    destination[destinationOffset + index] = max(destination[destinationOffset + index] + shift, 0.0);
  }`
        }),
        ...getSumNodes(`${label}-scale`, destination),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${label}-scale-apply`,
          operation: OPERATION,
          variant: 'rescale',
          bindings: [
            {name: 'zones', view: zones, type: 'u32', access: 'read'},
            {name: 'totals', view: totals, type: 'f32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            {name: 'zoneSums', view: zoneSums, type: 'f32', access: 'read'},
            {name: 'destination', view: destination, type: 'f32', access: 'read_write'}
          ],
          invocationCount: cellCount,
          declarations: sharedDeclarations,
          body: `let zone = zones[zonesOffset + index];
  if (zone < ZONE_COUNT) {
    let total = getTotal(zone);
    let sum = zoneSums[zoneSumsOffset + zone];
    // A zone whose clamped sum vanished is filled uniformly.
    destination[destinationOffset + index] = select(
      total / f32(counts[countsOffset + zone]),
      destination[destinationOffset + index] * (total / sum),
      sum > 0.0
    );
  }`
        })
      );
      source = destination;
    }
    return nodes;
  }
}
