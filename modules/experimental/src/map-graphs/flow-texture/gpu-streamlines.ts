// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createMapGraphFillNode, createMapGraphKernelNode} from '../map-graph-kernels';
import type {GPUMapGraphCompactOutput, GPUMapGraphRecipe} from '../map-graph-types';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMapGraphCompactOutput
} from '../map-graph-utils';
import {getFieldSamplingWGSL} from './flow-texture-field';
import {PHILOX_WGSL} from './flow-texture-random';
import {STREAMLINES_SEED_PURPOSE} from './streamlines-cpu';
import {
  GPU_STREAMLINES_PARAMETER_LENGTH,
  GPU_STREAMLINES_WORD_PARAMETER_LENGTH
} from './streamlines-parameters';

const OPERATION = 'GPUStreamlines';
const MAXIMUM_SEED_COUNT = 65535;
const UNDECIDED = 0;
const ACCEPTED = 1;
const REJECTED = 2;

/** Caller-owned outputs of {@link GPUStreamlines}. */
export type GPUStreamlinesOutput = {
  /**
   * Published lines in ascending seed index: `ids` holds the seed index of each line, `count` the
   * number of published lines, `overflow` is 1 when accepted lines were dropped because the line or
   * point capacity was full, and `totalCount` the number of accepted lines before capacity.
   */
  lines: GPUMapGraphCompactOutput;
  /**
   * CSR offsets into `points`, `lines.ids.length + 1` rows: line `i` is
   * `points[pathOffsets[i] .. pathOffsets[i + 1])`. Rows past `count + 1` are not written.
   */
  pathOffsets: GraphDataView<'uint32'>;
  /**
   * Polyline points, capacity rows, ordered backward end, seed, forward end per line. Rows past
   * `pointCount` are not written.
   */
  points: GraphDataView<'float32x2'>;
  /** One-row number of published points, `pathOffsets[count]`. */
  pointCount: GraphDataView<'uint32'>;
  /** One-row flag: 1 when `roundCount` rounds did not decide every seed (undecided seeds are dropped). */
  unconverged?: GraphDataView<'uint32'>;
  /**
   * Optional diagnostic outputs exposing the traced lines before pruning. `points` has
   * `seedCount * (2L + 1)` rows (slot `s * (2L + 1) + L + k` is step `k`, negative backward; the seed
   * point is NaN for an invalid seed; untraced slots are NaN) and `spans` has `seedCount * 4` rows
   * `[backwardSteps, forwardSteps, keptBackward, keptForward]` (the kept counts are final for
   * accepted lines only).
   */
  candidates?: {
    points: GraphDataView<'float32x2'>;
    spans: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUStreamlines}.
 *
 * Per-frame (no recompile): the contents of `parameters` (field and grid extents, step length,
 * minimum speed), `wordParameters` (seed, minimum points) and `velocities`. Topology: field, grid
 * and seed lattice sizes, `stepsPerDirection`, `roundCount`, output capacities.
 */
export type GPUStreamlinesProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'streamlines'`. */
  id?: string;
  /** Packed row-major `(u, v)` field, `fieldWidth * fieldHeight` rows; NaN marks no data. */
  velocities: GraphDataView<'float32x2'>;
  /** Field width in cells. Compile-time. */
  fieldWidth: number;
  /** Field height in cells. Compile-time. */
  fieldHeight: number;
  /** Occupancy grid width in cells. Compile-time. */
  gridWidth: number;
  /** Occupancy grid height in cells. Compile-time. */
  gridHeight: number;
  /** Seed lattice columns over the grid extent. Compile-time. */
  seedColumns: number;
  /** Seed lattice rows over the grid extent. Compile-time; `seedColumns * seedRows <= 65535`. */
  seedRows: number;
  /** Integration steps in each direction `L`. Compile-time. */
  stepsPerDirection: number;
  /** GPU-gated pruning rounds. Compile-time. Defaults to 32. */
  roundCount?: number;
  /** Per-frame float32 parameters from `getGPUStreamlinesParameterValues`. */
  parameters: GraphDataView<'float32'>;
  /** Per-frame uint32 parameters from `getGPUStreamlinesWordParameterValues`. */
  wordParameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUStreamlinesOutput;
};

/**
 * Evenly spaced streamlines (after Jobard and Lefer 1997) resolved by priority instead of
 * processing order, so the result is independent of GPU scheduling.
 *
 * 1. Trace: every seed of a jittered `seedColumns x seedRows` lattice over the grid extent is
 *    traced `L` RK2 midpoint steps backward and forward along the normalised field (`stepLength`
 *    world units per step), stopping at the grid or field edge, NaN data, or a speed that is zero
 *    or below `minimumSpeed`. Seed `s` uses `philox((s, 0, 0, 0), (seed, 2))`: words x and y
 *    jitter it inside its lattice cell and word z gives its priority key
 *    `(z >> 16) << 16 | (0xffff - s)`, unique, with ties favouring the lower seed index.
 * 2. Prune: the result equals this greedy pass in descending key: a line whose seed cell is
 *    occupied is rejected; each direction is cut before its first point in an occupied cell; the
 *    line is accepted when it keeps `minimumPoints` points, and its kept points occupy their cells.
 *    The occupancy grid cell is the separation distance. On the GPU, each of `roundCount` rounds
 *    clears a grid and runs two nodes. Claim: occupancy only grows, so an undecided line can keep
 *    at most the points before its first occupied cell in each direction; a line whose seed cell
 *    is occupied or whose remaining span is shorter than `minimumPoints` is rejected at once, and
 *    every other undecided line `atomicMax`es its key into the cells of its remaining span.
 *    Decide: a line holding the maximum in all of those cells is accepted with exactly that span.
 *    No undecided higher-key line can still occupy its cells, every decided one already has, and
 *    lower-key lines cannot have been accepted into them, so this reproduces the greedy pass. Two
 *    lines accepted in the same round touch disjoint cells, so marking accepted cells with plain
 *    stores is race-free. The highest undecided line is always decided, so every round makes
 *    progress; lines still undecided after `roundCount` rounds are dropped and `unconverged` set.
 * 3. Publish: prefix scans place the accepted lines in ascending seed order as CSR polylines,
 *    bounded by the line and point capacities.
 *
 * Determinism: integer atomics only (`atomicMax` of keys); counter-based jitter and priorities.
 * The pruning is exact given the traced points: grid cells use one subtraction and one
 * multiplication by a host-rounded reciprocal, both correctly rounded.
 */
export class GPUStreamlines implements GPUMapGraphRecipe {
  /** Prefix for graph node and transient IDs. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'streamlines';
  /** Number of seeds. */
  readonly seedCount: number;
  /** Number of pruning rounds. */
  readonly roundCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPUStreamlinesProps>;

  constructor(props: GPUStreamlinesProps) {
    const id = props.id ?? 'streamlines';
    for (const [name, value] of [
      ['fieldWidth', props.fieldWidth],
      ['fieldHeight', props.fieldHeight],
      ['gridWidth', props.gridWidth],
      ['gridHeight', props.gridHeight],
      ['seedColumns', props.seedColumns],
      ['seedRows', props.seedRows]
    ] as const) {
      if (!Number.isInteger(value) || value < 1 || value > 65535) {
        throw new Error(`${id} ${name} must be an integer in [1, 65535]`);
      }
    }
    const seedCount = props.seedColumns * props.seedRows;
    if (seedCount > MAXIMUM_SEED_COUNT) {
      throw new Error(`${id} seedColumns * seedRows must be at most ${MAXIMUM_SEED_COUNT}`);
    }
    if (!Number.isInteger(props.stepsPerDirection) || props.stepsPerDirection < 1) {
      throw new Error(`${id} stepsPerDirection must be a positive integer`);
    }
    const roundCount = props.roundCount ?? 32;
    if (!Number.isInteger(roundCount) || roundCount < 1) {
      throw new Error(`${id} roundCount must be a positive integer`);
    }
    validatePackedView(props.velocities, ['float32x2'], `${id} velocities`);
    if (props.velocities.length !== props.fieldWidth * props.fieldHeight) {
      throw new Error(`${id} velocities must have fieldWidth * fieldHeight rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_STREAMLINES_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_STREAMLINES_PARAMETER_LENGTH} float32 rows`
      );
    }
    validatePackedUint32View(props.wordParameters, `${id} wordParameters`);
    if (props.wordParameters.length < GPU_STREAMLINES_WORD_PARAMETER_LENGTH) {
      throw new Error(
        `${id} wordParameters must contain ${GPU_STREAMLINES_WORD_PARAMETER_LENGTH} uint32 rows`
      );
    }
    const {output} = props;
    validateMapGraphCompactOutput(id, output.lines);
    if (output.lines.ids.length < 1) {
      throw new Error(`${id} output.lines.ids must hold at least one line`);
    }
    validatePackedUint32View(output.pathOffsets, `${id} output.pathOffsets`);
    if (output.pathOffsets.length !== output.lines.ids.length + 1) {
      throw new Error(`${id} output.pathOffsets must have lines.ids.length + 1 rows`);
    }
    validatePackedView(output.points, ['float32x2'], `${id} output.points`);
    if (output.points.length < 1) {
      throw new Error(`${id} output.points must hold at least one point`);
    }
    for (const [name, view] of [
      ['pointCount', output.pointCount],
      ['unconverged', output.unconverged]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} output.${name}`);
        if (view.length < 1) {
          throw new Error(`${id} output.${name} must contain one uint32 row`);
        }
      }
    }
    if (output.candidates) {
      validatePackedView(output.candidates.points, ['float32x2'], `${id} candidates.points`);
      validatePackedUint32View(output.candidates.spans, `${id} candidates.spans`);
      if (
        output.candidates.points.length !== seedCount * (2 * props.stepsPerDirection + 1) ||
        output.candidates.spans.length !== seedCount * 4
      ) {
        throw new Error(
          `${id} candidates must have seedCount * (2L + 1) points and seedCount * 4 spans`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.lines.ids,
        output.lines.count,
        output.lines.overflow,
        output.lines.totalCount,
        output.pathOffsets,
        output.points,
        output.pointCount,
        output.unconverged,
        output.candidates?.points,
        output.candidates?.spans
      ],
      [props.velocities, props.parameters, props.wordParameters]
    );
    this.id = id;
    this.seedCount = seedCount;
    this.roundCount = roundCount;
    this.props = props;
  }

  /** Returns trace, pruning-round, scan, emit and finalize nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, seedCount, roundCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.velocities,
      props.parameters,
      props.wordParameters,
      output.lines.ids,
      output.lines.count,
      output.lines.overflow,
      output.lines.totalCount,
      output.pathOffsets,
      output.points,
      output.pointCount,
      output.unconverged,
      output.candidates?.points,
      output.candidates?.spans
    ]);
    const length = props.stepsPerDirection;
    const stride = 2 * length + 1;
    const gridCellCount = props.gridWidth * props.gridHeight;
    const lineCapacity = output.lines.ids.length;
    const pointCapacity = output.points.length;
    const candidates =
      output.candidates?.points ??
      createTransientView(graph, `${id}-candidates`, 'float32x2', seedCount * stride);
    const spans =
      output.candidates?.spans ??
      createTransientView(graph, `${id}-spans`, 'uint32', seedCount * 4);
    const keys = createTransientView(graph, `${id}-keys`, 'uint32', seedCount);
    const status = createTransientView(graph, `${id}-status`, 'uint32', seedCount);
    const undecidedMaximum = createTransientView(graph, `${id}-undecided`, 'uint32', gridCellCount);
    const accepted = createTransientView(graph, `${id}-accepted`, 'uint32', gridCellCount);
    const pointCounts = createTransientView(graph, `${id}-point-counts`, 'uint32', seedCount);
    const pointOffsets = createTransientView(graph, `${id}-point-offsets`, 'uint32', seedCount);
    const lineFlags = createTransientView(graph, `${id}-line-flags`, 'uint32', seedCount);
    const lineRanks = createTransientView(graph, `${id}-line-ranks`, 'uint32', seedCount);
    // [publishedLineCount, unconvergedFlag]
    const state = createTransientView(graph, `${id}-state`, 'uint32', 2);

    const gridDeclarations = /* wgsl */ `
const GRID_WIDTH: u32 = ${props.gridWidth}u;
const GRID_HEIGHT: u32 = ${props.gridHeight}u;
const STEPS: u32 = ${length}u;
const STRIDE: u32 = ${stride}u;
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getGridLocal(position: vec2<f32>) -> vec2<f32> {
  return (position - vec2<f32>(readParameter(4u), readParameter(5u))) *
    vec2<f32>(readParameter(8u), readParameter(9u));
}
fn isInsideGrid(local: vec2<f32>) -> bool {
  return local.x >= 0.0 && local.y >= 0.0 && local.x < f32(GRID_WIDTH) && local.y < f32(GRID_HEIGHT);
}
fn getGridCell(position: vec2<f32>) -> u32 {
  let local = getGridLocal(position);
  let column = min(u32(max(local.x, 0.0)), GRID_WIDTH - 1u);
  let row = min(u32(max(local.y, 0.0)), GRID_HEIGHT - 1u);
  return row * GRID_WIDTH + column;
}
fn readCandidate(slot: u32) -> vec2<f32> {
  return vec2<f32>(candidates[candidatesOffset + 2u * slot], candidates[candidatesOffset + 2u * slot + 1u]);
}`;

    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Trace every seed.
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-trace`,
        operation: OPERATION,
        variant: 'trace',
        bindings: [
          {
            name: 'velocities',
            view: props.velocities,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'parameters',
            view: props.parameters,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'words',
            view: props.wordParameters,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'candidates',
            view: candidates,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'spans', view: spans, type: 'u32', access: 'read_write'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'status', view: status, type: 'u32', access: 'read_write'}
        ],
        invocationCount: seedCount,
        declarations: `${PHILOX_WGSL}
${getFieldSamplingWGSL('velocities', props.fieldWidth, props.fieldHeight)}
${gridDeclarations}
const SEED_COLUMNS: u32 = ${props.seedColumns}u;
const SEED_ROWS: u32 = ${props.seedRows}u;
fn writeCandidate(slot: u32, position: vec2<f32>) {
  candidates[candidatesOffset + 2u * slot] = position.x;
  candidates[candidatesOffset + 2u * slot + 1u] = position.y;
}
fn getDirection(position: vec2<f32>, fieldExtent: vec4<f32>, directionSign: f32, minimumSpeed: f32) -> vec3<f32> {
  let sample = sampleField(position, fieldExtent);
  if (sample.z == 0.0) {
    return vec3<f32>(0.0);
  }
  let speed = length(sample.xy);
  if (!(speed > 0.0) || speed < minimumSpeed) {
    return vec3<f32>(0.0);
  }
  return vec3<f32>(directionSign * sample.xy / speed, 1.0);
}`,
        body: /* wgsl */ `
  let fieldExtent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let gridOrigin = vec2<f32>(readParameter(4u), readParameter(5u));
  let seedCell = vec2<f32>(readParameter(6u), readParameter(7u)) *
    vec2<f32>(f32(GRID_WIDTH), f32(GRID_HEIGHT)) / vec2<f32>(f32(SEED_COLUMNS), f32(SEED_ROWS));
  let stepLength = readParameter(10u);
  let minimumSpeed = readParameter(11u);
  let seed = words[wordsOffset];
  let random = philox4x32(vec4<u32>(index, 0u, 0u, 0u), vec2<u32>(seed, ${STREAMLINES_SEED_PURPOSE}u));
  keys[keysOffset + index] = ((random.z >> 16u) << 16u) | (0xffffu - index);
  let lattice = vec2<f32>(f32(index % SEED_COLUMNS), f32(index / SEED_COLUMNS));
  let start = gridOrigin + (lattice + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y))) * seedCell;
  let base = index * STRIDE + STEPS;
  var nanBits = 0x7fc00000u;
  let nanPoint = vec2<f32>(bitcast<f32>(nanBits));
  for (var slot = 0u; slot < STRIDE; slot = slot + 1u) {
    writeCandidate(index * STRIDE + slot, nanPoint);
  }
  spans[spansOffset + 4u * index] = 0u;
  spans[spansOffset + 4u * index + 1u] = 0u;
  spans[spansOffset + 4u * index + 2u] = 0u;
  spans[spansOffset + 4u * index + 3u] = 0u;
  if (getDirection(start, fieldExtent, 1.0, minimumSpeed).z == 0.0 || !isInsideGrid(getGridLocal(start))) {
    status[statusOffset + index] = ${REJECTED}u;
    return;
  }
  status[statusOffset + index] = ${UNDECIDED}u;
  writeCandidate(base, start);
  for (var direction = 0u; direction < 2u; direction = direction + 1u) {
    let directionSign = select(-1.0, 1.0, direction == 1u);
    var position = start;
    var steps = 0u;
    for (var stepIndex = 1u; stepIndex <= STEPS; stepIndex = stepIndex + 1u) {
      let first = getDirection(position, fieldExtent, directionSign, minimumSpeed);
      if (first.z == 0.0) {
        break;
      }
      let middle = getDirection(position + 0.5 * stepLength * first.xy, fieldExtent, directionSign, minimumSpeed);
      if (middle.z == 0.0) {
        break;
      }
      let next = position + stepLength * middle.xy;
      if (!isInsideGrid(getGridLocal(next))) {
        break;
      }
      position = next;
      steps = stepIndex;
      if (direction == 1u) {
        writeCandidate(base + stepIndex, position);
      } else {
        writeCandidate(base - stepIndex, position);
      }
    }
    spans[spansOffset + 4u * index + direction] = steps;
  }`
      })
    );

    // 2. GPU-gated pruning rounds.
    nodes.push(
      createMapGraphFillNode<Parameters>(graph, {
        id: `${id}-clear-accepted`,
        operation: OPERATION,
        view: accepted,
        type: 'u32',
        value: '0u'
      })
    );
    for (let round = 0; round < roundCount; round++) {
      nodes.push(
        createMapGraphFillNode<Parameters>(graph, {
          id: `${id}-round-${round}-clear`,
          operation: OPERATION,
          view: undecidedMaximum,
          type: 'u32',
          value: '0u'
        }),
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-round-${round}-claim`,
          operation: OPERATION,
          variant: 'claim',
          bindings: [
            {
              name: 'parameters',
              view: props.parameters,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'words',
              view: props.wordParameters,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'candidates',
              view: candidates,
              type: 'f32',
              access: 'read'
            },
            {name: 'spans', view: spans, type: 'u32', access: 'read_write'},
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'status', view: status, type: 'u32', access: 'read_write'},
            {name: 'accepted', view: accepted, type: 'u32', access: 'read'},
            {
              name: 'undecidedMaximum',
              view: undecidedMaximum,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: seedCount,
          declarations: gridDeclarations,
          body: /* wgsl */ `
  if (status[statusOffset + index] != ${UNDECIDED}u) {
    return;
  }
  // Occupancy only grows, so the line can keep at most the points before its first occupied
  // cell in each direction. Reject now when that cannot reach minimumPoints.
  let base = index * STRIDE + STEPS;
  if (accepted[acceptedOffset + getGridCell(readCandidate(base))] != 0u) {
    status[statusOffset + index] = ${REJECTED}u;
    return;
  }
  let backward = spans[spansOffset + 4u * index];
  let forward = spans[spansOffset + 4u * index + 1u];
  var keptBackward = 0u;
  while (keptBackward < backward &&
      accepted[acceptedOffset + getGridCell(readCandidate(base - keptBackward - 1u))] == 0u) {
    keptBackward = keptBackward + 1u;
  }
  var keptForward = 0u;
  while (keptForward < forward &&
      accepted[acceptedOffset + getGridCell(readCandidate(base + keptForward + 1u))] == 0u) {
    keptForward = keptForward + 1u;
  }
  if (1u + keptBackward + keptForward < words[wordsOffset + 1u]) {
    status[statusOffset + index] = ${REJECTED}u;
    return;
  }
  spans[spansOffset + 4u * index + 2u] = keptBackward;
  spans[spansOffset + 4u * index + 3u] = keptForward;
  let key = keys[keysOffset + index];
  for (var slot = base - keptBackward; slot <= base + keptForward; slot = slot + 1u) {
    atomicMax(&undecidedMaximum[undecidedMaximumOffset + getGridCell(readCandidate(slot))], key);
  }`
        }),
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-round-${round}-decide`,
          operation: OPERATION,
          variant: 'decide',
          bindings: [
            {
              name: 'parameters',
              view: props.parameters,
              type: 'f32',
              access: 'read'
            },
            {
              name: 'candidates',
              view: candidates,
              type: 'f32',
              access: 'read'
            },
            {name: 'spans', view: spans, type: 'u32', access: 'read'},
            {name: 'keys', view: keys, type: 'u32', access: 'read'},
            {name: 'status', view: status, type: 'u32', access: 'read_write'},
            {
              name: 'undecidedMaximum',
              view: undecidedMaximum,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'accepted',
              view: accepted,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: seedCount,
          declarations: gridDeclarations,
          body: /* wgsl */ `
  if (status[statusOffset + index] != ${UNDECIDED}u) {
    return;
  }
  let key = keys[keysOffset + index];
  let base = index * STRIDE + STEPS;
  let first = base - spans[spansOffset + 4u * index + 2u];
  let last = base + spans[spansOffset + 4u * index + 3u];
  for (var slot = first; slot <= last; slot = slot + 1u) {
    if (undecidedMaximum[undecidedMaximumOffset + getGridCell(readCandidate(slot))] != key) {
      return;
    }
  }
  // Ready: no undecided line with a higher key can still occupy these cells, every decided one
  // already has, and no line decided in this dispatch touches them, so the claimed span is the
  // greedy trim.
  status[statusOffset + index] = ${ACCEPTED}u;
  for (var slot = first; slot <= last; slot = slot + 1u) {
    accepted[acceptedOffset + getGridCell(readCandidate(slot))] = index + 1u;
  }`
        })
      );
    }

    // 3. Point counts, line flags and the convergence flag.
    nodes.push(
      createMapGraphFillNode<Parameters>(graph, {
        id: `${id}-clear-state`,
        operation: OPERATION,
        view: state,
        type: 'u32',
        value: '0u'
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-counts`,
        operation: OPERATION,
        variant: 'counts',
        bindings: [
          {name: 'spans', view: spans, type: 'u32', access: 'read'},
          {name: 'status', view: status, type: 'u32', access: 'read'},
          {
            name: 'pointCounts',
            view: pointCounts,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'lineFlags',
            view: lineFlags,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        invocationCount: seedCount,
        body: /* wgsl */ `
  let lineStatus = status[statusOffset + index];
  let isAccepted = lineStatus == ${ACCEPTED}u;
  pointCounts[pointCountsOffset + index] = select(
    0u,
    1u + spans[spansOffset + 4u * index + 2u] + spans[spansOffset + 4u * index + 3u],
    isAccepted
  );
  lineFlags[lineFlagsOffset + index] = select(0u, 1u, isAccepted);
  if (lineStatus == ${UNDECIDED}u) {
    state[stateOffset + 1u] = 1u;
  }`
      }),
      ...new GPUScan({
        id: `${id}-point-scan`,
        input: pointCounts,
        output: pointOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-line-scan`,
        input: lineFlags,
        output: lineRanks,
        mode: 'inclusive'
      }).getCommandNodes(graph)
    );

    // 4. Emit the lines that fit both capacities (a prefix in seed order).
    const fitsDeclaration = /* wgsl */ `
const LINE_CAPACITY: u32 = ${lineCapacity}u;
const POINT_CAPACITY: u32 = ${pointCapacity}u;
fn getFittingRank(index: u32) -> u32 {
  if (lineFlags[lineFlagsOffset + index] == 0u) {
    return 0xffffffffu;
  }
  let rank = lineRanks[lineRanksOffset + index] - 1u;
  let end = pointOffsets[pointOffsetsOffset + index] + pointCounts[pointCountsOffset + index];
  if (rank >= LINE_CAPACITY || end > POINT_CAPACITY) {
    return 0xffffffffu;
  }
  return rank;
}`;
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-emit-points`,
        operation: OPERATION,
        variant: 'emit-points',
        bindings: [
          {name: 'candidates', view: candidates, type: 'f32', access: 'read'},
          {name: 'spans', view: spans, type: 'u32', access: 'read'},
          {name: 'lineFlags', view: lineFlags, type: 'u32', access: 'read'},
          {name: 'lineRanks', view: lineRanks, type: 'u32', access: 'read'},
          {
            name: 'pointCounts',
            view: pointCounts,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'pointOffsets',
            view: pointOffsets,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'points',
            view: output.points,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: seedCount,
        declarations: `const STEPS: u32 = ${length}u;
const STRIDE: u32 = ${stride}u;
${fitsDeclaration}`,
        body: /* wgsl */ `
  if (getFittingRank(index) == 0xffffffffu) {
    return;
  }
  let first = index * STRIDE + STEPS - spans[spansOffset + 4u * index + 2u];
  let offset = pointOffsets[pointOffsetsOffset + index];
  let count = pointCounts[pointCountsOffset + index];
  for (var point = 0u; point < count; point = point + 1u) {
    points[pointsOffset + 2u * (offset + point)] = candidates[candidatesOffset + 2u * (first + point)];
    points[pointsOffset + 2u * (offset + point) + 1u] = candidates[candidatesOffset + 2u * (first + point) + 1u];
  }`
      }),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-emit-lines`,
        operation: OPERATION,
        variant: 'emit-lines',
        bindings: [
          {name: 'lineFlags', view: lineFlags, type: 'u32', access: 'read'},
          {name: 'lineRanks', view: lineRanks, type: 'u32', access: 'read'},
          {
            name: 'pointCounts',
            view: pointCounts,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'pointOffsets',
            view: pointOffsets,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'ids',
            view: output.lines.ids,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'pathOffsets',
            view: output.pathOffsets,
            type: 'u32',
            access: 'read_write'
          },
          {
            name: 'state',
            view: state,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        invocationCount: seedCount,
        declarations: fitsDeclaration,
        body: /* wgsl */ `
  let rank = getFittingRank(index);
  if (rank == 0xffffffffu) {
    return;
  }
  ids[idsOffset + rank] = index;
  pathOffsets[pathOffsetsOffset + rank + 1u] =
    pointOffsets[pointOffsetsOffset + index] + pointCounts[pointCountsOffset + index];
  atomicMax(&state[stateOffset], rank + 1u);`
      })
    );

    // 5. Counts and flags.
    const finalizeBindings = [
      {name: 'lineRanks', view: lineRanks, type: 'u32', access: 'read'},
      {name: 'state', view: state, type: 'u32', access: 'read'},
      {
        name: 'pathOffsets',
        view: output.pathOffsets,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'countOut',
        view: output.lines.count,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'overflowOut',
        view: output.lines.overflow,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'pointCountOut',
        view: output.pointCount,
        type: 'u32',
        access: 'read_write'
      }
    ] as const;
    const optionalBindings = [
      ...(output.lines.totalCount
        ? [
            {
              name: 'totalOut',
              view: output.lines.totalCount,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : []),
      ...(output.unconverged
        ? [
            {
              name: 'unconvergedOut',
              view: output.unconverged,
              type: 'u32',
              access: 'read_write'
            } as const
          ]
        : [])
    ];
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: 'finalize',
        bindings: [...finalizeBindings, ...optionalBindings],
        invocationCount: 1,
        body: /* wgsl */ `
  let total = lineRanks[lineRanksOffset + ${seedCount - 1}u];
  let count = state[stateOffset];
  pathOffsets[pathOffsetsOffset] = 0u;
  countOut[countOutOffset] = count;
  overflowOut[overflowOutOffset] = select(0u, 1u, count < total);
  pointCountOut[pointCountOutOffset] = select(0u, pathOffsets[pathOffsetsOffset + count], count > 0u);
  ${output.lines.totalCount ? 'totalOut[totalOutOffset] = total;' : ''}
  ${output.unconverged ? 'unconvergedOut[unconvergedOutOffset] = state[stateOffset + 1u];' : ''}`
      })
    );
    return nodes;
  }
}
