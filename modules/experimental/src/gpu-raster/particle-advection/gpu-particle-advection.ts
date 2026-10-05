// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  PARTICLE_ADVECTION_DROP_PURPOSE,
  PARTICLE_ADVECTION_SPAWN_PURPOSE
} from './particle-advection-cpu';
import {getFieldSamplingWGSL} from './particle-advection-field';
import {
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH
} from './particle-advection-parameters';
import {PHILOX_WGSL} from './particle-advection-random';

const OPERATION = 'GPUParticleAdvection';

/**
 * Optional trail history of {@link GPUParticleAdvection}: the last `length` positions of every
 * particle, ready for a `PathLayer` or `LineLayer` without CPU readback.
 */
export type GPUParticleAdvectionTrails = {
  /**
   * Ring buffer of `particleCount * length` rows. Particle `i` owns rows `i * length` to
   * `i * length + length - 1`. Frame `f` writes slot `f % length`, so the newest position is at
   * slot `frame % length` and the oldest at `(frame + 1) % length`. On a respawn frame every slot
   * of the particle is filled with the spawn position, so a trail never streaks across the map.
   */
  positions: GraphDataView<'float32x2'>;
  /** Ring length `L` in frames. Compile-time. */
  length: number;
};

/**
 * Properties for {@link GPUParticleAdvection}.
 *
 * Per-frame (no recompile): the contents of `parameters`, `wordParameters` and `velocities`.
 * Topology (needs a new graph): `fieldWidth`, `fieldHeight`, `particleCount` (view lengths), the
 * trail length and which optional outputs are present.
 */
export type GPUParticleAdvectionProps = {
  /** Prefix for generated node IDs. Defaults to `'particle-advection'`. */
  id?: string;
  /**
   * Packed row-major vector field, `fieldWidth * fieldHeight` rows of `(u, v)` in world units per
   * time unit. Row 0 has the smallest y. A NaN component marks a cell without data.
   */
  velocities: GraphDataView<'float32x2'>;
  /** Field width in cells. Compile-time. */
  fieldWidth: number;
  /** Field height in cells. Compile-time. */
  fieldHeight: number;
  /**
   * Per-frame float32 parameters (at least {@link GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH} rows),
   * written with `getGPUParticleAdvectionParameterValues`: field extent, time step, speed scale,
   * drop rate, minimum speed, spawn bounds.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Per-frame uint32 parameters (at least {@link GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH}
   * rows), written with `getGPUParticleAdvectionWordParameterValues`: seed, frame, maximum age,
   * reset flag.
   */
  wordParameters: GraphDataView<'uint32'>;
  /** Particle state, updated in place. All state views have `particleCount` rows. */
  state: {
    /** Current particle positions. */
    positions: GraphDataView<'float32x2'>;
    /** Frames since the particle spawned. */
    ages: GraphDataView<'uint32'>;
    /** Respawn count of each particle; keys its next spawn position. */
    generations: GraphDataView<'uint32'>;
  };
  /**
   * Optional output: position before this frame's step, equal to the new position on a respawn
   * frame, so `(previousPositions[i], positions[i])` is a zero-length segment after a jump.
   */
  previousPositions?: GraphDataView<'float32x2'>;
  /**
   * Optional output: `|v| * |speedScale|` sampled at the new position, for colouring. Zero when a
   * freshly spawned particle sits outside the field or on a NaN cell.
   */
  speeds?: GraphDataView<'float32'>;
  /** Optional trail ring buffer. */
  trails?: GPUParticleAdvectionTrails;
};

/**
 * Advects particles through a 2D vector field (wind, currents, flow directions), one frame per
 * encoding, with exact replays.
 *
 * Each frame every particle takes one second-order Runge-Kutta (midpoint) step
 * `p' = p + h * v(p + h/2 * v(p))` with `h = timeStep * speedScale`, sampling the field by manual
 * bilinear interpolation between cell centres. A particle respawns when the `reset` word is set,
 * when its age reaches `maximumAge`, when any sample (start, midpoint, landing) is outside the
 * field or touches a NaN cell, when it is slower than `minimumSpeed`, or when a per-frame drop test
 * fires with probability `dropRate`.
 *
 * Determinism: there are no atomics and no persistent RNG state. Random numbers come from a
 * Philox 4x32-10 counter-based generator. The spawn position of particle `i` in generation `g` is
 * `philox(counter = (i, g, 0, 0), key = (seed, 1))` (words x and y; word z staggers the initial age
 * on reset), and the drop test of frame `f` uses word x of `philox((i, f, 0, 0), (seed, 2))`. A
 * replay with the same parameter sequence is therefore bitwise identical, and a particle's spawn
 * points do not depend on the particle count or the dispatch order.
 *
 * State is updated in place rather than ping-ponged: every invocation reads and writes only its
 * own particle's rows, so there is no cross-row hazard, and one copy of the state halves memory.
 * The trail ring is written by a second node after the step.
 *
 * Precision: positions and the field extent are f32; use field-local or tile-local coordinates for
 * large worlds.
 */
export class GPUParticleAdvection implements GPUCommandNodeProducer {
  /** Prefix for graph node IDs. */
  readonly id: string;
  /** Number of particles. */
  readonly particleCount: number;
  /** Validated properties. */
  readonly props: Readonly<GPUParticleAdvectionProps>;

  constructor(props: GPUParticleAdvectionProps) {
    const id = props.id ?? 'particle-advection';
    const {fieldWidth, fieldHeight, state, trails} = props;
    for (const [name, value] of [
      ['fieldWidth', fieldWidth],
      ['fieldHeight', fieldHeight]
    ] as const) {
      if (!Number.isInteger(value) || value < 1 || value > 65535) {
        throw new Error(`${id} ${name} must be an integer in [1, 65535]`);
      }
    }
    validatePackedView(props.velocities, ['float32x2'], `${id} velocities`);
    if (props.velocities.length !== fieldWidth * fieldHeight) {
      throw new Error(`${id} velocities must have fieldWidth * fieldHeight rows`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must contain ${GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH} float32 rows`
      );
    }
    validatePackedUint32View(props.wordParameters, `${id} wordParameters`);
    if (props.wordParameters.length < GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH) {
      throw new Error(
        `${id} wordParameters must contain ${GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH} uint32 rows`
      );
    }
    validatePackedView(state.positions, ['float32x2'], `${id} state.positions`);
    const particleCount = state.positions.length;
    if (particleCount < 1) {
      throw new Error(`${id} needs at least one particle`);
    }
    validatePackedUint32View(state.ages, `${id} state.ages`);
    validatePackedUint32View(state.generations, `${id} state.generations`);
    if (props.previousPositions) {
      validatePackedView(props.previousPositions, ['float32x2'], `${id} previousPositions`);
    }
    if (props.speeds) {
      validatePackedView(props.speeds, ['float32'], `${id} speeds`);
    }
    for (const [name, view] of [
      ['state.ages', state.ages],
      ['state.generations', state.generations],
      ['previousPositions', props.previousPositions],
      ['speeds', props.speeds]
    ] as const) {
      if (view && view.length !== particleCount) {
        throw new Error(`${id} ${name} must have one row per particle`);
      }
    }
    if (trails) {
      if (!Number.isInteger(trails.length) || trails.length < 1) {
        throw new Error(`${id} trails.length must be a positive integer`);
      }
      validatePackedView(trails.positions, ['float32x2'], `${id} trails.positions`);
      if (trails.positions.length !== particleCount * trails.length) {
        throw new Error(`${id} trails.positions must have particleCount * trails.length rows`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        state.positions,
        state.ages,
        state.generations,
        props.previousPositions,
        props.speeds,
        trails?.positions
      ],
      [props.velocities, props.parameters, props.wordParameters]
    );
    this.id = id;
    this.particleCount = particleCount;
    this.props = props;
  }

  /** Returns the step node and, with trails, the trail-ring node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, particleCount} = this;
    const {state, trails} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.velocities,
      props.parameters,
      props.wordParameters,
      state.positions,
      state.ages,
      state.generations,
      props.previousPositions,
      props.speeds,
      trails?.positions
    ]);
    const bindings: WGSLKernelBinding[] = [
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
        name: 'positions',
        view: state.positions,
        type: 'f32',
        access: 'read_write'
      },
      {name: 'ages', view: state.ages, type: 'u32', access: 'read_write'},
      {
        name: 'generations',
        view: state.generations,
        type: 'u32',
        access: 'read_write'
      }
    ];
    if (props.previousPositions) {
      bindings.push({
        name: 'previousPositions',
        view: props.previousPositions,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.speeds) {
      bindings.push({
        name: 'speeds',
        view: props.speeds,
        type: 'f32',
        access: 'read_write'
      });
    }
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-step`,
        operation: OPERATION,
        variant: 'step',
        bindings,
        invocationCount: particleCount,
        declarations: `${PHILOX_WGSL}
${getFieldSamplingWGSL('velocities', props.fieldWidth, props.fieldHeight)}
fn readParameter(slot: u32) -> f32 { return parameters[parametersOffset + slot]; }
fn getSpeed(sample: vec2<f32>, speedScale: f32) -> f32 { return length(sample) * abs(speedScale); }`,
        body: /* wgsl */ `
  let extent = vec4<f32>(readParameter(0u), readParameter(1u), readParameter(2u), readParameter(3u));
  let timeStep = readParameter(4u);
  let speedScale = readParameter(5u);
  let dropRate = readParameter(6u);
  let minimumSpeed = readParameter(7u);
  let spawnMin = vec2<f32>(readParameter(8u), readParameter(9u));
  let spawnMax = vec2<f32>(readParameter(10u), readParameter(11u));
  let seed = words[wordsOffset];
  let frame = words[wordsOffset + 1u];
  let maximumAge = words[wordsOffset + 2u];
  let reset = words[wordsOffset + 3u] != 0u;
  let step = timeStep * speedScale;
  let halfStep = 0.5 * step;
  var position = vec2<f32>(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  var previous = position;
  var age = ages[agesOffset + index];
  var generation = generations[generationsOffset + index];
  var respawn = reset;
  var speed = 0.0;
  if (reset) {
    generation = 0u;
  } else {
    let k1 = sampleField(position, extent);
    let middle = position + halfStep * k1.xy;
    let k2 = sampleField(middle, extent);
    let next = position + step * k2.xy;
    let landing = sampleField(next, extent);
    let dropWord = philox4x32(vec4<u32>(index, frame, 0u, 0u), vec2<u32>(seed, ${PARTICLE_ADVECTION_DROP_PURPOSE}u)).x;
    let expired = maximumAge != 0u && age + 1u >= maximumAge;
    let tooSlow = k2.z != 0.0 && getSpeed(k2.xy, speedScale) < minimumSpeed;
    if (k1.z == 0.0 || k2.z == 0.0 || landing.z == 0.0 || expired || tooSlow ||
        philoxUnitFloat(dropWord) < dropRate) {
      respawn = true;
      generation = generation + 1u;
    } else {
      position = next;
      age = age + 1u;
      speed = getSpeed(landing.xy, speedScale);
    }
  }
  if (respawn) {
    let random = philox4x32(vec4<u32>(index, generation, 0u, 0u), vec2<u32>(seed, ${PARTICLE_ADVECTION_SPAWN_PURPOSE}u));
    position = spawnMin + vec2<f32>(philoxUnitFloat(random.x), philoxUnitFloat(random.y)) * (spawnMax - spawnMin);
    previous = position;
    age = 0u;
    if (reset) {
      age = min(u32(floor(philoxUnitFloat(random.z) * f32(maximumAge))), max(maximumAge, 1u) - 1u);
    }
    let landing = sampleField(position, extent);
    speed = select(0.0, getSpeed(landing.xy, speedScale), landing.z != 0.0);
  }
  positions[positionsOffset + 2u * index] = position.x;
  positions[positionsOffset + 2u * index + 1u] = position.y;
  ages[agesOffset + index] = age;
  generations[generationsOffset + index] = generation;
  ${
    props.previousPositions
      ? `previousPositions[previousPositionsOffset + 2u * index] = previous.x;
  previousPositions[previousPositionsOffset + 2u * index + 1u] = previous.y;`
      : '_ = previous;'
  }
  ${props.speeds ? 'speeds[speedsOffset + index] = speed;' : '_ = speed;'}`
      })
    ];
    if (trails) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-trails`,
          operation: OPERATION,
          variant: 'trails',
          bindings: [
            {
              name: 'words',
              view: props.wordParameters,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'positions',
              view: state.positions,
              type: 'f32',
              access: 'read'
            },
            {name: 'ages', view: state.ages, type: 'u32', access: 'read'},
            {
              name: 'trails',
              view: trails.positions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: particleCount,
          declarations: `const TRAIL_LENGTH: u32 = ${trails.length}u;`,
          body: /* wgsl */ `
  let frame = words[wordsOffset + 1u];
  let reset = words[wordsOffset + 3u] != 0u;
  let x = positions[positionsOffset + 2u * index];
  let y = positions[positionsOffset + 2u * index + 1u];
  let base = trailsOffset + 2u * index * TRAIL_LENGTH;
  if (reset || ages[agesOffset + index] == 0u) {
    for (var slot = 0u; slot < TRAIL_LENGTH; slot = slot + 1u) {
      trails[base + 2u * slot] = x;
      trails[base + 2u * slot + 1u] = y;
    }
  } else {
    let slot = frame % TRAIL_LENGTH;
    trails[base + 2u * slot] = x;
    trails[base + 2u * slot + 1u] = y;
  }`
        })
      );
    }
    return nodes;
  }
}
