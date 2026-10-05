// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createFillNode,
  createWGSLKernelNode,
  createPublishNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCompactOutput
} from '../../utils/gpu-contributor-utils';
import {TERRAIN_WGSL_HELPERS} from '../terrain-analysis/terrain-analysis-utils';

const OPERATION = 'GPUProfilePeaks';

/** Number of float32 values read from `GPUProfilePeaksProps.settings`. */
export const GPU_PROFILE_PEAKS_PARAMETER_LENGTH = 1;

/** Default compile-time cap on suppression rounds. */
export const GPU_PROFILE_PEAKS_DEFAULT_NMS_ROUNDS = 16;

/** Largest accepted `window` and (rounded up) `nms`, in samples. Bounds the per-sample loops. */
export const GPU_PROFILE_PEAKS_MAXIMUM_RADIUS = 4096;

/** Largest accepted `nmsRounds`. */
export const GPU_PROFILE_PEAKS_MAXIMUM_NMS_ROUNDS = 1024;

/**
 * CPU-side description packed by {@link getGPUProfilePeaksParameterValues}.
 *
 * Values are in profile-value units (for example metres for an elevation profile, degrees for a
 * horizon). The model is cell-size independent: every other distance of the recipe is measured in
 * profile samples.
 */
export type GPUProfilePeaksSettings = {
  /**
   * Minimum prominence, in profile-value units. A candidate with `prominence < minProminence` is
   * rejected. Must not be NaN; a value at or below zero accepts every candidate.
   */
  minProminence: number;
};

/**
 * Packs profile-peak settings into `[minProminence]`.
 *
 * @throws If `minProminence` is NaN or `target` is too short.
 */
export function getGPUProfilePeaksParameterValues(
  settings: GPUProfilePeaksSettings,
  target: Float32Array = new Float32Array(GPU_PROFILE_PEAKS_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_PROFILE_PEAKS_PARAMETER_LENGTH) {
    throw new Error('Profile peaks settings target must hold 1 value');
  }
  if (Number.isNaN(settings.minProminence)) {
    throw new Error('Profile peaks minProminence must not be NaN');
  }
  target[0] = settings.minProminence;
  return target;
}

/**
 * Properties for {@link GPUProfilePeaks}.
 *
 * Topology (compile time): `window`, `minSide`, `nms`, `wrap`, `nmsRounds`, which views exist and
 * their lengths. Per-frame: `settings` and the contents of `values`, `validity` and `offsets`.
 * Every distance is in profile samples and every value in profile-value units, so the recipe is
 * cell-size independent.
 */
export type GPUProfilePeaksProps = {
  /** Prefix for node and transient IDs. Defaults to `'profile-peaks'`. */
  id?: string;
  /**
   * Packed float32 profile samples, all profiles concatenated. A non-finite value (NaN or
   * infinity) is a gap. Length at least 1.
   */
  values: GraphDataView<'float32'>;
  /**
   * Optional packed uint32 validity, one word per sample; zero marks a gap. A gap's value is never
   * read, so a sentinel such as `-32768` or `1e30` stored there cannot influence any output.
   */
  validity?: GraphDataView<'uint32'>;
  /**
   * Packed uint32 CSR (compressed sparse row) span offsets with `profileCount + 1` rows: profile
   * `p` owns samples `[offsets[p], offsets[p + 1])`. Offsets must be non-decreasing and at most
   * `values.length`; empty profiles are allowed. Samples outside `[offsets[0], offsets[last])`
   * belong to no profile and are never reported. This is the layout of
   * `GPURasterProfile.pathSampleOffsets`.
   */
  offsets: GraphDataView<'uint32'>;
  /** Per-frame settings with at least 1 float32 value, see {@link getGPUProfilePeaksParameterValues}. */
  settings: GraphDataView<'float32'>;
  /** Half-window, in samples, of the prominence walk. Integer in `[1, 4096]`. */
  window: number;
  /**
   * Minimum number of valid samples counted on each side of a candidate. Integer `>= 0`. Default 4.
   */
  minSide?: number;
  /**
   * Non-maximum suppression radius in samples, measured between refined (fractional) positions.
   * Finite, in `[0, 4096]`. Default `max(2, round(window / 4))`.
   */
  nms?: number;
  /**
   * Treat each profile as circular (a 360 degree horizon). Default false. See
   * {@link GPUProfilePeaks} for the exact semantics.
   */
  wrap?: boolean;
  /** Compile-time cap on suppression rounds, in `[0, 1024]`. Default 16. */
  nmsRounds?: number;
  /** Optional per-sample prominence of kept peaks, NaN elsewhere. */
  prominence?: GraphDataView<'float32'>;
  /**
   * Optional per-sample refined fractional index of kept peaks inside their own profile (`0` is
   * the profile's first sample), NaN elsewhere.
   */
  refinedIndex?: GraphDataView<'float32'>;
  /** Optional per-sample parabola-refined value of kept peaks, NaN elsewhere. */
  refinedValue?: GraphDataView<'float32'>;
  /** Optional per-sample mask: 1 on a kept peak's sample row, otherwise 0. */
  peakMask?: GraphDataView<'uint32'>;
  /**
   * Optional compact result: sample row IDs (indices into `values`) of kept peaks in ascending
   * order, with a clamped count and an overflow flag.
   */
  output?: GPUCompactOutput;
  /**
   * Optional one-row scalar: 1 when suppression converged within `nmsRounds` rounds, otherwise 0.
   * When 0, undecided candidates are not reported (fail closed).
   */
  converged?: GraphDataView<'uint32'>;
};

// Per-candidate state codes of the suppression phase.
const STATE_INERT = 0;
const STATE_UNDECIDED = 1;
const STATE_KEPT = 2;
const STATE_SUPPRESSED = 3;

/**
 * Local maxima of many 1-D profiles with windowed topographic prominence, parabolic sub-sample
 * refinement and non-maximum suppression.
 *
 * Works on any profiles stored as CSR (compressed sparse row) spans, for example the
 * `sampleValues` and `pathSampleOffsets` of `GPURasterProfile`, or the apparent-horizon elevation
 * per azimuth. It ports the reference `profilePeaks` of mt-image, quirks included:
 *
 * - **Gaps.** A gap is a non-finite value or a zero validity word. A gap's value is never read or
 *   blended into anything.
 * - **Candidates.** Non-wrapped: samples `i` in `[1, n - 2]` of a profile with `n` samples whose
 *   value `v` is finite. Local maximum over `+-2`: for each `k` in `-2..2` except 0, skip gaps and
 *   out-of-range neighbours `w`; reject if `w > v`, or if `k < 0` and `w == v` (the first sample of
 *   a plateau wins).
 * - **Prominence.** For each side, `lo = v`, `valid = 0`; for `k = 1..window` step to
 *   `j = i +- k`, stop at the profile end or a gap, `valid++`, stop if `w > v` (that higher sample
 *   is counted in `valid` but not in `lo`), else `lo = min(lo, w)`. Reject if either side has
 *   `valid < minSide`. `prominence = v - max(loLeft, loRight)`; reject if below `minProminence`.
 *   The prominence is one f32 subtraction of two f32 values, so a CPU oracle using `Math.fround`
 *   reproduces it bit for bit.
 * - **Refinement.** With both neighbours valid, `den = a - 2v + c`; if `den < 0`,
 *   `di = clamp(0.5 (a - c) / den, -0.5, 0.5)` and `value = v - 0.25 (a - c) di`, otherwise
 *   `di = 0` and `value = v`. The refined index is `i + di`. The division is not correctly rounded
 *   on GPUs, so expect a few ULP against a float64 oracle.
 * - **Suppression.** Candidates are ranked by prominence descending, ties by ascending sample
 *   index. A candidate is kept iff no higher-ranked KEPT candidate of the same profile lies within
 *   `nms` samples (`|refined position difference| <= nms` suppresses; strictly greater keeps).
 *
 * **Wrap mode** (`wrap: true`, no counterpart in mt-image) treats each profile as a circle: every
 * sample is a candidate, neighbour indices are taken modulo `n` (a neighbour that is the sample
 * itself is skipped), side walks stop after `n - 1` steps, the parabola neighbours wrap, the
 * refined index is wrapped into `[0, n)` (a tiny negative offset on sample 0 rounds to 0 in f32),
 * and suppression uses circular distance `min(d, n - d)`. Rank ties still break on the local
 * sample index, so a plateau straddling the seam resolves to its first sample in index order that
 * lacks an equal predecessor within 2.
 *
 * **Parallel exact suppression.** Greedy suppression is sequential, but its result is the unique
 * fixed point of: `p` is kept iff no kept candidate of higher rank lies within `nms`. The GPU
 * computes that fixed point in rounds over a per-candidate state (undecided, kept, suppressed). In
 * a round an undecided `p` becomes suppressed if any higher-ranked candidate within `nms` is kept,
 * becomes kept if every higher-ranked candidate within `nms` is suppressed, and otherwise stays
 * undecided. Equivalence with greedy, by induction on rank: assume every candidate of rank below
 * `r` has the greedy decision whenever it is decided. A candidate `p` of rank `r` has greedy
 * decision "kept" iff all higher-ranked candidates within `nms` were rejected by greedy, and
 * "suppressed" iff one was kept; the round rule applies exactly these two tests to decided
 * higher-ranked neighbours, so it only ever decides `p` in agreement with greedy, and it decides
 * `p` as soon as the highest-ranked relevant neighbours are decided (the top-ranked candidate of
 * a profile has no higher neighbour and is kept in round 1). Decisions are monotone and never
 * revisited, and a decided state is final truth, so racing in-place updates are safe: a reader
 * sees either the old undecided state (and merely waits one more round) or the final decision.
 * State words are atomics, so no torn or speculative value is observable. Candidates within `nms`
 * in refined position lie within `floor(nms) + 1` raw samples because `|di| <= 0.5`, which bounds
 * the scan. If `nmsRounds` rounds do not decide every candidate (a chain of mutually suppressing
 * candidates longer than the cap), `converged` reads 0 and undecided candidates are not reported.
 *
 * Offsets must be non-decreasing; each sample finds its profile by binary search. All work is in
 * sample space and profile-value units, independent of any cell size. Chunked views are not
 * supported.
 */
export class GPUProfilePeaks implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUProfilePeaksProps;
  /** Minimum valid samples per side. */
  readonly minSide: number;
  /** Non-maximum suppression radius in samples. */
  readonly nms: number;
  /** Whether profiles are circular. */
  readonly wrap: boolean;
  /** Compile-time cap on suppression rounds. */
  readonly nmsRounds: number;

  constructor(props: GPUProfilePeaksProps) {
    this.id = props.id ?? 'profile-peaks';
    this.props = props;
    this.minSide = props.minSide ?? 4;
    this.nms = props.nms ?? Math.max(2, Math.round(props.window / 4));
    this.wrap = props.wrap ?? false;
    this.nmsRounds = props.nmsRounds ?? GPU_PROFILE_PEAKS_DEFAULT_NMS_ROUNDS;
    const {id} = this;
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length < 1) {
      throw new Error(`${id} values must contain at least one sample`);
    }
    const sampleCount = props.values.length;
    if (props.validity) {
      validatePackedView(props.validity, ['uint32'], `${id} validity`);
      if (props.validity.length !== sampleCount) {
        throw new Error(`${id} validity must contain one word per sample`);
      }
    }
    validatePackedView(props.offsets, ['uint32'], `${id} offsets`);
    if (props.offsets.length < 1) {
      throw new Error(`${id} offsets must contain profileCount + 1 rows`);
    }
    validatePackedView(props.settings, ['float32'], `${id} settings`);
    if (props.settings.length < GPU_PROFILE_PEAKS_PARAMETER_LENGTH) {
      throw new Error(`${id} settings must contain at least 1 float32 value`);
    }
    if (
      !Number.isInteger(props.window) ||
      props.window < 1 ||
      props.window > GPU_PROFILE_PEAKS_MAXIMUM_RADIUS
    ) {
      throw new Error(
        `${id} window must be an integer in [1, ${GPU_PROFILE_PEAKS_MAXIMUM_RADIUS}]`
      );
    }
    if (!Number.isInteger(this.minSide) || this.minSide < 0) {
      throw new Error(`${id} minSide must be an integer >= 0`);
    }
    if (!Number.isFinite(this.nms) || this.nms < 0 || this.nms > GPU_PROFILE_PEAKS_MAXIMUM_RADIUS) {
      throw new Error(`${id} nms must be in [0, ${GPU_PROFILE_PEAKS_MAXIMUM_RADIUS}]`);
    }
    if (
      !Number.isInteger(this.nmsRounds) ||
      this.nmsRounds < 0 ||
      this.nmsRounds > GPU_PROFILE_PEAKS_MAXIMUM_NMS_ROUNDS
    ) {
      throw new Error(
        `${id} nmsRounds must be an integer in [0, ${GPU_PROFILE_PEAKS_MAXIMUM_NMS_ROUNDS}]`
      );
    }
    if (
      !props.prominence &&
      !props.refinedIndex &&
      !props.refinedValue &&
      !props.peakMask &&
      !props.output &&
      !props.converged
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view, format] of [
      ['prominence', props.prominence, 'float32'],
      ['refinedIndex', props.refinedIndex, 'float32'],
      ['refinedValue', props.refinedValue, 'float32'],
      ['peakMask', props.peakMask, 'uint32']
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} ${name}`);
      if (view.length !== sampleCount) {
        throw new Error(`${id} ${name} must contain one value per sample`);
      }
    }
    if (props.output) {
      validateCompactOutput(id, props.output);
    }
    if (props.converged) {
      validatePackedView(props.converged, ['uint32'], `${id} converged`);
      if (props.converged.length < 1) {
        throw new Error(`${id} converged must contain one uint32 row`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.prominence,
        props.refinedIndex,
        props.refinedValue,
        props.peakMask,
        props.converged,
        props.output?.ids,
        props.output?.count,
        props.output?.overflow,
        props.output?.totalCount
      ],
      [props.values, props.validity, props.offsets, props.settings]
    );
  }

  /** Returns candidate, suppression, output and optional compaction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, minSide, nms, wrap, nmsRounds} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.offsets,
      props.settings,
      props.prominence,
      props.refinedIndex,
      props.refinedValue,
      props.peakMask,
      props.converged,
      props.output?.ids,
      props.output?.count,
      props.output?.overflow,
      props.output?.totalCount
    ]);
    const sampleCount = props.values.length;
    const profileCount = props.offsets.length - 1;
    const candidateProminence = createTransientView(
      graph,
      `${id}-candidate-prominence`,
      'float32',
      sampleCount
    );
    const candidateDelta = createTransientView(
      graph,
      `${id}-candidate-delta`,
      'float32',
      sampleCount
    );
    const candidateValue = createTransientView(
      graph,
      `${id}-candidate-value`,
      'float32',
      sampleCount
    );
    const candidateState = createTransientView(
      graph,
      `${id}-candidate-state`,
      'uint32',
      sampleCount
    );
    const common = getCommonDeclarations(sampleCount, profileCount, wrap);
    const nodes: GPUCommandNode<Parameters>[] = [];

    // Phase 1: candidates, prominence and refinement.
    const candidateBindings: WGSLKernelBinding[] = [
      {name: 'sampleValues', view: props.values, type: 'f32', access: 'read'}
    ];
    if (props.validity) {
      candidateBindings.push({
        name: 'sampleValidity',
        view: props.validity,
        type: 'u32',
        access: 'read'
      });
    }
    candidateBindings.push(
      {name: 'profileOffsets', view: props.offsets, type: 'u32', access: 'read'},
      {name: 'settings', view: props.settings, type: 'f32', access: 'read'},
      {name: 'candidateProminence', view: candidateProminence, type: 'f32', access: 'read_write'},
      {name: 'candidateDelta', view: candidateDelta, type: 'f32', access: 'read_write'},
      {name: 'candidateValue', view: candidateValue, type: 'f32', access: 'read_write'},
      {name: 'candidateState', view: candidateState, type: 'u32', access: 'read_write'}
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-candidates`,
        operation: OPERATION,
        variant: 'candidates',
        bindings: candidateBindings,
        invocationCount: sampleCount,
        declarations: `${common}
${TERRAIN_WGSL_HELPERS}
const WINDOW: u32 = ${props.window}u;
const MIN_SIDE: u32 = ${minSide}u;
const STATE_INERT: u32 = ${STATE_INERT}u;
const STATE_UNDECIDED: u32 = ${STATE_UNDECIDED}u;
${getNanFunction()}
fn isGap(sample: u32) -> bool {
  ${
    props.validity
      ? 'if (sampleValidity[sampleValidityOffset + sample] == 0u) { return true; }'
      : ''
  }
  return !isFiniteValue(sampleValues[sampleValuesOffset + sample]);
}
fn writeInert(sample: u32) {
  candidateProminence[candidateProminenceOffset + sample] = getNan();
  candidateDelta[candidateDeltaOffset + sample] = 0.0;
  candidateValue[candidateValueOffset + sample] = getNan();
  candidateState[candidateStateOffset + sample] = STATE_INERT;
}
struct Side {
  lowest: f32,
  valid: u32,
}
// Walks away from a candidate; the first higher sample is counted as valid but not as lowest.
fn walkSide(start: u32, count: u32, local: u32, peak: f32, direction: i32) -> Side {
  var side = Side(peak, 0u);
  var limit = WINDOW;
  if (WRAP) { limit = min(WINDOW, count - 1u); }
  for (var step = 1u; step <= limit; step++) {
    let neighbor = neighborLocal(local, direction * i32(step), count);
    if (neighbor < 0) { break; }
    let sample = start + u32(neighbor);
    if (isGap(sample)) { break; }
    side.valid = side.valid + 1u;
    let other = sampleValues[sampleValuesOffset + sample];
    if (other > peak) { break; }
    if (other < side.lowest) { side.lowest = other; }
  }
  return side;
}`,
        body: `let range = findProfile(index);
  let count = range.y - range.x;
  if (count == 0u) { writeInert(index); return; }
  let start = range.x;
  let local = index - start;
  if (!WRAP && (local < 1u || local + 1u >= count)) { writeInert(index); return; }
  if (isGap(index)) { writeInert(index); return; }
  let peak = sampleValues[sampleValuesOffset + index];
  for (var offset = -2; offset <= 2; offset++) {
    if (offset == 0) { continue; }
    let neighbor = neighborLocal(local, offset, count);
    if (neighbor < 0) { continue; }
    let sample = start + u32(neighbor);
    if (sample == index || isGap(sample)) { continue; }
    let other = sampleValues[sampleValuesOffset + sample];
    if (other > peak || (offset < 0 && other == peak)) { writeInert(index); return; }
  }
  let left = walkSide(start, count, local, peak, -1);
  let right = walkSide(start, count, local, peak, 1);
  if (left.valid < MIN_SIDE || right.valid < MIN_SIDE) { writeInert(index); return; }
  let prominence = peak - max(left.lowest, right.lowest);
  if (prominence < settings[settingsOffset]) { writeInert(index); return; }
  var delta = 0.0;
  var refined = peak;
  let previous = start + u32(neighborLocal(local, -1, count));
  let next = start + u32(neighborLocal(local, 1, count));
  if (!isGap(previous) && !isGap(next)) {
    let a = sampleValues[sampleValuesOffset + previous];
    let c = sampleValues[sampleValuesOffset + next];
    let denominator = a - 2.0 * peak + c;
    if (denominator < 0.0) {
      delta = clamp(0.5 * (a - c) / denominator, -0.5, 0.5);
      refined = peak - 0.25 * (a - c) * delta;
    }
  }
  candidateProminence[candidateProminenceOffset + index] = prominence;
  candidateDelta[candidateDeltaOffset + index] = delta;
  candidateValue[candidateValueOffset + index] = refined;
  candidateState[candidateStateOffset + index] = STATE_UNDECIDED;`
      })
    );

    // Phase 2: suppression rounds.
    const scanRadius = Math.floor(nms) + 1;
    for (let round = 0; round < nmsRounds; round++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-nms-${round}`,
          operation: OPERATION,
          variant: 'suppression',
          bindings: [
            {name: 'profileOffsets', view: props.offsets, type: 'u32', access: 'read'},
            {
              name: 'candidateProminence',
              view: candidateProminence,
              type: 'f32',
              access: 'read'
            },
            {name: 'candidateDelta', view: candidateDelta, type: 'f32', access: 'read'},
            {
              name: 'candidateState',
              view: candidateState,
              type: 'atomic<u32>',
              access: 'read_write'
            }
          ],
          invocationCount: sampleCount,
          declarations: `${common}
const NMS_RADIUS: f32 = ${getWGSLFloatLiteral(nms)};
const SCAN_RADIUS: u32 = ${scanRadius}u;
const STATE_UNDECIDED: u32 = ${STATE_UNDECIDED}u;
const STATE_KEPT: u32 = ${STATE_KEPT}u;
const STATE_SUPPRESSED: u32 = ${STATE_SUPPRESSED}u;`,
          body: `if (atomicLoad(&candidateState[candidateStateOffset + index]) != STATE_UNDECIDED) { return; }
  let range = findProfile(index);
  let start = range.x;
  let count = range.y - range.x;
  let local = index - start;
  let prominenceSelf = candidateProminence[candidateProminenceOffset + index];
  let deltaSelf = candidateDelta[candidateDeltaOffset + index];
  var undecided = false;
  for (var step = 1u; step <= SCAN_RADIUS; step++) {
    for (var side = 0; side < 2; side++) {
      let direction = select(-1, 1, side == 1);
      let neighbor = neighborLocal(local, direction * i32(step), count);
      if (neighbor < 0 || u32(neighbor) == local) { continue; }
      let sample = start + u32(neighbor);
      let state = atomicLoad(&candidateState[candidateStateOffset + sample]);
      if (state != STATE_UNDECIDED && state != STATE_KEPT) { continue; }
      let prominenceOther = candidateProminence[candidateProminenceOffset + sample];
      let higher = prominenceOther > prominenceSelf ||
        (prominenceOther == prominenceSelf && u32(neighbor) < local);
      if (!higher) { continue; }
      var distance = abs(f32(neighbor - i32(local)) +
        (candidateDelta[candidateDeltaOffset + sample] - deltaSelf));
      if (WRAP) { distance = min(distance, f32(count) - distance); }
      if (distance > NMS_RADIUS) { continue; }
      if (state == STATE_KEPT) {
        atomicStore(&candidateState[candidateStateOffset + index], STATE_SUPPRESSED);
        return;
      }
      undecided = true;
    }
  }
  if (!undecided) {
    atomicStore(&candidateState[candidateStateOffset + index], STATE_KEPT);
  }`
        })
      );
    }

    // Phase 3: convergence flag.
    if (props.converged) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-converged-reset`,
          operation: OPERATION,
          view: props.converged,
          type: 'u32',
          value: '1u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-converged`,
          operation: OPERATION,
          variant: 'converged',
          bindings: [
            {name: 'candidateState', view: candidateState, type: 'u32', access: 'read'},
            {name: 'convergedOut', view: props.converged, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: sampleCount,
          body: `if (candidateState[candidateStateOffset + index] == ${STATE_UNDECIDED}u) {
    atomicStore(&convergedOut[convergedOutOffset], 0u);
  }`
        })
      );
    }

    // Phase 4: outputs.
    const needsMask = Boolean(props.peakMask || props.output);
    const peakMask =
      props.peakMask ??
      (props.output
        ? createTransientView(graph, `${id}-peak-mask`, 'uint32', sampleCount)
        : undefined);
    const rowIds = props.output
      ? createTransientView(graph, `${id}-row-ids`, 'uint32', sampleCount)
      : undefined;
    if (props.prominence || props.refinedValue || needsMask) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'candidateState', view: candidateState, type: 'u32', access: 'read'}
      ];
      if (props.prominence) {
        bindings.push(
          {name: 'candidateProminence', view: candidateProminence, type: 'f32', access: 'read'},
          {name: 'prominenceOut', view: props.prominence, type: 'f32', access: 'read_write'}
        );
      }
      if (props.refinedValue) {
        bindings.push(
          {name: 'candidateValue', view: candidateValue, type: 'f32', access: 'read'},
          {name: 'refinedValueOut', view: props.refinedValue, type: 'f32', access: 'read_write'}
        );
      }
      if (peakMask) {
        bindings.push({name: 'peakMaskOut', view: peakMask, type: 'u32', access: 'read_write'});
      }
      if (rowIds) {
        bindings.push({name: 'rowIdsOut', view: rowIds, type: 'u32', access: 'read_write'});
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-outputs`,
          operation: OPERATION,
          variant: 'outputs',
          bindings,
          invocationCount: sampleCount,
          declarations: getNanFunction(),
          body: `let kept = candidateState[candidateStateOffset + index] == ${STATE_KEPT}u;
  ${
    props.prominence
      ? `prominenceOut[prominenceOutOffset + index] = select(getNan(), candidateProminence[candidateProminenceOffset + index], kept);`
      : ''
  }
  ${
    props.refinedValue
      ? `refinedValueOut[refinedValueOutOffset + index] = select(getNan(), candidateValue[candidateValueOffset + index], kept);`
      : ''
  }
  ${peakMask ? 'peakMaskOut[peakMaskOutOffset + index] = select(0u, 1u, kept);' : ''}
  ${rowIds ? 'rowIdsOut[rowIdsOutOffset + index] = index;' : ''}`
        })
      );
    }
    if (props.refinedIndex) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-refined-index`,
          operation: OPERATION,
          variant: 'refined-index',
          bindings: [
            {name: 'profileOffsets', view: props.offsets, type: 'u32', access: 'read'},
            {name: 'candidateDelta', view: candidateDelta, type: 'f32', access: 'read'},
            {name: 'candidateState', view: candidateState, type: 'u32', access: 'read'},
            {name: 'refinedIndexOut', view: props.refinedIndex, type: 'f32', access: 'read_write'}
          ],
          invocationCount: sampleCount,
          declarations: `${common}
${getNanFunction()}`,
          body: `if (candidateState[candidateStateOffset + index] != ${STATE_KEPT}u) {
    refinedIndexOut[refinedIndexOutOffset + index] = getNan();
    return;
  }
  let range = findProfile(index);
  let count = f32(range.y - range.x);
  var refined = f32(index - range.x) + candidateDelta[candidateDeltaOffset + index];
  if (WRAP) {
    if (refined < 0.0) { refined = refined + count; }
    if (refined >= count) { refined = refined - count; }
  }
  refinedIndexOut[refinedIndexOutOffset + index] = refined;`
        })
      );
    }
    if (props.output && peakMask && rowIds) {
      const compactRows = createTransientView(graph, `${id}-compact-rows`, 'uint32', sampleCount);
      const totalCount = createTransientView(graph, `${id}-peak-total`, 'uint32', 1);
      nodes.push(
        ...new GPUCompaction({
          id: `${id}-compaction`,
          input: rowIds,
          flags: peakMask,
          output: compactRows,
          count: totalCount
        }).getCommandNodes(graph),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          totalCount,
          compactIds: compactRows,
          output: props.output
        })
      );
    }
    return nodes;
  }
}

/** WGSL for the canonical NaN, built from a variable so it is not constant-evaluated. */
function getNanFunction(): string {
  return `fn getNan() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}`;
}

/** Shared WGSL: profile lookup by binary search over CSR offsets and local neighbour indexing. */
function getCommonDeclarations(sampleCount: number, profileCount: number, wrap: boolean): string {
  return `const SAMPLE_COUNT: u32 = ${sampleCount}u;
const PROFILE_COUNT: u32 = ${profileCount}u;
const WRAP: bool = ${wrap};
// Returns [start, end) of the profile owning \`sample\`, or [0, 0) when it belongs to none.
fn findProfile(sample: u32) -> vec2<u32> {
  var low = 0u;
  var high = PROFILE_COUNT;
  for (var step = 0u; step < 32u && low < high; step++) {
    let middle = low + (high - low) / 2u;
    if (profileOffsets[profileOffsetsOffset + middle + 1u] > sample) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  if (low >= PROFILE_COUNT) { return vec2<u32>(0u, 0u); }
  let start = profileOffsets[profileOffsetsOffset + low];
  let end = min(profileOffsets[profileOffsetsOffset + low + 1u], SAMPLE_COUNT);
  if (start > sample || sample >= end) { return vec2<u32>(0u, 0u); }
  return vec2<u32>(start, end);
}
// Local index of the neighbour \`delta\` samples away, or -1 when it is outside a non-wrapped profile.
fn neighborLocal(local: u32, delta: i32, count: u32) -> i32 {
  let position = i32(local) + delta;
  let signedCount = i32(count);
  if (WRAP) { return ((position % signedCount) + signedCount) % signedCount; }
  if (position < 0 || position >= signedCount) { return -1; }
  return position;
}`;
}
