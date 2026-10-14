// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
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
import {
  createRasterGatedLoopNodes,
  createRasterIterationResetNode,
  createRasterIterationState
} from '../../gpu-raster/cost-distance/raster-relaxation';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateCapacity,
  validateCompactOutput,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {RASTER_WGSL_HELPERS, validateParameterTarget} from '../../gpu-raster/raster-band-utils';

const OPERATION = 'GPUProfilePeaks';

/** Number of float32 values read from `GPUProfilePeaksProps.parameters`. */
export const GPU_PROFILE_PEAKS_PARAMETER_LENGTH = 1;

/** Default compile-time cap on non-maximum suppression rounds. */
export const GPU_PROFILE_PEAKS_DEFAULT_MAXIMUM_SUPPRESSION_ITERATIONS = 16;

/**
 * Largest accepted `window` and (rounded up) `suppressionRadius`, in samples. Bounds the
 * per-sample loops.
 */
export const GPU_PROFILE_PEAKS_MAXIMUM_RADIUS = 4096;

/** Largest accepted `maximumSuppressionIterations`. */
export const GPU_PROFILE_PEAKS_SUPPRESSION_ITERATION_LIMIT = 1024;

/**
 * Word written to `output.peakMask` at a candidate that suppression left undecided
 * (`maximumSuppressionIterations` rounds were not enough). A converged run only writes 0 or 1.
 */
export const GPU_PROFILE_PEAKS_UNRESOLVED_MASK = 0xffffffff;

/**
 * Value written to `output.prominence`, `output.refinedIndex` and `output.refinedValue` at a
 * candidate that suppression
 * left undecided: `+Infinity`. A converged run writes finite values at kept peaks and NaN
 * elsewhere, and a prominence or refined value is a difference of finite samples, so it never
 * produces `+Infinity`. Compare with `value === Infinity`, not `Number.isNaN`.
 */
export const GPU_PROFILE_PEAKS_UNRESOLVED_VALUE = Number.POSITIVE_INFINITY;

/**
 * Bit set in `output.peaks.overflow` when suppression left candidates undecided, so the compact
 * list of peaks is incomplete. Bit 0 (value 1) keeps its meaning, a full `output.peaks.ids`; the
 * flag is the OR of both and reads 0 only for a complete result. A converged run never sets this
 * bit.
 */
export const GPU_PROFILE_PEAKS_OVERFLOW_UNCONVERGED = 2;

/**
 * CPU-side description packed by {@link getGPUProfilePeaksParameterValues}.
 *
 * Values are in profile-value units (for example metres for an elevation profile, degrees for a
 * horizon). The model is cell-size independent: every other distance of the contributor is measured in
 * profile samples.
 */
export type GPUProfilePeaksSettings = {
  /**
   * Minimum prominence, in profile-value units. A candidate with
   * `prominence < minimumProminence` is rejected. Must not be NaN; a value at or below zero accepts
   * every candidate.
   */
  minimumProminence: number;
};

/**
 * Packs profile-peak settings into `[minimumProminence]`.
 *
 * @throws If `minimumProminence` is NaN or `target` is too short.
 */
export function getGPUProfilePeaksParameterValues(
  settings: GPUProfilePeaksSettings,
  target: Float32Array = new Float32Array(GPU_PROFILE_PEAKS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPUProfilePeaksParameterValues',
    target,
    GPU_PROFILE_PEAKS_PARAMETER_LENGTH
  );
  if (Number.isNaN(settings.minimumProminence)) {
    throw new Error('getGPUProfilePeaksParameterValues minimumProminence must not be NaN');
  }
  target[0] = settings.minimumProminence;
  return target;
}

/** Status words of {@link GPUProfilePeaks}. */
export type GPUProfilePeaksStatus = {
  /**
   * One-row scalar: 1 when suppression converged within `maximumSuppressionIterations` rounds,
   * otherwise 0. Optional because the data carries the answer: when 0, undecided candidates are
   * marked with the `GPU_PROFILE_PEAKS_UNRESOLVED_*` sentinels and the overflow bit, never reported
   * as kept or as absent.
   */
  converged?: GraphDataView<'uint32'>;
};

/** Caller-owned outputs of {@link GPUProfilePeaks}; at least one is required. */
export type GPUProfilePeaksOutput = {
  /**
   * Per-sample prominence of kept peaks, NaN elsewhere, and
   * {@link GPU_PROFILE_PEAKS_UNRESOLVED_VALUE} at undecided candidates.
   */
  prominence?: GraphDataView<'float32'>;
  /**
   * Per-sample refined fractional index of kept peaks inside their own profile (`0` is the
   * profile's first sample), NaN elsewhere, {@link GPU_PROFILE_PEAKS_UNRESOLVED_VALUE} at
   * undecided candidates.
   */
  refinedIndex?: GraphDataView<'float32'>;
  /**
   * Per-sample parabola-refined value of kept peaks, NaN elsewhere,
   * {@link GPU_PROFILE_PEAKS_UNRESOLVED_VALUE} at undecided candidates.
   */
  refinedValue?: GraphDataView<'float32'>;
  /**
   * Per-sample mask: 1 on a kept peak's sample row, otherwise 0, and
   * {@link GPU_PROFILE_PEAKS_UNRESOLVED_MASK} at undecided candidates.
   */
  peakMask?: GraphDataView<'uint32'>;
  /**
   * Compact result: sample row IDs (indices into `values`) of kept peaks in ascending order, with
   * a clamped count and an overflow flag. Undecided candidates are not listed; `overflow` then has
   * {@link GPU_PROFILE_PEAKS_OVERFLOW_UNCONVERGED} set.
   */
  peaks?: GPUCompactOutput;
  /** Optional status words. */
  status?: GPUProfilePeaksStatus;
};

/**
 * Properties for {@link GPUProfilePeaks}.
 *
 * Topology (compile time): `window`, `minimumSide`, `suppressionRadius`, `wrap`,
 * `maximumSuppressionIterations`, which views exist and their lengths. Per-frame: `parameters`
 * and the contents of `values`, `validity` and `offsets`.
 * Every distance is in profile samples and every value in profile-value units, so the contributor is
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
   * `GPURasterProfile.lineSampleOffsets`.
   */
  offsets: GraphDataView<'uint32'>;
  /** Per-frame parameters with at least 1 float32 value, see {@link getGPUProfilePeaksParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /** Half-window, in samples, of the prominence walk. Integer in `[1, 4096]`. */
  window: number;
  /**
   * Minimum number of valid samples counted on each side of a candidate. Integer `>= 0`. Default 4.
   */
  minimumSide?: number;
  /**
   * Non-maximum suppression radius in samples, measured between refined (fractional) positions.
   * Finite, in `[0, 4096]`. Default `max(2, round(window / 4))`.
   */
  suppressionRadius?: number;
  /**
   * Treat each profile as circular (a 360 degree horizon). Default false. See
   * {@link GPUProfilePeaks} for the exact semantics.
   */
  wrap?: boolean;
  /** Compile-time cap on suppression rounds, in `[0, 1024]`. Default 16. */
  maximumSuppressionIterations?: number;
  /** Caller-owned outputs. */
  output: GPUProfilePeaksOutput;
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
 * `sampleValues` and `lineSampleOffsets` of `GPURasterProfile`, or the apparent-horizon elevation
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
 *   `valid < minimumSide`. `prominence = v - max(loLeft, loRight)`; reject if below
 *   `minimumProminence`.
 *   The prominence is one f32 subtraction of two f32 values, so a CPU oracle using `Math.fround`
 *   reproduces it bit for bit.
 * - **Refinement.** With both neighbours valid, `den = a - 2v + c`; if `den < 0`,
 *   `di = clamp(0.5 (a - c) / den, -0.5, 0.5)` and `value = v - 0.25 (a - c) di`, otherwise
 *   `di = 0` and `value = v`. The refined index is `i + di`. The division is not correctly rounded
 *   on GPUs, so expect a few ULP against a float64 oracle.
 * - **Suppression.** Candidates are ranked by prominence descending, ties by ascending sample
 *   index. A candidate is kept iff no higher-ranked KEPT candidate of the same profile lies within
 *   `suppressionRadius` samples (`|refined position difference| <= suppressionRadius` suppresses; strictly greater keeps).
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
 * fixed point of: `p` is kept iff no kept candidate of higher rank lies within `suppressionRadius`. The GPU
 * computes that fixed point in rounds over a per-candidate state (undecided, kept, suppressed). In
 * a round an undecided `p` becomes suppressed if any higher-ranked candidate within `suppressionRadius` is kept,
 * becomes kept if every higher-ranked candidate within `suppressionRadius` is suppressed, and otherwise stays
 * undecided. Equivalence with greedy, by induction on rank: assume every candidate of rank below
 * `r` has the greedy decision whenever it is decided. A candidate `p` of rank `r` has greedy
 * decision "kept" iff all higher-ranked candidates within `suppressionRadius` were rejected by greedy, and
 * "suppressed" iff one was kept; the round rule applies exactly these two tests to decided
 * higher-ranked neighbours, so it only ever decides `p` in agreement with greedy, and it decides
 * `p` as soon as the highest-ranked relevant neighbours are decided (the top-ranked candidate of
 * a profile has no higher neighbour and is kept in round 1). Decisions are monotone and never
 * revisited, and a decided state is final truth, so racing in-place updates are safe: a reader
 * sees either the old undecided state (and merely waits one more round) or the final decision.
 * State words are atomics, so no torn or speculative value is observable. Candidates within `suppressionRadius`
 * in refined position lie within `floor(suppressionRadius) + 1` raw samples because `|di| <= 0.5`, which bounds
 * the scan. If `maximumSuppressionIterations` rounds do not decide every candidate (a chain of mutually suppressing
 * candidates longer than the cap), `output.status.converged` reads 0, undecided candidates carry the unresolved sentinels (see
 * {@link GPU_PROFILE_PEAKS_UNRESOLVED_VALUE}, {@link GPU_PROFILE_PEAKS_UNRESOLVED_MASK}) and the
 * compact list sets {@link GPU_PROFILE_PEAKS_OVERFLOW_UNCONVERGED}.
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
  readonly minimumSide: number;
  /** Non-maximum suppression radius in samples. */
  readonly suppressionRadius: number;
  /** Whether profiles are circular. */
  readonly wrap: boolean;
  /** Compile-time cap on suppression rounds. */
  readonly maximumSuppressionIterations: number;

  constructor(props: GPUProfilePeaksProps) {
    this.id = props.id ?? 'profile-peaks';
    this.props = props;
    this.minimumSide = props.minimumSide ?? 4;
    this.suppressionRadius = props.suppressionRadius ?? Math.max(2, Math.round(props.window / 4));
    this.wrap = props.wrap ?? false;
    this.maximumSuppressionIterations =
      props.maximumSuppressionIterations ??
      GPU_PROFILE_PEAKS_DEFAULT_MAXIMUM_SUPPRESSION_ITERATIONS;
    const {id} = this;
    const {output} = props;
    const converged = output.status?.converged;
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
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_PROFILE_PEAKS_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must contain at least 1 value`);
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
    validateCapacity(id, 'minimumSide', this.minimumSide, 0, Number.MAX_SAFE_INTEGER);
    if (
      !Number.isFinite(this.suppressionRadius) ||
      this.suppressionRadius < 0 ||
      this.suppressionRadius > GPU_PROFILE_PEAKS_MAXIMUM_RADIUS
    ) {
      throw new Error(
        `${id} suppressionRadius must be in [0, ${GPU_PROFILE_PEAKS_MAXIMUM_RADIUS}]`
      );
    }
    validateCapacity(
      id,
      'maximumSuppressionIterations',
      this.maximumSuppressionIterations,
      0,
      GPU_PROFILE_PEAKS_SUPPRESSION_ITERATION_LIMIT
    );
    if (
      !output.prominence &&
      !output.refinedIndex &&
      !output.refinedValue &&
      !output.peakMask &&
      !output.peaks &&
      !converged
    ) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, view, format] of [
      ['prominence', output.prominence, 'float32'],
      ['refinedIndex', output.refinedIndex, 'float32'],
      ['refinedValue', output.refinedValue, 'float32'],
      ['peakMask', output.peakMask, 'uint32']
    ] as const) {
      if (!view) {
        continue;
      }
      validatePackedView(view, [format], `${id} output.${name}`);
      if (view.length !== sampleCount) {
        throw new Error(`${id} output.${name} must contain one value per sample`);
      }
    }
    if (output.peaks) {
      validateCompactOutput(id, output.peaks);
    }
    if (converged) {
      validatePackedView(converged, ['uint32'], `${id} output.status.converged`);
      if (converged.length < 1) {
        throw new Error(`${id} output.status.converged must contain one row`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.prominence,
        output.refinedIndex,
        output.refinedValue,
        output.peakMask,
        converged,
        output.peaks?.ids,
        output.peaks?.status.count,
        output.peaks?.status.overflow,
        output.peaks?.status.requiredCount
      ],
      [props.values, props.validity, props.offsets, props.parameters]
    );
  }

  /** Returns candidate, suppression, output and optional compaction nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, minimumSide, suppressionRadius, wrap, maximumSuppressionIterations} = this;
    const {output} = props;
    const converged = output.status?.converged;
    validateGraphViewsBelongToGraph(id, graph, [
      props.values,
      props.validity,
      props.offsets,
      props.parameters,
      output.prominence,
      output.refinedIndex,
      output.refinedValue,
      output.peakMask,
      converged,
      output.peaks?.ids,
      output.peaks?.status.count,
      output.peaks?.status.overflow,
      output.peaks?.status.requiredCount
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
      {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
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
${RASTER_WGSL_HELPERS}
const WINDOW: u32 = ${props.window}u;
const MIN_SIDE: u32 = ${minimumSide}u;
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

    // Phase 2: suppression rounds. The rounds are GPU-gated: once a round decides no candidate the
    // remaining unrolled rounds are skipped by their indirect dispatch instead of re-scanning all
    // samples, so dense profiles pay for the rounds they need rather than for `maximumSuppressionIterations`.
    const scanRadius = Math.floor(suppressionRadius) + 1;
    if (maximumSuppressionIterations > 0) {
      const suppression = createRasterIterationState(graph, `${id}-nms`, OPERATION, sampleCount);
      nodes.push(
        createRasterIterationResetNode<Parameters>(graph, {
          id: `${id}-nms-reset`,
          operation: OPERATION,
          state: suppression
        }),
        ...createRasterGatedLoopNodes<Parameters>(graph, {
          id: `${id}-nms`,
          gateId: `${id}-nms-gate`,
          operation: OPERATION,
          state: suppression,
          maxIterations: maximumSuppressionIterations,
          createRound: ({nodeId, slotBinding, markChangedWGSL, condition, extraResources}) =>
            createWGSLKernelNode<Parameters>(graph, {
              id: nodeId,
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
                },
                {
                  name: 'status',
                  view: suppression.status,
                  type: 'atomic<u32>',
                  access: 'read_write'
                },
                slotBinding
              ],
              invocationCount: sampleCount,
              condition,
              extraResources,
              declarations: `${common}
const NMS_RADIUS: f32 = ${getWGSLFloatLiteral(suppressionRadius)};
const SCAN_RADIUS: u32 = ${scanRadius}u;
const STATE_UNDECIDED: u32 = ${STATE_UNDECIDED}u;
const STATE_KEPT: u32 = ${STATE_KEPT}u;
const STATE_SUPPRESSED: u32 = ${STATE_SUPPRESSED}u;`,
              // Every decision marks the round as changed; the gate stops after a round without one.
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
        ${markChangedWGSL}
        return;
      }
      undecided = true;
    }
  }
  if (!undecided) {
    atomicStore(&candidateState[candidateStateOffset + index], STATE_KEPT);
    ${markChangedWGSL}
  }`
            })
        })
      );
    }

    // Phase 3: convergence flag.
    if (converged) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-converged-reset`,
          operation: OPERATION,
          view: converged,
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
            {name: 'convergedOut', view: converged, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: sampleCount,
          body: `if (candidateState[candidateStateOffset + index] == ${STATE_UNDECIDED}u) {
    atomicStore(&convergedOut[convergedOutOffset], 0u);
  }`
        })
      );
    }

    // Phase 4: outputs. An undecided candidate (suppression hit maximumSuppressionIterations) writes the unresolved
    // sentinels, never "not a peak"; a converged run has no undecided candidate.
    // The compaction flags are separate from a caller's peakMask because that mask can hold the
    // unresolved word, which compaction would select.
    const peakMask =
      output.peakMask ??
      (output.peaks
        ? createTransientView(graph, `${id}-peak-mask`, 'uint32', sampleCount)
        : undefined);
    const compactFlags =
      output.peaks && output.peakMask
        ? createTransientView(graph, `${id}-peak-flags`, 'uint32', sampleCount)
        : peakMask;
    const rowIds = output.peaks
      ? createTransientView(graph, `${id}-row-ids`, 'uint32', sampleCount)
      : undefined;
    if (output.prominence || output.refinedValue || peakMask) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'candidateState', view: candidateState, type: 'u32', access: 'read'}
      ];
      if (output.prominence) {
        bindings.push(
          {name: 'candidateProminence', view: candidateProminence, type: 'f32', access: 'read'},
          {name: 'prominenceOut', view: output.prominence, type: 'f32', access: 'read_write'}
        );
      }
      if (output.refinedValue) {
        bindings.push(
          {name: 'candidateValue', view: candidateValue, type: 'f32', access: 'read'},
          {name: 'refinedValueOut', view: output.refinedValue, type: 'f32', access: 'read_write'}
        );
      }
      if (peakMask) {
        bindings.push({name: 'peakMaskOut', view: peakMask, type: 'u32', access: 'read_write'});
      }
      if (compactFlags && compactFlags !== peakMask) {
        bindings.push({name: 'flagsOut', view: compactFlags, type: 'u32', access: 'read_write'});
      }
      if (rowIds) {
        bindings.push({name: 'rowIdsOut', view: rowIds, type: 'u32', access: 'read_write'});
      }
      // A mask that only feeds the compaction never carries the sentinel.
      const maskMarksUnresolved = Boolean(output.peakMask);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-outputs`,
          operation: OPERATION,
          variant: 'outputs',
          bindings,
          invocationCount: sampleCount,
          declarations: `${getNanFunction()}
${getInfinityFunction()}`,
          body: `let state = candidateState[candidateStateOffset + index];
  let kept = state == ${STATE_KEPT}u;
  let unresolved = state == ${STATE_UNDECIDED}u;
  ${
    output.prominence
      ? `prominenceOut[prominenceOutOffset + index] = select(select(getNan(), candidateProminence[candidateProminenceOffset + index], kept), getInfinity(), unresolved);`
      : ''
  }
  ${
    output.refinedValue
      ? `refinedValueOut[refinedValueOutOffset + index] = select(select(getNan(), candidateValue[candidateValueOffset + index], kept), getInfinity(), unresolved);`
      : ''
  }
  ${
    peakMask
      ? maskMarksUnresolved
        ? `peakMaskOut[peakMaskOutOffset + index] = select(select(0u, 1u, kept), ${GPU_PROFILE_PEAKS_UNRESOLVED_MASK}u, unresolved);`
        : 'peakMaskOut[peakMaskOutOffset + index] = select(0u, 1u, kept);'
      : ''
  }
  ${compactFlags && compactFlags !== peakMask ? 'flagsOut[flagsOutOffset + index] = select(0u, 1u, kept);' : ''}
  ${rowIds ? 'rowIdsOut[rowIdsOutOffset + index] = index;' : ''}`
        })
      );
    }
    if (output.refinedIndex) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-refined-index`,
          operation: OPERATION,
          variant: 'refined-index',
          bindings: [
            {name: 'profileOffsets', view: props.offsets, type: 'u32', access: 'read'},
            {name: 'candidateDelta', view: candidateDelta, type: 'f32', access: 'read'},
            {name: 'candidateState', view: candidateState, type: 'u32', access: 'read'},
            {name: 'refinedIndexOut', view: output.refinedIndex, type: 'f32', access: 'read_write'}
          ],
          invocationCount: sampleCount,
          declarations: `${common}
${getNanFunction()}
${getInfinityFunction()}`,
          body: `let state = candidateState[candidateStateOffset + index];
  if (state != ${STATE_KEPT}u) {
    refinedIndexOut[refinedIndexOutOffset + index] =
      select(getNan(), getInfinity(), state == ${STATE_UNDECIDED}u);
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
    if (output.peaks && compactFlags && rowIds) {
      const compactRows = createTransientView(graph, `${id}-compact-rows`, 'uint32', sampleCount);
      const requiredCount = createTransientView(graph, `${id}-peak-total`, 'uint32', 1);
      const incomplete = createTransientView(graph, `${id}-incomplete`, 'uint32', 1);
      nodes.push(
        ...new GPUCompaction({
          id: `${id}-compaction`,
          input: rowIds,
          flags: compactFlags,
          output: compactRows,
          count: requiredCount
        }).getCommandNodes(graph),
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount,
          compactIds: compactRows,
          output: output.peaks
        }),
        // Publish only knows capacity overflow; add the unconverged bit when any candidate is
        // still undecided.
        createFillNode<Parameters>(graph, {
          id: `${id}-incomplete-clear`,
          operation: OPERATION,
          view: incomplete,
          type: 'u32',
          value: '0u',
          componentCount: 1
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-incomplete`,
          operation: OPERATION,
          variant: 'incomplete',
          bindings: [
            {name: 'candidateState', view: candidateState, type: 'u32', access: 'read'},
            {name: 'incomplete', view: incomplete, type: 'atomic<u32>', access: 'read_write'}
          ],
          invocationCount: sampleCount,
          body: `if (candidateState[candidateStateOffset + index] == ${STATE_UNDECIDED}u) {
    atomicStore(&incomplete[incompleteOffset], 1u);
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-unconverged`,
          operation: OPERATION,
          variant: 'publish-unconverged',
          bindings: [
            {name: 'incomplete', view: incomplete, type: 'u32', access: 'read'},
            {
              name: 'overflowOut',
              view: output.peaks.status.overflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: `overflowOut[overflowOutOffset] = overflowOut[overflowOutOffset] |
    select(0u, ${GPU_PROFILE_PEAKS_OVERFLOW_UNCONVERGED}u, incomplete[incompleteOffset] != 0u);`
        })
      );
    }
    return nodes;
  }
}

/** WGSL for `+Infinity`, built from a variable so it is not constant-evaluated. */
function getInfinityFunction(): string {
  return `fn getInfinity() -> f32 {
  var bits = 0x7f800000u;
  return bitcast<f32>(bits);
}`;
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
