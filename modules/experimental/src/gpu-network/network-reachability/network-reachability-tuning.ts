// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Smallest compile-time round count. */
const MINIMUM_ITERATIONS = 1;
/** Largest compile-time round count, matching `GPU_NETWORK_REACHABILITY_MAXIMUM_ITERATIONS`. */
const MAXIMUM_ITERATIONS = 1024;
/** Fraction of `localIterations` hops one round covers in practice (13.6 of 16, 28 of 32 on a grid). */
const HOPS_PER_ROUND_EFFICIENCY = 0.7;
/** Extra rounds for the first and last partial rounds and ragged wavefronts. */
const ROUND_SLACK = 4;
/** Headroom kept above the rounds a converged run used. */
const CONVERGED_MARGIN = 1.5;
/** Shrink only when the budget is at least this multiple of the rounds used, to avoid recompiles. */
const SHRINK_RATIO = 4;

/** Inputs of {@link recommendReachabilityIterations}. */
export type ReachabilityIterationsRecommendationProps = {
  /** Number of nodes in the network. */
  nodeCount: number;
  /**
   * Longest shortest path, in hops, from any source. Defaults to `sqrt(nodeCount)`, the measured
   * value for grid-like planar road networks (224 for 50k nodes, 700 for 490k). Random graphs are
   * far lower (about 12) and real road networks can be higher, so pass a measured value when known.
   */
  hopEccentricity?: number;
  /** The `localIterations` the contributor will be compiled with. */
  localIterations: number;
};

/** Result of {@link recommendReachabilityIterations}. */
export type ReachabilityIterationsRecommendation = {
  /** Echo of the input `localIterations`. */
  localIterations: number;
  /** Suggested `maxIterations` (compile-time round count), clamped to [1, 1024]. */
  maxIterations: number;
};

/** Inputs of {@link adaptReachabilityIterations}. */
export type ReachabilityIterationsAdaptationProps = {
  /** The `maxIterations` the last result was computed with. */
  maxIterations: number;
  /** The `iterationCount` output read back from that run. */
  iterationCount: number;
  /** Whether that run reported `converged`. */
  converged: boolean;
};

function clampIterations(value: number): number {
  return Math.min(MAXIMUM_ITERATIONS, Math.max(MINIMUM_ITERATIONS, Math.ceil(value)));
}

/**
 * Suggests a first-guess `maxIterations` from the network size, without a host-side probe.
 * Rounds needed are about `ceil(hopEccentricity / (0.7 * localIterations)) + 4`. Pure function.
 */
export function recommendReachabilityIterations(
  props: ReachabilityIterationsRecommendationProps
): ReachabilityIterationsRecommendation {
  const {nodeCount, localIterations} = props;
  if (!Number.isFinite(nodeCount) || nodeCount < 0) {
    throw new Error('recommendReachabilityIterations nodeCount must be a non-negative number');
  }
  if (!Number.isFinite(localIterations) || localIterations < 1) {
    throw new Error('recommendReachabilityIterations localIterations must be at least 1');
  }
  const hopEccentricity = props.hopEccentricity ?? Math.sqrt(nodeCount);
  if (!Number.isFinite(hopEccentricity) || hopEccentricity < 0) {
    throw new Error('recommendReachabilityIterations hopEccentricity must be non-negative');
  }
  const rounds = Math.ceil(hopEccentricity / (HOPS_PER_ROUND_EFFICIENCY * localIterations));
  return {localIterations, maxIterations: clampIterations(rounds + ROUND_SLACK)};
}

/**
 * Next `maxIterations` from the previous run's `iterationCount` and `converged` outputs.
 * Not converged: doubles the budget (at most 1024). Converged: shrinks to
 * `ceil(iterationCount * 1.5) + 2` once the budget is at least 4x the rounds used, otherwise keeps
 * it. Pure function; the caller owns the frame-to-frame value and decides when to recompile.
 */
export function adaptReachabilityIterations(props: ReachabilityIterationsAdaptationProps): number {
  const {maxIterations, iterationCount, converged} = props;
  const current = clampIterations(maxIterations);
  if (!converged) {
    return clampIterations(current * 2);
  }
  if (iterationCount * SHRINK_RATIO < current) {
    return clampIterations(Math.ceil(iterationCount * CONVERGED_MARGIN) + 2);
  }
  return current;
}
