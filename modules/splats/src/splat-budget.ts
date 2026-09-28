// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Greedy budget balancing for a Gaussian splat level-of-detail cut.
 *
 * A screen-space error threshold answers "is this node good enough?". Under a hard GPU budget that
 * is the wrong question: the right one is "where does my next hundred thousand splats buy the most
 * quality?". This plans a cut by repeatedly spending on the refinement with the best quality per
 * splat until the budget runs out.
 *
 * Three rules exist only to stop the cut oscillating between frames, and each of them costs a
 * little optimality to buy stability:
 *
 * 1. Gains are bucketed on a **fixed** geometric ladder rather than a range derived from the
 *    current frame. A per-frame range makes the ordering of two unchanged nodes depend on a third
 *    node that moved.
 * 2. The search **stops at the first refinement that does not fit** instead of skipping it for a
 *    cheaper one. Skipping makes the selected set depend on the exact remaining budget, which
 *    changes every frame.
 * 3. Only a node's **immediate** children become candidates once it is refined, so one frame can
 *    never descend an entire branch on the strength of a single gain estimate.
 */

/** Whether refined pages replace their parent's Gaussians or add to them. */
export type SplatBudgetRefinement = 'replace' | 'add';

/** Minimal node shape the planner needs; the hierarchy's own nodes satisfy it structurally. */
export type SplatBudgetNode = {
  /** Stable node identity. */
  id: string;
  /** Independently loadable child pages. */
  children?: readonly SplatBudgetNode[];
  /** World-space geometric approximation error represented by this page. */
  geometricError: number;
  /** Upper-bound logical splat rows this page contributes. */
  estimatedSplatCount?: number;
  /** Whether finer children replace this page or contribute additional detail. */
  refinement?: SplatBudgetRefinement;
};

/** View-dependent measurements the planner needs for one node. */
export type SplatBudgetView = {
  /** Conservative projected geometric error in physical pixels. */
  getScreenSpaceError: (node: SplatBudgetNode) => number;
  /** Fraction of the viewport this node covers, in `[0, 1]`. Ties the gain to visible area. */
  getCoverage: (node: SplatBudgetNode) => number;
  /** Whether the node intersects the view frustum at all. */
  isVisible: (node: SplatBudgetNode) => boolean;
};

/** Controls bounding how much a plan may spend and how finely it ranks candidates. */
export type SplatBudgetPlanProps = {
  /** Maximum splats the selected cut may contain. */
  splatBudget: number;
  /** Refinement stops once a node's projected error falls below this, however much budget remains. */
  maximumScreenSpaceError?: number;
  /**
   * Ratio between adjacent gain buckets. Defaults to `2`.
   *
   * Larger values make the ordering coarser and therefore steadier; smaller values rank more
   * finely and flicker more. The ladder is absolute, never derived from the current frame.
   */
  gainBucketScale?: number;
};

/** The refinement decisions one plan reached, and what it spent reaching them. */
export type SplatBudgetPlan = {
  /** Nodes whose children should be traversed. Every other visible node terminates the cut. */
  refinedNodeIds: ReadonlySet<string>;
  /** Estimated splat rows the resulting cut contains. */
  selectedSplatCount: number;
  /** Nodes terminating the selected cut. */
  selectedNodeCount: number;
  /** Refinements evaluated before the search stopped. */
  consideredUpgradeCount: number;
  /** Whether the search stopped because the next refinement did not fit. */
  budgetExhausted: boolean;
};

/** Smallest gain distinguishable on the fixed ladder, so a zero gain cannot take a bucket. */
const MINIMUM_BUDGET_GAIN = 1e-9;

type SplatBudgetUpgrade = {
  node: SplatBudgetNode;
  children: readonly SplatBudgetNode[];
  /** Additional splats the refinement costs, at least zero. */
  cost: number;
  /** Coverage-weighted error removed per additional splat. */
  gain: number;
  /** Fixed-ladder bucket of `gain`, used as the primary ordering key. */
  bucket: number;
};

/**
 * Plans which nodes to refine so the resulting cut fits a splat budget.
 *
 * @returns The refinement decisions; an empty set means render the roots and nothing finer.
 */
export function planSplatBudget(
  roots: readonly SplatBudgetNode[],
  view: SplatBudgetView,
  props: SplatBudgetPlanProps
): SplatBudgetPlan {
  const splatBudget = Math.max(props.splatBudget, 0);
  const maximumScreenSpaceError = Math.max(props.maximumScreenSpaceError ?? 0, 0);
  const gainBucketScale = Math.max(props.gainBucketScale ?? 2, 1 + Number.EPSILON);
  const logBucketScale = Math.log(gainBucketScale);

  const refinedNodeIds = new Set<string>();
  const cut: SplatBudgetNode[] = [];
  let selectedSplatCount = 0;
  const upgrades: SplatBudgetUpgrade[] = [];

  const admit = (node: SplatBudgetNode): void => {
    if (!view.isVisible(node)) {
      return;
    }
    cut.push(node);
    selectedSplatCount += getBudgetNodeSplatCount(node);
    const upgrade = getSplatBudgetUpgrade(node, view, maximumScreenSpaceError, logBucketScale);
    if (upgrade) {
      upgrades.push(upgrade);
    }
  };

  for (const root of roots) {
    admit(root);
  }

  let consideredUpgradeCount = 0;
  let budgetExhausted = false;
  while (upgrades.length > 0) {
    // Rule 1: rank on the fixed ladder, breaking ties by identity so the order is deterministic.
    let bestIndex = 0;
    for (let index = 1; index < upgrades.length; index++) {
      if (compareSplatBudgetUpgrades(upgrades[index], upgrades[bestIndex]) < 0) {
        bestIndex = index;
      }
    }
    const best = upgrades[bestIndex];
    upgrades.splice(bestIndex, 1);
    consideredUpgradeCount++;

    // Rule 2: stop at the first refinement that does not fit rather than looking for a cheaper one.
    if (selectedSplatCount + best.cost > splatBudget) {
      budgetExhausted = true;
      break;
    }

    refinedNodeIds.add(best.node.id);
    if (best.node.refinement !== 'add') {
      selectedSplatCount -= getBudgetNodeSplatCount(best.node);
      const cutIndex = cut.indexOf(best.node);
      if (cutIndex >= 0) {
        cut.splice(cutIndex, 1);
      }
    }
    // Rule 3: only the immediate children become candidates.
    for (const child of best.children) {
      admit(child);
    }
  }

  return {
    refinedNodeIds,
    selectedSplatCount,
    selectedNodeCount: cut.length,
    consideredUpgradeCount,
    budgetExhausted
  };
}

/** Splats one page contributes, defaulting to zero when the manifest does not estimate it. */
function getBudgetNodeSplatCount(node: SplatBudgetNode): number {
  return Math.max(node.estimatedSplatCount ?? 0, 0);
}

/** Builds the candidate refinement for one node, or `undefined` when refining cannot help. */
function getSplatBudgetUpgrade(
  node: SplatBudgetNode,
  view: SplatBudgetView,
  maximumScreenSpaceError: number,
  logBucketScale: number
): SplatBudgetUpgrade | undefined {
  const children = (node.children ?? []).filter(child => view.isVisible(child));
  if (children.length === 0) {
    return undefined;
  }
  const screenSpaceError = view.getScreenSpaceError(node);
  if (screenSpaceError <= maximumScreenSpaceError) {
    return undefined;
  }

  const childSplatCount = children.reduce(
    (total, child) => total + getBudgetNodeSplatCount(child),
    0
  );
  // Additive children add to their parent; replacing children stand in for it.
  const cost =
    node.refinement === 'add'
      ? childSplatCount
      : Math.max(childSplatCount - getBudgetNodeSplatCount(node), 0);
  const residualError = children.reduce(
    (worst, child) => Math.max(worst, view.getScreenSpaceError(child)),
    0
  );
  const errorRemoved = Math.max(screenSpaceError - residualError, 0);
  const coverage = Math.min(Math.max(view.getCoverage(node), 0), 1);
  const gain = (coverage * errorRemoved) / Math.max(cost, 1);

  return {
    node,
    children,
    cost,
    gain,
    bucket: getSplatBudgetGainBucket(gain, logBucketScale)
  };
}

/** Quantizes a gain onto the fixed ladder; higher buckets are better. */
function getSplatBudgetGainBucket(gain: number, logBucketScale: number): number {
  if (!(gain > MINIMUM_BUDGET_GAIN)) {
    return Number.NEGATIVE_INFINITY;
  }
  return Math.floor(Math.log(gain) / logBucketScale);
}

/** Orders refinements best-first, deterministically. */
function compareSplatBudgetUpgrades(left: SplatBudgetUpgrade, right: SplatBudgetUpgrade): number {
  if (left.bucket !== right.bucket) {
    return right.bucket - left.bucket;
  }
  if (left.cost !== right.cost) {
    return left.cost - right.cost;
  }
  return left.node.id < right.node.id ? -1 : left.node.id > right.node.id ? 1 : 0;
}

/**
 * Returns the opacity a node should fade in with as its parent crosses the refinement threshold.
 *
 * An exactly-once additive hierarchy has no refitted parent to interpolate against, so the only
 * continuous transition available is to fade a finer level in over a band of parent error rather
 * than switching it on at a single threshold. Everything this needs is already in the manifest.
 *
 * @param parentScreenSpaceError Projected error of the parent that caused this node to be selected.
 * @param maximumScreenSpaceError The refinement threshold.
 * @param fadeBand Width of the transition as a fraction of the threshold; `0` disables fading.
 * @returns An opacity multiplier in `[0, 1]`. Root-level nodes, which have no parent error, should
 * pass `Number.POSITIVE_INFINITY` and receive `1`.
 */
export function getSplatLevelFadeOpacity(
  parentScreenSpaceError: number,
  maximumScreenSpaceError: number,
  fadeBand: number
): number {
  if (!(fadeBand > 0) || !Number.isFinite(parentScreenSpaceError)) {
    return 1;
  }
  const bandWidth = Math.max(maximumScreenSpaceError * fadeBand, Number.EPSILON);
  const progress = (parentScreenSpaceError - maximumScreenSpaceError) / bandWidth;
  return Math.min(Math.max(progress, 0), 1);
}
