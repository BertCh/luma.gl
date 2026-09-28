// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSplatLevelFadeOpacity,
  planSplatBudget,
  type SplatBudgetNode,
  type SplatBudgetView
} from '../src/splat-budget';

/** A two-level tree whose two branches differ only in how much screen they cover. */
function makeTestHierarchy(): SplatBudgetNode {
  return {
    id: 'root',
    geometricError: 64,
    estimatedSplatCount: 1000,
    refinement: 'add',
    children: [
      {
        id: 'wide',
        geometricError: 32,
        estimatedSplatCount: 10_000,
        refinement: 'add',
        children: [
          {id: 'wide-a', geometricError: 1, estimatedSplatCount: 40_000},
          {id: 'wide-b', geometricError: 1, estimatedSplatCount: 40_000}
        ]
      },
      {
        id: 'narrow',
        geometricError: 32,
        estimatedSplatCount: 10_000,
        refinement: 'add',
        children: [
          {id: 'narrow-a', geometricError: 1, estimatedSplatCount: 40_000},
          {id: 'narrow-b', geometricError: 1, estimatedSplatCount: 40_000}
        ]
      }
    ]
  };
}

/** Screen-space error proportional to geometric error, with per-branch coverage. */
function makeTestView(overrides: Partial<SplatBudgetView> = {}): SplatBudgetView {
  return {
    getScreenSpaceError: node => node.geometricError,
    getCoverage: node => (node.id.startsWith('wide') ? 0.8 : 0.05),
    isVisible: () => true,
    ...overrides
  };
}

it('an unlimited budget refines everything above the error threshold', () => {
  const plan = planSplatBudget([makeTestHierarchy()], makeTestView(), {
    splatBudget: Number.MAX_SAFE_INTEGER,
    maximumScreenSpaceError: 8
  });

  expect(
    [...plan.refinedNodeIds].sort(),
    'every node whose error exceeds the threshold is refined'
  ).toEqual(['narrow', 'root', 'wide']);
  expect(plan.budgetExhausted, 'and nothing was left unspent for want of budget').toBe(false);
});

it('refinement stops at the error threshold even with budget to spare', () => {
  const plan = planSplatBudget([makeTestHierarchy()], makeTestView(), {
    splatBudget: Number.MAX_SAFE_INTEGER,
    maximumScreenSpaceError: 40
  });

  expect(
    [...plan.refinedNodeIds],
    'only the root exceeds a 40-pixel threshold, so only the root refines'
  ).toEqual(['root']);
  expect(plan.selectedSplatCount, 'the cut is the root plus both of its children').toBe(21_000);
});

it('a tight budget buys the refinement that covers the most screen', () => {
  // Both branches remove the same error for the same cost; only their coverage differs. Refining
  // one of them costs 80k splats on top of the 21k already selected, so the budget admits exactly
  // one - and it should be the one the viewer can actually see.
  const plan = planSplatBudget([makeTestHierarchy()], makeTestView(), {
    splatBudget: 110_000,
    maximumScreenSpaceError: 8
  });

  expect(plan.refinedNodeIds.has('wide'), 'the branch covering 80% of the screen is refined').toBe(
    true
  );
  expect(
    plan.refinedNodeIds.has('narrow'),
    'the branch covering 5% of it is not, because the budget ran out first'
  ).toBe(false);
  expect(plan.budgetExhausted, 'and the plan reports that it stopped for lack of budget').toBe(
    true
  );
  expect(plan.selectedSplatCount <= 110_000, 'the cut fits the budget').toBe(true);
});

it('the search stops at the first refinement that does not fit rather than skipping it', () => {
  // A cheap low-gain child sits alongside an expensive high-gain one. Skipping to the affordable
  // refinement would make the selected set depend on the exact remaining budget, which is what
  // makes a cut flicker between frames.
  const roots: SplatBudgetNode[] = [
    {
      id: 'root',
      geometricError: 64,
      estimatedSplatCount: 100,
      refinement: 'add',
      children: [
        {id: 'expensive', geometricError: 1, estimatedSplatCount: 1_000_000},
        {id: 'cheap', geometricError: 1, estimatedSplatCount: 10}
      ]
    },
    {
      id: 'other',
      geometricError: 64,
      estimatedSplatCount: 100,
      refinement: 'add',
      children: [{id: 'other-child', geometricError: 1, estimatedSplatCount: 10}]
    }
  ];
  const plan = planSplatBudget(roots, makeTestView({getCoverage: () => 0.5}), {
    splatBudget: 5_000,
    maximumScreenSpaceError: 8
  });

  expect(plan.refinedNodeIds.has('root'), 'the unaffordable refinement is not taken').toBe(false);
  expect(
    plan.consideredUpgradeCount <= 2,
    'and the search stops there instead of looking for something cheaper'
  ).toBe(true);
});

it('only a refined node’s immediate children become candidates', () => {
  const plan = planSplatBudget([makeTestHierarchy()], makeTestView(), {
    splatBudget: 21_000,
    maximumScreenSpaceError: 8
  });

  expect([...plan.refinedNodeIds], 'the root refines and its children become candidates').toEqual([
    'root'
  ]);
  expect(plan.consideredUpgradeCount, 'one taken refinement plus the first that did not fit').toBe(
    2
  );
});

it('replacing children cost only what they add over their parent', () => {
  const additive: SplatBudgetNode = {
    id: 'root',
    geometricError: 64,
    estimatedSplatCount: 1_000,
    refinement: 'add',
    children: [{id: 'child', geometricError: 1, estimatedSplatCount: 4_000}]
  };
  const replacing: SplatBudgetNode = {...additive, refinement: 'replace'};
  const view = makeTestView({getCoverage: () => 1});

  const additivePlan = planSplatBudget([additive], view, {splatBudget: 4_500});
  const replacingPlan = planSplatBudget([replacing], view, {splatBudget: 4_500});

  expect(
    additivePlan.refinedNodeIds.has('root'),
    'an additive child costs its whole splat count, which does not fit'
  ).toBe(false);
  expect(
    replacingPlan.refinedNodeIds.has('root'),
    'a replacing child costs only the difference, which does'
  ).toBe(true);
  expect(replacingPlan.selectedSplatCount, 'and the parent leaves the cut').toBe(4_000);
});

it('culled branches never enter the cut or consume budget', () => {
  const plan = planSplatBudget(
    [makeTestHierarchy()],
    makeTestView({
      isVisible: node => !node.id.startsWith('narrow')
    }),
    {
      splatBudget: Number.MAX_SAFE_INTEGER,
      maximumScreenSpaceError: 8
    }
  );

  expect(plan.refinedNodeIds.has('narrow'), 'a culled branch is not refined').toBe(false);
  expect(plan.selectedSplatCount, 'nor does it contribute splats').toBe(1_000 + 10_000 + 80_000);
});

it('the plan is deterministic across equal-gain refinements', () => {
  const view = makeTestView({getCoverage: () => 0.5});
  const first = planSplatBudget([makeTestHierarchy()], view, {
    splatBudget: 110_000,
    maximumScreenSpaceError: 8
  });
  const second = planSplatBudget([makeTestHierarchy()], view, {
    splatBudget: 110_000,
    maximumScreenSpaceError: 8
  });

  expect(
    [...first.refinedNodeIds].sort(),
    'two identical frames select exactly the same cut'
  ).toEqual([...second.refinedNodeIds].sort());
});

it('level fade derives a continuous transition from the parent error alone', () => {
  expect(
    getSplatLevelFadeOpacity(Number.POSITIVE_INFINITY, 8, 0.5),
    'a root has no parent error and is always fully visible'
  ).toBe(1);
  expect(getSplatLevelFadeOpacity(16, 8, 0), 'a zero band disables fading').toBe(1);
  expect(
    getSplatLevelFadeOpacity(8, 8, 0.5),
    'a level begins to fade in exactly at the threshold'
  ).toBe(0);
  expect(getSplatLevelFadeOpacity(10, 8, 0.5), 'and is halfway in at half the band').toBeCloseTo(
    0.5,
    6
  );
  expect(getSplatLevelFadeOpacity(12, 8, 0.5), 'reaching full opacity at the end of the band').toBe(
    1
  );
  expect(getSplatLevelFadeOpacity(100, 8, 0.5), 'and staying there well past it').toBe(1);
});
