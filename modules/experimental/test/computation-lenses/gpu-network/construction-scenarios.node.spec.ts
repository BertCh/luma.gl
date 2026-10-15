// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import {validateGPUComputationRecord, type GPUComputationRecord} from '@luma.gl/gpgpu';
import '../../../src/computation-lenses/scenarios/gpu-network/index';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import {
  extractLensExample,
  findNonPublicImports
} from '../../../src/computation-lenses/lens-example';
import {NETWORK_FORMULA_REFS} from '../../../src/gpu-network/computation-pack/network-formulas';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

type ConstructionCase = {
  kebab: string;
  element: string;
  /** Formula recorded on a kernel node, or undefined for `formula: none` elements. */
  formula?: string;
  /** Last line of the shown example. */
  outputs: RegExp;
  rungs: string[];
};

const CASES: ConstructionCase[] = [
  {
    kebab: 'network-cost-table',
    element: 'GPUNetworkCostTable',
    formula: 'gpu-network.cost-table.admission',
    outputs: /const \{offsets, targets, costs\} = table\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  },
  {
    kebab: 'network-cost-weights',
    element: 'GPUNetworkCostWeights',
    formula: 'gpu-network.cost-weights.decay',
    outputs: /const \{offsets, neighbors, weights\} = costWeights\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  },
  {
    kebab: 'network-noding',
    element: 'GPUNetworkNoding',
    outputs: /const \{pieces, nodes, edges, topology\} = noding\.props\.output;$/,
    rungs: ['result', 'datum', 'kernel', 'graph']
  },
  {
    kebab: 'map-matching',
    element: 'GPUMapMatching',
    outputs: /const \{matchedEdges, snappedPositions, breaks\} = matching\.props\.output;$/,
    rungs: ['result', 'datum', 'kernel', 'graph']
  },
  {
    kebab: 'network-k-function',
    element: 'GPUNetworkKFunction',
    formula: 'gpu-network.k-function.k',
    outputs: /const \{kValues, pairCounts, snappedEventCount\} = kFunction\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  },
  {
    kebab: 'edge-bundling',
    element: 'GPUEdgeBundling',
    outputs: /const \{paths, lineOffsets\} = bundling\.props\.output;$/,
    rungs: ['result', 'datum', 'iterations', 'kernel', 'graph']
  },
  {
    kebab: 'flow-aggregation',
    element: 'GPUFlowAggregation',
    formula: 'gpu-network.flow-aggregation.pair-sum',
    outputs:
      /const \{flowOriginZoneIds, flowDestinationZoneIds, flowWeights\} = flows\.props\.output;$/,
    rungs: ['result', 'datum', 'formula', 'kernel', 'graph']
  }
];

for (const constructionCase of CASES) {
  const scenarioId = `gpu-network.${constructionCase.kebab}`;
  const directory = '../../../src/computation-lenses/scenarios/gpu-network';

  test(`${scenarioId} is registered with the result rung first`, () => {
    const descriptor = getLensScenario(scenarioId);
    expect(descriptor?.element).toBe(constructionCase.element);
    expect(descriptor?.audience).toBe('analyst');
    expect(descriptor?.rungs).toEqual(constructionCase.rungs);
    expect(descriptor?.defaultRung).toBe('result');
  });

  test(`${scenarioId} describes a valid structure on a null device`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const lens: {describeStructure(device: Device): Promise<GPUComputationRecord>} = await import(
        `${directory}/${constructionCase.kebab}.lens`
      );
      const record = await lens.describeStructure(device);
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      const nodes = record.graphs[0].nodes;
      expect(nodes.length).toBeGreaterThan(0);
      expect(record.refs.sources.length).toBeGreaterThan(0);
      if (constructionCase.formula) {
        expect(nodes.some(node => node.formula === constructionCase.formula)).toBe(true);
        const formula = NETWORK_FORMULA_REFS[constructionCase.formula];
        // The element's own formula comes first so the formula rung shows it by default.
        expect(record.refs.formulas[0].id).toBe(formula.id);
        const sources = record.refs.sources.map(source => source.text ?? '');
        for (const symbol of formula.symbols) {
          const identifier = symbol.binding!.slice('wgsl:'.length);
          expect(
            sources.some(source => new RegExp(`\\b${identifier}\\b`).test(source)),
            `${formula.id}: ${identifier}`
          ).toBe(true);
        }
      }
      if (constructionCase.rungs.includes('iterations'))
        expect(nodes.some(node => node.loop?.iteration !== undefined)).toBe(true);
    } finally {
      device.destroy();
    }
  });

  test(`${scenarioId} example is short, public and ends with the displayed outputs`, () => {
    const source = readFileSync(
      new URL(`${directory}/${constructionCase.kebab}.example.ts`, import.meta.url),
      'utf8'
    );
    expect(findNonPublicImports(source)).toEqual([]);
    const example = extractLensExample(source);
    expect(example.lineCount).toBeLessThanOrEqual(25);
    expect(example.code).toMatch(constructionCase.outputs);
  });
}
