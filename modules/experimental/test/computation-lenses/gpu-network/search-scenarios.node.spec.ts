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

type SearchCase = {
  kebab: string;
  element: string;
  /** Formula recorded on a kernel node, or undefined for `formula: none` elements. */
  formula?: string;
  /** Last line of the shown example. */
  outputs: RegExp;
};

const CASES: SearchCase[] = [
  {
    kebab: 'network-service-areas',
    element: 'GPUNetworkServiceAreas',
    formula: 'gpu-network.service-areas.label',
    outputs: /const \{assignments, costs\} = serviceAreas\.props\.output;$/
  },
  {
    kebab: 'network-path-extraction',
    element: 'GPUNetworkPathExtraction',
    outputs: /const \{ids, lineOffsets, pathFound\} = paths\.props\.output;$/
  },
  {
    kebab: 'network-neighborhood',
    element: 'GPUNetworkNeighborhood',
    formula: 'gpu-network.neighborhood.hops',
    outputs: /const \{hopDistances\} = neighborhood\.props\.output;$/
  },
  {
    kebab: 'network-isochrones',
    element: 'GPUNetworkIsochrones',
    formula: 'gpu-network.isochrones.cost',
    outputs: /const \{triangles, triangleBands\} = isochrones\.props\.raster!\.output;$/
  },
  {
    kebab: 'network-snapping',
    element: 'GPUNetworkSnapping',
    formula: 'gpu-network.snapping.nearest',
    outputs: /const \{snappedEdges, snapDistances, snappedPositions\} = snapping\.props\.output;$/
  },
  {
    kebab: 'network-cost-matrix',
    element: 'GPUNetworkCostMatrix',
    outputs: /const \{costs\} = matrix\.props\.output;$/
  },
  {
    kebab: 'network-accessibility',
    element: 'GPUNetworkAccessibility',
    formula: 'gpu-network.accessibility.gravity',
    outputs: /const \{gravity\} = accessibility\.props;$/
  },
  {
    kebab: 'drive-time-catchment-recipe',
    element: 'addDriveTimeCatchmentRecipe',
    formula: 'gpu-network.catchment.band',
    outputs: /const \{demandTimes, demandBands, bands\} = catchment;$/
  }
];

for (const searchCase of CASES) {
  const scenarioId = `gpu-network.${searchCase.kebab}`;
  const directory = '../../../src/computation-lenses/scenarios/gpu-network';

  test(`${scenarioId} is registered with the result rung first`, () => {
    const descriptor = getLensScenario(scenarioId);
    expect(descriptor?.element).toBe(searchCase.element);
    expect(descriptor?.audience).toBe('analyst');
    expect(descriptor?.rungs[0]).toBe('result');
    expect(descriptor?.defaultRung).toBe('result');
  });

  test(`${scenarioId} describes a valid structure on a null device`, async () => {
    const device = createNullWebGPUDevice();
    try {
      const lens: {describeStructure(device: Device): Promise<GPUComputationRecord>} = await import(
        `${directory}/${searchCase.kebab}.lens`
      );
      const record = await lens.describeStructure(device);
      expect(validateGPUComputationRecord(record).issues).toEqual([]);
      const nodes = record.graphs[0].nodes;
      if (searchCase.formula) {
        expect(nodes.some(node => node.formula === searchCase.formula)).toBe(true);
        const formula = NETWORK_FORMULA_REFS[searchCase.formula];
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
    } finally {
      device.destroy();
    }
  });

  test(`${scenarioId} example is short, public and ends with the displayed outputs`, () => {
    const source = readFileSync(
      new URL(`${directory}/${searchCase.kebab}.example.ts`, import.meta.url),
      'utf8'
    );
    expect(findNonPublicImports(source)).toEqual([]);
    const example = extractLensExample(source);
    expect(example.lineCount).toBeLessThanOrEqual(25);
    expect(example.code).toMatch(searchCase.outputs);
  });
}
