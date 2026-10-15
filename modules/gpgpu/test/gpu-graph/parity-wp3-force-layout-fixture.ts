// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />

import {decodeOracleFixture} from '../../../experimental/test/oracles/oracle-fixture';
import equilibrium from '../../../experimental/test/oracles/fixtures/parity-wp3-force-layout-equilibrium.json?raw';
import synthetic from '../../../experimental/test/oracles/fixtures/parity-wp3-force-layout-synthetic.json?raw';
import flights3 from '../../../experimental/test/oracles/fixtures/parity-wp3-force-layout-openflights-seed-3.json?raw';
import flights17 from '../../../experimental/test/oracles/fixtures/parity-wp3-force-layout-openflights-seed-17.json?raw';
import flights41 from '../../../experimental/test/oracles/fixtures/parity-wp3-force-layout-openflights-seed-41.json?raw';

export type LayoutEvaluation = {
  positions: number[];
  energy: number;
  components: number[];
  gradient: number[];
  edgeLengths: number[];
  rootMeanSquareGradient: number;
};
export type LayoutFixture = {
  case: string;
  reference: string;
  versions: Record<string, string>;
  nodeCount: number;
  directed: boolean;
  edges: [number, number, number][];
  parameters: {
    repulsion: number;
    attraction: number;
    gravity: number;
    damping: number;
    timeStep: number;
    maxVelocity: number;
  };
  runs: {
    seed: number;
    initial: LayoutEvaluation;
    optimized: LayoutEvaluation;
    networkx: LayoutEvaluation;
    optimizer: {success: boolean; message: string; iterations: number};
  }[];
  iterationsPerFrame: number;
  frames: number;
  exactEquilibrium: boolean;
};
// Native JSON-module transforms can perturb f64 literals by one ulp (Vite 8 observed).
// Parse source text to preserve the reference bits, especially exact f32 input coordinates.
export const layoutFixtures = [equilibrium, synthetic, flights3, flights17, flights41].map(value =>
  decodeOracleFixture<LayoutFixture>(JSON.parse(value))
);

/** Composite scalar-energy definition and its derivative; checked against SciPy fixtures.
 * E = A/2 sum_edges d² + G/2 sum_vertices |x|² - R sum_pairs log(d).
 * The continuous quadratic continuation of log below d²=1e-4 is specified in
 * the generator. This evaluates an output; it does not implement a layout solver.
 */
export function evaluateLayout(
  fixture: LayoutFixture,
  positions: readonly number[]
): LayoutEvaluation {
  const {repulsion, attraction, gravity} = fixture.parameters;
  const gradient = positions.map(value => gravity * value);
  const components = [
    0,
    (gravity * positions.reduce((sum, value) => sum + value * value, 0)) / 2,
    0
  ];
  const edgeLengths: number[] = [];
  for (const [source, target] of fixture.edges) {
    const differenceX = positions[source * 2] - positions[target * 2];
    const differenceY = positions[source * 2 + 1] - positions[target * 2 + 1];
    const squared = differenceX ** 2 + differenceY ** 2;
    edgeLengths.push(Math.sqrt(squared));
    components[0] += (attraction * squared) / 2;
    for (const [axis, difference] of [
      [0, differenceX],
      [1, differenceY]
    ]) {
      gradient[source * 2 + axis] += attraction * difference;
      gradient[target * 2 + axis] -= attraction * difference;
    }
  }
  const minimumSquaredDistance = 1e-4;
  for (let first = 0; first < fixture.nodeCount; first++) {
    for (let second = first + 1; second < fixture.nodeCount; second++) {
      const differenceX = positions[first * 2] - positions[second * 2];
      const differenceY = positions[first * 2 + 1] - positions[second * 2 + 1];
      const squared = differenceX ** 2 + differenceY ** 2;
      const safeSquared = Math.max(squared, minimumSquaredDistance);
      components[2] -=
        (repulsion / 2) *
        (Math.log(safeSquared) + Math.min(squared / minimumSquaredDistance - 1, 0));
      for (const [axis, difference] of [
        [0, differenceX],
        [1, differenceY]
      ]) {
        const derivative = (repulsion * difference) / safeSquared;
        gradient[first * 2 + axis] -= derivative;
        gradient[second * 2 + axis] += derivative;
      }
    }
  }
  return {
    positions: Array.from(positions),
    components,
    energy: components.reduce((sum, value) => sum + value, 0),
    gradient,
    edgeLengths,
    rootMeanSquareGradient: Math.sqrt(
      gradient.reduce((sum, value) => sum + value * value, 0) / fixture.nodeCount
    )
  };
}
