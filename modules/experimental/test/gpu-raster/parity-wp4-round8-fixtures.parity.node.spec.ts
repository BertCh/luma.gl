// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, readdirSync, statSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';
import {decodeOracleFixture} from '../oracles/oracle-fixture';
import {compareRasterParity} from '../oracles/parity-wp4-compare';
import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionUint32ParameterValues,
  type GPUParticleAdvectionSettings,
  type GPUParticleAdvectionUint32Settings
} from '../../src/gpu-raster/particle-advection';
import {
  createParticleAdvectionCPUState,
  advanceParticlesOnCPU
} from '../../src/gpu-raster/particle-advection/particle-advection-cpu';
import {getGPULineIntegralConvolutionParameterValues} from '../../src/gpu-raster/flow-texture';
import {convolveLineIntegralOnCPU} from '../../src/gpu-raster/flow-texture/line-integral-convolution-cpu';
import {compareLICMode, type LICModeCase} from '../oracles/parity-wp4-round8-compare';

const directory = resolve('modules/experimental/test/oracles/fixtures');
const names = readdirSync(directory).filter(name =>
  /^parity-wp4-(particle-topology|overlay-showcase|lifecycle-|lic-modes-|cost-modes|change-modes|stitched|sieve-modes|polygon-modes|zonal-modes|streamline-modes-)/.test(
    name
  )
);
for (const name of names) {
  it(`Round 8 independent fixture provenance and size: ${name}`, () => {
    const path = resolve(directory, name);
    const fixture = JSON.parse(readFileSync(path, 'utf8'));
    expect(statSync(path).size).toBeLessThan(1500000);
    expect(fixture.versions.python).toBeTruthy();
    expect(fixture.versions.numpy).toBeTruthy();
    if (fixture.source)
      expect(createHash('sha256').update(readFileSync(fixture.source.path)).digest('hex')).toBe(
        fixture.source.sha256
      );
    expect((fixture.cases ?? fixture.frames).length).toBeGreaterThan(0);
  });
}

type LifecycleCase = {
  name: string;
  width: number;
  height: number;
  initial: number[];
  trailLength: number;
  outputMode: string;
  frames: {
    settings: GPUParticleAdvectionSettings;
    words: GPUParticleAdvectionUint32Settings;
    velocity: number[];
    positions: number[];
    previous: number[];
    speeds: number[];
    ages: number[];
    generations: number[];
    trails: number[];
  }[];
};
for (const name of names.filter(name => name.includes('lifecycle'))) {
  const fixture = decodeOracleFixture<{reference: string; cases: LifecycleCase[]}>(
    JSON.parse(readFileSync(resolve(directory, name), 'utf8'))
  );
  for (const scene of fixture.cases)
    it(`CPU mirror diagnostic, not GPU evidence: ${scene.name}`, () => {
      const count = scene.initial.length / 2;
      const state = createParticleAdvectionCPUState(count, scene.trailLength);
      state.positions.set(scene.initial);
      for (const frame of scene.frames) {
        advanceParticlesOnCPU(
          state,
          {
            width: scene.width,
            height: scene.height,
            velocities: Float32Array.from(
              {length: scene.width * scene.height * 2},
              (_, index) => frame.velocity[index % 2]
            )
          },
          getGPUParticleAdvectionParameterValues(frame.settings, [scene.width, scene.height]),
          getGPUParticleAdvectionUint32ParameterValues(frame.words),
          scene.trailLength
        );
        for (const metric of [
          'positions',
          'previousPositions',
          'speeds',
          'ages',
          'generations',
          'trails'
        ] as const) {
          const expected = metric === 'previousPositions' ? frame.previous : frame[metric];
          if (!expected.length) continue;
          compareRasterParity(
            'GPUParticleAdvection',
            `CPU diagnostic/${scene.name}/${frame.words.frame}`,
            fixture.reference,
            metric,
            state[metric]!,
            expected,
            metric === 'ages' || metric === 'generations' ? 'E' : 'L'
          );
        }
      }
    });
}
for (const name of names.filter(name => name.includes('lic-modes'))) {
  const fixture = decodeOracleFixture<{cases: LICModeCase[]}>(
    JSON.parse(readFileSync(resolve(directory, name), 'utf8'))
  );
  for (const scene of fixture.cases)
    it(`CPU mirror diagnostic, not GPU evidence: ${scene.name}`, () => {
      const actual = convolveLineIntegralOnCPU(
        {
          width: scene.fieldWidth,
          height: scene.fieldHeight,
          velocities: new Float32Array(scene.field)
        },
        scene.width,
        scene.height,
        scene.stepCount,
        getGPULineIntegralConvolutionParameterValues(scene.settings),
        19
      );
      compareLICMode(scene, actual.values, 'CPU diagnostic/');
    });
}

import costJSON from '../oracles/fixtures/parity-wp4-cost-modes.json';
import sieveJSON from '../oracles/fixtures/parity-wp4-sieve-modes.json';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {getRasterGridGeoreference} from '../../src/analysis-core/raster-grid';
import {
  computeCostDistance,
  computeBands,
  computeBandCounts
} from './cost-distance/cost-distance-oracle';
import {computeSieveOracle} from './raster-patches/raster-patches-oracle';

type CostCase = Omit<
  (typeof costJSON.cases)[number],
  'mode' | 'rowDirection' | 'costs' | 'costLimit'
> & {
  mode: 'uniform' | 'web-mercator' | 'geographic';
  rowDirection: 'north' | 'south';
  costs: number[];
  costLimit: number;
};
const costs = decodeOracleFixture<{reference: string; cases: CostCase[]}>(costJSON);
for (const scene of costs.cases)
  it(`CPU cost oracle diagnostic, not GPU evidence: ${scene.name}`, () => {
    const grid = makeGPURasterMetadata({
      ...scene,
      origin: [scene.origin[0], scene.origin[1]],
      cellSize: [scene.cellSize[0], scene.cellSize[1]]
    });
    const sources = scene.sources
      .slice(0, scene.sourceCount)
      .map((cell, index) => ({cell, cost: scene.sourceCosts[index]}));
    scene.sourceMask.forEach((active, cell) => {
      if (active) sources.push({cell, cost: 0});
    });
    const result = computeCostDistance({
      width: scene.width,
      height: scene.height,
      cellSizeMode: scene.mode,
      georeference: getRasterGridGeoreference(grid),
      friction: scene.raw.map((value, index) =>
        value === -9999 || !scene.validity[index]
          ? Infinity
          : value * scene.calibration[0] + scene.calibration[1]
      ),
      sources,
      costLimit: scene.costLimit
    });
    compareRasterParity(
      'GPUCostDistance',
      `CPU diagnostic/${scene.name}`,
      costs.reference,
      'all-costs',
      result,
      scene.costs,
      'L'
    );
    const bands = computeBands(result, scene.breaks);
    compareRasterParity(
      'GPUCostDistance',
      `CPU diagnostic/${scene.name}`,
      costs.reference,
      'all-bands',
      bands,
      scene.bands,
      'E'
    );
    compareRasterParity(
      'GPUCostDistance',
      `CPU diagnostic/${scene.name}`,
      costs.reference,
      'counts',
      computeBandCounts(bands, scene.breaks.length),
      scene.bandCounts,
      'E'
    );
  });
for (const scene of sieveJSON.cases)
  it(`CPU sieve oracle diagnostic, not GPU evidence: ${scene.name}`, () => {
    for (const frame of scene.frames) {
      if (!frame.converged || frame.overflow) continue;
      const labels = Uint32Array.from(scene.labels, label =>
        frame.valid && label <= frame.componentCount ? label : 0
      );
      const result = computeSieveOracle(
        labels,
        scene.width,
        scene.height,
        scene.capacity,
        frame.minimumPixels,
        scene.mode === 'merge' ? 'merge' : 'remove',
        scene.connectivity === 4 ? 4 : 8
      );
      for (const metric of ['labels', 'patchTargets', 'sievedCount'] as const)
        compareRasterParity(
          'GPURasterSieve',
          `CPU diagnostic/${scene.name}/${frame.minimumPixels}`,
          sieveJSON.reference,
          metric,
          metric === 'sievedCount' ? [result.sievedCount] : result[metric],
          frame.expected[metric],
          'E'
        );
    }
  });

import streamFirst from '../oracles/fixtures/parity-wp4-streamline-modes-0.json';
import streamSecond from '../oracles/fixtures/parity-wp4-streamline-modes-99.json';
import {decodeStreamlineCase, streamlineCases} from '../oracles/parity-wp4-round3-fixtures';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../src/gpu-raster/flow-texture';
import {generateStreamlinesOnCPU} from '../../src/gpu-raster/flow-texture/streamlines-cpu';
for (const encoded of [streamFirst, streamSecond].flatMap(
  fixture => decodeOracleFixture<{cases: typeof streamlineCases}>(fixture).cases
)) {
  it(`CPU streamline diagnostic, not GPU evidence: ${encoded.name}`, async () => {
    const scene = await decodeStreamlineCase(encoded);
    const result = generateStreamlinesOnCPU(
      {
        width: scene.fieldWidth,
        height: scene.fieldHeight,
        velocities: new Float32Array(scene.field)
      },
      scene.config,
      getGPUStreamlinesParameterValues(scene.settings),
      getGPUStreamlinesUint32ParameterValues(scene.settings)
    );
    compareRasterParity(
      'GPUStreamlines',
      `CPU diagnostic/${scene.name}`,
      scene.reference,
      'all-accepted-ids',
      result.ids,
      scene.ids,
      'E'
    );
    compareRasterParity(
      'GPUStreamlines',
      `CPU diagnostic/${scene.name}`,
      scene.reference,
      'all-points',
      result.points,
      scene.points,
      'L'
    );
    compareRasterParity(
      'GPUStreamlines',
      `CPU diagnostic/${scene.name}`,
      scene.reference,
      'all-offsets',
      result.pathOffsets,
      scene.offsets,
      'E'
    );
  });
}
