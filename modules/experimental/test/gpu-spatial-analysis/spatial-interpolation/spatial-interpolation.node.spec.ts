// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  getGPUKrigingParameterValues,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting,
  GPUKriging,
  type GPUFocalStatisticsProps,
  type GPUInverseDistanceWeightingProps,
  type GPUKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from './extent-grid';
import {KRIGING_ROBUSTNESS_REFERENCE} from './kriging-robustness-reference';
import {
  computeFocalStatisticsOnCPU,
  interpolateInverseDistanceWeightingOnCPU
} from './spatial-interpolation-oracle';

let serial = 0;

const GRID_4_BY_3 = makeExtentGrid(4, 3, [0, 0, 1, 1]);
const SHEARED_GRID = {...GRID_4_BY_3, affine: [0.25, 0.1, 0, 0, 1 / 3, 0] as const};

function view<Format extends 'uint32' | 'float32' | 'float32x2'>(
  graph: GPUCommandGraph,
  format: Format,
  length: number
) {
  return createTransientView(graph, `view-${serial++}`, format, length);
}

function createInterpolationProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUInverseDistanceWeightingProps> = {}
): GPUInverseDistanceWeightingProps {
  return {
    positions: view(graph, 'float32x2', 10),
    values: view(graph, 'float32', 10),
    parameters: view(graph, 'float32', 9),
    grid: GRID_4_BY_3,
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 1, 1],
    output: {
      values: view(graph, 'float32', 12),
      counts: view(graph, 'uint32', 12)
    },
    ...overrides
  };
}

function createKrigingProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUKrigingProps> = {}
): GPUKrigingProps {
  return {
    positions: view(graph, 'float32x2', 10),
    values: view(graph, 'float32', 10),
    parameters: view(graph, 'float32', 12),
    grid: GRID_4_BY_3,
    indexGridSize: [2, 2],
    indexBounds: [0, 0, 1, 1],
    output: {values: view(graph, 'float32', 12)},
    ...overrides
  };
}

function createFocalProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUFocalStatisticsProps> = {}
): GPUFocalStatisticsProps {
  return {
    values: view(graph, 'float32', 12),
    parameters: view(graph, 'float32', 4),
    grid: GRID_4_BY_3,
    maximumRadius: 2,
    output: {mean: view(graph, 'float32', 12)},
    ...overrides
  };
}

function withGraph(callback: (graph: GPUCommandGraph) => void): void {
  const device = createNullWebGPUDevice();
  callback(new GPUCommandGraph(device));
  device.destroy();
}

it('spatial interpolation parameter helpers pack layouts', () => {
  expect(
    Array.from(
      getGPUInverseDistanceWeightingParameterValues(
        {
          searchRadius: 5,
          power: 3,
          neighborCount: 4,
          minimumNeighborCount: 2
        },
        makeExtentGrid(4, 3, [0, 1, 2, 3])
      )
    )
  ).toEqual([0, 1, 2, 3, 5, 3, 4, 2, 0]);
  expect(
    Array.from(getGPUInverseDistanceWeightingParameterValues({searchRadius: 1}, GRID_4_BY_3))
  ).toEqual([0, 0, 1, 1, 1, 2, 0, 1, 0]);
  expect(() =>
    getGPUInverseDistanceWeightingParameterValues(
      {searchRadius: 1, neighborCount: 1.5},
      GRID_4_BY_3
    )
  ).toThrow(/neighborCount/);
  expect(() =>
    getGPUInverseDistanceWeightingParameterValues(
      {searchRadius: 1},
      GRID_4_BY_3,
      new Float32Array(4)
    )
  ).toThrow(/hold/);
  expect(
    Array.from(
      getGPUFocalStatisticsParameterValues({
        radius: 2.5,
        shape: 'circle',
        minimumCount: 3,
        propagateCenterNoData: true
      })
    )
  ).toEqual([2.5, 1, 3, 1]);
  expect(Array.from(getGPUFocalStatisticsParameterValues({radius: 1}))).toEqual([1, 0, 1, 0]);
  expect(() => getGPUFocalStatisticsParameterValues({radius: 1, minimumCount: -1})).toThrow(
    /minimumCount/
  );
});

it('GPUInverseDistanceWeighting validates its props', () => {
  const cases: [(graph: GPUCommandGraph) => Partial<GPUInverseDistanceWeightingProps>, RegExp][] = [
    [() => ({grid: {...GRID_4_BY_3, width: 0}}), /grid dimensions/],
    [() => ({grid: SHEARED_GRID}), /axis-aligned/],
    [() => ({indexGridSize: [2, 1.5]}), /indexGridSize/],
    [() => ({indexBounds: [0, 0, -1, 1]}), /indexBounds/],
    [() => ({indexBounds: [0, 0, Infinity, 1]}), /indexBounds/],
    [() => ({maximumNeighborCount: 65}), /maximumNeighborCount/],
    [graph => ({positions: view(graph, 'float32x2', 9)}), /positions length/],
    [graph => ({mask: view(graph, 'uint32', 9)}), /mask length/],
    [graph => ({parameters: view(graph, 'float32', 7)}), /parameters/],
    [graph => ({output: {values: view(graph, 'float32', 11)}}), /output.values/],
    [
      graph => ({
        output: {
          values: view(graph, 'float32', 12),
          counts: view(graph, 'uint32', 2)
        }
      }),
      /output.counts/
    ],
    [
      graph => ({
        values: view(graph, 'float32', 0),
        positions: view(graph, 'float32x2', 0)
      }),
      /at least one/
    ]
  ];
  for (const [overrides, message] of cases) {
    withGraph(graph => {
      expect(
        () => new GPUInverseDistanceWeighting(createInterpolationProps(graph, overrides(graph)))
      ).toThrow(message);
    });
  }
  withGraph(graph => {
    const values = view(graph, 'float32', 12);
    expect(
      () =>
        new GPUInverseDistanceWeighting(
          createInterpolationProps(graph, {
            values: view(graph, 'float32', 12),
            positions: view(graph, 'float32x2', 12),
            output: {values},
            parameters: values
          })
        )
    ).toThrow(/outputs must not share/);
  });
});

it('GPUInverseDistanceWeighting returns deterministic node IDs', () => {
  for (const maximumNeighborCount of [0, 8]) {
    const ids: string[][] = [];
    for (let run = 0; run < 2; run++) {
      withGraph(graph => {
        const contributor = new GPUInverseDistanceWeighting(
          createInterpolationProps(graph, {id: 'idw', maximumNeighborCount})
        );
        ids.push(contributor.getCommandNodes(graph).map(node => node.id));
      });
    }
    expect(ids[0]).toEqual(ids[1]);
    expect(ids[0].every(id => id.startsWith('idw-'))).toBe(true);
    expect(ids[0].slice(-2)).toEqual(['idw-sort-cells', 'idw-gather']);
  }
});

it('interpolation packers read the corners from the grid in grid order', () => {
  const extent = [10, 20, 18, 26] as const;
  const [minX, minY, maxX, maxY] = extent;
  // A grid whose rows advance north packs the old extent, so results do not change.
  const north = makeExtentGrid(4, 3, extent);
  // The north-up grid of the same rectangle starts at the top-left corner and ends at the bottom.
  const northUp = makeExtentGrid(4, 3, extent, 'south');
  // Point pixels are centered on their coordinates: the outer corner moves by half a cell.
  const pointPixels = {...north, pixelInterpretation: 'point' as const};
  const krigingSettings = {
    searchRadius: 5,
    variogram: {model: 'spherical' as const, nugget: 0, sill: 1, range: 4}
  };
  const packers = [
    (grid: typeof north) =>
      getGPUInverseDistanceWeightingParameterValues({searchRadius: 5}, grid).slice(0, 4),
    (grid: typeof north) => getGPUKrigingParameterValues(krigingSettings, grid).slice(0, 4)
  ];
  for (const pack of packers) {
    expect(Array.from(pack(north))).toEqual([minX, minY, maxX, maxY].map(Math.fround));
    expect(Array.from(pack(northUp))).toEqual([minX, maxY, maxX, minY].map(Math.fround));
    // Cell size 2 by 2: half a cell is 1 on each axis.
    expect(Array.from(pack(pointPixels))).toEqual(
      [minX - 1, minY - 1, maxX - 1, maxY - 1].map(Math.fround)
    );
  }
  for (const grid of [SHEARED_GRID, {...SHEARED_GRID, affine: [0.25, 0, 0, 0.1, 1, 0] as const}]) {
    expect(() => getGPUInverseDistanceWeightingParameterValues({searchRadius: 1}, grid)).toThrow(
      /axis-aligned/
    );
    expect(() => getGPUKrigingParameterValues(krigingSettings, grid)).toThrow(/axis-aligned/);
  }
});

it('inverse distance weighting packs and validates the taper', () => {
  const values = (taper: 'none' | 'modified-shepard' | undefined, searchRadius = 5) =>
    getGPUInverseDistanceWeightingParameterValues({searchRadius, taper}, GRID_4_BY_3);
  expect(values('modified-shepard')[8]).toBe(1);
  expect(values('none')[8]).toBe(0);
  expect(values(undefined)[8]).toBe(0);
  // The Franke-Nielson weight is undefined without a finite radius.
  expect(() => values('modified-shepard', Infinity)).toThrow(/finite searchRadius/);
  expect(values('none', Infinity)[4]).toBe(Infinity);
  expect(() => values('shepard' as never)).toThrow(/taper/);
});

it('interpolateInverseDistanceWeightingOnCPU matches numpy for the modified Shepard taper', () => {
  // Oracle: scratchpad/fix/work/D/interpolation/kriging_oracle.py F5 (numpy 2.5.4), Franke and
  // Nielson (1980) weights ((R - d)+ / (R d))^p versus 1 / d^p on a 5 x 3 raster.
  const reference = KRIGING_ROBUSTNESS_REFERENCE.idwTaper;
  for (const testCase of reference.cases) {
    const result = interpolateInverseDistanceWeightingOnCPU({
      positions: Float32Array.from(reference.positions),
      values: Float32Array.from(reference.values),
      width: reference.width,
      height: reference.height,
      indexBounds: [0, 0, 5, 3],
      extent: reference.extent as [number, number, number, number],
      searchRadius: testCase.radius,
      power: testCase.power,
      neighborCount: 0,
      minimumNeighborCount: 1,
      taper: testCase.taper as 'none' | 'modified-shepard'
    });
    for (let cell = 0; cell < result.values.length; cell++) {
      const expected = testCase.expected[cell];
      if (Number.isNaN(expected)) {
        expect(Number.isNaN(result.values[cell]), `cell ${cell}`).toBe(true);
      } else {
        expect(result.values[cell], `${testCase.taper} cell ${cell}`).toBeCloseTo(expected, 4);
      }
    }
  }
  // The taper really changes the surface: same radius, different weights.
  const [hardCutoff, tapered] = reference.cases;
  expect(tapered.expected).not.toEqual(hardCutoff.expected);
});

it('interpolation operators accept planar and great-circle contexts only', () => {
  const geographic = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  const projected = {
    coordinateSpace: 'planar',
    metric: 'native',
    units: 'native',
    frame: {id: 'utm-32632', kind: 'crs', crs: 'EPSG:32632'}
  } as const;
  withGraph(graph => {
    expect(
      new GPUKriging(createKrigingProps(graph, {spatialContext: geographic})).spatialContext.metric
    ).toBe('great-circle');
    expect(
      new GPUInverseDistanceWeighting(createInterpolationProps(graph, {spatialContext: geographic}))
        .spatialContext.metric
    ).toBe('great-circle');
    // Rhumb, ellipsoidal and absent metrics would silently differ from the sphere.
    for (const metric of ['none', 'rhumb', 'ellipsoidal'] as const) {
      const context = {coordinateSpace: 'longitude-latitude', metric} as const;
      expect(() => new GPUKriging(createKrigingProps(graph, {spatialContext: context}))).toThrow(
        /great-circle|metric/
      );
      expect(
        () =>
          new GPUInverseDistanceWeighting(
            createInterpolationProps(graph, {spatialContext: context})
          )
      ).toThrow(/great-circle|metric/);
    }
    // Discrete grids and topology-only contexts are not coordinates.
    expect(
      () =>
        new GPUKriging(
          createKrigingProps(graph, {
            spatialContext: {coordinateSpace: 'topology-only', metric: 'none'}
          })
        )
    ).toThrow(/does not support/);
    expect(new GPUKriging(createKrigingProps(graph)).spatialContext.coordinateSpace).toBe('planar');
    expect(
      new GPUKriging(createKrigingProps(graph, {spatialContext: projected})).spatialContext.frame
        ?.id
    ).toBe('utm-32632');
    expect(
      new GPUInverseDistanceWeighting(createInterpolationProps(graph, {spatialContext: projected}))
        .spatialContext.frame?.id
    ).toBe('utm-32632');
    // The great-circle kernels still fit eight storage bindings.
    const full = new GPUKriging(createKrigingProps(graph, {id: 'kg', spatialContext: geographic}));
    expect(
      full
        .getCommandNodes(graph)
        .map(node => node.id)
        .slice(-1)
    ).toEqual(['kg-solve']);
  });
});

it('GPUKriging validates its props', () => {
  const cases: [(graph: GPUCommandGraph) => Partial<GPUKrigingProps>, RegExp][] = [
    [() => ({grid: {...GRID_4_BY_3, width: 0}}), /grid dimensions/],
    [() => ({grid: SHEARED_GRID}), /axis-aligned/],
    [() => ({maximumNeighborCount: 17}), /maximumNeighborCount/],
    [() => ({indexBounds: [0, 0, -1, 1]}), /indexBounds/],
    [graph => ({mask: view(graph, 'uint32', 9)}), /mask length/],
    [graph => ({parameters: view(graph, 'float32', 11)}), /parameters/],
    [graph => ({output: {values: view(graph, 'float32', 11)}}), /output.values/],
    [
      graph => ({
        output: {
          values: view(graph, 'float32', 12),
          crossValidationVariance: view(graph, 'float32', 10)
        }
      }),
      /needs output.crossValidationPrediction/
    ],
    [
      graph => ({
        output: {
          values: view(graph, 'float32', 12),
          crossValidationPrediction: view(graph, 'float32', 9)
        }
      }),
      /one row per sample/
    ],
    [
      graph => ({
        output: {
          values: view(graph, 'float32', 12),
          status: {uncertainCount: view(graph, 'uint32', 0)}
        }
      }),
      /uncertainCount must contain one uint32 row/
    ],
    [
      graph => ({
        output: {
          values: view(graph, 'float32', 12),
          status: {invalidCount: view(graph, 'float32', 1) as never}
        }
      }),
      /invalidCount/
    ]
  ];
  for (const [overrides, message] of cases) {
    withGraph(graph => {
      expect(() => new GPUKriging(createKrigingProps(graph, overrides(graph)))).toThrow(message);
    });
  }
});

it('GPUKriging schedules only requested nodes within eight storage bindings', () => {
  withGraph(graph => {
    const minimal = new GPUKriging(createKrigingProps(graph, {id: 'kr'}));
    expect(
      minimal
        .getCommandNodes(graph)
        .map(node => node.id)
        .slice(-1)
    ).toEqual(['kr-solve']);
  });
  withGraph(graph => {
    // The maximum combination (mask, variance, both counters, cross-validation) must still fit
    // the default eight bindings per kernel; createWGSLKernelNode throws otherwise.
    const full = new GPUKriging(
      createKrigingProps(graph, {
        id: 'kr',
        mask: view(graph, 'uint32', 10),
        output: {
          values: view(graph, 'float32', 12),
          variance: view(graph, 'float32', 12),
          crossValidationPrediction: view(graph, 'float32', 10),
          crossValidationVariance: view(graph, 'float32', 10),
          status: {
            invalidCount: view(graph, 'uint32', 1),
            uncertainCount: view(graph, 'uint32', 1)
          }
        }
      })
    );
    expect(
      full
        .getCommandNodes(graph)
        .map(node => node.id)
        .slice(-6)
    ).toEqual([
      'kr-apply-mask',
      'kr-solve',
      'kr-cross-validation',
      'kr-clear-invalid',
      'kr-clear-uncertain',
      'kr-count-status'
    ]);
  });
});

it('GPUFocalStatistics validates its props and schedules only requested kernels', () => {
  const cases: [(graph: GPUCommandGraph) => Partial<GPUFocalStatisticsProps>, RegExp][] = [
    [() => ({grid: {...GRID_4_BY_3, height: -1}}), /grid dimensions/],
    [() => ({maximumRadius: 65}), /maximumRadius/],
    [() => ({maximumRadius: 1.5}), /maximumRadius/],
    [() => ({noDataValue: NaN}), /noDataValue/],
    [() => ({output: {}}), /at least one output/],
    [graph => ({values: view(graph, 'float32', 11)}), /values must hold/],
    [graph => ({validity: view(graph, 'uint32', 11)}), /validity must hold/],
    [graph => ({parameters: view(graph, 'float32', 3)}), /parameters/],
    [graph => ({output: {counts: view(graph, 'float32', 12) as never}}), /output.counts/],
    [graph => ({output: {min: view(graph, 'float32', 3)}}), /output.min/]
  ];
  for (const [overrides, message] of cases) {
    withGraph(graph => {
      expect(() => new GPUFocalStatistics(createFocalProps(graph, overrides(graph)))).toThrow(
        message
      );
    });
  }
  withGraph(graph => {
    const shared = view(graph, 'float32', 12);
    expect(
      () => new GPUFocalStatistics(createFocalProps(graph, {output: {mean: shared, sum: shared}}))
    ).toThrow(/share buffers/);
  });
  withGraph(graph => {
    const moments = new GPUFocalStatistics(createFocalProps(graph, {id: 'focal'}));
    expect(moments.getCommandNodes(graph).map(node => node.id)).toEqual(['focal-moments']);
    const extremes = new GPUFocalStatistics(
      createFocalProps(graph, {
        id: 'focal2',
        output: {range: view(graph, 'float32', 12)}
      })
    );
    expect(extremes.getCommandNodes(graph).map(node => node.id)).toEqual(['focal2-extremes']);
    const both = new GPUFocalStatistics(
      createFocalProps(graph, {
        id: 'focal3',
        output: {
          mean: view(graph, 'float32', 12),
          sum: view(graph, 'float32', 12),
          min: view(graph, 'float32', 12),
          max: view(graph, 'float32', 12),
          range: view(graph, 'float32', 12),
          standardDeviation: view(graph, 'float32', 12),
          counts: view(graph, 'uint32', 12)
        }
      })
    );
    expect(both.getCommandNodes(graph).map(node => node.id)).toEqual([
      'focal3-moments',
      'focal3-extremes'
    ]);
  });
});

it('interpolateInverseDistanceWeightingOnCPU weights, limits, and resolves exact hits', () => {
  // Two samples at distance 1 and 3 from the single cell center (0.5, 0.5).
  const base = {
    positions: Float32Array.from([1.5, 0.5, 0.5, 3.5, 0.5, 0.5, 0.5, 0.5]),
    values: Float32Array.from([10, 20, 30, 40]),
    width: 1,
    height: 1,
    indexBounds: [0, 0, 4, 4] as const,
    extent: [0, 0, 1, 1] as const,
    searchRadius: 5,
    power: 2,
    neighborCount: 0,
    minimumNeighborCount: 1
  };
  // Rows 2 and 3 are exact hits: the smaller row wins and all four samples count.
  let result = interpolateInverseDistanceWeightingOnCPU(base);
  expect(result.values[0]).toBe(30);
  expect(result.counts[0]).toBe(4);
  // Without exact hits: (10 * 1 + 20 / 9) / (1 + 1 / 9) = 11.
  const mask = [1, 1, 0, 0];
  result = interpolateInverseDistanceWeightingOnCPU({...base, mask});
  expect(result.values[0]).toBeCloseTo(11, 5);
  // Power zero is the plain mean; k = 1 keeps only the nearest sample.
  expect(interpolateInverseDistanceWeightingOnCPU({...base, mask, power: 0}).values[0]).toBe(15);
  result = interpolateInverseDistanceWeightingOnCPU({
    ...base,
    mask,
    neighborCount: 1
  });
  expect([result.values[0], result.counts[0]]).toEqual([10, 1]);
  // Radius 2 excludes the far sample; minimumNeighborCount 2 then yields nodata.
  result = interpolateInverseDistanceWeightingOnCPU({
    ...base,
    mask,
    searchRadius: 2,
    minimumNeighborCount: 2
  });
  expect(Number.isNaN(result.values[0])).toBe(true);
  expect(result.counts[0]).toBe(1);
});

it('computeFocalStatisticsOnCPU clips edges and honors nodata', () => {
  const values = Float32Array.from([1, 2, 3, 4, NaN, 6, 7, 8, -9]);
  const result = computeFocalStatisticsOnCPU({
    values,
    noDataValue: -9,
    width: 3,
    height: 3,
    maximumRadius: 4,
    radius: 1,
    shape: 'square',
    minimumCount: 1,
    propagateCenterNoData: false
  });
  // Corner (0, 0) sees 1, 2, 4 (center 4 is NaN).
  expect([result.count[0], result.sum[0], result.min[0], result.max[0]]).toEqual([3, 7, 1, 4]);
  // Center sees 7 valid cells (NaN center and the -9 sentinel are skipped).
  expect(result.count[4]).toBe(7);
  expect(result.mean[4]).toBeCloseTo(31 / 7, 5);
  const propagated = computeFocalStatisticsOnCPU({
    values,
    width: 3,
    height: 3,
    maximumRadius: 4,
    radius: 1,
    shape: 'circle',
    minimumCount: 1,
    propagateCenterNoData: true
  });
  expect(Number.isNaN(propagated.mean[4])).toBe(true);
  // Circle radius 1 excludes diagonals: corner sees 1, 2, 4.
  expect(propagated.count[0]).toBe(3);
});

it('IDW and kriging kernels parse and declare each shared NaN helper once', () => {
  withGraph(graph => {
    const nodes = [
      ...new GPUInverseDistanceWeighting(
        createInterpolationProps(graph, {id: 'idw-parse'})
      ).getCommandNodes(graph),
      ...new GPUKriging(createKrigingProps(graph, {id: 'kriging-parse'})).getCommandNodes(graph)
    ];
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources.length).toBeGreaterThan(0);
    for (const {id, source} of sources) {
      // wgsl_reflect rejects `const ORIGIN: vec2f = DOMAIN_MIN;` (valid WGSL) in the kriging solve
      // kernel; every other parse error is a real failure.
      try {
        new WgslReflect(source);
      } catch (error) {
        const message = String((error as {message?: string}).message ?? error);
        expect(id.endsWith('-solve') && /Invalid cast from f32 to vec2f/.test(message), id).toBe(
          true
        );
      }
      // wgsl_reflect accepts duplicate declarations, Dawn does not.
      for (const name of ['isNanBits', 'isFiniteValue', 'getNaN']) {
        expect(source.split(`fn ${name}(`).length - 1, `${id} ${name}`).toBeLessThanOrEqual(1);
      }
      expect(source).not.toContain('isNanValue');
    }
    expect(sources.some(({source}) => source.includes('fn isNanBits('))).toBe(true);
  });
});

const DISTANCE_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-azimuthal-equidistant'}
} as const;
const AREA_FRAME = {
  coordinateSpace: 'planar',
  metric: 'native',
  units: 'meters',
  frame: {id: 'f', kind: 'local-lambert-equal-area'}
} as const;

it('interpolation operators validate distance frames', () => {
  withGraph(graph => {
    const inverseDistanceProps = createInterpolationProps(graph);
    const krigingProps = createKrigingProps(graph);
    for (const spatialContext of [DISTANCE_FRAME, AREA_FRAME]) {
      const constructors = [
        () => new GPUInverseDistanceWeighting({...inverseDistanceProps, spatialContext}),
        () => new GPUKriging({...krigingProps, spatialContext})
      ];
      for (const construct of constructors) {
        if (spatialContext === AREA_FRAME) {
          expect(construct).toThrow(
            /measures distance; project into an azimuthal-equidistant frame/
          );
        } else {
          expect(construct).not.toThrow();
        }
      }
    }
  });
});
