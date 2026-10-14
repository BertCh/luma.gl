// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUChangeDetection,
  GPUCostDistance,
  GPUCostDistancePath,
  GPUDistanceField,
  GPULineIntegralConvolution,
  GPUStreamlines,
  GPUIsolines,
  GPUIsobands,
  GPUIsobandRings,
  GPUParticleAdvection,
  GPUPolygonRasterization,
  getGPURasterJoinMaximumSummedZoneCount,
  getGPURasterJoinMaximumZoneCount,
  GPURasterJoin,
  GPURasterReclassify,
  GPUWeightedOverlay,
  GPURasterCellStatistics,
  GPURasterConditional,
  GPURasterArithmetic,
  GPURasterPatchMetrics,
  GPURasterSieve,
  GPURasterExtremaPyramid,
  GPURasterSampling,
  getGPURasterProfileMaximumLineCount,
  GPURasterProfile,
  GPURasterStretch,
  getGPURasterZonalStatisticsMaximumSortedZoneCapacity,
  getGPURasterZonalStatisticsMaximumZoneCapacity,
  GPURasterZonalStatistics,
  GPU_RASTER_MAXIMUM_ITERATIONS,
  getGPURasterExtremaPyramidLayout,
  makeGPURasterBand,
  makeGPURasterMetadata,
  type GPURasterBand,
  type GPURasterCellStatisticsOutput,
  type GPURasterMetadata,
  type GPURasterScalarFormat
} from '../../../src/gpu-raster';
import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const MODERATE = {width: 100, height: 80};
/** 4096 x 4096 float32 cells are 64 MiB, half of one storage binding. */
const LARGE_SIDE = 4096;
const LARGE_CELLS = LARGE_SIDE * LARGE_SIDE;
/**
 * Largest texture side the fake core device accepts (its \`maxTextureDimension2D\` is 2048, below
 * the core 8192); texture inputs and outputs are only wired on grids up to this size.
 */
const TEXTURE_SIDE = 2048;

type Context = KernelAuditContext;

function makeGrid(width: number, height: number, cellSize: number | [number, number] = 1) {
  return makeGPURasterMetadata({width, height, cellSize, origin: [0, height]});
}

/** A buffer band with every optional field (validity, nodata, calibration) set. */
function makeBufferBand<Format extends GPURasterScalarFormat>(
  context: Context,
  grid: GPURasterMetadata,
  format: Format,
  id = 'band'
): GPURasterBand {
  const cellCount = grid.width * grid.height;
  return makeGPURasterBand(context.view(format, cellCount, `${id}-values`), grid, {
    id,
    validity: context.view('uint32', cellCount, `${id}-validity`),
    noDataValue: format === 'uint32' ? 0xffffffff : -9999,
    scale: 0.5,
    offset: 2
  });
}

/** A single-channel float32 texture band with validity, nodata and calibration. */
function makeTextureBand(context: Context, grid: GPURasterMetadata, id = 'texture-band') {
  const cellCount = grid.width * grid.height;
  return {
    id,
    format: 'float32',
    storage: {
      kind: 'texture',
      view: context.graph.createTextureView(
        context.texture('r32float', grid.width, grid.height, id),
        {mipLevelCount: 1}
      ),
      channel: 0
    },
    validity: context.view('uint32', cellCount, `${id}-validity`),
    noDataValue: -9999,
    scale: 0.5,
    offset: 2
  } as GPURasterBand;
}

/* ------------------------------------------------------------------------------------------ */
/* GPUChangeDetection                                                                         */
/* ------------------------------------------------------------------------------------------ */

type ChangeDetectionOptions = {
  cellCount: number;
  sliceCount: number;
  bandCount?: number;
  significanceSource?: 't-test' | 'mann-kendall';
  mask?: boolean;
};

function addChangeDetection(
  {graph, view}: Context,
  {cellCount, sliceCount, bandCount = 1, significanceSource, mask = true}: ChangeDetectionOptions,
  kind: 'single' | 'single-without-sen' | 'multiband'
): void {
  const pair = cellCount * bandCount;
  const singleBand = kind !== 'multiband';
  graph.add(
    new GPUChangeDetection({
      slices: view('float32', cellCount * sliceCount * bandCount),
      mask: mask ? view('uint32', cellCount) : undefined,
      parameters: view('float32', 5),
      cellCount,
      sliceCount,
      bandCount,
      significanceSource,
      output: {
        difference: view('float32', pair),
        logRatio: view('float32', pair),
        percentChange: view('float32', pair),
        ...(singleBand
          ? {
              tStatistic: view('float32', cellCount),
              tDegreesOfFreedom: view('float32', cellCount),
              tPValue: view('float32', cellCount),
              ...(kind === 'single' ? {senSlope: view('float32', cellCount)} : {}),
              mannKendallS: view('sint32', cellCount),
              mannKendallZ: view('float32', cellCount),
              mannKendallP: view('float32', cellCount),
              significance: view('uint32', cellCount)
            }
          : {
              changeMagnitude: view('float32', cellCount),
              changeDirection:
                bandCount === 2 ? view('float32', cellCount) : view('uint32', cellCount)
            })
      }
    })
  );
}

const CHANGE_DETECTION_CASES: KernelAuditCase[] = [
  ...(['t-test', 'mann-kendall'] as const).map(
    (significanceSource): KernelAuditCase => ({
      exportName: 'GPUChangeDetection',
      name: `single band every output, 64 slices, ${significanceSource}`,
      build: context =>
        addChangeDetection(context, {cellCount: 300, sliceCount: 64, significanceSource}, 'single')
    })
  ),
  {
    exportName: 'GPUChangeDetection',
    name: 'single band 256 slices without Sen slope',
    build: context =>
      addChangeDetection(
        context,
        {cellCount: 300, sliceCount: 256, significanceSource: 'mann-kendall', mask: false},
        'single-without-sen'
      )
  },
  {
    exportName: 'GPUChangeDetection',
    name: 'two bands, float direction',
    build: context =>
      addChangeDetection(context, {cellCount: 300, sliceCount: 6, bandCount: 2}, 'multiband')
  },
  {
    exportName: 'GPUChangeDetection',
    name: 'three bands, bit-pattern direction',
    build: context =>
      addChangeDetection(context, {cellCount: 300, sliceCount: 6, bandCount: 3}, 'multiband')
  },
  {
    exportName: 'GPUChangeDetection',
    name: '16 bands, bit-pattern direction',
    build: context =>
      addChangeDetection(context, {cellCount: 300, sliceCount: 6, bandCount: 16}, 'multiband')
  },
  {
    exportName: 'GPUChangeDetection',
    name: 'maximum cells, 64 slices with Sen slope',
    build: context =>
      addChangeDetection(
        context,
        {cellCount: context.maximumRows(4 * 64), sliceCount: 64},
        'single'
      )
  },
  {
    exportName: 'GPUChangeDetection',
    name: 'maximum cells, 256 slices',
    build: context =>
      addChangeDetection(
        context,
        {cellCount: context.maximumRows(4 * 256), sliceCount: 256},
        'single-without-sen'
      )
  },
  {
    exportName: 'GPUChangeDetection',
    name: 'maximum cells, 16 bands',
    build: context =>
      addChangeDetection(
        context,
        {cellCount: context.maximumRows(4 * 16 * 4), sliceCount: 4, bandCount: 16},
        'multiband'
      )
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPUCostDistance and GPUCostDistancePath                                                    */
/* ------------------------------------------------------------------------------------------ */

type CostDistanceOptions = {
  width: number;
  height: number;
  cellSizeMode?: 'uniform' | 'web-mercator' | 'geographic';
  friction: 'float32' | 'uint32' | 'sint32' | 'texture';
  sourceCapacity: number;
  breakCount: number;
  maximumIterations?: number;
  maximumTieIterations?: number;
  sweepAfterIteration?: number;
};

function addCostDistance(context: Context, options: CostDistanceOptions): void {
  const {graph, view} = context;
  const {width, height, breakCount} = options;
  const grid = makeGrid(width, height);
  const cellCount = width * height;
  graph.add(
    new GPUCostDistance({
      grid,
      friction:
        options.friction === 'texture'
          ? makeTextureBand(context, grid, 'friction')
          : makeBufferBand(context, grid, options.friction, 'friction'),
      frictionParameters: view('float32', 2),
      parameters: view('float32', 8),
      cellSizeMode: options.cellSizeMode,
      sources: view('uint32', options.sourceCapacity),
      sourceCosts: view('float32', options.sourceCapacity),
      sourceCount: view('uint32', 1),
      sourceMask: view('uint32', cellCount),
      maximumIterations: options.maximumIterations,
      maximumTieIterations: options.maximumTieIterations,
      sweepAfterIteration: options.sweepAfterIteration,
      costBreaks: view('float32', breakCount),
      output: {
        costs: view('float32', cellCount),
        backLinks: view('uint32', cellCount),
        bands: view('uint32', cellCount),
        bandCounts: view('uint32', breakCount),
        status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
      }
    })
  );
}

const COST_DISTANCE_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUCostDistance',
    name: 'every option, maximum iterations, uniform cells, float32 friction',
    build: context =>
      addCostDistance(context, {
        ...MODERATE,
        friction: 'float32',
        sourceCapacity: 64,
        breakCount: 8,
        maximumIterations: GPU_RASTER_MAXIMUM_ITERATIONS,
        maximumTieIterations: GPU_RASTER_MAXIMUM_ITERATIONS,
        sweepAfterIteration: 1
      })
  },
  {
    exportName: 'GPUCostDistance',
    name: 'web-mercator cells, uint32 friction',
    build: context =>
      addCostDistance(context, {
        ...MODERATE,
        cellSizeMode: 'web-mercator',
        friction: 'uint32',
        sourceCapacity: 64,
        breakCount: 8
      })
  },
  {
    exportName: 'GPUCostDistance',
    name: 'geographic cells, sint32 friction',
    build: context =>
      addCostDistance(context, {
        ...MODERATE,
        cellSizeMode: 'geographic',
        friction: 'sint32',
        sourceCapacity: 64,
        breakCount: 8
      })
  },
  {
    exportName: 'GPUCostDistance',
    name: 'texture friction band',
    build: context =>
      addCostDistance(context, {
        ...MODERATE,
        friction: 'texture',
        sourceCapacity: 64,
        breakCount: 8,
        sweepAfterIteration: 1024
      })
  },
  {
    exportName: 'GPUCostDistance',
    name: 'costs from a source mask only',
    build: ({graph, view}) => {
      const grid = makeGrid(MODERATE.width, MODERATE.height);
      graph.add(
        new GPUCostDistance({
          grid,
          friction: makeGPURasterBand(view('float32', 8000), grid),
          parameters: view('float32', 8),
          sourceMask: view('uint32', 8000),
          output: {costs: view('float32', 8000)}
        })
      );
    }
  },
  {
    exportName: 'GPUCostDistance',
    name: 'maximum grid, every output, maximum iterations',
    build: context =>
      addCostDistance(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        friction: 'float32',
        sourceCapacity: context.maximumRows(4),
        breakCount: 4096,
        maximumIterations: GPU_RASTER_MAXIMUM_ITERATIONS,
        maximumTieIterations: GPU_RASTER_MAXIMUM_ITERATIONS
      })
  },
  {
    exportName: 'GPUCostDistance',
    name: 'largest texture friction band, default iterations',
    build: context =>
      addCostDistance(context, {
        width: TEXTURE_SIDE,
        height: TEXTURE_SIDE,
        friction: 'texture',
        sourceCapacity: 1024,
        breakCount: 16
      })
  }
];

function addCostDistancePath(
  {graph, view}: Context,
  width: number,
  height: number,
  capacity: number
): void {
  graph.add(
    new GPUCostDistancePath({
      grid: makeGrid(width, height),
      backLinks: view('uint32', width * height),
      target: view('uint32', 1),
      output: {
        path: {
          ids: view('uint32', capacity),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            requiredCount: view('uint32', 1)
          }
        },
        status: {iterationLimitReached: view('uint32', 1)}
      }
    })
  );
}

const COST_DISTANCE_PATH_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUCostDistancePath',
    name: 'every output',
    build: context => addCostDistancePath(context, MODERATE.width, MODERATE.height, 500)
  },
  {
    exportName: 'GPUCostDistancePath',
    name: 'maximum grid and path capacity',
    build: context => addCostDistancePath(context, LARGE_SIDE, LARGE_SIDE, context.maximumRows(4))
  },
  {
    exportName: 'GPUCostDistancePath',
    name: 'minimal outputs',
    build: ({graph, view}) =>
      graph.add(
        new GPUCostDistancePath({
          grid: makeGrid(20, 10),
          backLinks: view('uint32', 200),
          target: view('uint32', 1),
          output: {
            path: {
              ids: view('uint32', 1),
              status: {count: view('uint32', 1), overflow: view('uint32', 1)}
            }
          }
        })
      )
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPUDistanceField                                                                           */
/* ------------------------------------------------------------------------------------------ */

function addDistanceField(
  {graph, view, texture}: Context,
  options: {
    width: number;
    height: number;
    cellSize?: number | [number, number];
    mode: 'exact' | 'jump-flood';
    refinement?: 0 | 1 | 2;
    seedCapacity: number;
    seedMask?: boolean;
    withTexture?: boolean;
  }
): void {
  const {width, height} = options;
  const cellCount = width * height;
  graph.add(
    new GPUDistanceField({
      grid: makeGrid(width, height, options.cellSize),
      parameters: view('float32', 8),
      seedPositions: view('float32x2', options.seedCapacity),
      seedIds: view('uint32', options.seedCapacity),
      seedCount: view('uint32', 1),
      seedMask: options.seedMask === false ? undefined : view('uint32', cellCount),
      mode: options.mode,
      jumpFloodRefinementIterationCount: options.refinement,
      output: {
        distances: view('float32', cellCount),
        allocation: view('uint32', cellCount),
        nearestCells: view('uint32', cellCount),
        withinDistance: view('uint32', cellCount),
        texture:
          options.withTexture === false
            ? undefined
            : graph.createTextureView(texture('r32float', width, height, 'distance-field'), {
                mipLevelCount: 1
              })
      }
    })
  );
}

const DISTANCE_FIELD_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUDistanceField',
    name: 'exact, every option, isotropic cells',
    build: context => addDistanceField(context, {...MODERATE, mode: 'exact', seedCapacity: 256})
  },
  {
    exportName: 'GPUDistanceField',
    name: 'exact, anisotropic cells, seed points only',
    build: context =>
      addDistanceField(context, {
        ...MODERATE,
        cellSize: [2, 3],
        mode: 'exact',
        seedCapacity: 256,
        seedMask: false,
        withTexture: false
      })
  },
  ...([0, 1, 2] as const).map(
    (refinement): KernelAuditCase => ({
      exportName: 'GPUDistanceField',
      name: `jump-flood, refinement ${refinement}, every option`,
      build: context =>
        addDistanceField(context, {
          ...MODERATE,
          mode: 'jump-flood',
          refinement,
          seedCapacity: 256
        })
    })
  ),
  {
    exportName: 'GPUDistanceField',
    name: 'exact, maximum grid',
    build: context =>
      addDistanceField(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        mode: 'exact',
        seedCapacity: context.maximumRows(8),
        withTexture: false
      })
  },
  {
    exportName: 'GPUDistanceField',
    name: 'exact, largest texture output',
    build: context =>
      addDistanceField(context, {
        width: TEXTURE_SIDE,
        height: TEXTURE_SIDE,
        mode: 'exact',
        seedCapacity: context.maximumRows(8)
      })
  },
  {
    exportName: 'GPUDistanceField',
    name: 'jump-flood, maximum grid, refinement 2',
    build: context =>
      addDistanceField(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        mode: 'jump-flood',
        refinement: 2,
        seedCapacity: context.maximumRows(8),
        withTexture: false
      })
  },
  {
    exportName: 'GPUDistanceField',
    name: 'jump-flood, largest texture output, refinement 2',
    build: context =>
      addDistanceField(context, {
        width: TEXTURE_SIDE,
        height: TEXTURE_SIDE,
        mode: 'jump-flood',
        refinement: 2,
        seedCapacity: context.maximumRows(8)
      })
  },
  ...(['exact', 'jump-flood'] as const).flatMap((mode): KernelAuditCase[] => [
    {
      exportName: 'GPUDistanceField',
      name: `${mode}, maximum width 32768`,
      build: context =>
        addDistanceField(context, {
          width: 32768,
          height: context.maximumRows(4) / 32768,
          mode,
          seedCapacity: 1024,
          withTexture: false
        })
    },
    {
      exportName: 'GPUDistanceField',
      name: `${mode}, maximum height 32768`,
      build: context =>
        addDistanceField(context, {
          width: context.maximumRows(4) / 32768,
          height: 32768,
          mode,
          seedCapacity: 1024,
          withTexture: false
        })
    }
  ])
];

/* ------------------------------------------------------------------------------------------ */
/* GPULineIntegralConvolution, GPUStreamlines                                                 */
/* ------------------------------------------------------------------------------------------ */

function addLineIntegralConvolution(
  {graph, view, texture}: Context,
  options: {
    fieldWidth: number;
    fieldHeight: number;
    width: number;
    height: number;
    stepCount?: number;
    outputs: boolean;
    texture?: boolean;
  }
): void {
  const {width, height} = options;
  graph.add(
    new GPULineIntegralConvolution({
      velocities: view('float32x2', options.fieldWidth * options.fieldHeight),
      fieldWidth: options.fieldWidth,
      fieldHeight: options.fieldHeight,
      grid: makeGrid(width, height),
      stepCount: options.stepCount,
      parameters: view('float32', 12),
      uint32Parameters: view('uint32', 4),
      output: {
        values: view('float32', width * height),
        ...(options.outputs ? {speeds: view('float32', width * height)} : {}),
        ...(options.texture
          ? {
              texture: graph.createTextureView(texture('r32float', width, height, 'lic'), {
                mipLevelCount: 1
              })
            }
          : {})
      }
    })
  );
}

const LINE_INTEGRAL_CONVOLUTION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPULineIntegralConvolution',
    name: 'every output, maximum step count',
    build: context =>
      addLineIntegralConvolution(context, {
        fieldWidth: 100,
        fieldHeight: 80,
        width: 300,
        height: 200,
        stepCount: 256,
        outputs: true,
        texture: true
      })
  },
  {
    exportName: 'GPULineIntegralConvolution',
    name: 'values only, default steps',
    build: context =>
      addLineIntegralConvolution(context, {
        fieldWidth: 100,
        fieldHeight: 80,
        width: 300,
        height: 200,
        outputs: false
      })
  },
  {
    exportName: 'GPULineIntegralConvolution',
    name: 'zero steps',
    build: context =>
      addLineIntegralConvolution(context, {
        fieldWidth: 10,
        fieldHeight: 10,
        width: 30,
        height: 20,
        stepCount: 0,
        outputs: true,
        texture: true
      })
  },
  {
    exportName: 'GPULineIntegralConvolution',
    name: 'maximum field and output grid',
    build: context =>
      addLineIntegralConvolution(context, {
        fieldWidth: LARGE_SIDE,
        fieldHeight: LARGE_SIDE,
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        stepCount: 256,
        outputs: true
      })
  },
  {
    exportName: 'GPULineIntegralConvolution',
    name: 'largest texture output',
    build: context =>
      addLineIntegralConvolution(context, {
        fieldWidth: LARGE_SIDE,
        fieldHeight: LARGE_SIDE,
        width: TEXTURE_SIDE,
        height: TEXTURE_SIDE,
        stepCount: 256,
        outputs: true,
        texture: true
      })
  }
];

function addStreamlines(
  {graph, view}: Context,
  options: {
    fieldWidth: number;
    fieldHeight: number;
    gridWidth: number;
    gridHeight: number;
    seedColumns: number;
    seedRows: number;
    stepsPerDirection: number;
    maximumIterations?: number;
    lineCapacity: number;
    positionCapacity: number;
  }
): void {
  const seedCount = options.seedColumns * options.seedRows;
  graph.add(
    new GPUStreamlines({
      velocities: view('float32x2', options.fieldWidth * options.fieldHeight),
      fieldWidth: options.fieldWidth,
      fieldHeight: options.fieldHeight,
      gridWidth: options.gridWidth,
      gridHeight: options.gridHeight,
      seedColumns: options.seedColumns,
      seedRows: options.seedRows,
      stepsPerDirection: options.stepsPerDirection,
      maximumIterations: options.maximumIterations,
      parameters: view('float32', 12),
      uint32Parameters: view('uint32', 4),
      output: {
        lines: {
          ids: view('uint32', options.lineCapacity),
          status: {
            count: view('uint32', 1),
            overflow: view('uint32', 1),
            requiredCount: view('uint32', 1)
          }
        },
        lineOffsets: view('uint32', options.lineCapacity + 1),
        positions: view('float32x2', options.positionCapacity),
        positionCount: view('uint32', 1),
        status: {iterationLimitReached: view('uint32', 1)},
        candidates: {
          positions: view('float32x2', seedCount * (2 * options.stepsPerDirection + 1)),
          spans: view('uint32', seedCount * 4)
        }
      }
    })
  );
}

const STREAMLINES_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUStreamlines',
    name: 'every output',
    build: context =>
      addStreamlines(context, {
        fieldWidth: 100,
        fieldHeight: 80,
        gridWidth: 200,
        gridHeight: 160,
        seedColumns: 20,
        seedRows: 15,
        stepsPerDirection: 30,
        lineCapacity: 300,
        positionCapacity: 5000
      })
  },
  {
    exportName: 'GPUStreamlines',
    name: 'many pruning rounds',
    build: context =>
      addStreamlines(context, {
        fieldWidth: 100,
        fieldHeight: 80,
        gridWidth: 200,
        gridHeight: 160,
        seedColumns: 20,
        seedRows: 15,
        stepsPerDirection: 30,
        maximumIterations: 512,
        lineCapacity: 300,
        positionCapacity: 5000
      })
  },
  {
    exportName: 'GPUStreamlines',
    name: 'maximum seeds, field and capacities',
    build: context =>
      addStreamlines(context, {
        fieldWidth: LARGE_SIDE,
        fieldHeight: LARGE_SIDE,
        gridWidth: LARGE_SIDE,
        gridHeight: LARGE_SIDE,
        seedColumns: 255,
        seedRows: 257,
        // candidates.positions holds seedCount * (2 * steps + 1) float32x2 rows in one binding
        stepsPerDirection: Math.floor((context.maximumRows(8) / 65535 - 1) / 2),
        lineCapacity: 65535,
        positionCapacity: context.maximumRows(8)
      })
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPUIsolines, GPUIsobands, GPUIsobandRings                                                  */
/* ------------------------------------------------------------------------------------------ */

function addIsolines(
  {graph, view}: Context,
  options: {
    width: number;
    height: number;
    levelCount: number;
    segmentCapacity: number;
    vertexCapacity: number;
    polylines: boolean;
  }
): void {
  const {segmentCapacity} = options;
  const cellCount = options.width * options.height;
  graph.add(
    new GPUIsolines({
      grid: makeGrid(options.width, options.height),
      values: view('float32', cellCount),
      validity: view('uint32', cellCount),
      noDataValue: -9999,
      levels: view('float32', options.levelCount),
      parameters: view('float32', 12),
      output: {
        segments: view('float32x4', segmentCapacity),
        segmentLevels: view('uint32', segmentCapacity),
        segmentEdges: view('uint32x2', segmentCapacity),
        polylines: options.polylines
          ? {
              vertices: view('float32x2', options.vertexCapacity),
              polylineOffsets: view('uint32', segmentCapacity + 1),
              polylineLevels: view('uint32', segmentCapacity),
              polylineClosed: view('uint32', segmentCapacity),
              polylineCount: view('uint32', 1),
              vertexCount: view('uint32', 1)
            }
          : undefined,
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1),
          polylineOverflow: options.polylines ? view('uint32', 1) : undefined
        }
      }
    })
  );
}

const ISOLINES_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUIsolines',
    name: 'every output with polylines',
    build: context =>
      addIsolines(context, {
        width: 200,
        height: 150,
        levelCount: 16,
        segmentCapacity: 4000,
        vertexCapacity: 8000,
        polylines: true
      })
  },
  {
    exportName: 'GPUIsolines',
    name: 'segments only',
    build: context =>
      addIsolines(context, {
        width: 200,
        height: 150,
        levelCount: 1,
        segmentCapacity: 4000,
        vertexCapacity: 0,
        polylines: false
      })
  },
  {
    exportName: 'GPUIsolines',
    name: 'maximum grid, levels and capacities with polylines',
    build: context =>
      addIsolines(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        levelCount: 256,
        segmentCapacity: context.maximumRows(16),
        vertexCapacity: context.maximumRows(8),
        polylines: true
      })
  },
  {
    exportName: 'GPUIsolines',
    name: 'maximum grid, segments only',
    build: context =>
      addIsolines(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        levelCount: 256,
        segmentCapacity: context.maximumRows(16),
        vertexCapacity: 0,
        polylines: false
      })
  }
];

type IsobandsOptions = {
  width: number;
  height: number;
  breakCount: number;
  classes: boolean;
  triangleCapacity: number;
  edgeCapacity: number;
};

function addIsobands({graph, view}: Context, options: IsobandsOptions): void {
  const cellCount = options.width * options.height;
  const {triangleCapacity, edgeCapacity} = options;
  graph.add(
    new GPUIsobands({
      grid: makeGrid(options.width, options.height),
      values: view('float32', cellCount),
      validity: view('uint32', cellCount),
      noDataValue: -9999,
      breaks: view('float32', options.breakCount),
      parameters: view('float32', 12),
      output: {
        bandClasses: options.classes ? view('uint32', cellCount) : undefined,
        ...(triangleCapacity > 0
          ? {
              triangles: view('float32x2', triangleCapacity * 3),
              triangleBands: view('uint32', triangleCapacity)
            }
          : {}),
        ...(edgeCapacity > 0
          ? {edges: view('float32x4', edgeCapacity), edgeBands: view('uint32', edgeCapacity)}
          : {}),
        status: {
          ...(triangleCapacity > 0
            ? {
                count: view('uint32', 1),
                requiredCount: view('uint32', 1),
                vertexCount: view('uint32', 1),
                overflow: view('uint32', 1)
              }
            : {}),
          ...(edgeCapacity > 0
            ? {
                edgeCount: view('uint32', 1),
                edgeRequiredCount: view('uint32', 1),
                edgeOverflow: view('uint32', 1)
              }
            : {})
        }
      }
    })
  );
}

const ISOBANDS_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUIsobands',
    name: 'every output',
    build: context =>
      addIsobands(context, {
        width: 200,
        height: 150,
        breakCount: 16,
        classes: true,
        triangleCapacity: 20000,
        edgeCapacity: 10000
      })
  },
  {
    exportName: 'GPUIsobands',
    name: 'band classes only',
    build: context =>
      addIsobands(context, {
        width: 200,
        height: 150,
        breakCount: 16,
        classes: true,
        triangleCapacity: 0,
        edgeCapacity: 0
      })
  },
  {
    exportName: 'GPUIsobands',
    name: 'triangles only',
    build: context =>
      addIsobands(context, {
        width: 200,
        height: 150,
        breakCount: 16,
        classes: false,
        triangleCapacity: 20000,
        edgeCapacity: 0
      })
  },
  {
    exportName: 'GPUIsobands',
    name: 'edges only',
    build: context =>
      addIsobands(context, {
        width: 200,
        height: 150,
        breakCount: 16,
        classes: false,
        triangleCapacity: 0,
        edgeCapacity: 10000
      })
  },
  {
    exportName: 'GPUIsobands',
    name: 'maximum grid, breaks and capacities',
    build: context =>
      addIsobands(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        breakCount: 256,
        classes: true,
        triangleCapacity: context.maximumRows(24),
        edgeCapacity: context.maximumRows(16)
      })
  },
  {
    exportName: 'GPUIsobands',
    name: 'maximum grid, triangles only',
    build: context =>
      addIsobands(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        breakCount: 256,
        classes: false,
        triangleCapacity: context.maximumRows(24),
        edgeCapacity: 0
      })
  }
];

function addIsobandRings(
  {graph, view}: Context,
  options: {
    width: number;
    height: number;
    breakCount: number;
    edgeCapacity: number;
    ringCapacity: number;
    vertexCapacity: number;
    splitTouchingRings?: boolean;
    vertexTolerance?: number;
  }
): void {
  const cellCount = options.width * options.height;
  const {ringCapacity, vertexCapacity} = options;
  graph.add(
    new GPUIsobandRings({
      grid: makeGrid(options.width, options.height),
      values: view('float32', cellCount),
      validity: view('uint32', cellCount),
      noDataValue: -9999,
      breaks: view('float32', options.breakCount),
      parameters: view('float32', 12),
      edgeCapacity: options.edgeCapacity,
      splitTouchingRings: options.splitTouchingRings,
      vertexTolerance: options.vertexTolerance,
      output: {
        ringOffsets: view('uint32', ringCapacity + 1),
        positions: view('float32x2', vertexCapacity),
        ringAreas: view('float32', ringCapacity),
        ringIsHole: view('uint32', ringCapacity),
        ringShells: view('uint32', ringCapacity),
        ringGroups: view('uint32', ringCapacity),
        segmentRings: view('uint32', options.edgeCapacity),
        segmentVertices: view('uint32', options.edgeCapacity),
        segmentFlags: view('uint32', options.edgeCapacity),
        polygons: {
          kind: 'polygons',
          positions: view('float32x2', vertexCapacity),
          ringOffsets: view('uint32', ringCapacity + 1),
          polygonOffsets: view('uint32', ringCapacity + 1),
          featureOffsets: view('uint32', ringCapacity + 1),
          sourceIds: view('uint32', ringCapacity)
        },
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1),
          openSegmentCount: view('uint32', 1),
          touchingSegmentCount: view('uint32', 1),
          edgeCount: view('uint32', 1),
          edgeOverflow: view('uint32', 1)
        }
      }
    })
  );
}

const ISOBAND_RINGS_CASES: KernelAuditCase[] = [
  ...([true, false] as const).map(
    (splitTouchingRings): KernelAuditCase => ({
      exportName: 'GPUIsobandRings',
      name: `every output, splitTouchingRings ${splitTouchingRings}`,
      build: context =>
        addIsobandRings(context, {
          width: 200,
          height: 150,
          breakCount: 16,
          edgeCapacity: 6000,
          ringCapacity: 400,
          vertexCapacity: 6000,
          splitTouchingRings,
          vertexTolerance: 1e-3
        })
    })
  ),
  {
    exportName: 'GPUIsobandRings',
    name: 'maximum grid, edge and ring capacities',
    build: context =>
      addIsobandRings(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        breakCount: 256,
        edgeCapacity: context.maximumRows(16),
        ringCapacity: context.maximumRows(16),
        vertexCapacity: context.maximumRows(8)
      })
  },
  {
    exportName: 'GPUIsobandRings',
    name: 'edge capacity above the maximum',
    build: context =>
      addIsobandRings(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        breakCount: 256,
        edgeCapacity: context.maximumRows(16) + 1,
        ringCapacity: context.maximumRows(16),
        vertexCapacity: context.maximumRows(8)
      })
  },
  {
    exportName: 'GPUIsobandRings',
    name: 'ring capacity above the maximum',
    build: context =>
      addIsobandRings(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        breakCount: 256,
        edgeCapacity: context.maximumRows(16),
        ringCapacity: context.maximumRows(16) + 1,
        vertexCapacity: context.maximumRows(8)
      })
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPUParticleAdvection                                                                       */
/* ------------------------------------------------------------------------------------------ */

function addParticleAdvection(
  {graph, view}: Context,
  options: {
    fieldWidth: number;
    fieldHeight: number;
    particleCount: number;
    trailLength?: number;
    outputs: boolean;
  }
): void {
  const {particleCount, trailLength} = options;
  graph.add(
    new GPUParticleAdvection({
      velocities: view('float32x2', options.fieldWidth * options.fieldHeight),
      fieldWidth: options.fieldWidth,
      fieldHeight: options.fieldHeight,
      parameters: view('float32', 16),
      uint32Parameters: view('uint32', 4),
      state: {
        positions: view('float32x2', particleCount),
        ages: view('uint32', particleCount),
        generations: view('uint32', particleCount)
      },
      output: options.outputs
        ? {
            previousPositions: view('float32x2', particleCount),
            speeds: view('float32', particleCount),
            trails: trailLength
              ? {positions: view('float32x2', particleCount * trailLength), length: trailLength}
              : undefined
          }
        : undefined
    })
  );
}

const PARTICLE_ADVECTION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUParticleAdvection',
    name: 'every output with trails',
    build: context =>
      addParticleAdvection(context, {
        fieldWidth: 100,
        fieldHeight: 80,
        particleCount: 5000,
        trailLength: 16,
        outputs: true
      })
  },
  {
    exportName: 'GPUParticleAdvection',
    name: 'state only',
    build: context =>
      addParticleAdvection(context, {
        fieldWidth: 100,
        fieldHeight: 80,
        particleCount: 5000,
        outputs: false
      })
  },
  {
    exportName: 'GPUParticleAdvection',
    name: 'maximum particles and field, no trails',
    build: context =>
      addParticleAdvection(context, {
        fieldWidth: LARGE_SIDE,
        fieldHeight: LARGE_SIDE,
        particleCount: context.maximumRows(8),
        outputs: true
      })
  },
  {
    exportName: 'GPUParticleAdvection',
    name: 'maximum particles with trails',
    build: context =>
      addParticleAdvection(context, {
        fieldWidth: LARGE_SIDE,
        fieldHeight: LARGE_SIDE,
        particleCount: Math.floor(context.maximumRows(8) / 64),
        trailLength: 64,
        outputs: true
      })
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPUPolygonRasterization, GPURasterJoin                                                     */
/* ------------------------------------------------------------------------------------------ */

function addPolygonRasterization(
  {graph, view}: Context,
  options: {
    width: number;
    height: number;
    featureCount: number;
    vertexCount: number;
    crossingCapacity: number;
    longitudeLatitude?: boolean;
  }
): void {
  const cellCount = options.width * options.height;
  const {featureCount} = options;
  graph.add(
    new GPUPolygonRasterization({
      spatialContext: options.longitudeLatitude
        ? {coordinateSpace: 'longitude-latitude', metric: 'none'}
        : undefined,
      grid: makeGrid(options.width, options.height),
      parameters: view('float32', 4),
      positions: view('float32x2', options.vertexCount),
      featureOffsets: view('uint32', featureCount + 1),
      polygonOffsets: view('uint32', featureCount + 1),
      ringOffsets: view('uint32', featureCount + 1),
      crossingCapacity: options.crossingCapacity,
      output: {
        zones: view('uint32', cellCount),
        boundary: view('uint32', cellCount),
        status: {overflow: view('uint32', 1), crossingCount: view('uint32', 1)}
      }
    })
  );
}

const POLYGON_RASTERIZATION_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUPolygonRasterization',
    name: 'longitude copies, every output',
    build: context =>
      addPolygonRasterization(context, {
        ...MODERATE,
        featureCount: 10,
        vertexCount: 400,
        crossingCapacity: 10000,
        longitudeLatitude: true
      })
  },
  {
    exportName: 'GPUPolygonRasterization',
    name: 'every output, single sort',
    build: context =>
      addPolygonRasterization(context, {
        ...MODERATE,
        featureCount: 10,
        vertexCount: 400,
        crossingCapacity: 5000
      })
  },
  {
    exportName: 'GPUPolygonRasterization',
    name: 'every output, two sorts',
    build: context =>
      addPolygonRasterization(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        featureCount: 5000,
        vertexCount: 20000,
        crossingCapacity: 100000
      })
  },
  {
    exportName: 'GPUPolygonRasterization',
    name: 'maximum grid, features and crossing capacity',
    build: context =>
      addPolygonRasterization(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        featureCount: context.maximumRows(4) - 1,
        vertexCount: context.maximumRows(8),
        crossingCapacity: context.maximumRows(4)
      })
  },
  {
    exportName: 'GPUPolygonRasterization',
    name: 'required outputs only',
    build: ({graph, view}) =>
      graph.add(
        new GPUPolygonRasterization({
          grid: makeGrid(20, 10),
          parameters: view('float32', 4),
          positions: view('float32x2', 20),
          featureOffsets: view('uint32', 2),
          polygonOffsets: view('uint32', 2),
          ringOffsets: view('uint32', 2),
          crossingCapacity: 64,
          output: {zones: view('uint32', 200), status: {overflow: view('uint32', 1)}}
        })
      )
  }
];

function addRasterJoin(
  {graph, view}: Context,
  options: {
    width: number;
    height: number;
    pointCount: number;
    zoneCount: number;
    sums: boolean;
    boundary: boolean;
  }
): void {
  const cellCount = options.width * options.height;
  const {pointCount, zoneCount} = options;
  graph.add(
    new GPURasterJoin({
      grid: makeGrid(options.width, options.height),
      parameters: view('float32', 4),
      points: view('float32x2', pointCount),
      values: options.sums ? view('float32', pointCount) : undefined,
      zones: view('uint32', cellCount),
      boundary: options.boundary ? view('uint32', cellCount) : undefined,
      zoneCount,
      output: {
        counts: view('uint32', zoneCount),
        sums: options.sums ? view('float32', zoneCount) : undefined,
        outsideCount: view('uint32', 1),
        pointZones: view('uint32', pointCount),
        ...(options.boundary
          ? {
              boundaryCounts: view('uint32', zoneCount),
              unassignedBoundaryCount: view('uint32', 1),
              pointBoundaryMask: view('uint32', pointCount)
            }
          : {})
      }
    })
  );
}

const RASTER_JOIN_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterJoin',
    name: 'every output',
    build: context =>
      addRasterJoin(context, {
        ...MODERATE,
        pointCount: 5000,
        zoneCount: 200,
        sums: true,
        boundary: true
      })
  },
  {
    exportName: 'GPURasterJoin',
    name: 'counts only',
    build: context =>
      addRasterJoin(context, {
        ...MODERATE,
        pointCount: 5000,
        zoneCount: 200,
        sums: false,
        boundary: false
      })
  },
  {
    exportName: 'GPURasterJoin',
    name: 'maximum points and zones, every output',
    build: context =>
      addRasterJoin(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        pointCount: context.maximumRows(8),
        zoneCount: getGPURasterJoinMaximumSummedZoneCount(context.device),
        sums: true,
        boundary: true
      })
  },
  {
    exportName: 'GPURasterJoin',
    name: 'zones above the maximum with sums, every output',
    build: context =>
      addRasterJoin(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        pointCount: context.maximumRows(8),
        zoneCount: getGPURasterJoinMaximumSummedZoneCount(context.device) + 1,
        sums: true,
        boundary: true
      })
  },
  {
    exportName: 'GPURasterJoin',
    name: 'maximum points and zones, counts only',
    build: context =>
      addRasterJoin(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        pointCount: context.maximumRows(8),
        zoneCount: getGPURasterJoinMaximumZoneCount(context.device),
        sums: false,
        boundary: true
      })
  },
  {
    exportName: 'GPURasterJoin',
    name: 'zones above the maximum, counts only',
    build: context =>
      addRasterJoin(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        pointCount: context.maximumRows(8),
        zoneCount: getGPURasterJoinMaximumZoneCount(context.device) + 1,
        sums: false,
        boundary: true
      })
  }
];

/* ------------------------------------------------------------------------------------------ */
/* Raster algebra                                                                             */
/* ------------------------------------------------------------------------------------------ */

function addReclassify(
  {graph, view}: Context,
  rowCount: number,
  breakCount: number,
  partialRows: boolean
): void {
  graph.add(
    new GPURasterReclassify({
      values: view('float32', rowCount),
      rowCount: partialRows ? rowCount - 1 : undefined,
      validity: view('uint32', rowCount),
      noDataValue: -9999,
      breaks: view('float32', breakCount),
      classValues: view('float32', breakCount + 1),
      parameters: view('float32', 4),
      output: {
        classes: view('uint32', rowCount),
        reclassified: view('float32', rowCount),
        classCounts: view('uint32', breakCount + 1)
      }
    })
  );
}

const RECLASSIFY_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterReclassify',
    name: 'every output, privatized class counts',
    build: context => addReclassify(context, 5000, 15, true)
  },
  {
    exportName: 'GPURasterReclassify',
    name: 'every output, largest privatized class counts',
    build: context => addReclassify(context, 5000, 4095, false)
  },
  {
    exportName: 'GPURasterReclassify',
    name: 'every output, global class counts',
    build: context => addReclassify(context, 5000, 4096, false)
  },
  {
    exportName: 'GPURasterReclassify',
    name: 'classes only without validity',
    build: ({graph, view}) =>
      graph.add(
        new GPURasterReclassify({
          values: view('float32', 5000),
          breaks: view('float32', 8),
          parameters: view('float32', 4),
          output: {classes: view('uint32', 5000)}
        })
      )
  },
  {
    exportName: 'GPURasterReclassify',
    name: 'maximum rows, every output, many breaks',
    build: context => addReclassify(context, context.maximumRows(4), 65535, false)
  }
];

function addWeightedOverlay(
  {graph, view}: Context,
  layerCount: number,
  cellCount: number,
  tables: number | undefined
): void {
  graph.add(
    new GPUWeightedOverlay({
      stack: view('float32', layerCount * cellCount),
      layerCount,
      cellCount,
      noDataValue: -9999,
      parameters: view('float32', 8 + 8 * layerCount),
      ...(tables
        ? {
            remapBreaks: view('float32', layerCount * tables),
            remapValues: view('float32', layerCount * (tables + 1)),
            maximumBreakCount: tables
          }
        : {}),
      output: {
        score: view('float32', cellCount),
        outputValidity: view('uint32', cellCount),
        scoreRange: view('float32', 2)
      }
    })
  );
}

const WEIGHTED_OVERLAY_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUWeightedOverlay',
    name: 'every output, 16 layers, remap tables',
    build: context => addWeightedOverlay(context, 16, 300, 32)
  },
  {
    exportName: 'GPUWeightedOverlay',
    name: 'every output, 16 layers, no tables',
    build: context => addWeightedOverlay(context, 16, 300, undefined)
  },
  {
    exportName: 'GPUWeightedOverlay',
    name: 'one layer with maximum remap tables',
    build: context => addWeightedOverlay(context, 1, 300, 1 << 16)
  },
  {
    exportName: 'GPUWeightedOverlay',
    name: 'maximum cells, 16 layers, maximum remap tables',
    build: context => addWeightedOverlay(context, 16, context.maximumRows(4 * 16), 1 << 16)
  }
];

const CELL_STATISTICS_OUTPUTS = [
  'minimum',
  'maximum',
  'range',
  'sum',
  'mean',
  'standardDeviation',
  'majority',
  'minority'
] as const;

function addCellStatistics(
  {graph, view}: Context,
  layerCount: number,
  cellCount: number,
  only?: (typeof CELL_STATISTICS_OUTPUTS)[number] | 'variety' | 'counts'
): void {
  const all: Record<string, GraphDataView<'float32'> | GraphDataView<'uint32'>> = {
    ...Object.fromEntries(CELL_STATISTICS_OUTPUTS.map(name => [name, view('float32', cellCount)])),
    variety: view('uint32', cellCount),
    counts: view('uint32', cellCount)
  };
  graph.add(
    new GPURasterCellStatistics({
      stack: view('float32', layerCount * cellCount),
      layerCount,
      cellCount,
      noDataValue: -9999,
      parameters: view('float32', 4),
      output: (only ? {[only]: all[only]} : all) as GPURasterCellStatisticsOutput
    })
  );
}

const CELL_STATISTICS_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterCellStatistics',
    name: 'every output, 64 layers',
    build: context => addCellStatistics(context, 64, 300)
  },
  ...(
    [
      'minimum',
      'maximum',
      'range',
      'sum',
      'mean',
      'standardDeviation',
      'majority',
      'minority',
      'variety',
      'counts'
    ] as const
  ).map(
    (only): KernelAuditCase => ({
      exportName: 'GPURasterCellStatistics',
      name: `${only} only, 64 layers`,
      build: context => addCellStatistics(context, 64, 300, only)
    })
  ),
  {
    exportName: 'GPURasterCellStatistics',
    name: 'maximum cells, every output, 64 layers',
    build: context => addCellStatistics(context, 64, context.maximumRows(4 * 64))
  },
  {
    exportName: 'GPURasterCellStatistics',
    name: 'maximum cells, every output, 1 layer',
    build: context => addCellStatistics(context, 1, context.maximumRows(4))
  }
];

const CONDITIONAL_CASES: KernelAuditCase[] = [
  ...[
    {mask: true, a: true, b: true},
    {mask: true, a: false, b: false},
    {mask: false, a: true, b: false},
    {mask: false, a: false, b: true}
  ].map(
    ({mask, a, b}): KernelAuditCase => ({
      exportName: 'GPURasterConditional',
      name: `${mask ? 'mask' : 'condition values'}, a ${a}, b ${b}, output mask`,
      build: ({graph, view}) =>
        graph.add(
          new GPURasterConditional({
            cellCount: 5000,
            mask: mask ? view('uint32', 5000) : undefined,
            conditionValues: mask ? undefined : view('float32', 5000),
            a: a ? view('float32', 5000) : undefined,
            b: b ? view('float32', 5000) : undefined,
            noDataValue: -9999,
            parameters: view('float32', 8),
            output: {values: view('float32', 5000), outputMask: view('uint32', 5000)}
          })
        )
    })
  ),
  {
    exportName: 'GPURasterConditional',
    name: 'maximum cells, every input and output',
    build: ({graph, view, maximumRows}) => {
      const cellCount = maximumRows(4);
      graph.add(
        new GPURasterConditional({
          cellCount,
          conditionValues: view('float32', cellCount),
          a: view('float32', cellCount),
          b: view('float32', cellCount),
          noDataValue: -9999,
          parameters: view('float32', 8),
          output: {values: view('float32', cellCount), outputMask: view('uint32', cellCount)}
        })
      );
    }
  }
];

const ARITHMETIC_CASES: KernelAuditCase[] = [
  ...[true, false].map(
    (hasB): KernelAuditCase => ({
      exportName: 'GPURasterArithmetic',
      name: hasB ? 'two rasters with nodata' : 'one raster and a constant',
      build: ({graph, view}) =>
        graph.add(
          new GPURasterArithmetic({
            cellCount: 5000,
            a: view('float32', 5000),
            b: hasB ? view('float32', 5000) : undefined,
            noDataValue: -9999,
            parameters: view('float32', 8),
            output: {values: view('float32', 5000)}
          })
        )
    })
  ),
  {
    exportName: 'GPURasterArithmetic',
    name: 'maximum cells, two rasters',
    build: ({graph, view, maximumRows}) => {
      const cellCount = maximumRows(4);
      graph.add(
        new GPURasterArithmetic({
          cellCount,
          a: view('float32', cellCount),
          b: view('float32', cellCount),
          noDataValue: -9999,
          parameters: view('float32', 8),
          output: {values: view('float32', cellCount)}
        })
      );
    }
  }
];

/* ------------------------------------------------------------------------------------------ */
/* Raster patches                                                                             */
/* ------------------------------------------------------------------------------------------ */

function patchLabels({view}: Context, width: number, height: number) {
  return {
    grid: makeGrid(width, height),
    labels: view('uint32', width * height),
    validity: view('uint32', width * height),
    status: {
      converged: view('uint32', 1),
      overflow: view('uint32', 1),
      componentCount: view('uint32', 1)
    }
  };
}

function addPatchMetrics(
  context: Context,
  width: number,
  height: number,
  capacity: number,
  countRasterBorder: boolean
): void {
  const {graph, view} = context;
  graph.add(
    new GPURasterPatchMetrics({
      ...patchLabels(context, width, height),
      countRasterBorder,
      output: {
        pixelCounts: view('uint32', capacity),
        areas: view('float32', capacity),
        perimeterFaceCounts: view('uint32', capacity),
        perimeters: view('float32', capacity),
        minColumns: view('uint32', capacity),
        minRows: view('uint32', capacity),
        maxColumns: view('uint32', capacity),
        maxRows: view('uint32', capacity)
      }
    })
  );
}

const PATCH_METRICS_CASES: KernelAuditCase[] = [
  ...[true, false].map(
    (countRasterBorder): KernelAuditCase => ({
      exportName: 'GPURasterPatchMetrics',
      name: `every output, countRasterBorder ${countRasterBorder}`,
      build: context => addPatchMetrics(context, 100, 80, 500, countRasterBorder)
    })
  ),
  {
    exportName: 'GPURasterPatchMetrics',
    name: 'maximum grid and patch capacity',
    build: context => addPatchMetrics(context, LARGE_SIDE, LARGE_SIDE, context.maximumRows(4), true)
  },
  {
    exportName: 'GPURasterPatchMetrics',
    name: 'pixel counts only',
    build: context => {
      context.graph.add(
        new GPURasterPatchMetrics({
          grid: makeGrid(100, 80),
          labels: context.view('uint32', 8000),
          output: {pixelCounts: context.view('uint32', 100)}
        })
      );
    }
  }
];

function addSieve(
  context: Context,
  width: number,
  height: number,
  patchCapacity: number,
  mode: 'remove' | 'merge',
  connectivity: 4 | 8
): void {
  const {graph, view} = context;
  graph.add(
    new GPURasterSieve({
      ...patchLabels(context, width, height),
      patchCapacity,
      parameters: view('uint32', 4),
      mode,
      connectivity,
      output: {
        labels: view('uint32', width * height),
        patchTargets: view('uint32', patchCapacity),
        sievedCount: view('uint32', 1)
      }
    })
  );
}

const SIEVE_CASES: KernelAuditCase[] = [
  ...(['remove', 'merge'] as const).flatMap(mode =>
    ([4, 8] as const).map(
      (connectivity): KernelAuditCase => ({
        exportName: 'GPURasterSieve',
        name: `${mode}, connectivity ${connectivity}, every output`,
        build: context => addSieve(context, 100, 80, 500, mode, connectivity)
      })
    )
  ),
  ...(['remove', 'merge'] as const).map(
    (mode): KernelAuditCase => ({
      exportName: 'GPURasterSieve',
      name: `${mode}, maximum grid and patch capacity`,
      build: context => addSieve(context, LARGE_SIDE, LARGE_SIDE, LARGE_CELLS, mode, 8)
    })
  ),
  {
    exportName: 'GPURasterSieve',
    name: 'labels output only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPURasterSieve({
          grid: makeGrid(100, 80),
          labels: view('uint32', 8000),
          patchCapacity: 100,
          parameters: view('uint32', 4),
          output: {labels: view('uint32', 8000)}
        })
      );
    }
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPURasterExtremaPyramid                                                                    */
/* ------------------------------------------------------------------------------------------ */

type ExtremaPyramidOptions = {
  width: number;
  height: number;
  firstBlockSize?: number;
  levelCount?: number;
  footprint?: 'cell' | 'bilinear';
  combined: boolean;
  textureBand?: boolean;
};

function addExtremaPyramid(context: Context, options: ExtremaPyramidOptions): void {
  const {graph, view} = context;
  const {firstBlockSize, levelCount, footprint} = options;
  const grid = makeGrid(options.width, options.height);
  const {length} = getGPURasterExtremaPyramidLayout(grid, {firstBlockSize, levelCount, footprint});
  graph.add(
    new GPURasterExtremaPyramid({
      grid,
      firstBlockSize,
      levelCount,
      footprint,
      input: options.textureBand
        ? makeTextureBand(context, grid, 'pyramid-input')
        : makeBufferBand(context, grid, 'float32', 'pyramid-input'),
      output: options.combined
        ? {combined: view('float32', 2 * length)}
        : {maximum: view('float32', length), minimum: view('float32', length)}
    })
  );
}

const EXTREMA_PYRAMID_CASES: KernelAuditCase[] = [
  ...([1, 4, 256] as const).flatMap(firstBlockSize =>
    (['cell', 'bilinear'] as const).map(
      (footprint): KernelAuditCase => ({
        exportName: 'GPURasterExtremaPyramid',
        name: `maximum and minimum, first block ${firstBlockSize}, ${footprint}`,
        build: context =>
          addExtremaPyramid(context, {
            ...MODERATE,
            firstBlockSize,
            footprint,
            combined: false
          })
      })
    )
  ),
  {
    exportName: 'GPURasterExtremaPyramid',
    name: 'combined, bilinear, limited levels',
    build: context =>
      addExtremaPyramid(context, {
        ...MODERATE,
        firstBlockSize: 2,
        levelCount: 3,
        footprint: 'bilinear',
        combined: true
      })
  },
  {
    exportName: 'GPURasterExtremaPyramid',
    name: 'texture input band, combined',
    build: context => addExtremaPyramid(context, {...MODERATE, combined: true, textureBand: true})
  },
  {
    exportName: 'GPURasterExtremaPyramid',
    name: 'maximum grid, first block 1, maximum and minimum',
    build: context =>
      addExtremaPyramid(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        firstBlockSize: 1,
        combined: false
      })
  },
  {
    exportName: 'GPURasterExtremaPyramid',
    name: 'maximum grid, first block 2, combined',
    build: context =>
      addExtremaPyramid(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        firstBlockSize: 2,
        combined: true
      })
  },
  {
    exportName: 'GPURasterExtremaPyramid',
    name: 'largest texture input, first block 4',
    build: context =>
      addExtremaPyramid(context, {
        width: TEXTURE_SIDE,
        height: TEXTURE_SIDE,
        combined: true,
        textureBand: true
      })
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPURasterSampling, GPURasterProfile                                                        */
/* ------------------------------------------------------------------------------------------ */

function addSampling(
  {graph, view}: Context,
  width: number,
  height: number,
  pointCapacity: number
): void {
  graph.add(
    new GPURasterSampling({
      grid: makeGrid(width, height),
      values: view('float32', width * height),
      noDataValue: -9999,
      validity: view('uint32', width * height),
      positions: view('float32x2', pointCapacity),
      pointCount: view('uint32', 1),
      parameters: view('float32', 12),
      output: {
        values: view('float32', pointCapacity),
        outputValidity: view('uint32', pointCapacity)
      }
    })
  );
}

const SAMPLING_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterSampling',
    name: 'every option',
    build: context => addSampling(context, 200, 150, 5000)
  },
  {
    exportName: 'GPURasterSampling',
    name: 'values only',
    build: ({graph, view}) =>
      graph.add(
        new GPURasterSampling({
          grid: makeGrid(200, 150),
          values: view('float32', 30000),
          positions: view('float32x2', 5000),
          parameters: view('float32', 12),
          output: {values: view('float32', 5000)}
        })
      )
  },
  {
    exportName: 'GPURasterSampling',
    name: 'maximum grid and point capacity',
    build: context => addSampling(context, LARGE_SIDE, LARGE_SIDE, context.maximumRows(8))
  }
];

type ProfileOptions = {
  width: number;
  height: number;
  vertexCount: number;
  lineCount: number;
  sampleCapacity: number;
};

function addProfile({graph, view}: Context, options: ProfileOptions): void {
  const {sampleCapacity, lineCount} = options;
  graph.add(
    new GPURasterProfile({
      grid: makeGrid(options.width, options.height),
      values: view('float32', options.width * options.height),
      noDataValue: -9999,
      validity: view('uint32', options.width * options.height),
      positions: view('float32x2', options.vertexCount),
      lineOffsets: view('uint32', lineCount + 1),
      sampleCapacity,
      parameters: view('float32', 12),
      output: {
        samplePositions: view('float32x2', sampleCapacity),
        sampleDistances: view('float32', sampleCapacity),
        sampleValues: view('float32', sampleCapacity),
        sampleLineIds: view('uint32', sampleCapacity),
        sampleCumulativeGain: view('float32', sampleCapacity),
        sampleCumulativeLoss: view('float32', sampleCapacity),
        lineSampleOffsets: view('uint32', lineCount + 1),
        lineLengths: view('float32', lineCount),
        lineGains: view('float32', lineCount),
        lineLosses: view('float32', lineCount),
        lineMinimums: view('float32', lineCount),
        lineMaximums: view('float32', lineCount),
        status: {
          count: view('uint32', 1),
          requiredCount: view('uint32', 1),
          overflow: view('uint32', 1)
        }
      }
    })
  );
}

const PROFILE_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterProfile',
    name: 'every output',
    build: context =>
      addProfile(context, {
        width: 200,
        height: 150,
        vertexCount: 500,
        lineCount: 20,
        sampleCapacity: 5000
      })
  },
  {
    exportName: 'GPURasterProfile',
    name: 'status only with explicit sample capacity',
    build: ({graph, view}) =>
      graph.add(
        new GPURasterProfile({
          grid: makeGrid(200, 150),
          values: view('float32', 30000),
          positions: view('float32x2', 500),
          lineOffsets: view('uint32', 21),
          sampleCapacity: 5000,
          parameters: view('float32', 12),
          output: {status: {count: view('uint32', 1), overflow: view('uint32', 1)}}
        })
      )
  },
  {
    exportName: 'GPURasterProfile',
    name: 'maximum vertices, paths and sample capacity',
    build: context =>
      addProfile(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        vertexCount: context.maximumRows(8),
        lineCount: getGPURasterProfileMaximumLineCount(context.device),
        sampleCapacity: context.maximumRows(8)
      })
  },
  {
    exportName: 'GPURasterProfile',
    name: 'paths above the maximum',
    build: context =>
      addProfile(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        vertexCount: context.maximumRows(8),
        lineCount: getGPURasterProfileMaximumLineCount(context.device) + 1,
        sampleCapacity: context.maximumRows(8)
      })
  },
  {
    exportName: 'GPURasterProfile',
    name: 'a single path at maximum capacities',
    build: context =>
      addProfile(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        vertexCount: context.maximumRows(8),
        lineCount: 1,
        sampleCapacity: context.maximumRows(8)
      })
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPURasterStretch                                                                           */
/* ------------------------------------------------------------------------------------------ */

type StretchOptions = {
  width: number;
  height: number;
  binCount?: number;
  lutSize?: number;
  paletteSize?: number;
};

function addStretch({graph, view}: Context, options: StretchOptions): void {
  const cellCount = options.width * options.height;
  const binCount = options.binCount ?? 1024;
  const lutSize = options.lutSize ?? 256;
  graph.add(
    new GPURasterStretch({
      values: view('float32', cellCount),
      grid: makeGrid(options.width, options.height),
      noDataValue: -9999,
      validity: view('uint32', cellCount),
      regionMask: view('uint32', cellCount),
      parameters: view('float32', 16),
      palette: options.paletteSize ? view('uint32', options.paletteSize) : undefined,
      binCount: options.binCount,
      lutSize: options.lutSize,
      output: {
        stretched: view('float32', cellCount),
        ...(options.paletteSize
          ? {colors: view('uint32', cellCount), lutColors: view('uint32', lutSize)}
          : {}),
        lut: view('float32', lutSize),
        histogram: view('uint32', binCount),
        statistics: view('float32', 8)
      }
    })
  );
}

const STRETCH_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPURasterStretch',
    name: 'every output, default sizes',
    build: context => addStretch(context, {...MODERATE, paletteSize: 256})
  },
  {
    exportName: 'GPURasterStretch',
    name: 'every output, minimum bins and LUT',
    build: context => addStretch(context, {...MODERATE, binCount: 16, lutSize: 2, paletteSize: 2})
  },
  {
    exportName: 'GPURasterStretch',
    name: 'every output, largest privatized histogram',
    build: context =>
      addStretch(context, {...MODERATE, binCount: 4096, lutSize: 4096, paletteSize: 1024})
  },
  {
    exportName: 'GPURasterStretch',
    name: 'every output, maximum bins, LUT and palette',
    build: context =>
      addStretch(context, {
        ...MODERATE,
        binCount: 65536,
        lutSize: 65536,
        paletteSize: 65536
      })
  },
  {
    exportName: 'GPURasterStretch',
    name: 'maximum grid, every output, maximum bins',
    build: context =>
      addStretch(context, {
        width: LARGE_SIDE,
        height: LARGE_SIDE,
        binCount: 65536,
        lutSize: 65536,
        paletteSize: 65536
      })
  },
  {
    exportName: 'GPURasterStretch',
    name: 'maximum width 2^24, stretched output only',
    build: ({graph, view, maximumRows}) => {
      const width = 1 << 24;
      const height = Math.floor(maximumRows(4) / width);
      graph.add(
        new GPURasterStretch({
          values: view('float32', width * height),
          grid: makeGrid(width, height),
          parameters: view('float32', 16),
          output: {stretched: view('float32', width * height)}
        })
      );
    }
  },
  {
    exportName: 'GPURasterStretch',
    name: 'maximum height 2^24, stretched output only',
    build: ({graph, view, maximumRows}) => {
      const height = 1 << 24;
      const width = Math.floor(maximumRows(4) / height);
      graph.add(
        new GPURasterStretch({
          values: view('float32', width * height),
          grid: makeGrid(width, height),
          parameters: view('float32', 16),
          output: {stretched: view('float32', width * height)}
        })
      );
    }
  }
];

/* ------------------------------------------------------------------------------------------ */
/* GPURasterZonalStatistics                                                                   */
/* ------------------------------------------------------------------------------------------ */

type ZonalOptions = {
  width: number;
  height: number;
  zoneCapacity: number;
  sumOrder: 'atomic' | 'sorted';
  cellSizeMode: 'uniform' | 'geographic';
  values: 'float32' | 'uint32' | 'sint32' | 'texture';
  ignoredZone?: number;
  /** Also wire weightSums, variances and standardDeviations. Defaults to true. */
  moments?: boolean;
};

function addZonalStatistics(context: Context, options: ZonalOptions): void {
  const {graph, view} = context;
  const {zoneCapacity} = options;
  const grid = makeGrid(options.width, options.height);
  const uintColumn = () => view('uint32', zoneCapacity);
  const floatColumn = () => view('float32', zoneCapacity);
  graph.add(
    new GPURasterZonalStatistics({
      grid,
      zones: view('uint32', options.width * options.height),
      values:
        options.values === 'texture'
          ? makeTextureBand(context, grid, 'zonal-values')
          : makeBufferBand(context, grid, options.values, 'zonal-values'),
      zoneCapacity,
      ignoredZone: options.ignoredZone,
      cellSizeMode: options.cellSizeMode,
      parameters: options.cellSizeMode === 'geographic' ? view('float32', 8) : undefined,
      sumOrder: options.sumOrder,
      output: {
        cellCounts: uintColumn(),
        valueCounts: uintColumn(),
        sums: floatColumn(),
        means: floatColumn(),
        minimums: floatColumn(),
        maximums: floatColumn(),
        ...(options.moments === false
          ? {}
          : {
              weightSums: floatColumn(),
              variances: floatColumn(),
              standardDeviations: floatColumn()
            }),
        status: {overflow: view('uint32', 1)}
      }
    })
  );
}

const ZONAL_STATISTICS_CASES: KernelAuditCase[] = [
  ...(['atomic', 'sorted'] as const).flatMap(sumOrder =>
    (['uniform', 'geographic'] as const).map(
      (cellSizeMode): KernelAuditCase => ({
        exportName: 'GPURasterZonalStatistics',
        name: `every output, ${sumOrder} sums, ${cellSizeMode} cells`,
        build: context =>
          addZonalStatistics(context, {
            ...MODERATE,
            zoneCapacity: 200,
            sumOrder,
            cellSizeMode,
            values: 'float32',
            ignoredZone: 0
          })
      })
    )
  ),
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'uint32 values',
    build: context =>
      addZonalStatistics(context, {
        ...MODERATE,
        zoneCapacity: 200,
        sumOrder: 'sorted',
        cellSizeMode: 'uniform',
        values: 'uint32'
      })
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'sint32 values',
    build: context =>
      addZonalStatistics(context, {
        ...MODERATE,
        zoneCapacity: 200,
        sumOrder: 'atomic',
        cellSizeMode: 'geographic',
        values: 'sint32'
      })
  },
  {
    exportName: 'GPURasterZonalStatistics',
    name: 'texture values',
    build: context =>
      addZonalStatistics(context, {
        ...MODERATE,
        zoneCapacity: 200,
        sumOrder: 'sorted',
        cellSizeMode: 'geographic',
        values: 'texture'
      })
  },
  ...(['atomic', 'sorted'] as const).flatMap(sumOrder =>
    (['uniform', 'geographic'] as const).flatMap(cellSizeMode =>
      [true, false].flatMap(moments =>
        [0, 1].map(
          (excess): KernelAuditCase => ({
            exportName: 'GPURasterZonalStatistics',
            name: `${excess ? 'zones above the maximum' : 'maximum grid and zones'}${
              moments ? '' : ' without moments'
            }, ${sumOrder} sums, ${cellSizeMode} cells`,
            build: context =>
              addZonalStatistics(context, {
                width: LARGE_SIDE,
                height: LARGE_SIDE,
                zoneCapacity:
                  (sumOrder === 'sorted'
                    ? getGPURasterZonalStatisticsMaximumSortedZoneCapacity(context.device)
                    : getGPURasterZonalStatisticsMaximumZoneCapacity(context.device)) + excess,
                sumOrder,
                cellSizeMode,
                values: 'float32',
                moments
              })
          })
        )
      )
    )
  )
];

/** Kernel-audit cases for raster. */
export const RASTER_CASES: KernelAuditCase[] = [
  ...CHANGE_DETECTION_CASES,
  ...COST_DISTANCE_CASES,
  ...COST_DISTANCE_PATH_CASES,
  ...DISTANCE_FIELD_CASES,
  ...LINE_INTEGRAL_CONVOLUTION_CASES,
  ...STREAMLINES_CASES,
  ...ISOLINES_CASES,
  ...ISOBANDS_CASES,
  ...ISOBAND_RINGS_CASES,
  ...PARTICLE_ADVECTION_CASES,
  ...POLYGON_RASTERIZATION_CASES,
  ...RASTER_JOIN_CASES,
  ...RECLASSIFY_CASES,
  ...WEIGHTED_OVERLAY_CASES,
  ...CELL_STATISTICS_CASES,
  ...CONDITIONAL_CASES,
  ...ARITHMETIC_CASES,
  ...PATCH_METRICS_CASES,
  ...SIEVE_CASES,
  ...EXTREMA_PYRAMID_CASES,
  ...SAMPLING_CASES,
  ...PROFILE_CASES,
  ...STRETCH_CASES,
  ...ZONAL_STATISTICS_CASES
];
