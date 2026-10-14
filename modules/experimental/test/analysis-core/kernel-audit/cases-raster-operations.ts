// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GraphBufferHandle,
  GraphDataView,
  type DrawCommandBufferView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterBandMath,
  GPURasterBufferToTexture,
  GPURasterCategoricalOverview,
  GPURasterConnectedComponents,
  GPURasterContourClassifier,
  GPURasterContours,
  GPURasterContrast,
  GPURasterCrossTileComponents,
  GPURasterDenseComponents,
  GPURasterGlobalHistogramMerge,
  GPURasterGlobalInitialize,
  GPURasterGlobalPercentile,
  GPURasterGlobalStatisticsMerge,
  GPURasterGradientMagnitude,
  GPURasterHistogram,
  GPURasterMorphology,
  GPURasterNDVI,
  GPURasterNeighborhood,
  GPURasterOtsuThreshold,
  GPURasterOverview,
  GPURasterRegionMeasurements,
  GPURasterStatistics,
  GPURasterTextureToBuffer,
  GPURasterThreshold,
  GPURasterTileCoreExtract,
  GPURasterTileHaloFill,
  type GPURasterBufferBand,
  type GPURasterCrossTile,
  type GPURasterMetadata,
  type GPURasterRegionMeasurementOutputs,
  type GPURasterScalarFormat
} from '../../../src/gpu-raster';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

/** Moderate raster used by the "every option" cases. */
const WIDTH = 24;
const HEIGHT = 16;
const PIXELS = WIDTH * HEIGHT;
/** Largest square grid whose float32 samples fit one 128 MiB binding (64 MiB per band). */
const MAXIMUM_SIDE = 4096;
const MAXIMUM_PIXELS = MAXIMUM_SIDE * MAXIMUM_SIDE;
/** Elements one 1-D dispatch covers on the core limits: 65535 workgroups x 256 invocations. */
const DISPATCH_ELEMENTS = 65535 * 256;
/** Three fixed-point words per output region, in one 128 MiB binding. */
const CROSS_TILE_CAPACITY = Math.floor(134217728 / 12);
const DISPATCH_WIDTH = 65535;
const DISPATCH_HEIGHT = 256;
const DISPATCH_PIXELS = DISPATCH_WIDTH * DISPATCH_HEIGHT;

function metadata(width: number, height: number): GPURasterMetadata {
  return {width, height, affine: [2, 0, 100, 0, -2, 200], pixelInterpretation: 'area'};
}

function band<Format extends GPURasterScalarFormat>(
  context: KernelAuditContext,
  id: string,
  format: Format,
  length: number,
  options: {validity?: boolean; noDataValue?: number; scale?: number; offset?: number} = {}
): GPURasterBufferBand<Format> {
  const {validity, ...calibration} = options;
  return {
    id,
    format,
    storage: {kind: 'buffer', values: context.view(format, length, id)},
    ...(validity ? {validity: context.view('uint32', length, `${id}-validity`)} : {}),
    ...calibration
  } as GPURasterBufferBand<Format>;
}

/** A fully calibrated band: validity flags, nodata, scale and offset. */
function calibrated<Format extends GPURasterScalarFormat>(
  context: KernelAuditContext,
  id: string,
  format: Format,
  length: number
): GPURasterBufferBand<Format> {
  return band(context, id, format, length, {
    validity: true,
    noDataValue: format === 'uint32' ? 4294967295 : format === 'sint32' ? -2147483648 : -9999,
    scale: 0.5,
    offset: 4
  });
}

function textureView<Format extends 'r32float' | 'rgba32float' | 'r32uint' | 'rgba32uint'>(
  context: KernelAuditContext,
  format: Format,
  width: number,
  height: number,
  name: string
): GraphTextureView<Format> {
  return context.graph.createTextureView(context.texture(format, width, height, name), {
    mipLevelCount: 1
  }) as GraphTextureView<Format>;
}

function drawCommands(context: KernelAuditContext, capacity: number): DrawCommandBufferView {
  const words = context.view('uint32', capacity * 4, 'draw');
  const buffer: GraphBufferHandle = words.buffer;
  return {
    type: 'draw',
    capacity,
    recordByteLength: 16,
    buffer,
    words,
    instanceCounts: new GraphDataView(buffer, {
      format: 'uint32',
      length: capacity,
      byteOffset: 4,
      byteStride: 16,
      rowByteLength: 4
    }),
    firstInstances: new GraphDataView(buffer, {
      format: 'uint32',
      length: capacity,
      byteOffset: 12,
      byteStride: 16,
      rowByteLength: 4
    })
  } as DrawCommandBufferView;
}

function measurements(
  context: KernelAuditContext,
  id: string,
  capacity: number
): GPURasterRegionMeasurementOutputs {
  return {
    pixelCounts: context.view('uint32', capacity, `${id}-pixels`),
    intensityCounts: context.view('uint32', capacity, `${id}-intensity-counts`),
    intensitySums: context.view('float32', capacity, `${id}-sums`),
    intensityMinimums: context.view('float32', capacity, `${id}-minimums`),
    intensityMaximums: context.view('float32', capacity, `${id}-maximums`),
    intensityMeans: context.view('float32', capacity, `${id}-means`),
    columnSums: context.view('float32', capacity, `${id}-column-sums`),
    rowSums: context.view('float32', capacity, `${id}-row-sums`),
    centroidColumns: context.view('float32', capacity, `${id}-centroid-columns`),
    centroidRows: context.view('float32', capacity, `${id}-centroid-rows`),
    areas: context.view('float32', capacity, `${id}-areas`)
  };
}

function accumulator(context: KernelAuditContext, bins: number) {
  return {
    extent: context.view('float32', 2, 'global-extent'),
    count: context.view('uint32', 1, 'global-count'),
    sum: context.view('float32', 1, 'global-sum'),
    histogram: context.view('uint32', bins, 'global-histogram'),
    overflow: context.view('uint32', 1, 'global-overflow')
  };
}

function crossTile(
  context: KernelAuditContext,
  global: GPURasterMetadata,
  bounds: [number, number, number, number],
  regionCapacity: number,
  id: string
): GPURasterCrossTile {
  const width = bounds[2] - bounds[0];
  const height = bounds[3] - bounds[1];
  const pixelCount = width * height;
  const [a, b, c, d, e, f] = global.affine;
  return {
    metadata: {
      width,
      height,
      affine: [a, b, c + a * bounds[0] + b * bounds[1], d, e, f + d * bounds[0] + e * bounds[1]],
      pixelInterpretation: global.pixelInterpretation
    },
    pixelBounds: bounds,
    labels: context.view('uint32', pixelCount, `${id}-labels`),
    labelValidity: context.view('uint32', pixelCount, `${id}-validity`),
    componentCount: context.view('uint32', 1, `${id}-count`),
    converged: context.view('uint32', 1, `${id}-converged`),
    overflow: context.view('uint32', 1, `${id}-overflow`),
    measurements: measurements(context, `${id}-local`, regionCapacity),
    outputLabels: context.view('uint32', pixelCount, `${id}-global-labels`),
    outputValidity: context.view('uint32', pixelCount, `${id}-global-validity`)
  };
}

/** Foreground flags: validity and nodata, but the identity calibration the mask kernels require. */
function flags(context: KernelAuditContext, id: string, length: number) {
  return band(context, id, 'uint32', length, {validity: true, noDataValue: 4294967295});
}

const FORMATS = ['float32', 'uint32', 'sint32'] as const;

/** Kernel-audit cases for raster-operations. */
export const RASTER_OPERATIONS_CASES: KernelAuditCase[] = [
  // ---- largest sizes one 1-D dispatch covers (65535 workgroups x 256 invocations) ----
  {
    exportName: 'GPURasterStatistics',
    name: 'largest one-dispatch size 65535x256',
    build: context =>
      new GPURasterStatistics({
        width: DISPATCH_WIDTH,
        height: DISPATCH_HEIGHT,
        input: calibrated(context, 'input', 'float32', DISPATCH_PIXELS),
        count: context.view('uint32', 1, 'count'),
        sum: context.view('float32', 1, 'sum'),
        mean: context.view('float32', 1, 'mean'),
        extent: context.view('float32', 2, 'extent')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterGlobalStatisticsMerge',
    name: 'largest one-dispatch size 65535x256',
    build: context =>
      new GPURasterGlobalStatisticsMerge({
        width: DISPATCH_WIDTH,
        height: DISPATCH_HEIGHT,
        input: calibrated(context, 'tile', 'float32', DISPATCH_PIXELS),
        accumulator: accumulator(context, 256)
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterGlobalHistogramMerge',
    name: 'largest one-dispatch size 65535x256',
    build: context =>
      new GPURasterGlobalHistogramMerge({
        width: DISPATCH_WIDTH,
        height: DISPATCH_HEIGHT,
        input: calibrated(context, 'tile', 'float32', DISPATCH_PIXELS),
        accumulator: accumulator(context, 256)
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterHistogram',
    name: 'largest one-dispatch size 65535x256, 256 bins',
    build: context =>
      new GPURasterHistogram({
        input: calibrated(context, 'input', 'float32', DISPATCH_PIXELS),
        output: context.view('uint32', 256, 'histogram'),
        domain: 'valid-auto',
        domainOutput: context.view('float32', 2, 'domain-output')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterRegionMeasurements',
    name: 'largest one-dispatch region capacity',
    build: context =>
      new GPURasterRegionMeasurements({
        metadata: metadata(MAXIMUM_SIDE, MAXIMUM_SIDE),
        labels: context.view('uint32', MAXIMUM_PIXELS, 'labels'),
        labelValidity: context.view('uint32', MAXIMUM_PIXELS, 'labels-validity'),
        converged: context.view('uint32', 1, 'converged'),
        componentCount: context.view('uint32', 1, 'count'),
        overflow: context.view('uint32', 1, 'overflow'),
        intensity: calibrated(context, 'intensity', 'float32', MAXIMUM_PIXELS),
        output: measurements(context, 'regions', DISPATCH_ELEMENTS),
        capacity: DISPATCH_ELEMENTS
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterCrossTileComponents',
    name: 'largest one-dispatch candidate and global capacity',
    build: context => {
      const global = metadata(2 * MAXIMUM_SIDE, MAXIMUM_SIDE);
      new GPURasterCrossTileComponents({
        metadata: global,
        tiles: [
          crossTile(
            context,
            global,
            [0, 0, MAXIMUM_SIDE, MAXIMUM_SIDE],
            CROSS_TILE_CAPACITY / 2,
            'west'
          ),
          crossTile(
            context,
            global,
            [MAXIMUM_SIDE, 0, 2 * MAXIMUM_SIDE, MAXIMUM_SIDE],
            CROSS_TILE_CAPACITY / 2,
            'east'
          )
        ],
        connectivity: 8,
        maximumIterations: 64,
        componentCount: context.view('uint32', 1, 'global-count'),
        requiredComponentCount: context.view('uint32', 1, 'global-required'),
        converged: context.view('uint32', 1, 'global-converged'),
        overflow: context.view('uint32', 1, 'global-overflow'),
        output: measurements(context, 'global', CROSS_TILE_CAPACITY),
        capacity: CROSS_TILE_CAPACITY
      }).addToGraph(context.graph);
    }
  },
  {
    exportName: 'GPURasterTextureToBuffer',
    name: 'largest texture of the fake device (2048x2048 rgba32float)',
    build: context =>
      new GPURasterTextureToBuffer({
        input: {
          id: 'texture-band',
          format: 'float32',
          storage: {
            kind: 'texture',
            view: textureView(context, 'rgba32float', 2048, 2048, 'source'),
            channel: 3
          }
        },
        output: context.view('float32', 2048 * 2048, 'buffer'),
        outputValidity: context.view('uint32', 2048 * 2048, 'buffer-validity')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterBufferToTexture',
    name: 'largest texture of the fake device (2048x2048 rgba32float)',
    build: context =>
      new GPURasterBufferToTexture({
        input: calibrated(context, 'source', 'float32', 2048 * 2048),
        output: textureView(context, 'rgba32float', 2048, 2048, 'destination'),
        outputValidity: context.view('uint32', 2048 * 2048, 'destination-validity'),
        channel: 3,
        applyCalibration: true
      }).addToGraph(context.graph)
  },

  // ---- tile halo ----
  ...FORMATS.map(
    (format): KernelAuditCase => ({
      exportName: 'GPURasterTileCoreExtract',
      name: `${format} calibrated`,
      build: context =>
        new GPURasterTileCoreExtract({
          availablePixelBounds: [0, 0, WIDTH, HEIGHT],
          corePixelBounds: [4, 3, 20, 13],
          input: calibrated(context, 'available', format, PIXELS),
          output: context.view(format, 16 * 10, 'core'),
          outputValidity: context.view('uint32', 16 * 10, 'core-validity')
        } as never).addToGraph(context.graph)
    })
  ),
  {
    exportName: 'GPURasterTileCoreExtract',
    name: 'maximum size',
    build: context =>
      new GPURasterTileCoreExtract({
        availablePixelBounds: [0, 0, MAXIMUM_SIDE, MAXIMUM_SIDE],
        corePixelBounds: [0, 0, MAXIMUM_SIDE, MAXIMUM_SIDE],
        input: calibrated(context, 'available', 'float32', MAXIMUM_PIXELS),
        output: context.view('float32', MAXIMUM_PIXELS, 'core'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'core-validity')
      }).addToGraph(context.graph)
  },
  ...FORMATS.map(
    (format): KernelAuditCase => ({
      exportName: 'GPURasterTileHaloFill',
      name: `${format} four calibrated sources`,
      build: context =>
        new GPURasterTileHaloFill({
          pixelBounds: [0, 0, 16, 16],
          sources: [
            [0, 0, 8, 8],
            [8, 0, 16, 8],
            [0, 8, 8, 16],
            [8, 8, 16, 16]
          ].map(pixelBounds => ({
            pixelBounds: pixelBounds as [number, number, number, number],
            input: calibrated(context, 'source', format, 64)
          })),
          output: context.view(format, 256, 'halo'),
          outputValidity: context.view('uint32', 256, 'halo-validity')
        } as never).addToGraph(context.graph)
    })
  ),
  {
    exportName: 'GPURasterTileHaloFill',
    name: 'maximum size',
    build: context =>
      new GPURasterTileHaloFill({
        pixelBounds: [0, 0, MAXIMUM_SIDE, MAXIMUM_SIDE],
        sources: [
          {
            pixelBounds: [0, 0, MAXIMUM_SIDE, MAXIMUM_SIDE],
            input: calibrated(context, 'source', 'float32', MAXIMUM_PIXELS)
          }
        ],
        output: context.view('float32', MAXIMUM_PIXELS, 'halo'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'halo-validity')
      }).addToGraph(context.graph)
  },

  // ---- overview ----
  ...[1, [2, 3], 8].map(
    (scale): KernelAuditCase => ({
      exportName: 'GPURasterOverview',
      name: `scale ${JSON.stringify(scale)} weighted calibrated`,
      build: context => {
        const [horizontal, vertical] = typeof scale === 'number' ? [scale, scale] : scale;
        const outputLength = Math.ceil(WIDTH / horizontal) * Math.ceil(HEIGHT / vertical);
        new GPURasterOverview({
          metadata: metadata(WIDTH, HEIGHT),
          scale: scale as never,
          sourcePixelOrigin: [0, 0],
          input: calibrated(context, 'samples', 'float32', PIXELS),
          inputSum: context.view('float32', PIXELS, 'input-sum'),
          inputValidCount: context.view('uint32', PIXELS, 'input-count'),
          maximumInputValidCount: 64,
          output: context.view('float32', outputLength, 'means'),
          outputValidity: context.view('uint32', outputLength, 'validity'),
          sum: context.view('float32', outputLength, 'sums'),
          validCount: context.view('uint32', outputLength, 'counts')
        }).addToGraph(context.graph);
      }
    })
  ),
  {
    exportName: 'GPURasterOverview',
    name: 'direct calibrated',
    build: context =>
      new GPURasterOverview({
        metadata: metadata(WIDTH, HEIGHT),
        scale: [2, 3],
        sourcePixelOrigin: [0, 0],
        input: calibrated(context, 'samples', 'float32', PIXELS),
        output: context.view('float32', 72, 'means'),
        outputValidity: context.view('uint32', 72, 'validity'),
        sum: context.view('float32', 72, 'sums'),
        validCount: context.view('uint32', 72, 'counts')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterOverview',
    name: 'maximum size weighted, scale 8 and exact maximum valid count',
    build: context => {
      const outputLength = (MAXIMUM_SIDE / 8) ** 2;
      new GPURasterOverview({
        metadata: metadata(MAXIMUM_SIDE, MAXIMUM_SIDE),
        scale: 8,
        sourcePixelOrigin: [0, 0],
        input: calibrated(context, 'samples', 'float32', MAXIMUM_PIXELS),
        inputSum: context.view('float32', MAXIMUM_PIXELS, 'input-sum'),
        inputValidCount: context.view('uint32', MAXIMUM_PIXELS, 'input-count'),
        maximumInputValidCount: Math.floor(0xffffffff / 64),
        output: context.view('float32', outputLength, 'means'),
        outputValidity: context.view('uint32', outputLength, 'validity'),
        sum: context.view('float32', outputLength, 'sums'),
        validCount: context.view('uint32', outputLength, 'counts')
      }).addToGraph(context.graph);
    }
  },
  ...(['uint32', 'sint32'] as const).flatMap(format =>
    (['nearest', 'mode'] as const).flatMap(policy =>
      [[2, 3] as const, [8, 8] as const].map(
        (scale): KernelAuditCase => ({
          exportName: 'GPURasterCategoricalOverview',
          name: `${format} ${policy} scale ${scale.join('x')} with coverage`,
          build: context => {
            const outputLength = Math.ceil(WIDTH / scale[0]) * Math.ceil(HEIGHT / scale[1]);
            new GPURasterCategoricalOverview({
              metadata: metadata(WIDTH, HEIGHT),
              scale: scale as never,
              sourcePixelOrigin: [0, 0],
              input: calibrated(context, 'labels', format, PIXELS),
              policy,
              output: context.view(format, outputLength, 'selected'),
              outputValidity: context.view('uint32', outputLength, 'selected-validity'),
              validCount: context.view('uint32', outputLength, 'coverage')
            } as never).addToGraph(context.graph);
          }
        })
      )
    )
  ),
  {
    exportName: 'GPURasterCategoricalOverview',
    name: 'maximum size mode scale 1',
    build: context =>
      new GPURasterCategoricalOverview({
        metadata: metadata(MAXIMUM_SIDE, MAXIMUM_SIDE),
        scale: 1,
        sourcePixelOrigin: [0, 0],
        input: calibrated(context, 'labels', 'uint32', MAXIMUM_PIXELS),
        policy: 'mode',
        output: context.view('uint32', MAXIMUM_PIXELS, 'selected'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'selected-validity'),
        validCount: context.view('uint32', MAXIMUM_PIXELS, 'coverage')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterCategoricalOverview',
    name: 'maximum size mode scale 8',
    build: context =>
      new GPURasterCategoricalOverview({
        metadata: metadata(MAXIMUM_SIDE, MAXIMUM_SIDE),
        scale: 8,
        sourcePixelOrigin: [0, 0],
        input: calibrated(context, 'labels', 'sint32', MAXIMUM_PIXELS),
        policy: 'mode',
        output: context.view('sint32', (MAXIMUM_SIDE / 8) ** 2, 'selected'),
        outputValidity: context.view('uint32', (MAXIMUM_SIDE / 8) ** 2, 'selected-validity'),
        validCount: context.view('uint32', (MAXIMUM_SIDE / 8) ** 2, 'coverage')
      }).addToGraph(context.graph)
  },

  // ---- global statistics ----
  ...[1, 256].map(
    (bins): KernelAuditCase => ({
      exportName: 'GPURasterGlobalInitialize',
      name: `${bins} bins`,
      build: context =>
        new GPURasterGlobalInitialize({accumulator: accumulator(context, bins)}).addToGraph(
          context.graph
        )
    })
  ),
  ...[1, 256].map(
    (bins): KernelAuditCase => ({
      exportName: 'GPURasterGlobalStatisticsMerge',
      name: `${bins} bins calibrated`,
      build: context =>
        new GPURasterGlobalStatisticsMerge({
          width: WIDTH,
          height: HEIGHT,
          input: calibrated(context, 'tile', 'float32', PIXELS),
          accumulator: accumulator(context, bins)
        }).addToGraph(context.graph)
    })
  ),
  {
    exportName: 'GPURasterGlobalStatisticsMerge',
    name: 'size above the maximum',
    build: context =>
      new GPURasterGlobalStatisticsMerge({
        width: DISPATCH_PIXELS + 1,
        height: 1,
        input: calibrated(context, 'tile', 'float32', DISPATCH_PIXELS + 1),
        accumulator: accumulator(context, 256)
      }).addToGraph(context.graph)
  },
  ...[1, 256].map(
    (bins): KernelAuditCase => ({
      exportName: 'GPURasterGlobalHistogramMerge',
      name: `${bins} bins calibrated`,
      build: context =>
        new GPURasterGlobalHistogramMerge({
          width: WIDTH,
          height: HEIGHT,
          input: calibrated(context, 'tile', 'float32', PIXELS),
          accumulator: accumulator(context, bins)
        }).addToGraph(context.graph)
    })
  ),
  {
    exportName: 'GPURasterGlobalHistogramMerge',
    name: 'size above the maximum',
    build: context =>
      new GPURasterGlobalHistogramMerge({
        width: DISPATCH_PIXELS + 1,
        height: 1,
        input: calibrated(context, 'tile', 'float32', DISPATCH_PIXELS + 1),
        accumulator: accumulator(context, 256)
      }).addToGraph(context.graph)
  },
  ...[0, 0.5, 1].flatMap(percentile =>
    [false, true].map(
      (withValidity): KernelAuditCase => ({
        exportName: 'GPURasterGlobalPercentile',
        name: `percentile ${percentile}${withValidity ? ' with validity' : ''}, 256 bins`,
        build: context =>
          new GPURasterGlobalPercentile({
            accumulator: accumulator(context, 256),
            percentile,
            output: context.view('float32', 1, 'percentile'),
            ...(withValidity
              ? {outputValidity: context.view('uint32', 1, 'percentile-validity')}
              : {})
          }).addToGraph(context.graph)
      })
    )
  ),

  // ---- connected components ----
  ...([4, 8] as const).flatMap(connectivity =>
    [undefined, 64].map(
      (maximumIterations): KernelAuditCase => ({
        exportName: 'GPURasterConnectedComponents',
        name: `connectivity ${connectivity}, iterations ${maximumIterations ?? 'default'}`,
        build: context =>
          new GPURasterConnectedComponents({
            width: WIDTH,
            height: HEIGHT,
            input: flags(context, 'mask', PIXELS),
            output: context.view('uint32', PIXELS, 'labels'),
            outputValidity: context.view('uint32', PIXELS, 'labels-validity'),
            converged: context.view('uint32', 1, 'converged'),
            iterationCount: context.view('uint32', 1, 'iterations'),
            connectivity,
            maximumIterations
          }).addToGraph(context.graph)
      })
    )
  ),
  ...([4, 8] as const).map(
    (connectivity): KernelAuditCase => ({
      exportName: 'GPURasterConnectedComponents',
      name: `maximum size connectivity ${connectivity}, 64 iterations`,
      build: context =>
        new GPURasterConnectedComponents({
          width: MAXIMUM_SIDE,
          height: MAXIMUM_SIDE,
          input: flags(context, 'mask', MAXIMUM_PIXELS),
          output: context.view('uint32', MAXIMUM_PIXELS, 'labels'),
          outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'labels-validity'),
          converged: context.view('uint32', 1, 'converged'),
          iterationCount: context.view('uint32', 1, 'iterations'),
          connectivity,
          maximumIterations: 64
        }).addToGraph(context.graph)
    })
  ),

  // ---- dense components ----
  {
    exportName: 'GPURasterDenseComponents',
    name: 'every output, capacity below pixel count',
    build: context =>
      new GPURasterDenseComponents({
        width: WIDTH,
        height: HEIGHT,
        input: context.view('uint32', PIXELS, 'labels'),
        inputValidity: context.view('uint32', PIXELS, 'labels-validity'),
        converged: context.view('uint32', 1, 'converged'),
        output: context.view('uint32', PIXELS, 'dense'),
        outputValidity: context.view('uint32', PIXELS, 'dense-validity'),
        componentCount: context.view('uint32', 1, 'count'),
        overflow: context.view('uint32', 1, 'overflow'),
        requiredComponentCount: context.view('uint32', 1, 'required'),
        capacity: 100
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterDenseComponents',
    name: 'default capacity, no required count',
    build: context =>
      new GPURasterDenseComponents({
        width: WIDTH,
        height: HEIGHT,
        input: context.view('uint32', PIXELS, 'labels'),
        inputValidity: context.view('uint32', PIXELS, 'labels-validity'),
        converged: context.view('uint32', 1, 'converged'),
        output: context.view('uint32', PIXELS, 'dense'),
        outputValidity: context.view('uint32', PIXELS, 'dense-validity'),
        componentCount: context.view('uint32', 1, 'count'),
        overflow: context.view('uint32', 1, 'overflow')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterDenseComponents',
    name: 'maximum size, capacity equals pixel count',
    build: context =>
      new GPURasterDenseComponents({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: context.view('uint32', MAXIMUM_PIXELS, 'labels'),
        inputValidity: context.view('uint32', MAXIMUM_PIXELS, 'labels-validity'),
        converged: context.view('uint32', 1, 'converged'),
        output: context.view('uint32', MAXIMUM_PIXELS, 'dense'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'dense-validity'),
        componentCount: context.view('uint32', 1, 'count'),
        overflow: context.view('uint32', 1, 'overflow'),
        requiredComponentCount: context.view('uint32', 1, 'required'),
        capacity: MAXIMUM_PIXELS
      }).addToGraph(context.graph)
  },

  // ---- region measurements ----
  ...[
    {name: 'calibrated float intensity', pixels: PIXELS, side: [WIDTH, HEIGHT], capacity: 64},
    {
      name: 'maximum size, capacity above the maximum',
      pixels: MAXIMUM_PIXELS,
      side: [MAXIMUM_SIDE, MAXIMUM_SIDE],
      capacity: DISPATCH_ELEMENTS + 1
    }
  ].map(
    (config): KernelAuditCase => ({
      exportName: 'GPURasterRegionMeasurements',
      name: config.name,
      build: context => {
        const capacity = config.capacity;
        new GPURasterRegionMeasurements({
          metadata: metadata(config.side[0], config.side[1]),
          labels: context.view('uint32', config.pixels, 'labels'),
          labelValidity: context.view('uint32', config.pixels, 'labels-validity'),
          converged: context.view('uint32', 1, 'converged'),
          componentCount: context.view('uint32', 1, 'count'),
          overflow: context.view('uint32', 1, 'overflow'),
          intensity: calibrated(context, 'intensity', 'float32', config.pixels),
          output: measurements(context, 'regions', capacity),
          capacity
        }).addToGraph(context.graph);
      }
    })
  ),

  // ---- cross tile components ----
  ...([4, 8] as const).flatMap(connectivity =>
    [undefined, 64].map(
      (maximumIterations): KernelAuditCase => ({
        exportName: 'GPURasterCrossTileComponents',
        name: `two tiles, connectivity ${connectivity}, iterations ${maximumIterations ?? 'default'}`,
        build: context => {
          const global = metadata(WIDTH, HEIGHT);
          new GPURasterCrossTileComponents({
            metadata: global,
            tiles: [
              crossTile(context, global, [0, 0, 12, HEIGHT], 16, 'west'),
              crossTile(context, global, [12, 0, WIDTH, HEIGHT], 16, 'east')
            ],
            connectivity,
            maximumIterations,
            componentCount: context.view('uint32', 1, 'global-count'),
            requiredComponentCount: context.view('uint32', 1, 'global-required'),
            converged: context.view('uint32', 1, 'global-converged'),
            overflow: context.view('uint32', 1, 'global-overflow'),
            output: measurements(context, 'global', 32),
            capacity: 20
          }).addToGraph(context.graph);
        }
      })
    )
  ),
  {
    exportName: 'GPURasterCrossTileComponents',
    name: 'four tiles with corners, no required count',
    build: context => {
      const global = metadata(WIDTH, HEIGHT);
      new GPURasterCrossTileComponents({
        metadata: global,
        tiles: [
          crossTile(context, global, [0, 0, 12, 8], 8, 'a'),
          crossTile(context, global, [12, 0, WIDTH, 8], 8, 'b'),
          crossTile(context, global, [0, 8, 12, HEIGHT], 8, 'c'),
          crossTile(context, global, [12, 8, WIDTH, HEIGHT], 8, 'd')
        ],
        connectivity: 8,
        componentCount: context.view('uint32', 1, 'global-count'),
        converged: context.view('uint32', 1, 'global-converged'),
        overflow: context.view('uint32', 1, 'global-overflow'),
        output: measurements(context, 'global', 32)
      }).addToGraph(context.graph);
    }
  },
  ...(
    [
      ['global capacity above the maximum', CROSS_TILE_CAPACITY / 2, CROSS_TILE_CAPACITY + 1],
      [
        'total tile region capacity above the maximum',
        CROSS_TILE_CAPACITY / 2 + 1,
        CROSS_TILE_CAPACITY
      ]
    ] as const
  ).flatMap(([name, tileCapacity, globalCapacity]) =>
    ([4, 8] as const).map(
      (connectivity): KernelAuditCase => ({
        exportName: 'GPURasterCrossTileComponents',
        name: `${name}, two ${MAXIMUM_SIDE}x${MAXIMUM_SIDE} tiles, connectivity ${connectivity}`,
        build: context => {
          const global = metadata(2 * MAXIMUM_SIDE, MAXIMUM_SIDE);
          new GPURasterCrossTileComponents({
            metadata: global,
            tiles: [
              crossTile(context, global, [0, 0, MAXIMUM_SIDE, MAXIMUM_SIDE], tileCapacity, 'west'),
              crossTile(
                context,
                global,
                [MAXIMUM_SIDE, 0, 2 * MAXIMUM_SIDE, MAXIMUM_SIDE],
                tileCapacity,
                'east'
              )
            ],
            connectivity,
            maximumIterations: 64,
            componentCount: context.view('uint32', 1, 'global-count'),
            requiredComponentCount: context.view('uint32', 1, 'global-required'),
            converged: context.view('uint32', 1, 'global-converged'),
            overflow: context.view('uint32', 1, 'global-overflow'),
            output: measurements(context, 'global', globalCapacity),
            capacity: globalCapacity
          }).addToGraph(context.graph);
        }
      })
    )
  ),

  // ---- texture <-> buffer ----
  ...(
    [
      ['r32float', 'float32', 0],
      ['rgba32float', 'float32', 3],
      ['r32uint', 'uint32', 0],
      ['rgba32uint', 'uint32', 2]
    ] as const
  ).flatMap(([textureFormat, format, channel]) =>
    [false, true].map(
      (applyCalibration): KernelAuditCase => ({
        exportName: 'GPURasterTextureToBuffer',
        name: `${textureFormat} channel ${channel} calibration ${applyCalibration}`,
        build: context =>
          new GPURasterTextureToBuffer({
            input: {
              id: 'texture-band',
              format,
              storage: {
                kind: 'texture',
                view: textureView(context, textureFormat, WIDTH, HEIGHT, 'source'),
                channel
              },
              validity: context.view('uint32', PIXELS, 'source-validity'),
              noDataValue: format === 'float32' ? -9999 : 4294967295,
              scale: 0.5,
              offset: 4
            } as never,
            output: context.view(applyCalibration ? 'float32' : format, PIXELS, 'buffer'),
            outputValidity: context.view('uint32', PIXELS, 'buffer-validity'),
            applyCalibration
          }).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterTextureToBuffer',
    name: 'maximum size 4096x4096 rgba32float',
    build: context =>
      new GPURasterTextureToBuffer({
        input: {
          id: 'texture-band',
          format: 'float32',
          storage: {
            kind: 'texture',
            view: textureView(context, 'rgba32float', MAXIMUM_SIDE, MAXIMUM_SIDE, 'source'),
            channel: 3
          },
          validity: context.view('uint32', MAXIMUM_PIXELS, 'source-validity'),
          noDataValue: -9999,
          scale: 0.5,
          offset: 4
        },
        output: context.view('float32', MAXIMUM_PIXELS, 'buffer'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'buffer-validity'),
        applyCalibration: true
      }).addToGraph(context.graph)
  },
  ...(
    [
      ['r32float', 'float32', 0],
      ['rgba32float', 'float32', 3],
      ['r32uint', 'uint32', 0],
      ['rgba32uint', 'uint32', 2]
    ] as const
  ).flatMap(([textureFormat, format, channel]) =>
    [false, true].map(
      (applyCalibration): KernelAuditCase => ({
        exportName: 'GPURasterBufferToTexture',
        name: `${textureFormat} channel ${channel} calibration ${applyCalibration}`,
        build: context =>
          new GPURasterBufferToTexture({
            input: calibrated(context, 'source', format, PIXELS),
            output: textureView(
              context,
              applyCalibration ? (channel === 0 ? 'r32float' : 'rgba32float') : textureFormat,
              WIDTH,
              HEIGHT,
              'destination'
            ),
            outputValidity: context.view('uint32', PIXELS, 'destination-validity'),
            channel,
            applyCalibration
          } as never).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterBufferToTexture',
    name: 'maximum size 4096x4096 rgba32float',
    build: context =>
      new GPURasterBufferToTexture({
        input: calibrated(context, 'source', 'float32', MAXIMUM_PIXELS),
        output: textureView(context, 'rgba32float', MAXIMUM_SIDE, MAXIMUM_SIDE, 'destination'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'destination-validity'),
        channel: 3,
        applyCalibration: true
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterBufferToTexture',
    name: 'maximum texture extent 8192x4096 r32float',
    build: context =>
      new GPURasterBufferToTexture({
        input: calibrated(context, 'source', 'float32', 8192 * 4096),
        output: textureView(context, 'r32float', 8192, 4096, 'destination'),
        outputValidity: context.view('uint32', 8192 * 4096, 'destination-validity')
      }).addToGraph(context.graph)
  },

  // ---- band math ----
  ...(['add', 'subtract', 'multiply', 'divide', 'normalized-difference'] as const).flatMap(
    operation =>
      FORMATS.map(
        (format): KernelAuditCase => ({
          exportName: 'GPURasterBandMath',
          name: `${operation} ${format} clamped with epsilon`,
          build: context =>
            new GPURasterBandMath({
              width: WIDTH,
              height: HEIGHT,
              left: calibrated(context, 'left', format, PIXELS),
              right: calibrated(
                context,
                'right',
                format === 'float32' ? 'sint32' : 'float32',
                PIXELS
              ),
              operation,
              output: context.view('float32', PIXELS, 'output'),
              outputValidity: context.view('uint32', PIXELS, 'output-validity'),
              epsilon: 1e-6,
              clamp: [-1, 1]
            }).addToGraph(context.graph)
        })
      )
  ),
  {
    exportName: 'GPURasterBandMath',
    name: 'maximum size normalized-difference',
    build: context =>
      new GPURasterBandMath({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        left: calibrated(context, 'left', 'float32', MAXIMUM_PIXELS),
        right: calibrated(context, 'right', 'float32', MAXIMUM_PIXELS),
        operation: 'normalized-difference',
        output: context.view('float32', MAXIMUM_PIXELS, 'output'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'output-validity'),
        epsilon: 1e-6,
        clamp: [-1, 1]
      }).addToGraph(context.graph)
  },

  // ---- neighborhood ----
  ...(['clamp', 'reflect', 'constant', 'nodata'] as const).flatMap(borderMode =>
    (['propagate', 'ignore-renormalize'] as const).map(
      (noDataPolicy): KernelAuditCase => ({
        exportName: 'GPURasterNeighborhood',
        name: `radius 8x8, border ${borderMode}, ${noDataPolicy}`,
        build: context =>
          new GPURasterNeighborhood({
            width: WIDTH,
            height: HEIGHT,
            input: calibrated(context, 'input', 'float32', PIXELS),
            output: context.view('float32', PIXELS, 'output'),
            outputValidity: context.view('uint32', PIXELS, 'output-validity'),
            radius: 8,
            kernel: Array.from({length: 17 * 17}, (_, index) => 1 + (index % 3)),
            borderMode,
            borderValue: 2.5,
            noDataPolicy,
            normalize: true
          }).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterNeighborhood',
    name: 'anisotropic radius [8, 0] unnormalized',
    build: context =>
      new GPURasterNeighborhood({
        width: WIDTH,
        height: HEIGHT,
        input: calibrated(context, 'input', 'sint32', PIXELS),
        output: context.view('float32', PIXELS, 'output'),
        outputValidity: context.view('uint32', PIXELS, 'output-validity'),
        radius: [8, 0],
        kernel: Array.from({length: 17}, (_, index) => index - 8)
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterNeighborhood',
    name: 'maximum size radius 8',
    build: context =>
      new GPURasterNeighborhood({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: calibrated(context, 'input', 'uint32', MAXIMUM_PIXELS),
        output: context.view('float32', MAXIMUM_PIXELS, 'output'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'output-validity'),
        radius: 8,
        kernel: Array.from({length: 17 * 17}, () => 1),
        borderMode: 'reflect',
        noDataPolicy: 'ignore-renormalize',
        normalize: true
      }).addToGraph(context.graph)
  },

  // ---- gradient magnitude ----
  ...(['sobel', 'scharr'] as const).flatMap(operator =>
    (['clamp', 'reflect', 'constant', 'nodata'] as const).map(
      (borderMode): KernelAuditCase => ({
        exportName: 'GPURasterGradientMagnitude',
        name: `${operator} border ${borderMode} scaled`,
        build: context =>
          new GPURasterGradientMagnitude({
            width: WIDTH,
            height: HEIGHT,
            input: calibrated(context, 'input', 'float32', PIXELS),
            output: context.view('float32', PIXELS, 'output'),
            outputValidity: context.view('uint32', PIXELS, 'output-validity'),
            operator,
            borderMode,
            borderValue: 1.5,
            scale: 0.25
          }).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterGradientMagnitude',
    name: 'maximum size default operator',
    build: context =>
      new GPURasterGradientMagnitude({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: calibrated(context, 'input', 'float32', MAXIMUM_PIXELS),
        output: context.view('float32', MAXIMUM_PIXELS, 'output'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'output-validity')
      }).addToGraph(context.graph)
  },

  // ---- morphology ----
  ...(['dilate', 'erode'] as const).flatMap(operation =>
    (['binary', 'grayscale'] as const).flatMap(mode =>
      (['square', 'cross'] as const).map(
        (structuringElement): KernelAuditCase => ({
          exportName: 'GPURasterMorphology',
          name: `${mode} ${operation} ${structuringElement} radius 8`,
          build: context =>
            new GPURasterMorphology({
              width: WIDTH,
              height: HEIGHT,
              radius: 8,
              operation,
              mode,
              structuringElement,
              borderMode: structuringElement === 'square' ? 'reflect' : 'constant',
              borderValue: 1,
              noDataPolicy: operation === 'dilate' ? 'ignore' : 'propagate',
              input:
                mode === 'binary'
                  ? flags(context, 'input', PIXELS)
                  : calibrated(context, 'input', 'float32', PIXELS),
              output: context.view(mode === 'binary' ? 'uint32' : 'float32', PIXELS, 'output'),
              outputValidity: context.view('uint32', PIXELS, 'output-validity')
            } as never).addToGraph(context.graph)
        })
      )
    )
  ),
  ...(['binary', 'grayscale'] as const).map(
    (mode): KernelAuditCase => ({
      exportName: 'GPURasterMorphology',
      name: `maximum size ${mode} erode square radius 8`,
      build: context =>
        new GPURasterMorphology({
          width: MAXIMUM_SIDE,
          height: MAXIMUM_SIDE,
          radius: 8,
          operation: 'erode',
          mode,
          structuringElement: 'square',
          input:
            mode === 'binary'
              ? flags(context, 'input', MAXIMUM_PIXELS)
              : calibrated(context, 'input', 'float32', MAXIMUM_PIXELS),
          output: context.view(mode === 'binary' ? 'uint32' : 'float32', MAXIMUM_PIXELS, 'output'),
          outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'output-validity')
        } as never).addToGraph(context.graph)
    })
  ),

  // ---- contrast ----
  ...(['linear', 'gamma'] as const).flatMap(mode =>
    ['range', 'view'].map(
      (domainKind): KernelAuditCase => ({
        exportName: 'GPURasterContrast',
        name: `${mode} domain ${domainKind}`,
        build: context =>
          new GPURasterContrast({
            width: WIDTH,
            height: HEIGHT,
            input: calibrated(context, 'input', 'float32', PIXELS),
            output: context.view('float32', PIXELS, 'output'),
            outputValidity: context.view('uint32', PIXELS, 'output-validity'),
            domain: domainKind === 'range' ? [0, 1000] : context.view('float32', 2, 'domain'),
            contrast: 1.5,
            gamma: 2.2,
            mode
          }).addToGraph(context.graph)
      })
    )
  ),
  ...[2, 256, 65536].map(
    (bins): KernelAuditCase => ({
      exportName: 'GPURasterContrast',
      name: `equalize ${bins} bins domain view`,
      build: context =>
        new GPURasterContrast({
          width: WIDTH,
          height: HEIGHT,
          input: calibrated(context, 'input', 'float32', PIXELS),
          output: context.view('float32', PIXELS, 'output'),
          outputValidity: context.view('uint32', PIXELS, 'output-validity'),
          domain: context.view('float32', 2, 'domain'),
          mode: 'equalize',
          histogram: context.view('uint32', bins, 'histogram')
        }).addToGraph(context.graph)
    })
  ),
  {
    exportName: 'GPURasterContrast',
    name: 'maximum size equalize, histogram at binding limit',
    build: context =>
      new GPURasterContrast({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: calibrated(context, 'input', 'float32', MAXIMUM_PIXELS),
        output: context.view('float32', MAXIMUM_PIXELS, 'output'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'output-validity'),
        domain: context.view('float32', 2, 'domain'),
        mode: 'equalize',
        histogram: context.view('uint32', context.maximumRows(4), 'histogram')
      }).addToGraph(context.graph)
  },

  // ---- contours ----
  ...['number', 'view'].flatMap(levelKind =>
    FORMATS.map(
      (format): KernelAuditCase => ({
        exportName: 'GPURasterContourClassifier',
        name: `${format} level ${levelKind}`,
        build: context =>
          new GPURasterContourClassifier({
            width: WIDTH,
            height: HEIGHT,
            input: calibrated(context, 'input', format, PIXELS),
            level: levelKind === 'number' ? 12.5 : context.view('float32', 1, 'level'),
            cases: context.view('uint32', (WIDTH - 1) * (HEIGHT - 1), 'cases'),
            segmentCounts: context.view('uint32', (WIDTH - 1) * (HEIGHT - 1), 'segments')
          }).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterContourClassifier',
    name: 'maximum size',
    build: context =>
      new GPURasterContourClassifier({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: calibrated(context, 'input', 'float32', MAXIMUM_PIXELS),
        level: context.view('float32', 1, 'level'),
        cases: context.view('uint32', (MAXIMUM_SIDE - 1) ** 2, 'cases'),
        segmentCounts: context.view('uint32', (MAXIMUM_SIDE - 1) ** 2, 'segments')
      }).addToGraph(context.graph)
  },
  ...['number', 'view'].flatMap(levelKind =>
    FORMATS.map(
      (format): KernelAuditCase => ({
        exportName: 'GPURasterContours',
        name: `${format} level ${levelKind} every output, indirect draw`,
        build: context =>
          new GPURasterContours({
            width: WIDTH,
            height: HEIGHT,
            input: calibrated(context, 'input', format, PIXELS),
            level: levelKind === 'number' ? 12.5 : context.view('float32', 1, 'level'),
            vertices: context.view('float32x2', 2 * 400, 'vertices'),
            segmentCount: context.view('uint32', 1, 'segment-count'),
            overflow: context.view('uint32', 1, 'overflow'),
            requiredSegmentCount: context.view('uint32', 1, 'required'),
            draw: drawCommands(context, 4),
            drawCommandIndex: 3,
            capacity: 300,
            metadata: metadata(WIDTH, HEIGHT)
          }).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterContours',
    name: 'minimal outputs',
    build: context =>
      new GPURasterContours({
        width: WIDTH,
        height: HEIGHT,
        input: band(context, 'input', 'float32', PIXELS),
        level: 3,
        vertices: context.view('float32x2', 100, 'vertices'),
        segmentCount: context.view('uint32', 1, 'segment-count'),
        overflow: context.view('uint32', 1, 'overflow')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterContours',
    name: 'maximum size, vertices at binding limit',
    build: context => {
      const vertices = context.maximumRows(8);
      new GPURasterContours({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: calibrated(context, 'input', 'float32', MAXIMUM_PIXELS),
        level: context.view('float32', 1, 'level'),
        vertices: context.view('float32x2', vertices, 'vertices'),
        segmentCount: context.view('uint32', 1, 'segment-count'),
        overflow: context.view('uint32', 1, 'overflow'),
        requiredSegmentCount: context.view('uint32', 1, 'required'),
        draw: drawCommands(context, 1),
        capacity: Math.floor(vertices / 2),
        metadata: metadata(MAXIMUM_SIDE, MAXIMUM_SIDE)
      }).addToGraph(context.graph);
    }
  },

  // ---- NDVI ----
  {
    exportName: 'GPURasterNDVI',
    name: 'calibrated bands, epsilon and clamp',
    build: context =>
      new GPURasterNDVI({
        width: WIDTH,
        height: HEIGHT,
        nearInfrared: calibrated(context, 'nir', 'float32', PIXELS),
        red: calibrated(context, 'red', 'float32', PIXELS),
        output: context.view('float32', PIXELS, 'output'),
        outputValidity: context.view('uint32', PIXELS, 'output-validity'),
        epsilon: 1e-6,
        clamp: [-1, 1]
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterNDVI',
    name: 'integer bands',
    build: context =>
      new GPURasterNDVI({
        width: WIDTH,
        height: HEIGHT,
        nearInfrared: calibrated(context, 'nir', 'uint32', PIXELS),
        red: calibrated(context, 'red', 'sint32', PIXELS),
        output: context.view('float32', PIXELS, 'output'),
        outputValidity: context.view('uint32', PIXELS, 'output-validity')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterNDVI',
    name: 'maximum size',
    build: context =>
      new GPURasterNDVI({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        nearInfrared: calibrated(context, 'nir', 'float32', MAXIMUM_PIXELS),
        red: calibrated(context, 'red', 'float32', MAXIMUM_PIXELS),
        output: context.view('float32', MAXIMUM_PIXELS, 'output'),
        outputValidity: context.view('uint32', MAXIMUM_PIXELS, 'output-validity'),
        epsilon: 1e-6,
        clamp: [-1, 1]
      }).addToGraph(context.graph)
  },

  // ---- statistics ----
  {
    exportName: 'GPURasterStatistics',
    name: 'calibrated every output',
    build: context =>
      new GPURasterStatistics({
        width: WIDTH,
        height: HEIGHT,
        input: calibrated(context, 'input', 'float32', PIXELS),
        count: context.view('uint32', 1, 'count'),
        sum: context.view('float32', 1, 'sum'),
        mean: context.view('float32', 1, 'mean'),
        extent: context.view('float32', 2, 'extent')
      }).addToGraph(context.graph)
  },
  {
    exportName: 'GPURasterStatistics',
    name: 'size above the maximum',
    build: context =>
      new GPURasterStatistics({
        width: DISPATCH_PIXELS + 1,
        height: 1,
        input: calibrated(context, 'input', 'float32', DISPATCH_PIXELS + 1),
        count: context.view('uint32', 1, 'count'),
        sum: context.view('float32', 1, 'sum'),
        mean: context.view('float32', 1, 'mean'),
        extent: context.view('float32', 2, 'extent')
      }).addToGraph(context.graph)
  },

  // ---- histogram ----
  ...FORMATS.flatMap(format =>
    ['range', 'view', 'valid-auto', 'valid-auto-output'].map(
      (domainKind): KernelAuditCase => ({
        exportName: 'GPURasterHistogram',
        name: `${format} 256 bins domain ${domainKind}`,
        build: context =>
          new GPURasterHistogram({
            input: calibrated(context, 'input', format, PIXELS),
            output: context.view('uint32', 256, 'histogram'),
            domain:
              domainKind === 'range'
                ? [0, 1000]
                : domainKind === 'view'
                  ? context.view(format, 2, 'domain')
                  : 'valid-auto',
            ...(domainKind === 'valid-auto-output'
              ? {domainOutput: context.view(format, 2, 'domain-output')}
              : {})
          } as never).addToGraph(context.graph)
      })
    )
  ),
  {
    exportName: 'GPURasterHistogram',
    name: 'largest one-dispatch size, 4096 bins valid-auto with domain output',
    build: context =>
      new GPURasterHistogram({
        input: calibrated(context, 'input', 'float32', DISPATCH_PIXELS),
        output: context.view('uint32', 4096, 'histogram'),
        domain: 'valid-auto',
        domainOutput: context.view('float32', 2, 'domain-output')
      }).addToGraph(context.graph)
  },

  // ---- thresholds ----
  ...(['above', 'below'] as const).flatMap(operation =>
    [false, true].flatMap(inclusive =>
      ['number', 'view'].map(
        (valueKind): KernelAuditCase => ({
          exportName: 'GPURasterThreshold',
          name: `${operation} inclusive ${inclusive} ${valueKind}`,
          build: context =>
            new GPURasterThreshold({
              width: WIDTH,
              height: HEIGHT,
              input: calibrated(context, 'input', 'float32', PIXELS),
              output: context.view('uint32', PIXELS, 'output'),
              threshold: valueKind === 'number' ? 3 : context.view('float32', 1, 'threshold'),
              operation,
              inclusive
            }).addToGraph(context.graph)
        })
      )
    )
  ),
  ...[false, true].flatMap(inclusive =>
    ['tuple', 'view'].flatMap(valueKind =>
      FORMATS.map(
        (format): KernelAuditCase => ({
          exportName: 'GPURasterThreshold',
          name: `range ${format} inclusive ${inclusive} ${valueKind}`,
          build: context =>
            new GPURasterThreshold({
              width: WIDTH,
              height: HEIGHT,
              input: calibrated(context, 'input', format, PIXELS),
              output: context.view('uint32', PIXELS, 'output'),
              threshold: valueKind === 'tuple' ? [1, 5] : context.view('float32', 2, 'threshold'),
              operation: 'range',
              inclusive
            }).addToGraph(context.graph)
        })
      )
    )
  ),
  {
    exportName: 'GPURasterThreshold',
    name: 'maximum size range',
    build: context =>
      new GPURasterThreshold({
        width: MAXIMUM_SIDE,
        height: MAXIMUM_SIDE,
        input: calibrated(context, 'input', 'float32', MAXIMUM_PIXELS),
        output: context.view('uint32', MAXIMUM_PIXELS, 'output'),
        threshold: context.view('float32', 2, 'threshold'),
        operation: 'range',
        inclusive: true
      }).addToGraph(context.graph)
  },
  ...[1, 2, 256].flatMap(bins =>
    ['tuple', 'view'].map(
      (domainKind): KernelAuditCase => ({
        exportName: 'GPURasterOtsuThreshold',
        name: `${bins} bins domain ${domainKind}`,
        build: context =>
          new GPURasterOtsuThreshold({
            histogram: context.view('uint32', bins, 'histogram'),
            domain: domainKind === 'tuple' ? [0, 255] : context.view('float32', 2, 'domain'),
            output: context.view('float32', 1, 'threshold')
          }).addToGraph(context.graph)
      })
    )
  )
];
