// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  getGPUIsobandsParameterValues,
  getGPUIsolinesParameterValues,
  GPU_ISOBANDS_PARAMETER_LENGTH,
  GPU_ISOLINES_PARAMETER_LENGTH,
  GPUIsobandRings,
  GPUIsobands,
  GPUIsolines
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPUTerrainDerivatives
} from '@luma.gl/experimental/gpu-terrain';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {RAMP_STOPS, type RampName} from '../../engine/ramps';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {createCanyonGrid, flipRows, formatInteger, getCellAt} from './b15-common';
import {IsobandTriangleLayer, PolylineLayer} from './b15-contour-layers';

/** Option state of the contours scene. */
export type ContoursOptions = {
  source: 'elevation' | 'slope';
  elevationInterval: number;
  slopeInterval: number;
  elevationWindow: readonly [number, number];
  slopeWindow: readonly [number, number];
  showBands: boolean;
  bandOpacity: number;
  ramp: Extract<RampName, 'cividis' | 'viridis' | 'magma' | 'inferno' | 'grayscale'>;
  showLines: boolean;
  stitchLines: boolean;
  indexEvery: number;
  lineWidth: number;
  showRings: boolean;
  ringTolerance: 'fine' | 'half-cell' | 'cell';
  splitTouchingRings: boolean;
  showRelief: boolean;
};

/** Raster reduction: 2 averages 2 x 2 blocks of the 15 m DEM to a 31 m grid. */
const GRID_STRIDE = 2;
/** Compile-time number of level slots (isolines levels and isobands breaks). */
export const LEVEL_SLOT_COUNT = 64;
/** Compile-time capacities; the readouts flag overflow. */
const SEGMENT_CAPACITY = 1_200_000;
const TRIANGLE_CAPACITY = 3_000_000;
const RING_EDGE_CAPACITY = 1_200_000;
const RING_CAPACITY = 32_768;
const PALETTE_SIZE = 256;
/** Upper end of the slope domain used for slope levels, degrees. */
export const SLOPE_DOMAIN_MAXIMUM = 85;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
const RECORD_SEGMENTS = 0;
const RECORD_POLYLINES = 1;
const RECORD_BANDS = 2;
const RECORD_BYTE_LENGTH = 16;
const SHELL_COLOR = [255, 255, 255, 235] as const;
const HOLE_COLOR = [255, 60, 200, 235] as const;
const LINE_COLORS = {
  light: {line: [40, 40, 50, 170], index: [10, 10, 20, 255]},
  dark: {line: [235, 235, 245, 160], index: [255, 255, 255, 255]}
} as const;
/** Vertex matching distance of the ring assembly in meters, by option. */
const RING_TOLERANCES = {fine: 1e-4, 'half-cell': 15, cell: 31} as const;

type RingKey = string;

/**
 * Contours of the Grand Canyon: `GPUIsolines` (segments, optionally stitched polylines),
 * `GPUIsobands` (filled bands) and `GPUIsobandRings` (shell and hole rings) over the elevation or the
 * slope raster. Levels, band window, palette and source are buffer writes; the graphs are compiled
 * once (plain and stitched lines in two graphs, rings in a third compiled for the current ring
 * options) and encoded only when an input changed. The raster is south-first so the contributors'
 * row 0 at minimum y matches the extent in ground meters.
 */
export async function createContours(
  ctx: SceneContext<ContoursOptions>
): Promise<SceneInstance<ContoursOptions>> {
  const {device} = ctx;
  const grid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), GRID_STRIDE);
  const {width, height, cellCount} = grid;
  const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
  const extent = grid.bounds;
  const resources = new SpatialAnalysisResources(device, 'contours');
  let destroyed = false;

  let minimumElevation = Infinity;
  let maximumElevation = -Infinity;
  for (const value of grid.elevation) {
    if (value < minimumElevation) minimumElevation = value;
    if (value > maximumElevation) maximumElevation = value;
  }

  // --- Buffers -------------------------------------------------------------------------------
  const cells = (name: string) => resources.createBuffer(name, cellCount * 4);
  const elevationBuffer = resources.createBuffer(
    'elevation-south-first',
    flipRows(grid.elevation, width, height)
  );
  const slopeBuffer = cells('slope');
  const hillshadeBuffer = cells('hillshade');
  const valuesBuffer = cells('values');
  const derivativesSettings = resources.createParameterBuffer(
    'derivatives-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  const isolineParameters = resources.createParameterBuffer(
    'isoline-parameters',
    'float32',
    GPU_ISOLINES_PARAMETER_LENGTH
  );
  const isobandParameters = resources.createParameterBuffer(
    'isoband-parameters',
    'float32',
    GPU_ISOBANDS_PARAMETER_LENGTH
  );
  const levelsBuffer = resources.createBuffer('levels', LEVEL_SLOT_COUNT * 4);
  const levelStyleBuffer = resources.createBuffer('level-style', LEVEL_SLOT_COUNT * 4);
  const paletteBuffer = resources.createBuffer('palette', new Uint32Array(PALETTE_SIZE));
  const segmentsBuffer = resources.createBuffer('segments', SEGMENT_CAPACITY * 16);
  const segmentLevelsBuffer = resources.createBuffer('segment-levels', SEGMENT_CAPACITY * 4);
  const segmentCountBuffer = resources.createBuffer('segment-count', 4);
  const segmentOverflowBuffer = resources.createBuffer('segment-overflow', 4);
  const trianglesBuffer = resources.createBuffer('triangles', TRIANGLE_CAPACITY * 3 * 8);
  const triangleBandsBuffer = resources.createBuffer('triangle-bands', TRIANGLE_CAPACITY * 4);
  const triangleCountBuffer = resources.createBuffer('triangle-count', 4);
  const triangleOverflowBuffer = resources.createBuffer('triangle-overflow', 4);
  const bandVertexCountBuffer = resources.createBuffer('band-vertex-count', 4);
  const polylineVerticesBuffer = resources.createBuffer(
    'polyline-vertices',
    SEGMENT_CAPACITY * 2 * 8
  );
  const polylineOffsetsBuffer = resources.createBuffer(
    'polyline-offsets',
    (SEGMENT_CAPACITY + 1) * 4
  );
  const polylineLevelsBuffer = resources.createBuffer('polyline-levels', SEGMENT_CAPACITY * 4);
  const polylineClosedBuffer = resources.createBuffer('polyline-closed', SEGMENT_CAPACITY * 4);
  const polylineCountBuffer = resources.createBuffer('polyline-count', 4);
  const polylineVertexCountBuffer = resources.createBuffer('polyline-vertex-count', 4);
  const polylineOverflowBuffer = resources.createBuffer('polyline-overflow', 4);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', (RING_CAPACITY + 1) * 4);
  const ringPositionsBuffer = resources.createBuffer('ring-positions', RING_EDGE_CAPACITY * 8);
  const ringBandsBuffer = resources.createBuffer('ring-bands', RING_CAPACITY * 4);
  const ringIsHoleBuffer = resources.createBuffer('ring-is-hole', RING_CAPACITY * 4);
  const ringCountBuffer = resources.createBuffer('ring-count', 4);
  const ringOverflowBuffer = resources.createBuffer('ring-overflow', 4);
  const ringOpenBuffer = resources.createBuffer('ring-open', 4);
  const ringEdgeCountBuffer = resources.createBuffer('ring-edge-count', 4);
  const ringEdgeOverflowBuffer = resources.createBuffer('ring-edge-overflow', 4);
  const ringStyleBuffer = resources.createBuffer('ring-style', Uint32Array.of(0, 1));
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'contours-draw',
      type: 'draw',
      commands: [
        {vertexCount: 6, instanceCount: 0},
        {vertexCount: 6, instanceCount: 0},
        {vertexCount: 0, instanceCount: 1}
      ]
    })
  );

  // --- Setup graph: slope and hillshade, once ------------------------------------------------
  const setupGraph = new GPUCommandGraph<void>(device, {id: 'contours-setup'});
  setupGraph.add(
    new GPUTerrainDerivatives({
      id: 'derivatives',
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(setupGraph, 'elevation', elevationBuffer, 'float32', cellCount)
        }
      },
      settings: derivativesSettings.importToGraph(setupGraph),
      slope: importGraphBuffer(setupGraph, 'slope', slopeBuffer, 'float32', cellCount),
      hillshade: importGraphBuffer(setupGraph, 'hillshade', hillshadeBuffer, 'float32', cellCount),
      cellSizeMode: grid.cellSizeMode,
      rowDirection: 'north'
    })
  );
  const setup = resources.track(setupGraph.compile());

  // --- Analysis graphs -------------------------------------------------------------------------
  const compileAnalysis = (withPolylines: boolean): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {
      id: withPolylines ? 'contours-stitched' : 'contours-plain'
    });
    const values = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount);
    const levels = importGraphBuffer(graph, 'levels', levelsBuffer, 'float32', LEVEL_SLOT_COUNT);
    graph.add(
      new GPUIsobands({
        id: 'bands',
        width,
        height,
        values,
        breaks: levels,
        parameters: isobandParameters.importToGraph(graph),
        output: {
          triangles: importGraphBuffer(
            graph,
            'triangles',
            trianglesBuffer,
            'float32x2',
            TRIANGLE_CAPACITY * 3
          ),
          triangleBands: importGraphBuffer(
            graph,
            'triangle-bands',
            triangleBandsBuffer,
            'uint32',
            TRIANGLE_CAPACITY
          ),
          count: importGraphBuffer(graph, 'triangle-count', triangleCountBuffer, 'uint32', 1),
          overflow: importGraphBuffer(
            graph,
            'triangle-overflow',
            triangleOverflowBuffer,
            'uint32',
            1
          ),
          vertexCount: importGraphBuffer(
            graph,
            'band-vertex-count',
            bandVertexCountBuffer,
            'uint32',
            1
          )
        }
      })
    );
    graph.add(
      new GPUIsolines({
        id: 'lines',
        width,
        height,
        values,
        levels,
        parameters: isolineParameters.importToGraph(graph),
        output: {
          segments: importGraphBuffer(
            graph,
            'segments',
            segmentsBuffer,
            'float32x4',
            SEGMENT_CAPACITY
          ),
          segmentLevels: importGraphBuffer(
            graph,
            'segment-levels',
            segmentLevelsBuffer,
            'uint32',
            SEGMENT_CAPACITY
          ),
          count: importGraphBuffer(graph, 'segment-count', segmentCountBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'segment-overflow', segmentOverflowBuffer, 'uint32', 1)
        },
        polylines: withPolylines
          ? {
              vertices: importGraphBuffer(
                graph,
                'polyline-vertices',
                polylineVerticesBuffer,
                'float32x2',
                SEGMENT_CAPACITY * 2
              ),
              polylineOffsets: importGraphBuffer(
                graph,
                'polyline-offsets',
                polylineOffsetsBuffer,
                'uint32',
                SEGMENT_CAPACITY + 1
              ),
              polylineLevels: importGraphBuffer(
                graph,
                'polyline-levels',
                polylineLevelsBuffer,
                'uint32',
                SEGMENT_CAPACITY
              ),
              polylineClosed: importGraphBuffer(
                graph,
                'polyline-closed',
                polylineClosedBuffer,
                'uint32',
                SEGMENT_CAPACITY
              ),
              polylineCount: importGraphBuffer(
                graph,
                'polyline-count',
                polylineCountBuffer,
                'uint32',
                1
              ),
              vertexCount: importGraphBuffer(
                graph,
                'polyline-vertex-count',
                polylineVertexCountBuffer,
                'uint32',
                1
              ),
              overflow: importGraphBuffer(
                graph,
                'polyline-overflow',
                polylineOverflowBuffer,
                'uint32',
                1
              )
            }
          : undefined
      })
    );
    return resources.track(graph.compile());
  };
  const plainGraph = compileAnalysis(false);
  const stitchedGraph = compileAnalysis(true);

  /** Ring graph for one tolerance and splitting choice (both compile-time). */
  const compileRings = (
    tolerance: ContoursOptions['ringTolerance'],
    split: boolean
  ): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'contours-rings'});
    graph.add(
      new GPUIsobandRings({
        id: 'band-rings',
        width,
        height,
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', cellCount),
        breaks: importGraphBuffer(graph, 'levels', levelsBuffer, 'float32', LEVEL_SLOT_COUNT),
        parameters: isobandParameters.importToGraph(graph),
        edgeCapacity: RING_EDGE_CAPACITY,
        vertexTolerance: RING_TOLERANCES[tolerance],
        splitTouchingRings: split,
        output: {
          ringOffsets: importGraphBuffer(
            graph,
            'ring-offsets',
            ringOffsetsBuffer,
            'uint32',
            RING_CAPACITY + 1
          ),
          positions: importGraphBuffer(
            graph,
            'ring-positions',
            ringPositionsBuffer,
            'float32x2',
            RING_EDGE_CAPACITY
          ),
          ringGroups: importGraphBuffer(
            graph,
            'ring-bands',
            ringBandsBuffer,
            'uint32',
            RING_CAPACITY
          ),
          ringIsHole: importGraphBuffer(
            graph,
            'ring-is-hole',
            ringIsHoleBuffer,
            'uint32',
            RING_CAPACITY
          ),
          count: importGraphBuffer(graph, 'ring-count', ringCountBuffer, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'ring-overflow', ringOverflowBuffer, 'uint32', 1),
          openSegmentCount: importGraphBuffer(graph, 'ring-open', ringOpenBuffer, 'uint32', 1),
          edgeCount: importGraphBuffer(graph, 'edge-count', ringEdgeCountBuffer, 'uint32', 1),
          edgeOverflow: importGraphBuffer(
            graph,
            'edge-overflow',
            ringEdgeOverflowBuffer,
            'uint32',
            1
          )
        }
      })
    );
    return graph.compile();
  };

  // --- State --------------------------------------------------------------------------------
  let dirty = true;
  let setupEncoded = false;
  let lastStyleKey = '';
  let builtRingKey: RingKey = '';
  let ringsGraph: CompiledGraphOrNull = null;
  let ringsEncoded = false;
  let measuring = false;

  type CompiledGraphOrNull = CompiledGPUCommandGraph<void> | null;

  function getRingKey(options: ContoursOptions): RingKey {
    return `${options.ringTolerance}-${options.splitTouchingRings}`;
  }

  function ensureRingsGraph(): void {
    const options = ctx.options;
    const key = getRingKey(options);
    if (ringsGraph && key === builtRingKey) return;
    if (ringsGraph) resources.release(ringsGraph);
    ringsGraph = resources.track(compileRings(options.ringTolerance, options.splitTouchingRings));
    builtRingKey = key;
    dirty = true;
  }

  function writePalette(): void {
    const stops = RAMP_STOPS[ctx.options.ramp];
    const packed = new Uint32Array(PALETTE_SIZE);
    for (let index = 0; index < PALETTE_SIZE; index++) {
      const position = (index / (PALETTE_SIZE - 1)) * (stops.length - 1);
      const lower = Math.min(Math.floor(position), stops.length - 2);
      const fraction = position - lower;
      const channels = [0, 1, 2].map(channel =>
        Math.round(stops[lower][channel] * (1 - fraction) + stops[lower + 1][channel] * fraction)
      );
      packed[index] = (channels[0] | (channels[1] << 8) | (channels[2] << 16) | (255 << 24)) >>> 0;
    }
    paletteBuffer.write(packed);
  }

  /** Levels at multiples of the interval inside the source's domain. */
  function computeLevels(options: ContoursOptions): {values: number[]; offset: number} {
    const elevation = options.source === 'elevation';
    const interval = elevation ? options.elevationInterval : options.slopeInterval;
    const low = elevation ? minimumElevation : 0;
    const high = elevation ? maximumElevation : SLOPE_DOMAIN_MAXIMUM;
    const first = Math.floor(low / interval) + 1;
    const last = Math.max(first, Math.floor(high / interval));
    const count = Math.min(LEVEL_SLOT_COUNT, last - first + 1);
    return {
      values: Array.from({length: count}, (_, index) => interval * (first + index)),
      offset: first
    };
  }

  /** Band window of the current source as `[firstBand, lastBand]`. */
  function computeBandWindow(
    options: ContoursOptions,
    values: readonly number[]
  ): [number, number] {
    const [low, high] =
      options.source === 'elevation' ? options.elevationWindow : options.slopeWindow;
    const bandOf = (value: number) => values.filter(level => level <= value).length;
    return [bandOf(low), Math.min(values.length, bandOf(high))];
  }

  function writeLevels(): void {
    const options = ctx.options;
    const {values, offset} = computeLevels(options);
    const count = values.length;
    const next = new Float32Array(LEVEL_SLOT_COUNT);
    const styles = new Uint32Array(LEVEL_SLOT_COUNT);
    const interval =
      options.source === 'elevation' ? options.elevationInterval : options.slopeInterval;
    for (let index = 0; index < LEVEL_SLOT_COUNT; index++) {
      // Unused slots stay ascending so the break tables remain sorted.
      next[index] =
        index < count ? values[index] : values[count - 1] + (index - count + 1) * interval;
      styles[index] = options.indexEvery > 0 && (offset + index) % options.indexEvery === 0 ? 1 : 0;
    }
    const [firstBand, lastBand] = computeBandWindow(options, values);
    const key = `${options.source}|${Array.from(next).join(',')}|${styles.join('')}|${firstBand}|${lastBand}`;
    if (key === lastStyleKey) return;
    lastStyleKey = key;
    levelsBuffer.write(next);
    levelStyleBuffer.write(styles);
    isolineParameters.write(
      getGPUIsolinesParameterValues({width, height, levelCount: count, extent})
    );
    isobandParameters.write(
      getGPUIsobandsParameterValues({width, height, breakCount: count, extent, firstBand, lastBand})
    );
    ctx.setReadout(
      'levels',
      `${count} levels every ${interval} ${options.source === 'elevation' ? 'm' : '°'}${count >= LEVEL_SLOT_COUNT ? ' (capped at 64)' : ''}`
    );
    ctx.setReadout(
      'bandWindow',
      `bands ${firstBand} to ${lastBand} of ${count} drawn as filled geometry`
    );
    dirty = true;
  }

  function copySource(commandEncoder: CommandEncoder): void {
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: ctx.options.source === 'elevation' ? elevationBuffer : slopeBuffer,
      sourceOffset: 0,
      destinationBuffer: valuesBuffer,
      destinationOffset: 0,
      size: cellCount * 4
    });
  }

  const summary = new SummaryReader(
    resources,
    'contours',
    [
      {buffer: segmentCountBuffer, size: 4},
      {buffer: segmentOverflowBuffer, size: 4},
      {buffer: triangleCountBuffer, size: 4},
      {buffer: triangleOverflowBuffer, size: 4},
      {buffer: polylineCountBuffer, size: 4},
      {buffer: polylineVertexCountBuffer, size: 4},
      {buffer: polylineOverflowBuffer, size: 4},
      {buffer: ringCountBuffer, size: 4},
      {buffer: ringOverflowBuffer, size: 4},
      {buffer: ringOpenBuffer, size: 4},
      {buffer: ringEdgeCountBuffer, size: 4},
      {buffer: ringEdgeOverflowBuffer, size: 4},
      {buffer: ringIsHoleBuffer, size: RING_CAPACITY * 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const options = ctx.options;
      ctx.setReadout(
        'segments',
        `${formatInteger(words[0])} of ${formatInteger(SEGMENT_CAPACITY)}${words[1] ? ' OVERFLOW' : ''}`
      );
      ctx.setReadout(
        'triangles',
        `${formatInteger(words[2])} of ${formatInteger(TRIANGLE_CAPACITY)}${words[3] ? ' OVERFLOW' : ''}`
      );
      ctx.setReadout(
        'polylines',
        options.stitchLines
          ? `${formatInteger(words[4])} lines, ${formatInteger(words[5])} vertices${words[6] ? ' OVERFLOW' : ''}`
          : 'off (plain segments)'
      );
      if (options.showRings && ringsEncoded) {
        const ringCount = Math.min(words[7], RING_CAPACITY);
        let holes = 0;
        for (let ring = 0; ring < ringCount; ring++) holes += words[12 + ring] ? 1 : 0;
        ctx.setReadout(
          'rings',
          `${formatInteger(ringCount - holes)} shells, ${formatInteger(holes)} holes${words[8] ? ' OVERFLOW' : ''}`
        );
        ctx.setReadout(
          'ringEdges',
          `${formatInteger(words[10])} boundary edges of ${formatInteger(RING_EDGE_CAPACITY)}${words[11] ? ' OVERFLOW' : ''}; ${formatInteger(words[9])} open segments`
        );
      } else {
        ctx.setReadout('rings', 'off');
        ctx.setReadout('ringEdges', 'off');
      }
    }
  );

  async function measure(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timings', 'measuring...');
    try {
      const run = async (graph: CompiledGPUCommandGraph<void>) =>
        measureCompiledGraph(device, graph, {
          parameters: undefined,
          completionBuffer: segmentCountBuffer,
          signal: ctx.signal,
          runs: 3,
          warmUpRuns: 1,
          repetitions: 2
        });
      const plain = await run(plainGraph);
      const stitched = await run(stitchedGraph);
      ensureRingsGraph();
      const rings = await run(ringsGraph!);
      ctx.setReadout(
        'timings',
        `lines + bands ${formatCompiledGraphTiming(plain)} · stitched ${formatCompiledGraphTiming(stitched)} · rings ${formatCompiledGraphTiming(rings)}`
      );
    } catch {
      // Aborted or destroyed while measuring.
    } finally {
      measuring = false;
      dirty = true;
    }
  }

  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues({
      cellSize: grid.cellSize,
      northEdge: grid.southEdge,
      southEdge: grid.northEdge,
      azimuthDegrees: SUN_AZIMUTH_DEGREES,
      altitudeDegrees: SUN_ALTITUDE_DEGREES
    })
  );
  writePalette();
  writeLevels();
  ctx.setReadout('grid', `${width} × ${height} cells of ${grid.groundCellSize[0].toFixed(0)} m`);
  ctx.setReadout(
    'domain',
    `${formatInteger(minimumElevation)} to ${formatInteger(maximumElevation)} m elevation, 0 to ${SLOPE_DOMAIN_MAXIMUM}° slope`
  );
  if (ctx.options.showRings) ensureRingsGraph();

  return {
    getCompiledGraphs: () => [
      setup,
      ctx.options.stitchLines ? stitchedGraph : plainGraph,
      ...(ringsGraph && ctx.options.showRings ? [ringsGraph] : [])
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'ramp':
          writePalette();
          dirty = true;
          break;
        case 'source':
        case 'elevationInterval':
        case 'slopeInterval':
        case 'elevationWindow':
        case 'slopeWindow':
        case 'indexEvery':
          writeLevels();
          dirty = true;
          break;
        case 'showRings':
        case 'ringTolerance':
        case 'splitTouchingRings':
          if (state.showRings) ensureRingsGraph();
          dirty = true;
          break;
        case 'stitchLines':
          dirty = true;
          break;
        default:
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'time') void measure();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!setupEncoded) {
        setup.encode(commandEncoder, {parameters: undefined});
        setupEncoded = true;
        dirty = true;
      }
      if (!dirty) {
        summary.flush(commandEncoder);
        return;
      }
      copySource(commandEncoder);
      const compiled = ctx.options.stitchLines ? stitchedGraph : plainGraph;
      const encoding = compiled.encode(commandEncoder, {parameters: undefined});
      const copyWord = (sourceBuffer: Buffer, record: number, wordIndex: number) =>
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset: 0,
          destinationBuffer: drawCommands.buffer,
          destinationOffset: record * RECORD_BYTE_LENGTH + wordIndex * 4,
          size: 4
        });
      copyWord(segmentCountBuffer, RECORD_SEGMENTS, 1);
      copyWord(polylineVertexCountBuffer, RECORD_POLYLINES, 1);
      copyWord(bandVertexCountBuffer, RECORD_BANDS, 0);
      ringsEncoded = false;
      if (ctx.options.showRings && ringsGraph) {
        ringsGraph.encode(commandEncoder, {parameters: undefined});
        ringsEncoded = true;
      }
      ctx.setReadout('encode', `${encoding.stats.cpuEncodeTimeMilliseconds.toFixed(2)} ms CPU`);
      dirty = false;
      summary.request(commandEncoder);
      void frame;
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showRelief) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'contours-hillshade',
            coordinateOrigin: origin,
            gridSize: [width, height],
            bounds: extent,
            rowOrigin: 'south',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, dark ? 190 : 175]
          })
        );
      }
      if (options.showBands) {
        layers.push(
          new IsobandTriangleLayer({
            id: 'contours-bands',
            coordinateOrigin: origin,
            gridSize: [1, 1],
            bounds: extent,
            triangles: trianglesBuffer,
            triangleBands: triangleBandsBuffer,
            values: paletteBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            extent: isobandParameters.buffer,
            drawCommands,
            drawCommandIndex: RECORD_BANDS,
            opacity: options.bandOpacity
          })
        );
      }
      if (options.showRings) {
        layers.push(
          new PolylineLayer({
            id: 'contours-ring-outlines',
            coordinateOrigin: origin,
            segments: ringPositionsBuffer,
            polylineOffsets: ringOffsetsBuffer,
            valueIndices: ringIsHoleBuffer,
            extent: ringCountBuffer,
            values: ringStyleBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [SHELL_COLOR, HOLE_COLOR],
            instanceCount: RING_EDGE_CAPACITY,
            widthPixels: 1.8
          })
        );
      }
      if (options.showLines) {
        const colors = dark ? LINE_COLORS.dark : LINE_COLORS.light;
        const lineStyle = {
          coordinateOrigin: origin,
          values: levelStyleBuffer,
          valueFormat: 'uint32' as const,
          colormap: 'category' as const,
          palette: [colors.line, colors.index],
          widthPixels: options.lineWidth
        };
        layers.push(
          options.stitchLines
            ? new PolylineLayer({
                id: 'contours-polylines',
                ...lineStyle,
                segments: polylineVerticesBuffer,
                polylineOffsets: polylineOffsetsBuffer,
                valueIndices: polylineLevelsBuffer,
                extent: polylineCountBuffer,
                drawCommands,
                drawCommandIndex: RECORD_POLYLINES
              })
            : new SpatialAnalysisSegmentLayer({
                id: 'contours-segments',
                ...lineStyle,
                segments: segmentsBuffer,
                valueIndices: segmentLevelsBuffer,
                drawCommands,
                drawCommandIndex: RECORD_SEGMENTS
              })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const cell = getCellAt(grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0) return null;
      const elevation = grid.elevation[cell];
      const interval = ctx.options.elevationInterval;
      const low = Math.floor(elevation / interval) * interval;
      return `Elevation ${formatInteger(elevation)} m\nBand ${formatInteger(low)} to ${formatInteger(low + interval)} m`;
    },

    destroy() {
      destroyed = true;
      summary.stop();
      resources.destroy();
    }
  };
}
