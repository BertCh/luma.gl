// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPULineChunkParameterValues,
  getGPULineLocateParameterValues,
  getGPULineSegmentizeParameterValues,
  GPULineChunk,
  GPULineLocate,
  GPULinearReferencing,
  GPULineSegmentize,
  GPU_LINE_CHUNK_PARAMETER_LENGTH,
  GPU_LINE_LOCATE_PARAMETER_LENGTH,
  GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext} from '../scene';
import {
  copyCountToDrawRecord,
  createGraphImporter,
  createPathOutputBuffers,
  createStaticPaths,
  formatDistance,
  importPathOutput,
  type GeometryView,
  type PathOutputBuffers
} from './b3-common';
import type {PathSet} from './b3-city-data';
import {PairSegmentLayer, PathOutputLayer} from './b3-layers';
import type {LineOperationsOptions} from './b3-line-options';

type Options = LineOperationsOptions;

const LOCAL = COORDINATE_SYSTEM.METER_OFFSETS;
const LNGLAT = COORDINATE_SYSTEM.LNGLAT;

/** Inputs of the Chicago views, prepared once. */
export type CityLineEnvironment = {
  ctx: SceneContext<Options>;
  coordinateOrigin: [number, number, number];
  rail: PathSet;
  streets: PathSet;
  places: {
    local: Float32Array;
    categories: Uint8Array;
    count: number;
  };
};

const MINIMUM_DENSIFY_LENGTH = 10;
const MINIMUM_CHUNK_LENGTH = 100;
const EVENTS_PER_LINE = 40;

/** Sum of segment lengths of the paths of a set, in local meters. */
function getTotalLength(paths: PathSet): number {
  let total = 0;
  for (let path = 0; path < paths.pathCount; path++) {
    for (let row = paths.offsets[path]; row + 1 < paths.offsets[path + 1]; row++) {
      total += Math.hypot(
        paths.local[row * 2 + 2] - paths.local[row * 2],
        paths.local[row * 2 + 3] - paths.local[row * 2 + 1]
      );
    }
  }
  return total;
}

// ---------------------------------------------------------------------------------------------
// Reshape: densify, chunk, substring and locate on the L routes
// ---------------------------------------------------------------------------------------------

/**
 * The eight L routes put through four line tools: `GPULineSegmentize` (densify),
 * `GPULineChunk` (chunk and substring modes) and `GPULineLocate` (events by distance or fraction).
 * Each tool and coordinate system is compiled on first use and cached.
 */
export function createReshapeView(env: CityLineEnvironment): GeometryView<Options> {
  const {ctx, rail} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'reshape');
  const inputVertices = rail.vertexCount;
  const pathCount = rail.pathCount;
  const totalLength = getTotalLength(rail);
  const localPositions = resources.createBuffer('rail-local', rail.local);
  const lngLatPositions = resources.createBuffer('rail-lnglat', rail.lngLat);
  const railOffsets = resources.createBuffer('rail-offsets', rail.offsets);
  const staticRail = createStaticPaths(resources, 'rail-static', rail.local, rail.offsets);
  let maximumPathLength = 0;
  for (let path = 0; path < pathCount; path++) {
    let length = 0;
    for (let row = rail.offsets[path]; row + 1 < rail.offsets[path + 1]; row++) {
      length += Math.hypot(
        rail.local[row * 2 + 2] - rail.local[row * 2],
        rail.local[row * 2 + 3] - rail.local[row * 2 + 1]
      );
    }
    maximumPathLength = Math.max(maximumPathLength, length);
  }

  ctx.setLegendExtent('measure', [0, maximumPathLength]);

  type Tool = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    output?: PathOutputBuffers;
    reader?: SummaryReader;
    encode: (
      commandEncoder: Parameters<GeometryView<Options>['encode']>[0],
      frameTime: number
    ) => void;
    layers: (options: Options, dark: boolean) => Layer[];
    write: (options: Options) => void;
    dirty: boolean;
  };
  const tools = new Map<string, Tool>();
  let active: Tool | null = null;

  const buildDensify = (options: Options): Tool => {
    const spherical = options.lineSystem === 'spherical';
    let capacity = inputVertices;
    for (let path = 0; path < pathCount; path++) {
      for (let row = rail.offsets[path]; row + 1 < rail.offsets[path + 1]; row++) {
        const length = Math.hypot(
          rail.local[row * 2 + 2] - rail.local[row * 2],
          rail.local[row * 2 + 3] - rail.local[row * 2 + 1]
        );
        capacity += Math.min(1024, Math.max(1, Math.ceil(length / MINIMUM_DENSIFY_LENGTH))) - 1;
      }
    }
    capacity += 16;
    const output = createPathOutputBuffers(resources, `densify-${spherical}`, capacity, pathCount);
    const parameters = resources.createParameterBuffer(
      'densify-parameters',
      'float32',
      GPU_LINE_SEGMENTIZE_PARAMETER_LENGTH
    );
    const graph = new GPUCommandGraph<void>(device, {id: `densify-${spherical}`});
    const imp = createGraphImporter(graph);
    graph.add(
      new GPULineSegmentize({
        id: 'densify',
        positions: imp(
          'positions',
          spherical ? lngLatPositions : localPositions,
          'float32x2',
          inputVertices
        ),
        pathOffsets: imp('offsets', railOffsets, 'uint32', pathCount + 1),
        coordinateSystem: spherical ? 'spherical' : 'planar',
        parameters: parameters.importToGraph(graph),
        output: importPathOutput(graph, output, {measures: true})
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `densify-${spherical}`,
      [
        {buffer: output.count, size: 4},
        {buffer: output.overflow, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        ctx.setReadout(
          'reshapeVertices',
          `${formatCount(words[0])} (${(words[0] / inputVertices).toFixed(2)}× the ${formatCount(inputVertices)} input vertices)`
        );
        ctx.setReadout(
          'reshapeSpacing',
          formatDistance(totalLength / Math.max(1, words[0] - pathCount))
        );
        ctx.setReadout('reshapeOverflow', words[1] ? 'yes' : 'no');
      }
    );
    const tool: Tool = {
      key: `densify-${spherical}`,
      compiled,
      output,
      reader,
      dirty: true,
      write: opts => {
        parameters.write(
          getGPULineSegmentizeParameterValues({maximumSegmentLength: 10 ** opts.densifyLength})
        );
        tool.dirty = true;
      },
      encode: commandEncoder => {
        if (tool.dirty) {
          compiled.encode(commandEncoder, {parameters: undefined});
          copyCountToDrawRecord(commandEncoder, output.count, output.drawCommands);
          reader.request(commandEncoder);
          tool.dirty = false;
        } else {
          reader.flush(commandEncoder);
        }
      },
      layers: (opts, dark) => {
        const system = opts.lineSystem === 'spherical' ? LNGLAT : LOCAL;
        const layers: Layer[] = [];
        if (opts.showOriginal) {
          layers.push(originalLayer(dark));
        }
        layers.push(
          new PathOutputLayer({
            id: 'densify-path',
            coordinateSystem: system,
            coordinateOrigin: env.coordinateOrigin,
            positions: output.positions,
            pathOffsets: output.offsets,
            pathOffsetCount: output.offsetCount,
            vertexCount: output.count,
            drawCommands: output.drawCommands,
            values: output.measures,
            colorSource: 'vertex-value',
            colormap: 'ylgnbu',
            valueRange: [0, maximumPathLength],
            color: [255, 255, 255, 255],
            widthPixels: 2.2
          }),
          new SpatialAnalysisPointLayer({
            id: 'densify-vertices',
            coordinateSystem: system,
            coordinateOrigin: env.coordinateOrigin,
            positions: output.positions,
            drawCommands: output.drawCommands,
            values: output.measures,
            valueFormat: 'float32',
            colormap: 'ylgnbu',
            valueRange: [0, maximumPathLength],
            radiusPixels: 2.2,
            color: [255, 255, 255, 235]
          })
        );
        return layers;
      }
    };
    return tool;
  };

  const originalLayer = (dark: boolean) =>
    new PathOutputLayer({
      id: 'rail-original',
      coordinateSystem: LOCAL,
      coordinateOrigin: env.coordinateOrigin,
      positions: staticRail.positions,
      pathOffsets: staticRail.offsets,
      pathOffsetCount: staticRail.offsetCount,
      vertexCount: staticRail.vertexCount,
      drawCommands: staticRail.drawCommands,
      color: dark ? [150, 160, 180, 90] : [90, 100, 120, 90],
      widthPixels: 4
    });

  const buildChunk = (options: Options, mode: 'chunk' | 'substring'): Tool => {
    const spherical = options.lineSystem === 'spherical';
    const chunkMode = mode === 'chunk';
    const pieceCapacity = Math.ceil(totalLength / MINIMUM_CHUNK_LENGTH) + pathCount + 16;
    const vertexCapacity = inputVertices + 2 * pieceCapacity + 16;
    const output = createPathOutputBuffers(
      resources,
      `${mode}-${spherical}`,
      vertexCapacity,
      pieceCapacity
    );
    const parameters = resources.createParameterBuffer(
      `${mode}-parameters`,
      'float32',
      GPU_LINE_CHUNK_PARAMETER_LENGTH
    );
    const graph = new GPUCommandGraph<void>(device, {id: `${mode}-${spherical}`});
    const imp = createGraphImporter(graph);
    graph.add(
      new GPULineChunk({
        id: mode,
        positions: imp(
          'positions',
          spherical ? lngLatPositions : localPositions,
          'float32x2',
          inputVertices
        ),
        pathOffsets: imp('offsets', railOffsets, 'uint32', pathCount + 1),
        mode,
        coordinateSystem: spherical ? 'spherical' : 'planar',
        parameters: parameters.importToGraph(graph),
        output: importPathOutput(graph, output, {pathCount: true, sourcePaths: chunkMode})
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `${mode}-${spherical}`,
      [
        {buffer: output.count, size: 4},
        {buffer: output.overflow, size: 4},
        {buffer: output.pathCount, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        ctx.setReadout('reshapeVertices', formatCount(words[0]));
        ctx.setReadout('reshapeOverflow', words[1] ? 'yes (capacity)' : 'no');
        ctx.setReadout(
          'reshapePieces',
          `${formatCount(words[2])} of ${formatCount(pieceCapacity)} slots`
        );
      }
    );
    const tool: Tool = {
      key: `${mode}-${spherical}`,
      compiled,
      output,
      reader,
      dirty: true,
      write: opts => {
        parameters.write(
          getGPULineChunkParameterValues(
            chunkMode
              ? {chunkLength: 10 ** opts.chunkLength}
              : {
                  startMeasure: opts.substringStart * 1000,
                  endMeasure: Math.max(opts.substringStart, opts.substringEnd) * 1000
                }
          )
        );
        tool.dirty = true;
      },
      encode: commandEncoder => {
        if (tool.dirty) {
          compiled.encode(commandEncoder, {parameters: undefined});
          copyCountToDrawRecord(commandEncoder, output.count, output.drawCommands);
          reader.request(commandEncoder);
          tool.dirty = false;
        } else {
          reader.flush(commandEncoder);
        }
      },
      layers: (opts, dark) => {
        const system = opts.lineSystem === 'spherical' ? LNGLAT : LOCAL;
        const layers: Layer[] = [];
        if (!chunkMode || opts.showOriginal) layers.push(originalLayer(dark));
        layers.push(
          new PathOutputLayer({
            id: `${mode}-output`,
            coordinateSystem: system,
            coordinateOrigin: env.coordinateOrigin,
            positions: output.positions,
            pathOffsets: output.offsets,
            pathOffsetCount: output.offsetCount,
            vertexCount: output.count,
            drawCommands: output.drawCommands,
            colorSource: chunkMode ? 'path-index' : 'uniform',
            color: chunkMode ? [255, 255, 255, 235] : [240, 150, 60, 255],
            widthPixels: chunkMode ? 3 : 4
          })
        );
        return layers;
      }
    };
    return tool;
  };

  const buildLocate = (options: Options): Tool => {
    const fraction = options.locateMode === 'fraction';
    const eventCount = pathCount * EVENTS_PER_LINE;
    const eventPaths = new Uint32Array(eventCount);
    const eventMeasures = new Float32Array(eventCount);
    for (let path = 0; path < pathCount; path++) {
      for (let event = 0; event < EVENTS_PER_LINE; event++) {
        const row = path * EVENTS_PER_LINE + event;
        eventPaths[row] = path;
        // Distance mode: a marker every kilometre (scaled per frame). Fraction mode: even spread.
        eventMeasures[row] = fraction ? event / EVENTS_PER_LINE : event * 1000;
      }
    }
    const pathBuffer = resources.createBuffer('event-paths', eventPaths);
    const measureBuffer = resources.createBuffer('event-measures', eventMeasures);
    const offsetBuffer = resources.createBuffer('event-offsets', eventCount * 4);
    const parameters = resources.createParameterBuffer(
      'locate-parameters',
      'float32',
      GPU_LINE_LOCATE_PARAMETER_LENGTH
    );
    const positions = resources.createBuffer(`event-positions-${fraction}`, eventCount * 8);
    const tangents = resources.createBuffer(`event-tangents-${fraction}`, eventCount * 8);
    const statuses = resources.createBuffer(`event-statuses-${fraction}`, eventCount * 4);
    const graph = new GPUCommandGraph<void>(device, {id: `locate-${fraction}`});
    const imp = createGraphImporter(graph);
    graph.add(
      new GPULineLocate({
        id: 'locate',
        positions: imp('positions', localPositions, 'float32x2', inputVertices),
        pathOffsets: imp('offsets', railOffsets, 'uint32', pathCount + 1),
        eventPaths: imp('event-paths', pathBuffer, 'uint32', eventCount),
        eventMeasures: imp('event-measures', measureBuffer, 'float32', eventCount),
        eventOffsets: imp('event-offsets', offsetBuffer, 'float32', eventCount),
        measureMode: fraction ? 'fraction' : 'distance',
        parameters: parameters.importToGraph(graph),
        output: {
          positions: imp('event-positions', positions, 'float32x2', eventCount),
          tangents: imp('event-tangents', tangents, 'float32x2', eventCount),
          statuses: imp('event-statuses', statuses, 'uint32', eventCount)
        }
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `locate-${fraction}`,
      [{buffer: statuses, size: eventCount * 4}],
      bytes => {
        let clamped = 0;
        let invalid = 0;
        for (const status of new Uint32Array(bytes)) {
          if (status === 1) clamped++;
          else if (status === 2) invalid++;
        }
        ctx.setReadout(
          'reshapePieces',
          `${formatCount(eventCount)} events, ${formatCount(clamped)} clamped to a route end, ${formatCount(invalid)} invalid`
        );
      }
    );
    let lastTime = 0;
    const tool: Tool = {
      key: `locate-${fraction}`,
      compiled,
      reader,
      dirty: true,
      write: opts => {
        // Alternate lanes so the offset reads as two tracks.
        const lanes = new Float32Array(eventCount);
        for (let row = 0; row < eventCount; row++)
          lanes[row] = (row % 2 === 0 ? 1 : -1) * opts.locateLateral;
        offsetBuffer.write(lanes);
        tool.dirty = true;
      },
      encode: (commandEncoder, time) => {
        const opts = ctx.options;
        const spacing = opts.locateSpacing;
        const phase = opts.locateAnimate ? (time / 12) % 1 : 0;
        const offset = fraction ? (phase / EVENTS_PER_LINE) * spacing : phase * 1000 * spacing;
        if (opts.locateAnimate || tool.dirty) {
          parameters.write(
            getGPULineLocateParameterValues({measureScale: spacing, measureOffset: offset})
          );
          compiled.encode(commandEncoder, {parameters: undefined});
          if (time - lastTime > 0.5 || tool.dirty) {
            reader.request(commandEncoder);
            lastTime = time;
          }
          tool.dirty = false;
        }
        reader.flush(commandEncoder);
      },
      layers: (opts, dark) => {
        const layers: Layer[] = [];
        layers.push(originalLayer(dark));
        layers.push(
          new PairSegmentLayer({
            id: 'locate-ticks',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            starts: positions,
            ends: tangents,
            directionScale: 140,
            instanceCount: eventCount,
            color: dark ? [255, 255, 255, 230] : [30, 40, 60, 230],
            widthPixels: 1.6
          }),
          new SpatialAnalysisPointLayer({
            id: 'locate-events',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions,
            instanceCount: eventCount,
            radiusPixels: 4,
            color: [240, 150, 60, 255]
          })
        );
        return layers;
      }
    };
    return tool;
  };

  const select = (options: Options): Tool => {
    const spherical = options.lineSystem === 'spherical';
    const key =
      options.lineTool === 'locate'
        ? `locate-${options.locateMode === 'fraction'}`
        : `${options.lineTool}-${spherical}`;
    let tool = tools.get(key);
    if (!tool) {
      tool =
        options.lineTool === 'densify'
          ? buildDensify(options)
          : options.lineTool === 'locate'
            ? buildLocate(options)
            : buildChunk(options, options.lineTool);
      tools.set(key, tool);
    }
    tool.write(options);
    active = tool;
    ctx.setReadout(
      'reshapeInputs',
      `${pathCount} routes, ${formatCount(inputVertices)} vertices, ${formatDistance(totalLength)}`
    );
    ctx.setReadout('reshapeVertices', null);
    ctx.setReadout('reshapeSpacing', null);
    ctx.setReadout('reshapePieces', null);
    ctx.setReadout('reshapeOverflow', null);
    tool.dirty = true;
    return tool;
  };
  select(ctx.options);

  return {
    getCompiledGraphs: () =>
      [...tools.values()].map(tool => tool.compiled as CompiledGPUCommandGraph<never>),
    setOption(id, options) {
      if (id === 'lineTool' || id === 'lineSystem' || id === 'locateMode' || id === 'view') {
        select(options);
      } else {
        active?.write(options);
      }
    },
    encode(commandEncoder, frame) {
      active?.encode(commandEncoder, frame.timeSeconds);
    },
    getLayers() {
      return active ? active.layers(ctx.options, ctx.theme() === 'dark') : [];
    },
    destroy() {
      for (const tool of tools.values()) tool.reader?.stop();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Snap: GPULinearReferencing (eligible community places to streets)
// ---------------------------------------------------------------------------------------------

/**
 * Eligible community places snapped to the nearest street polyline. The source has no asserted street
 * edge, so this is a nearest-segment demonstration rather than an address-match benchmark.
 */
export function createSnapView(env: CityLineEnvironment): GeometryView<Options> {
  const {ctx, streets, places} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'snap');
  const pointCount = places.count;
  const pathCount = streets.pathCount;
  const vertexCount = streets.vertexCount;
  const staticStreets = createStaticPaths(
    resources,
    'snap-streets',
    streets.local,
    streets.offsets
  );
  const points = resources.createBuffer('place-points', places.local);
  const streetPositions = resources.createBuffer('street-positions', streets.local);
  const streetOffsets = resources.createBuffer('street-offsets', streets.offsets);
  const radius = resources.createParameterBuffer('radius', 'float32', 1);
  const footPoints = resources.createBuffer('foot-points', pointCount * 8);
  const distances = resources.createBuffer('distances', pointCount * 4);
  const measures = resources.createBuffer('measures', pointCount * 4);
  const signedOffsets = resources.createBuffer('signed-offsets', pointCount * 4);
  const pathIndices = resources.createBuffer('path-indices', pointCount * 4);
  const overflow = resources.createBuffer('overflow', 4);
  const candidateCount = resources.createBuffer('candidate-count', 4);
  const candidateCapacity = 1 << 22;

  const graph = new GPUCommandGraph<void>(device, {id: 'snap'});
  const imp = createGraphImporter(graph);
  graph.add(
    new GPULinearReferencing({
      id: 'snap',
      points: imp('points', points, 'float32x2', pointCount),
      positions: imp('positions', streetPositions, 'float32x2', vertexCount),
      pathOffsets: imp('offsets', streetOffsets, 'uint32', pathCount + 1),
      radius: radius.importToGraph(graph),
      candidateCapacity,
      spatialSort: true,
      output: {
        footPoints: imp('foot-points', footPoints, 'float32x2', pointCount),
        distances: imp('distances', distances, 'float32', pointCount),
        measures: imp('measures', measures, 'float32', pointCount),
        signedOffsets: imp('signed-offsets', signedOffsets, 'float32', pointCount),
        pathIndices: imp('path-indices', pathIndices, 'uint32', pointCount)
      },
      overflow: imp('overflow', overflow, 'uint32', 1),
      candidateCount: imp('candidate-count', candidateCount, 'uint32', 1)
    })
  );
  const compiled = resources.track(graph.compile());

  const categoryCodes: Record<Exclude<Options['placeCategory'], 'all'>, number> = {
    grocery: 2,
    school_education: 4,
    park_recreation: 5,
    arts_culture: 10,
    worship_community: 11
  };
  const allowedCategoryCodes = new Set(Object.values(categoryCodes));
  const categoryMarks = Object.entries(categoryCodes).map(([category, code], index) => {
    const positions = new Float32Array(pointCount * 2);
    for (let row = 0; row < pointCount; row++) {
      const matches = places.categories[row] === code;
      positions[row * 2] = matches ? places.local[row * 2] : -1e8;
      positions[row * 2 + 1] = matches ? places.local[row * 2 + 1] : -1e8;
    }
    return {
      category,
      code,
      buffer: resources.createBuffer(`place-category-${category}`, positions),
      shape: (['square', 'triangle', 'diamond', 'star', 'cross'] as const)[index]
    };
  });
  let selectedPlaceCount = 0;
  const selectedRows = new Uint8Array(pointCount);

  let dirty = true;
  const reader = new SummaryReader(
    resources,
    'snap',
    [
      {buffer: signedOffsets, size: pointCount * 4},
      {buffer: distances, size: pointCount * 4},
      {buffer: pathIndices, size: pointCount * 4},
      {buffer: overflow, size: 4},
      {buffer: candidateCount, size: 4}
    ],
    bytes => {
      const offsets = new Float32Array(bytes, 0, pointCount);
      const distanceValues = new Float32Array(bytes, pointCount * 4, pointCount);
      const paths = new Uint32Array(bytes, pointCount * 8, pointCount);
      const words = new Uint32Array(bytes, pointCount * 12, 2);
      let matched = 0;
      let left = 0;
      let right = 0;
      const matchedDistances: number[] = [];
      let withinTenMeters = 0;
      for (let index = 0; index < pointCount; index++) {
        if (selectedRows[index] && Number.isFinite(offsets[index]) && paths[index] !== 0xffffffff) {
          matched++;
          matchedDistances.push(distanceValues[index]);
          if (distanceValues[index] <= 10) withinTenMeters++;
          if (offsets[index] > 0) left++;
          else if (offsets[index] < 0) right++;
        }
      }
      matchedDistances.sort((leftDistance, rightDistance) => leftDistance - rightDistance);
      const percentile = (fraction: number) =>
        matchedDistances[
          Math.min(
            matchedDistances.length - 1,
            Math.floor((matchedDistances.length - 1) * fraction)
          )
        ];
      ctx.setReadout(
        'snapMatched',
        `${formatCount(matched)} of ${formatCount(selectedPlaceCount)}`
      );
      ctx.setReadout('snapSides', `${formatCount(left)} left / ${formatCount(right)} right`);
      ctx.setReadout('snapDistance', matched ? formatDistance(percentile(0.5)) : 'n/a');
      ctx.setReadout('snapP90Distance', matched ? formatDistance(percentile(0.9)) : 'n/a');
      ctx.setReadout(
        'snapWithinTen',
        matched ? `${((100 * withinTenMeters) / matched).toFixed(1)}%` : 'n/a'
      );
      ctx.setReadout(
        'snapCandidates',
        `${formatCount(words[1])} of ${formatCount(candidateCapacity)}${words[0] ? ' (overflow)' : ''}`
      );
    }
  );

  const writePoints = (options: Options) => {
    const requestedCategory =
      options.placeCategory === 'all' ? undefined : categoryCodes[options.placeCategory];
    const selectedPositions = new Float32Array(pointCount * 2);
    for (let index = 0; index < pointCount; index++) {
      const selected =
        allowedCategoryCodes.has(places.categories[index]) &&
        (requestedCategory === undefined || places.categories[index] === requestedCategory);
      selectedRows[index] = Number(selected);
      if (selected) {
        selectedPositions[index * 2] = places.local[index * 2];
        selectedPositions[index * 2 + 1] = places.local[index * 2 + 1];
      } else {
        // A remote coordinate keeps excluded rows out of the local spatial search and view.
        selectedPositions[index * 2] = -1e8;
        selectedPositions[index * 2 + 1] = -1e8;
      }
    }
    selectedPlaceCount = selectedRows.reduce((total, selected) => total + selected, 0);
    points.write(selectedPositions);
    for (const mark of categoryMarks) {
      const markPositions = new Float32Array(pointCount * 2);
      for (let index = 0; index < pointCount; index++) {
        const selected = selectedRows[index] && places.categories[index] === mark.code;
        markPositions[index * 2] = selected ? places.local[index * 2] : -1e8;
        markPositions[index * 2 + 1] = selected ? places.local[index * 2 + 1] : -1e8;
      }
      mark.buffer.write(markPositions);
    }
    ctx.setReadout(
      'snapInputs',
      `${formatCount(selectedPlaceCount)} eligible places, ${formatCount(pathCount)} streets`
    );
  };
  const writeRadius = (options: Options) => {
    radius.write(Float32Array.of(options.snapRadius));
    dirty = true;
  };
  writePoints(ctx.options);
  writeRadius(ctx.options);
  ctx.setReadout(
    'snapInputs',
    `${formatCount(selectedPlaceCount)} eligible places, ${formatCount(pathCount)} streets`
  );

  return {
    getCompiledGraphs: () => [compiled as CompiledGPUCommandGraph<never>],
    setOption(id, options) {
      if (id === 'snapRadius') writeRadius(options);
      if (id === 'placeCategory') {
        writePoints(options);
        dirty = true;
      }
    },
    encode(commandEncoder) {
      if (dirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        reader.request(commandEncoder);
        dirty = false;
      } else {
        reader.flush(commandEncoder);
      }
    },
    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const limit = Math.min(options.snapRadius, 40);
      const valueRange: [number, number] =
        options.snapColor === 'side'
          ? [-limit, limit]
          : options.snapColor === 'measure'
            ? [0, 600]
            : [0, limit];
      const values: Buffer =
        options.snapColor === 'side'
          ? signedOffsets
          : options.snapColor === 'measure'
            ? measures
            : distances;
      return [
        new PathOutputLayer({
          id: 'snap-streets',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: staticStreets.positions,
          pathOffsets: staticStreets.offsets,
          pathOffsetCount: staticStreets.offsetCount,
          vertexCount: staticStreets.vertexCount,
          drawCommands: staticStreets.drawCommands,
          color: dark ? [150, 160, 180, 90] : [90, 100, 120, 90],
          widthPixels: 0.8
        }),
        new PairSegmentLayer({
          id: 'snap-lines',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          starts: points,
          ends: footPoints,
          instanceCount: pointCount,
          values,
          colormap:
            options.snapColor === 'side'
              ? 'diverging'
              : options.snapColor === 'measure'
                ? 'ylgnbu'
                : 'ylorbr',
          valueRange,
          color: [255, 255, 255, 255],
          widthPixels: 1.4,
          opacity: 0.9
        }),
        ...categoryMarks.map(
          mark =>
            new SpatialAnalysisPointLayer({
              id: `snap-places-${mark.category}`,
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              positions: mark.buffer,
              instanceCount: pointCount,
              radiusPixels: 2.4,
              shape: mark.shape,
              outlineColor: dark ? [255, 255, 255, 220] : [20, 30, 50, 220],
              outlineWidthPixels: 0.8,
              color: dark ? [120, 200, 235, 210] : [0, 105, 150, 210]
            })
        )
      ];
    },
    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
