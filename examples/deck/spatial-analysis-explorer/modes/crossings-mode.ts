// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Crossings: `GPUSegmentIntersection` in self mode finds every crossing, touch and overlap between the
 * New York road polylines. Hits are drawn straight from the `points` and `kinds` outputs (the instance
 * count is copied from the pair count into an indirect draw record) and colored by kind.
 * "Inject test roads" writes eight extra lines (diagonals, a collinear extension and a duplicate road)
 * so every kind lights up; the slider moves the diagonals. The graph is compiled once and encodes only
 * when an input changed.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {GPUSegmentIntersection} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {NEW_YORK_ORIGIN} from '../spatial-analysis-data';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {SummaryReader} from './summary-reader';

const PAIR_CAPACITY = 1 << 18;
const KIND_NAMES = ['', 'proper', 'touch', 'collinear touch', 'overlap', 'uncertain'] as const;
/** Palette by kind (index = kind). */
const KIND_COLORS = [
  [0, 0, 0, 0],
  [255, 70, 70, 255],
  [60, 200, 255, 255],
  [255, 214, 80, 255],
  [255, 90, 220, 255],
  [255, 255, 255, 255]
] as const;
const TOUCH_ONLY_PALETTE = [
  [0, 0, 0, 0],
  [255, 70, 70, 255],
  [0, 0, 0, 0],
  [255, 214, 80, 255],
  [255, 90, 220, 255],
  [255, 255, 255, 255]
] as const;
type RoadScene = {
  /** Vertex positions: road polylines followed by eight injected two-vertex lines. */
  positions: Float32Array;
  lineOffsets: Uint32Array;
  baseVertexCount: number;
  /** Fill for the injected lines; `active` false collapses them to zero-length. */
  writeInjected: (active: boolean, slide: number) => void;
  injectedSegments: Float32Array;
  span: number;
};

/** Chains consecutive road segments sharing an endpoint into polylines and appends test lines. */
function createRoadScene(segments: Float32Array): RoadScene {
  const segmentCount = segments.length / 4;
  const vertices: number[] = [];
  const offsets: number[] = [0];
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let row = 0; row < segmentCount; row++) {
    const x0 = segments[row * 4];
    const y0 = segments[row * 4 + 1];
    const x1 = segments[row * 4 + 2];
    const y1 = segments[row * 4 + 3];
    minX = Math.min(minX, x0, x1);
    maxX = Math.max(maxX, x0, x1);
    minY = Math.min(minY, y0, y1);
    maxY = Math.max(maxY, y0, y1);
    const continues =
      row > 0 &&
      vertices[vertices.length - 2] === x0 &&
      vertices[vertices.length - 1] === y0 &&
      offsets[offsets.length - 1] !== vertices.length / 2;
    if (!continues) {
      if (row > 0) offsets.push(vertices.length / 2);
      vertices.push(x0, y0);
    }
    vertices.push(x1, y1);
  }
  offsets.push(vertices.length / 2);
  const baseVertexCount = vertices.length / 2;
  const injectedLineCount = 8;
  for (let line = 0; line < injectedLineCount; line++) {
    vertices.push(0, 0, 0, 0);
    offsets.push(vertices.length / 2);
  }
  const positions = Float32Array.from(vertices);
  const injectedSegments = new Float32Array(injectedLineCount * 4);
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const span = Math.min(spanX, spanY);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  // Roads 0 and 5 define the collinear extension and the duplicate.
  const writeInjected = (active: boolean, slide: number) => {
    const lines: [number, number, number, number][] = [];
    for (let diagonal = 0; diagonal < 6; diagonal++) {
      const shift = (diagonal - 2.5) * 0.09 * span + slide * 0.2 * span;
      lines.push([
        centerX - 0.4 * spanX + shift,
        centerY - 0.4 * spanY,
        centerX + 0.4 * spanX + shift * 0.5,
        centerY + 0.4 * spanY
      ]);
    }
    // Duplicate of road segment 3: a full overlap.
    lines.push([segments[3 * 4], segments[3 * 4 + 1], segments[3 * 4 + 2], segments[3 * 4 + 3]]);
    // Collinear continuation of road segment 8 beyond its end: a collinear touch.
    const ax = segments[8 * 4];
    const ay = segments[8 * 4 + 1];
    const bx = segments[8 * 4 + 2];
    const by = segments[8 * 4 + 3];
    lines.push([bx, by, bx + (bx - ax), by + (by - ay)]);
    lines.forEach((line, index) => {
      // Inactive lines collapse to one point, which the contributor skips as degenerate.
      const values = active ? line : [line[0], line[1], line[0], line[1]];
      for (let component = 0; component < 4; component++) {
        positions[(baseVertexCount + index * 2) * 2 + component] = values[component];
        injectedSegments[index * 4 + component] = values[component];
      }
    });
  };
  return {
    positions,
    lineOffsets: Uint32Array.from(offsets),
    baseVertexCount,
    writeInjected,
    injectedSegments,
    span
  };
}

export const crossingsMode: SpatialAnalysisModeDefinition = {
  id: 'crossings',
  title: 'Crossings',
  contributors: ['GPUSegmentIntersection'],
  description:
    'Self-intersections between New York road polylines, colored by kind (proper, touch, collinear, overlap). ' +
    'Inject test roads and slide the diagonals to watch the marks update.',
  initialViewState: {longitude: -73.985, latitude: 40.745, zoom: 12.6},

  async create(context) {
    const roads = await context.data.getNewYorkRoads();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'crossings');

    // ------------------------------------------------------------------ road crossings
    const roadScene = createRoadScene(roads.segments);
    const roadLineCount = roadScene.lineOffsets.length - 1;
    const roadVertexCount = roadScene.positions.length / 2;
    const roadSegmentCount = roads.segments.length / 4;
    roadScene.writeInjected(false, 0);
    const roadPositions = resources.createBuffer('road-positions', roadScene.positions);
    const roadLineOffsets = resources.createBuffer('road-line-offsets', roadScene.lineOffsets);
    const roadSegments = resources.createBuffer('road-segments', roads.segments);
    const injectedBuffer = resources.createBuffer('injected-segments', roadScene.injectedSegments);
    const pairLeft = resources.createBuffer('pair-left', PAIR_CAPACITY * 4);
    const pairRight = resources.createBuffer('pair-right', PAIR_CAPACITY * 4);
    const pairCount = resources.createBuffer('pair-count', 4);
    const pairOverflow = resources.createBuffer('pair-overflow', 4);
    const pairTotal = resources.createBuffer('pair-total', 4);
    const pairKinds = resources.createBuffer('pair-kinds', PAIR_CAPACITY * 4);
    const pairPoints = resources.createBuffer('pair-points', PAIR_CAPACITY * 8);
    const uncertainCount = resources.createBuffer('uncertain-count', 4);
    const markerDraw = resources.track(
      new DrawCommandBuffer(device, {
        id: 'crossings-marker-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const roadGraph = (() => {
      const graph = new GPUCommandGraph<void>(device, {id: 'crossings-roads'});
      graph.add(
        new GPUSegmentIntersection({
          id: 'road-crossings',
          left: {
            kind: 'lines',
            positions: importGraphBuffer(
              graph,
              'road-positions',
              roadPositions,
              'float32x2',
              roadVertexCount
            ),
            lineOffsets: importGraphBuffer(
              graph,
              'road-line-offsets',
              roadLineOffsets,
              'uint32',
              roadLineCount + 1
            )
          },
          pairs: {
            leftIds: importGraphBuffer(graph, 'pair-left', pairLeft, 'uint32', PAIR_CAPACITY),
            rightIds: importGraphBuffer(graph, 'pair-right', pairRight, 'uint32', PAIR_CAPACITY),
            count: importGraphBuffer(graph, 'pair-count', pairCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'pair-overflow', pairOverflow, 'uint32', 1),
            totalCount: importGraphBuffer(graph, 'pair-total', pairTotal, 'uint32', 1)
          },
          kinds: importGraphBuffer(graph, 'pair-kinds', pairKinds, 'uint32', PAIR_CAPACITY),
          points: importGraphBuffer(graph, 'pair-points', pairPoints, 'float32x2', PAIR_CAPACITY),
          uncertainCount: importGraphBuffer(graph, 'uncertain-count', uncertainCount, 'uint32', 1)
        })
      );
      return resources.track(graph.compile());
    })();

    // ------------------------------------------------------------------ state and controls
    let injectRoads = false;
    let slide = 0;
    let showTouches = true;
    let roadsDirty = true;
    let destroyed = false;

    context.controls.addToggle({
      label: 'Inject test roads (diagonals, extension, duplicate)',
      value: injectRoads,
      onChange: value => {
        injectRoads = value;
        roadScene.writeInjected(injectRoads, slide);
        roadPositions.write(roadScene.positions);
        injectedBuffer.write(roadScene.injectedSegments);
        roadsDirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Slide injected diagonals (re-runs the join)',
      min: -1,
      max: 1,
      step: 0.05,
      value: slide,
      format: value => value.toFixed(2),
      onChange: value => {
        slide = value;
        if (!injectRoads) return;
        roadScene.writeInjected(true, slide);
        roadPositions.write(roadScene.positions);
        injectedBuffer.write(roadScene.injectedSegments);
        roadsDirty = true;
      }
    });
    context.controls.addToggle({
      label: 'Show touch marks (junctions)',
      value: showTouches,
      onChange: value => {
        showTouches = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Road hit kind',
      entries: [1, 2, 3, 4, 5].map(kind => ({
        color: KIND_COLORS[kind],
        label: KIND_NAMES[kind]
      }))
    });
    const roadReadout = context.controls.addReadout('Road hits', '...');
    const kindReadout = context.controls.addReadout('By kind', '...');
    const roadFlagReadout = context.controls.addReadout('Overflow / uncertain', '...');
    context.controls.addNote('The graph encodes only when an input changed (inject, slide).');
    context.controls.addReadout('Data', roads.attribution);

    const roadReader = new SummaryReader(
      resources,
      'crossings-roads',
      [
        {buffer: pairCount, size: 4},
        {buffer: pairOverflow, size: 4},
        {buffer: pairTotal, size: 4},
        {buffer: uncertainCount, size: 4},
        {buffer: pairKinds, size: PAIR_CAPACITY * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const count = Math.min(words[0], PAIR_CAPACITY);
        const kinds = words.subarray(4, 4 + count);
        const histogram = [0, 0, 0, 0, 0, 0];
        for (const kind of kinds) if (kind < histogram.length) histogram[kind]++;
        roadReadout.setValue(
          `${formatCount(words[2])} pairs among ${formatCount(roadSegmentCount)} road segments`
        );
        kindReadout.setValue(
          [1, 2, 3, 4, 5]
            .map(kind => `${KIND_NAMES[kind]} ${formatCount(histogram[kind])}`)
            .join(', ')
        );
        roadFlagReadout.setValue(`${words[1] ? 'OVERFLOW' : 'no'} / ${formatCount(words[3])}`);
      }
    );
    resources.track({
      destroy: () => {
        roadReader.stop();
      }
    });

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [roadGraph] as readonly CompiledGPUCommandGraph<never>[],
      encode(commandEncoder) {
        if (roadsDirty) {
          roadsDirty = false;
          roadGraph.encode(commandEncoder, {parameters: undefined});
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: pairCount,
            destinationBuffer: markerDraw.buffer,
            destinationOffset: 4,
            size: 4
          });
          roadReader.request(commandEncoder);
        } else {
          roadReader.flush(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [
          NEW_YORK_ORIGIN[0],
          NEW_YORK_ORIGIN[1],
          0
        ];
        const layers: Layer[] = [];
        {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'crossings-roads',
              coordinateOrigin,
              segments: roadSegments,
              instanceCount: roadSegmentCount,
              color: [150, 170, 200, 120],
              widthPixels: 1.2
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'crossings-injected',
              coordinateOrigin,
              segments: injectedBuffer,
              instanceCount: injectRoads ? roadScene.injectedSegments.length / 4 : 0,
              color: [255, 255, 255, 200],
              widthPixels: 2
            }),
            new SpatialAnalysisPointLayer({
              id: 'crossings-marks',
              coordinateOrigin,
              positions: pairPoints,
              drawCommands: markerDraw,
              values: pairKinds,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: showTouches ? KIND_COLORS : TOUCH_ONLY_PALETTE,
              radiusPixels: 3
            }),
            new SpatialAnalysisPointLayer({
              id: 'crossings-marks-emphasis',
              coordinateOrigin,
              positions: pairPoints,
              drawCommands: markerDraw,
              values: pairKinds,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: TOUCH_ONLY_PALETTE,
              radiusPixels: 6
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};
