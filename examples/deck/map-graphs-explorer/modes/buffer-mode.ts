// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Buffer select: `GPUBufferSelection` picks the New York points of interest that lie within a
 * per-frame distance of a road subset. The road subset is chosen at runtime WITHOUT recompiling:
 * the feature capacity is fixed to every road segment, and rows outside the subset are "parked" as
 * a degenerate segment far outside the data by rewriting the starts/ends buffers. The distance
 * slider rewrites a parameter buffer every frame. Selected POIs are drawn straight from the
 * compact `ids` output with a GPU-written instance count (no readback); only a small summary
 * (selected count, overflow flags) is read back. `spatialSort` is the compile-time option and is
 * timed on vs off outside the frame.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {GPUBufferSelection} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {MapGraphsPointLayer, MapGraphsSegmentLayer} from '../map-graphs-layers';
import {LocalMetricProjection} from '../map-graphs-data';
import type {MapGraphsModeDefinition, MapGraphsModeInstance} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

const MAXIMUM_DISTANCE = 200;
const DEFAULT_DISTANCE = 30;
/** Frames between summary readbacks. */
const READBACK_INTERVAL = 10;
/** Frame at which the spatialSort comparison runs once. */
const AUTO_MEASURE_FRAME = 40;
const CANDIDATES_PER_POINT = 128;
/**
 * Parked rows sit this many meters beyond the north-east corner of the data, far more than the
 * maximum distance, so no POI is ever in range. Keeping them close to the data (rather than at a
 * huge coordinate) keeps the Morton domain of `spatialSort` tight around the real segments.
 */
const PARKED_MARGIN = 2000;
/** Click selection: segments within this many meters of the click form the subset. */
const CLICK_RADIUS = 25;

type SubsetKind = 'major' | 'tertiary' | 'minor' | 'all' | 'click';

const SUBSET_OPTIONS: readonly {value: SubsetKind; label: string}[] = [
  {value: 'major', label: 'Major roads (motorway–secondary)'},
  {value: 'tertiary', label: 'Tertiary'},
  {value: 'minor', label: 'Minor streets'},
  {value: 'all', label: 'All roads'}
];

export const bufferMode: MapGraphsModeDefinition = {
  id: 'buffer',
  title: 'Buffer select',
  recipes: ['GPUBufferSelection'],
  description:
    'Select the points of interest within a distance of a road subset, on the GPU. The distance ' +
    'and the road subset change every frame without recompiling; click a road to buffer only ' +
    'around it.',
  initialViewState: {longitude: -73.985, latitude: 40.755, zoom: 15},

  async create(context) {
    const [pois, roads] = await Promise.all([
      context.data.getNewYorkPointsOfInterest(),
      context.data.getNewYorkRoads()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new MapGraphsResources(device, 'buffer');
    const projection = new LocalMetricProjection(roads.origin);
    const pointCount = pois.positions.length / 2;
    const segmentCount = roads.segments.length / 4;

    // Fixed-capacity feature buffers: rewritten when the subset changes, never recompiled.
    const starts = new Float32Array(segmentCount * 2);
    const ends = new Float32Array(segmentCount * 2);
    const selectedFlags = new Uint32Array(segmentCount);
    const positionsBuffer = resources.createBuffer('positions', pois.positions);
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const startsBuffer = resources.createBuffer('starts', starts);
    const endsBuffer = resources.createBuffer('ends', ends);
    const flagsBuffer = resources.createBuffer('segment-flags', selectedFlags);
    const distance = resources.createParameterBuffer(
      'distance',
      'float32',
      1,
      Float32Array.of(DEFAULT_DISTANCE)
    );
    const outputMask = resources.createBuffer('output-mask', pointCount * 4);
    const outputIds = resources.createBuffer('output-ids', pointCount * 4);
    const outputCount = resources.createBuffer('output-count', 4);
    const outputOverflow = resources.createBuffer('output-overflow', 4);
    const outputTotal = resources.createBuffer('output-total', 4);
    const distances = resources.createBuffer('distances', pointCount * 4);
    const joinOverflow = resources.createBuffer('join-overflow', 4);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'buffer-selected-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    /** Compiles a graph for one `spatialSort` value over the shared caller-owned buffers. */
    const buildGraph = (spatialSort: boolean): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `buffer-${spatialSort ? 'sorted' : 'plain'}`
      });
      graph.add(
        new GPUBufferSelection({
          id: 'buffer',
          points: importGraphBuffer(graph, 'points', positionsBuffer, 'float32x2', pointCount),
          features: {
            kind: 'segments',
            starts: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', segmentCount),
            ends: importGraphBuffer(graph, 'ends', endsBuffer, 'float32x2', segmentCount)
          },
          distance: distance.importToGraph(graph),
          candidateCapacity: pointCount * CANDIDATES_PER_POINT,
          spatialSort,
          outputMask: importGraphBuffer(graph, 'output-mask', outputMask, 'uint32', pointCount),
          output: {
            ids: importGraphBuffer(graph, 'output-ids', outputIds, 'uint32', pointCount),
            count: importGraphBuffer(graph, 'output-count', outputCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'output-overflow', outputOverflow, 'uint32', 1),
            totalCount: importGraphBuffer(graph, 'output-total', outputTotal, 'uint32', 1)
          },
          distances: importGraphBuffer(graph, 'distances', distances, 'float32', pointCount),
          overflow: importGraphBuffer(graph, 'join-overflow', joinOverflow, 'uint32', 1)
        })
      );
      return graph.compile();
    };

    let spatialSort = false;
    let compiled = resources.track(buildGraph(spatialSort));

    const readbackRing = new GPUReadbackRing(device, {id: 'buffer-readback', byteLength: 16});
    resources.track({destroy: () => readbackRing.destroy()});

    let currentDistance = DEFAULT_DISTANCE;
    let subsetKind: SubsetKind = 'major';
    let subsetDirty = true;
    let subsetCount = 0;
    let colorByDistance = true;
    let destroyed = false;
    let readbackPending = false;
    let measuring = false;
    let autoMeasureScheduled = false;

    let parkedX = -Infinity;
    let parkedY = -Infinity;
    for (let index = 0; index < roads.segments.length; index += 2) {
      parkedX = Math.max(parkedX, roads.segments[index]);
      parkedY = Math.max(parkedY, roads.segments[index + 1]);
    }
    for (let index = 0; index < pois.positions.length; index += 2) {
      parkedX = Math.max(parkedX, pois.positions[index]);
      parkedY = Math.max(parkedY, pois.positions[index + 1]);
    }
    parkedX += PARKED_MARGIN;
    parkedY += PARKED_MARGIN;

    /** Rewrites the feature buffers for the chosen subset; unselected rows are parked. */
    const writeSubset = (isSelected: (row: number) => boolean) => {
      subsetCount = 0;
      for (let row = 0; row < segmentCount; row++) {
        const selected = isSelected(row);
        selectedFlags[row] = selected ? 1 : 0;
        if (selected) {
          subsetCount++;
          starts.set(roads.segments.subarray(row * 4, row * 4 + 2), row * 2);
          ends.set(roads.segments.subarray(row * 4 + 2, row * 4 + 4), row * 2);
        } else {
          starts[row * 2] = ends[row * 2] = parkedX;
          starts[row * 2 + 1] = ends[row * 2 + 1] = parkedY;
        }
      }
      startsBuffer.write(starts);
      endsBuffer.write(ends);
      flagsBuffer.write(selectedFlags);
      subsetReadout.setValue(
        `${formatCount(subsetCount)} of ${formatCount(segmentCount)} segments`
      );
    };

    const applyClassSubset = (kind: Exclude<SubsetKind, 'click'>) => {
      const {segmentClasses} = roads;
      const test = {
        major: (row: number) => segmentClasses[row] <= 1,
        tertiary: (row: number) => segmentClasses[row] === 2,
        minor: (row: number) => segmentClasses[row] >= 3,
        all: () => true
      }[kind];
      writeSubset(test);
    };

    const applyClickSubset = (x: number, y: number) => {
      const {segments} = roads;
      writeSubset(row => {
        const x0 = segments[row * 4];
        const y0 = segments[row * 4 + 1];
        const dx = segments[row * 4 + 2] - x0;
        const dy = segments[row * 4 + 3] - y0;
        const lengthSquared = dx * dx + dy * dy;
        const t = lengthSquared === 0 ? 0 : ((x - x0) * dx + (y - y0) * dy) / lengthSquared;
        const clamped = Math.min(1, Math.max(0, t));
        return Math.hypot(x - (x0 + clamped * dx), y - (y0 + clamped * dy)) <= CLICK_RADIUS;
      });
    };

    const subsetSelect = context.controls.addSelect<SubsetKind>({
      label: 'Road subset (per-frame: feature buffer rewrite, no recompile)',
      options: SUBSET_OPTIONS,
      value: subsetKind,
      onChange: value => {
        subsetKind = value;
        subsetDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Buffer distance (per-frame parameter)',
      min: 0,
      max: MAXIMUM_DISTANCE,
      step: 5,
      value: currentDistance,
      format: value => `${value} m`,
      onChange: value => {
        currentDistance = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Color selected POIs by distance',
      value: colorByDistance,
      onChange: value => {
        colorByDistance = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'Reset subset',
      onClick: () => {
        subsetKind = 'major';
        subsetSelect.setValue('major');
        subsetDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addNote('Click a road to buffer only around the segments within 25 m of it.');
    context.controls.addToggle({
      label: 'spatialSort (compile-time: rebuilds graph)',
      value: spatialSort,
      onChange: value => {
        if (value === spatialSort) return;
        spatialSort = value;
        const previous = compiled;
        compiled = resources.track(buildGraph(spatialSort));
        // Deck may still encode the previous graph this frame; destroy it two frames later.
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            if (!destroyed) resources.release(previous);
          })
        );
        context.updateLayers();
      }
    });
    const measureButton = context.controls.addButton({
      label: 'Measure spatialSort on vs off',
      onClick: () => void measureSpatialSort()
    });
    const sortOnReadout = context.controls.addReadout('spatialSort on', '...');
    const sortOffReadout = context.controls.addReadout('spatialSort off', '...');
    const speedupReadout = context.controls.addReadout('Speedup', '...');
    context.controls.addNote(
      'Timed outside the frame: GPU timestamps when the device has timestamp-query, otherwise wall clock / 8 repetitions (upper bound).'
    );
    context.controls.addLegend({
      title: 'POI color: distance to nearest selected road',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0 m',
        maximumLabel: 'distance'
      }
    });
    context.controls.addLegend({
      title: 'Overlays',
      entries: [
        {color: [255, 150, 40], label: 'Selected road subset'},
        {color: [130, 140, 160], label: 'Other roads and unselected POIs'}
      ]
    });
    context.controls.addReadout('Points of interest', formatCount(pointCount));
    const subsetReadout = context.controls.addReadout('Road subset', '...');
    const selectedReadout = context.controls.addReadout('Selected POIs', '...');
    const outputOverflowReadout = context.controls.addReadout('Output overflow', '...');
    const joinOverflowReadout = context.controls.addReadout('Join overflow', '...');
    context.controls.addReadout('Data', `${pois.attribution}; ${roads.attribution}`);

    /** Compiles a temporary graph for the other `spatialSort` value and times both. */
    async function measureSpatialSort(): Promise<void> {
      if (measuring || destroyed) return;
      measuring = true;
      measureButton.setDisabled(true);
      const other = buildGraph(!spatialSort);
      try {
        const current = spatialSort;
        const optionsFor = {parameters: undefined, completionBuffer: outputCount};
        const currentTiming = await measureCompiledGraph(device, compiled, optionsFor);
        const otherTiming = await measureCompiledGraph(device, other, optionsFor);
        if (destroyed) return;
        const [on, off]: [CompiledGraphTiming, CompiledGraphTiming] = current
          ? [currentTiming, otherTiming]
          : [otherTiming, currentTiming];
        sortOnReadout.setValue(formatCompiledGraphTiming(on));
        sortOffReadout.setValue(formatCompiledGraphTiming(off));
        speedupReadout.setValue(formatSpeedup(off.milliseconds, on.milliseconds));
      } catch {
        // Device destroyed or measurement aborted.
      } finally {
        other.destroy();
        measuring = false;
        if (!destroyed) measureButton.setDisabled(false);
      }
    }

    const readSummary = async (commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      [outputCount, outputOverflow, joinOverflow].forEach((sourceBuffer, index) => {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: index * 4,
          size: 4
        });
      });
      ticket.markEncoded({byteLength: 12});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 3);
        const percent = pointCount ? (100 * words[0]) / pointCount : 0;
        selectedReadout.setValue(
          `${formatCount(words[0])} of ${formatCount(pointCount)} (${percent.toFixed(1)}%)`
        );
        outputOverflowReadout.setValue(words[1] ? 'YES' : 'no');
        joinOverflowReadout.setValue(words[2] ? 'YES' : 'no');
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        if (subsetDirty) {
          subsetDirty = false;
          if (subsetKind !== 'click') applyClassSubset(subsetKind);
        }
        distance.write(Float32Array.of(currentDistance));
        compiled.encode(commandEncoder, {parameters: undefined});
        // The recipe has no drawInstanceCount: copy the stored count into the instance-count word
        // (second uint32) of the indirect draw record.
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: outputCount,
          sourceOffset: 0,
          destinationBuffer: drawCommands.buffer,
          destinationOffset: 4,
          size: 4
        });
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL === 0) {
          void readSummary(commandEncoder);
        }
        if (!autoMeasureScheduled && frame.frameIndex >= AUTO_MEASURE_FRAME) {
          autoMeasureScheduled = true;
          void measureSpatialSort();
        }
      },
      onClick({coordinate}) {
        if (!coordinate) return false;
        const [x, y] = projection.project(coordinate[0], coordinate[1]);
        subsetKind = 'click';
        applyClickSubset(x, y);
        if (subsetCount === 0) {
          subsetKind = 'major';
          subsetDirty = true;
        } else {
          context.setStatus(`Buffering around ${subsetCount} segments near the click`);
        }
        context.updateLayers();
        return true;
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const layers: Layer[] = [
          new MapGraphsSegmentLayer({
            id: 'buffer-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            color: [130, 145, 170, 70],
            widthPixels: 1
          }),
          new MapGraphsSegmentLayer({
            id: 'buffer-subset-roads',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: flagsBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [255, 150, 40, 235],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 2.5
          }),
          new MapGraphsPointLayer({
            id: 'buffer-unselected',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            values: outputMask,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [0, 0, 0, 0],
            noDataColor: [150, 160, 180, 130],
            radiusPixels: 2.5
          }),
          new MapGraphsPointLayer({
            id: 'buffer-selected',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: outputIds,
            drawCommands,
            values: colorByDistance ? distances : outputMask,
            valueFormat: colorByDistance ? 'float32' : 'uint32',
            colormap: colorByDistance ? 'viridis' : 'mask',
            valueRange: [0, Math.max(currentDistance, 1)],
            color: [90, 255, 190, 255],
            radiusPixels: 5
          })
        ];
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
