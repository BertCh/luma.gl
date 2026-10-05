// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {getGPUDistanceFieldParameterValues, GPU_DISTANCE_FIELD_NONE, GPUDistanceField, type GPUDistanceFieldMode} from '@luma.gl/experimental/gpu-raster';
import {importGraphBuffer} from '@luma.gl/experimental/UNRESOLVED';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsPointLayer, MapGraphsRasterLayer, type MapGraphsColor} from '../map-graphs-layers';
import type {
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent
} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {
  formatCompiledGraphTiming,
  measureCompiledGraph,
  type CompiledGraphTiming
} from './vector-timing';

/** Cells per side of the square analysis grid. */
const GRID_SIZE = 512;
/** Compile-time seed capacity; the live facility count is a per-frame buffer value. */
const SEED_CAPACITY = 64;
const INITIAL_SEED_COUNT = 24;
/** Slider value that means "no distance limit". */
const MAXIMUM_DISTANCE_SLIDER = 8000;
/** A seed within this many CSS pixels of the pointer is grabbed by a drag. */
const GRAB_RADIUS_PIXELS = 28;
/** Neighbors a seed must differ in color from, see {@link assignZoneColors}. */
const COLOR_NEIGHBOR_COUNT = 8;

type View = 'distance' | 'allocation';
type AlgorithmChoice = GPUDistanceFieldMode | 'auto';

const ZONE_PALETTE: readonly MapGraphsColor[] = [
  [78, 201, 255, 200],
  [255, 148, 72, 200],
  [189, 122, 255, 200],
  [87, 235, 168, 200],
  [255, 105, 168, 200],
  [245, 220, 87, 200],
  [107, 158, 255, 200],
  [255, 92, 92, 200]
];

/**
 * Euclidean distance and nearest-facility allocation from San Francisco bike-parking locations.
 *
 * `GPUDistanceField` is compiled twice (exact and jump-flood) over shared seed, settings and
 * output buffers. Dragging a facility, clicking to move it, the facility count, the distance band
 * and the algorithm are buffer writes; the active graph is re-encoded only when an input changed
 * (the result persists in the output buffers), and every frame while a seed is being dragged.
 */
export const distanceFieldMode: MapGraphsModeDefinition = {
  id: 'distance-field',
  title: 'Distance',
  recipes: ['GPUDistanceField'],
  description:
    'Euclidean distance and nearest-facility zones from bike-parking facilities. Drag a ' +
    'facility, or click to move the selected one: seeds are rewritten every frame. Switch the ' +
    'exact and jump-flood graphs, both compiled once.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const parking = await context.data.getSanFranciscoBikeParking();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(parking.origin);
    const resources = new MapGraphsResources(device, 'distance-field');
    const cellCount = GRID_SIZE * GRID_SIZE;

    // Candidate facilities in farthest-point order, so the first N are always well spread.
    const candidates = chooseSpreadSeeds(parking.positions, SEED_CAPACITY);
    const seedCapacity = candidates.length / 2;
    const seedPositions = Float32Array.from(candidates);
    const seedIds = new Uint32Array(seedCapacity);
    let seedCount = Math.min(INITIAL_SEED_COUNT, seedCapacity);
    const bounds = getSquareBounds(seedPositions);
    const cellSize = (bounds[2] - bounds[0]) / GRID_SIZE;

    let view: View = 'distance';
    let algorithm: AlgorithmChoice = 'auto';
    let maximumDistanceSlider = MAXIMUM_DISTANCE_SLIDER;
    let selectedSeed = 0;
    let draggingSeed = -1;
    let dirty = true;
    let destroyed = false;
    let measuring = false;
    const timings: Record<GPUDistanceFieldMode, CompiledGraphTiming | null> = {
      exact: null,
      'jump-flood': null
    };

    const seedPositionsBuffer = resources.createBuffer('seed-positions', seedPositions);
    const seedIdsBuffer = resources.createBuffer('seed-ids', seedIds);
    const seedCountBuffer = resources.createParameterBuffer(
      'seed-count',
      'uint32',
      1,
      Uint32Array.of(seedCount)
    );
    const settingsBuffer = resources.createParameterBuffer('settings', 'float32', 8);
    const selectedPositionBuffer = resources.createBuffer('selected-position', 8);
    const distances = resources.createBuffer('distances', cellCount * 4);
    const allocation = resources.createBuffer('allocation', cellCount * 4);

    const compileGraph = (mode: GPUDistanceFieldMode): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {id: `distance-field-${mode}`});
      graph.add(
        new GPUDistanceField({
          id: `distance-field-${mode}`,
          width: GRID_SIZE,
          height: GRID_SIZE,
          mode,
          settings: settingsBuffer.importToGraph(graph),
          seedPositions: importGraphBuffer(
            graph,
            'seed-positions',
            seedPositionsBuffer,
            'float32x2',
            seedCapacity
          ),
          seedIds: importGraphBuffer(graph, 'seed-ids', seedIdsBuffer, 'uint32', seedCapacity),
          seedCount: seedCountBuffer.importToGraph(graph),
          output: {
            distances: importGraphBuffer(graph, 'distances', distances, 'float32', cellCount),
            allocation: importGraphBuffer(graph, 'allocation', allocation, 'uint32', cellCount)
          }
        })
      );
      return resources.track(graph.compile());
    };
    const compiledGraphs: Record<GPUDistanceFieldMode, CompiledGPUCommandGraph<void>> = {
      exact: compileGraph('exact'),
      'jump-flood': compileGraph('jump-flood')
    };

    const getMaximumDistance = () =>
      maximumDistanceSlider >= MAXIMUM_DISTANCE_SLIDER ? Infinity : maximumDistanceSlider;
    const getActiveMode = (): GPUDistanceFieldMode =>
      algorithm === 'auto' ? (draggingSeed >= 0 ? 'jump-flood' : 'exact') : algorithm;

    const writeSettings = () => {
      settingsBuffer.write(
        getGPUDistanceFieldParameterValues({
          bounds,
          gridSize: [GRID_SIZE, GRID_SIZE],
          maxDistance: getMaximumDistance()
        })
      );
      dirty = true;
    };
    const writeSeeds = () => {
      seedPositionsBuffer.write(seedPositions);
      seedIdsBuffer.write(seedIds);
      seedCountBuffer.write(Uint32Array.of(seedCount));
      selectedPositionBuffer.write(seedPositions.subarray(selectedSeed * 2, selectedSeed * 2 + 2));
      dirty = true;
    };
    const recolorZones = () => {
      assignZoneColors(seedPositions, seedCount, ZONE_PALETTE.length, seedIds);
    };

    const moveSeed = (seed: number, coordinate: readonly [number, number]) => {
      const [x, y] = projection.project(coordinate[0], coordinate[1]);
      seedPositions[seed * 2] = x;
      seedPositions[seed * 2 + 1] = y;
      writeSeeds();
    };
    const findSeedNear = (pixel: readonly [number, number]): number => {
      const viewport = context.getViewport();
      if (!viewport) return -1;
      let best = -1;
      let bestDistance = GRAB_RADIUS_PIXELS;
      for (let seed = 0; seed < seedCount; seed++) {
        const [longitude, latitude] = projection.unproject(
          seedPositions[seed * 2],
          seedPositions[seed * 2 + 1]
        );
        const [pixelX, pixelY] = viewport.project([longitude, latitude]);
        const distance = Math.hypot(pixelX - pixel[0], pixelY - pixel[1]);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = seed;
        }
      }
      return best;
    };

    context.controls.addSelect<View>({
      label: 'Show',
      options: [
        {value: 'distance', label: 'Distance to nearest facility'},
        {value: 'allocation', label: 'Allocation (nearest-facility zones)'}
      ],
      value: view,
      onChange: value => {
        view = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Maximum distance band (per-frame setting)',
      min: 250,
      max: MAXIMUM_DISTANCE_SLIDER,
      step: 250,
      value: maximumDistanceSlider,
      format: value => (value >= MAXIMUM_DISTANCE_SLIDER ? 'unlimited' : `${value} m`),
      onChange: value => {
        maximumDistanceSlider = value;
        writeSettings();
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Facilities (per-frame seed count)',
      min: 2,
      max: seedCapacity,
      step: 1,
      value: seedCount,
      onChange: value => {
        seedCount = value;
        selectedSeed = Math.min(selectedSeed, seedCount - 1);
        recolorZones();
        writeSeeds();
        showFacilityCount();
        context.updateLayers();
      }
    });
    context.controls.addSelect<AlgorithmChoice>({
      label: 'Algorithm (both graphs compiled once)',
      options: [
        {value: 'auto', label: 'Auto: jump-flood while dragging, exact at rest'},
        {value: 'exact', label: 'Exact (Felzenszwalb-Huttenlocher)'},
        {value: 'jump-flood', label: 'Jump flood (approximate)'}
      ],
      value: algorithm,
      onChange: value => {
        algorithm = value;
        dirty = true;
      }
    });
    context.controls.addLegend({
      title: 'Distance (viridis, near to far; transparent beyond the band)',
      gradient: {
        colors: [
          [68, 1, 84],
          [59, 82, 139],
          [33, 145, 140],
          [94, 201, 98],
          [253, 231, 37]
        ],
        minimumLabel: '0 m',
        maximumLabel: 'band limit'
      }
    });
    context.controls.addNote(
      'Drag a white facility to move it live, or click the map to move the selected (larger) ' +
        'facility there. Pan from empty map.'
    );
    const facilityReadout = context.controls.addReadout('Facilities');
    const showFacilityCount = () =>
      facilityReadout.setValue(`${formatCount(seedCount)} of ${seedCapacity}`);
    showFacilityCount();
    context.controls.addReadout(
      'Grid',
      `${GRID_SIZE} × ${GRID_SIZE} cells of ${cellSize.toFixed(1)} m`
    );
    const activeReadout = context.controls.addReadout('Active graph');
    const frameReadout = context.controls.addReadout('Last encode (CPU)');
    const exactReadout = context.controls.addReadout('Exact');
    const jumpFloodReadout = context.controls.addReadout('Jump flood');
    context.controls.addButton({label: 'Re-measure both graphs', onClick: () => void measure()});
    context.controls.addReadout('Data', parking.attribution);

    async function measure(): Promise<void> {
      if (measuring || destroyed) return;
      measuring = true;
      try {
        for (const mode of ['exact', 'jump-flood'] as const) {
          // Runs outside Deck's frame; the output is rewritten by the next frame encode.
          timings[mode] = await measureCompiledGraph(device, compiledGraphs[mode], {
            parameters: undefined,
            completionBuffer: distances,
            signal: context.signal
          });
          if (destroyed) return;
          (mode === 'exact' ? exactReadout : jumpFloodReadout).setValue(
            formatCompiledGraphTiming(timings[mode])
          );
        }
      } catch {
        // Aborted or destroyed while measuring.
      } finally {
        measuring = false;
        dirty = true;
      }
    }

    recolorZones();
    writeSettings();
    writeSeeds();
    void measure();

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => [compiledGraphs.exact, compiledGraphs['jump-flood']],
      encode(commandEncoder) {
        if (!dirty && draggingSeed < 0) return;
        const mode = getActiveMode();
        const encoding = compiledGraphs[mode].encode(commandEncoder, {parameters: undefined});
        activeReadout.setValue(mode === 'exact' ? 'exact' : 'jump-flood');
        frameReadout.setValue(`${encoding.stats.cpuEncodeTimeMilliseconds.toFixed(2)} ms`);
        dirty = false;
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [
          parking.origin[0],
          parking.origin[1],
          0
        ];
        const maximumDistance = getMaximumDistance();
        const range = Math.min(maximumDistance, (bounds[2] - bounds[0]) * 0.5);
        const layers: Layer[] = [
          new MapGraphsRasterLayer({
            id: `distance-field-${view}`,
            coordinateOrigin,
            gridSize: [GRID_SIZE, GRID_SIZE],
            bounds,
            values: view === 'distance' ? distances : allocation,
            valueFormat: view === 'distance' ? 'float32' : 'uint32',
            colormap: view === 'distance' ? 'viridis' : 'category',
            valueRange: [0, range],
            palette: ZONE_PALETTE,
            noDataValue: GPU_DISTANCE_FIELD_NONE,
            noDataColor: [0, 0, 0, 0],
            color: [255, 255, 255, 190]
          }),
          new MapGraphsPointLayer({
            id: 'distance-field-seeds',
            coordinateOrigin,
            positions: seedPositionsBuffer,
            instanceCount: seedCount,
            radiusPixels: 5,
            color: [255, 255, 255, 255]
          }),
          new MapGraphsPointLayer({
            id: 'distance-field-selected',
            coordinateOrigin,
            positions: selectedPositionBuffer,
            instanceCount: 1,
            radiusPixels: 9,
            color: [255, 80, 60, 255]
          })
        ];
        return layers;
      },
      onClick(event: MapGraphsPointerEvent) {
        if (!event.coordinate) return false;
        const grabbed = findSeedNear(event.pixel);
        if (grabbed >= 0) {
          selectedSeed = grabbed;
          writeSeeds();
        } else {
          moveSeed(selectedSeed, event.coordinate);
          recolorZones();
          writeSeeds();
        }
        return true;
      },
      onDragStart(event) {
        const grabbed = findSeedNear(event.pixel);
        if (grabbed < 0) return false;
        draggingSeed = grabbed;
        selectedSeed = grabbed;
        writeSeeds();
        return true;
      },
      onDrag(event) {
        if (draggingSeed >= 0 && event.coordinate) moveSeed(draggingSeed, event.coordinate);
      },
      onDragEnd() {
        draggingSeed = -1;
        recolorZones();
        writeSeeds();
      },
      destroy() {
        destroyed = true;
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Returns a square `[minX, minY, maxX, maxY]` centered on the seeds with 12% padding. */
function getSquareBounds(positions: Float32Array): [number, number, number, number] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minX = Math.min(minX, positions[index]);
    maxX = Math.max(maxX, positions[index]);
    minY = Math.min(minY, positions[index + 1]);
    maxY = Math.max(maxY, positions[index + 1]);
  }
  const half = Math.max(maxX - minX, maxY - minY) * 0.56;
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  return [centerX - half, centerY - half, centerX + half, centerY + half];
}

/**
 * Picks up to `count` points in farthest-point order from the central 96% of the data, so the
 * first N are always spread out and outliers do not stretch the grid.
 */
function chooseSpreadSeeds(positions: Float32Array, count: number): number[] {
  const pointCount = positions.length / 2;
  const xs = Float32Array.from({length: pointCount}, (_, index) => positions[index * 2]).sort();
  const ys = Float32Array.from({length: pointCount}, (_, index) => positions[index * 2 + 1]).sort();
  const low = Math.floor(pointCount * 0.02);
  const high = Math.max(low, Math.ceil(pointCount * 0.98) - 1);
  const eligible: number[] = [];
  for (let point = 0; point < pointCount; point++) {
    const x = positions[point * 2];
    const y = positions[point * 2 + 1];
    if (x >= xs[low] && x <= xs[high] && y >= ys[low] && y <= ys[high]) eligible.push(point);
  }
  if (eligible.length === 0) return [];
  const nearest = new Float64Array(eligible.length).fill(Infinity);
  const chosen: number[] = [];
  let next = eligible[0];
  while (chosen.length < Math.min(count, eligible.length)) {
    const x = positions[next * 2];
    const y = positions[next * 2 + 1];
    chosen.push(x, y);
    let farthest = -1;
    let farthestDistance = -1;
    eligible.forEach((point, slot) => {
      const distance = (positions[point * 2] - x) ** 2 + (positions[point * 2 + 1] - y) ** 2;
      nearest[slot] = Math.min(nearest[slot], distance);
      if (nearest[slot] > farthestDistance) {
        farthestDistance = nearest[slot];
        farthest = point;
      }
    });
    next = farthest;
  }
  return chosen;
}

/**
 * Greedy graph coloring of the facilities: each gets the lowest color not used by a nearby
 * (already colored) facility, so neighboring zones rarely share a category color. The color is
 * written as the seed ID, so same-colored zones that touch would merge; this keeps that rare.
 */
function assignZoneColors(
  positions: Float32Array,
  seedCount: number,
  colorCount: number,
  target: Uint32Array
): void {
  target.fill(GPU_DISTANCE_FIELD_NONE);
  for (let seed = 0; seed < seedCount; seed++) {
    const neighbors = Array.from({length: seed}, (_, other) => other)
      .map(other => ({
        other,
        distance:
          (positions[other * 2] - positions[seed * 2]) ** 2 +
          (positions[other * 2 + 1] - positions[seed * 2 + 1]) ** 2
      }))
      .sort((left, right) => left.distance - right.distance)
      .slice(0, COLOR_NEIGHBOR_COUNT);
    // Also avoid later seeds' colors being unknown: nearest earlier seeds are a good proxy.
    const used = new Set(neighbors.map(({other}) => target[other]));
    let color = 0;
    while (used.has(color) && color < colorCount - 1) color++;
    target[seed] = color;
  }
}
