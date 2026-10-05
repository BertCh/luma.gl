// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Dasymetric dot density over the San Francisco ZIP codes. `GPUDotDensity` places
 * `value * dotsPerUnit` dots per (ZIP, category) with counter-based random numbers, so a dot is a
 * pure function of (seed, ZIP, category, rank): changing the dot value, or tying it to the map
 * zoom, only appends or removes dots at the end of each slot and never moves one. Every control is
 * a buffer write (the parameter words, the per-slot values, the mask weights, the random-point
 * counts), so the graphs compile once.
 *
 * - Values: bike-parking spaces per ZIP split into three rack-size classes (counted on the CPU
 *   once at load), or a synthetic population (area times a hash-driven density and shares).
 * - Mask: a water/steepness weight derived from the terrain raster, an illustrative dasymetric
 *   proxy rather than a real built-up layer. "Off" rewrites the weights to 1 (no recompile).
 * - Random points: a second graph (`GPURandomPointsInPolygon`) places N uniform points per ZIP.
 * - "Check dot stability" runs the graph at two dot values outside the frame and reads positions
 *   back to prove that the coarser dots are a bitwise prefix of the finer ones, per slot.
 */

import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUDotDensityParameterValues,
  GPU_DOT_DENSITY_PARAMETER_LENGTH,
  GPUDotDensity,
  GPURandomPointsInPolygon
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

const CATEGORY_COUNT = 3;
/** Dot capacity of the dot-density output. A total above it is clamped and flagged. */
const DOT_CAPACITY = 1 << 19;
/** Random-point capacity: the slider maximum times a generous feature count. */
const MAXIMUM_POINTS_PER_FEATURE = 2000;
const REFERENCE_ZOOM = 12;
const READBACK_INTERVAL = 15;
const SEED = 20260;
const CATEGORY_COLORS = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255]
] as const;

type ValueSource = 'bike' | 'synthetic';
type CategoryNames = readonly [string, string, string];

const CATEGORY_NAMES: Record<ValueSource, CategoryNames> = {
  bike: ['Racks of 1-2 spaces', 'Racks of 3-9 spaces', 'Racks of 10+ spaces'],
  synthetic: ['Group A', 'Group B', 'Group C']
};
const SOURCE_UNIT: Record<ValueSource, string> = {bike: 'spaces', synthetic: 'people'};
/** Default log10 of units per dot at the reference zoom. */
const DEFAULT_LOG_UNITS_PER_DOT: Record<ValueSource, number> = {bike: 0.5, synthetic: 2.2};

export const dotDensityMode: SpatialAnalysisModeDefinition = {
  id: 'dot-density',
  title: 'Dot density',
  contributors: ['GPUDotDensity', 'GPURandomPointsInPolygon'],
  description:
    'One dot per N units inside each San Francisco ZIP, colored by category, generated on the ' +
    'GPU. Zoom in or change the dot value: dots are added or removed but never move.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 12},

  async create(context) {
    const [parking, zips, terrain] = await Promise.all([
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoZipCodes(),
      context.data.getSanFranciscoTerrain()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'dot-density');
    const featureCount = zips.featureOffsets.length - 1;
    const slotCount = featureCount * CATEGORY_COUNT;
    const segmentCount = zips.outlineSegments.length / 4;

    const valuesBySource: Record<ValueSource, Float32Array> = {
      bike: getBikeParkingValues(parking.positions, parking.spaces, zips),
      synthetic: getSyntheticPopulationValues(zips)
    };
    const maskOn = getTerrainMask(terrain);
    const maskOff = new Float32Array(maskOn.length).fill(1);

    // Inputs. Values, mask weights, parameters and per-feature counts are rewritten at runtime.
    const polygonPositions = resources.createBuffer('polygon-positions', zips.polygonPositions);
    const featureOffsets = resources.createBuffer('feature-offsets', zips.featureOffsets);
    const polygonOffsets = resources.createBuffer('polygon-offsets', zips.polygonOffsets);
    const ringOffsets = resources.createBuffer('ring-offsets', zips.ringOffsets);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);
    const valuesBuffer = resources.createBuffer('values', valuesBySource.bike);
    const maskBuffer = resources.createBuffer('mask-weights', maskOff);
    const parameters = resources.createParameterBuffer(
      'dot-parameters',
      'uint32',
      GPU_DOT_DENSITY_PARAMETER_LENGTH
    );
    const randomParameters = resources.createParameterBuffer(
      'random-parameters',
      'uint32',
      GPU_DOT_DENSITY_PARAMETER_LENGTH
    );
    const randomCountsBuffer = resources.createBuffer('random-counts', featureCount * 4);

    // Dot-density outputs.
    const dotPositions = resources.createBuffer('dot-positions', DOT_CAPACITY * 8);
    const dotIds = resources.createBuffer('dot-ids', DOT_CAPACITY * 4);
    const dotCategories = resources.createBuffer('dot-categories', DOT_CAPACITY * 4);
    const dotOverflow = resources.createBuffer('dot-overflow', 4);
    const dotTotal = resources.createBuffer('dot-total', 4);
    const dotFailed = resources.createBuffer('dot-failed', 4);
    const slotCounts = resources.createBuffer('slot-counts', slotCount * 4);
    const slotOffsets = resources.createBuffer('slot-offsets', slotCount * 4);
    const dotDrawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'dot-density-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    // Random-point outputs.
    const randomCapacity = featureCount * MAXIMUM_POINTS_PER_FEATURE;
    const randomPositions = resources.createBuffer('random-positions', randomCapacity * 8);
    const randomIds = resources.createBuffer('random-ids', randomCapacity * 4);
    const randomOverflow = resources.createBuffer('random-overflow', 4);
    const randomFailed = resources.createBuffer('random-failed', 4);
    const randomDrawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'random-points-draw',
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    const importPolygons = <Parameters>(graph: GPUCommandGraph<Parameters>) => ({
      polygonPositions: importGraphBuffer(
        graph,
        'polygon-positions',
        polygonPositions,
        'float32x2',
        zips.polygonPositions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'feature-offsets',
        featureOffsets,
        'uint32',
        zips.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygon-offsets',
        polygonOffsets,
        'uint32',
        zips.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        ringOffsets,
        'uint32',
        zips.ringOffsets.length
      )
    });
    const importMask = <Parameters>(graph: GPUCommandGraph<Parameters>) => ({
      weights: importGraphBuffer(
        graph,
        'mask-weights',
        maskBuffer,
        'float32',
        terrain.width * terrain.height
      ),
      width: terrain.width,
      height: terrain.height
    });

    const dotGraph = new GPUCommandGraph<void>(device, {id: 'dot-density'});
    dotGraph.add(
      new GPUDotDensity({
        id: 'dots',
        ...importPolygons(dotGraph),
        values: importGraphBuffer(dotGraph, 'values', valuesBuffer, 'float32', slotCount),
        categoryCount: CATEGORY_COUNT,
        parameters: parameters.importToGraph(dotGraph),
        mask: importMask(dotGraph),
        output: {
          positions: importGraphBuffer(
            dotGraph,
            'dot-positions',
            dotPositions,
            'float32x2',
            DOT_CAPACITY
          ),
          dots: {
            ids: importGraphBuffer(dotGraph, 'dot-ids', dotIds, 'uint32', DOT_CAPACITY),
            count: dotGraph.importGPUData('dot-count', dotDrawCommands.getInstanceCountData(0)),
            overflow: importGraphBuffer(dotGraph, 'dot-overflow', dotOverflow, 'uint32', 1),
            totalCount: importGraphBuffer(dotGraph, 'dot-total', dotTotal, 'uint32', 1)
          },
          categories: importGraphBuffer(
            dotGraph,
            'dot-categories',
            dotCategories,
            'uint32',
            DOT_CAPACITY
          ),
          failedCount: importGraphBuffer(dotGraph, 'dot-failed', dotFailed, 'uint32', 1),
          slotCounts: importGraphBuffer(dotGraph, 'slot-counts', slotCounts, 'uint32', slotCount),
          slotOffsets: importGraphBuffer(dotGraph, 'slot-offsets', slotOffsets, 'uint32', slotCount)
        }
      })
    );
    const compiledDots = resources.track(dotGraph.compile());

    const randomGraph = new GPUCommandGraph<void>(device, {id: 'random-points'});
    randomGraph.add(
      new GPURandomPointsInPolygon({
        id: 'random-points',
        ...importPolygons(randomGraph),
        counts: importGraphBuffer(
          randomGraph,
          'random-counts',
          randomCountsBuffer,
          'uint32',
          featureCount
        ),
        parameters: randomParameters.importToGraph(randomGraph),
        mask: importMask(randomGraph),
        output: {
          positions: importGraphBuffer(
            randomGraph,
            'random-positions',
            randomPositions,
            'float32x2',
            randomCapacity
          ),
          dots: {
            ids: importGraphBuffer(randomGraph, 'random-ids', randomIds, 'uint32', randomCapacity),
            count: randomGraph.importGPUData(
              'random-count',
              randomDrawCommands.getInstanceCountData(0)
            ),
            overflow: importGraphBuffer(randomGraph, 'random-overflow', randomOverflow, 'uint32', 1)
          },
          failedCount: importGraphBuffer(randomGraph, 'random-failed', randomFailed, 'uint32', 1)
        }
      })
    );
    const compiledRandom = resources.track(randomGraph.compile());

    // Summary: dot count, overflow, total, failed, random count, random failed, slot counts.
    const summaryWords = 6 + slotCount;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'dot-density-summary', byteLength: summaryWords * 4})
    );

    // Controls. State is plain variables read by `encode`.
    let source: ValueSource = 'bike';
    let logUnitsPerDot = DEFAULT_LOG_UNITS_PER_DOT[source];
    let zoomCoupling = 1;
    let useMask = false;
    let showRandomPoints = false;
    let randomPointsPerFeature = 200;
    let dotRadius = 1.6;
    let randomSeed = SEED;
    let checking = false;
    let destroyed = false;
    let readbackPending = false;

    context.controls.addSelect<ValueSource>({
      label: 'Values (buffer rewrite)',
      options: [
        {value: 'bike', label: 'Bike-parking spaces by rack size'},
        {value: 'synthetic', label: 'Synthetic population by area'}
      ],
      value: source,
      onChange: value => {
        source = value;
        valuesBuffer.write(valuesBySource[source]);
        logUnitsPerDot = DEFAULT_LOG_UNITS_PER_DOT[source];
        unitsControl.setValue(logUnitsPerDot);
        updateLegendNote();
      }
    });
    const unitsControl = context.controls.addSlider({
      label: 'Units per dot at zoom 12 (per-frame parameter)',
      min: 0,
      max: 4,
      step: 0.1,
      value: logUnitsPerDot,
      format: value => `${formatUnits(10 ** value)} ${SOURCE_UNIT[source]}`,
      onChange: value => {
        logUnitsPerDot = value;
      }
    });
    context.controls.addSlider({
      label: 'Zoom coupling (dots per unit double every 1/c zoom)',
      min: 0,
      max: 2,
      step: 0.25,
      value: zoomCoupling,
      format: value =>
        value === 0 ? 'off (fixed dot value)' : `x${(2 ** value).toFixed(2)} per zoom`,
      onChange: value => {
        zoomCoupling = value;
      }
    });
    context.controls.addToggle({
      label: 'Dasymetric mask (terrain proxy: water 0, hills thinner)',
      value: useMask,
      onChange: value => {
        useMask = value;
        maskBuffer.write(useMask ? maskOn : maskOff);
      }
    });
    context.controls.addSlider({
      label: 'Dot radius (pixels)',
      min: 0.8,
      max: 4,
      step: 0.2,
      value: dotRadius,
      onChange: value => {
        dotRadius = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'New seed (all dots move)',
      onClick: () => {
        randomSeed = (randomSeed * 1664525 + 1013904223) >>> 0;
      }
    });
    context.controls.addToggle({
      label: 'Show random points in polygon instead (second graph)',
      value: showRandomPoints,
      onChange: value => {
        showRandomPoints = value;
        context.updateLayers();
      }
    });
    context.controls.addSlider({
      label: 'Random points per ZIP (per-frame counts)',
      min: 10,
      max: MAXIMUM_POINTS_PER_FEATURE,
      step: 10,
      value: randomPointsPerFeature,
      onChange: value => {
        randomPointsPerFeature = value;
        randomCountsBuffer.write(new Uint32Array(featureCount).fill(randomPointsPerFeature));
      }
    });
    randomCountsBuffer.write(new Uint32Array(featureCount).fill(randomPointsPerFeature));
    context.controls.addButton({
      label: 'Check dot stability (GPU readback)',
      onClick: () => void checkStability()
    });

    context.controls.addButton({
      label: 'Measure graphs (outside frame)',
      onClick: () => void measureGraphs()
    });

    context.controls.addLegend({
      title: 'Category (dot color)',
      entries: CATEGORY_COLORS.map((color, index) => ({
        color,
        label: `Category ${index + 1}`
      }))
    });
    const legendNote = context.controls.addNote('');
    const updateLegendNote = () => {
      legendNote.setValue(`Cyan / orange / violet: ${CATEGORY_NAMES[source].join(' / ')}`);
    };
    updateLegendNote();
    context.controls.addReadout('ZIP codes', formatCount(featureCount));
    const dotsReadout = context.controls.addReadout('Dots drawn');
    const dotValueReadout = context.controls.addReadout('Effective dot value');
    const categoryReadout = context.controls.addReadout('Dots per category');
    const overflowReadout = context.controls.addReadout('Overflow / failed dots');
    const randomReadout = context.controls.addReadout('Random points');
    const stabilityReadout = context.controls.addReadout('Dot stability');
    const dotTimingReadout = context.controls.addReadout('GPUDotDensity graph');
    const randomTimingReadout = context.controls.addReadout('GPURandomPointsInPolygon graph');
    context.controls.addReadout(
      'Mask',
      `${terrain.width} x ${terrain.height} f32 weights (${((maskOn.length * 4) / 1e6).toFixed(1)} MB)`
    );
    context.controls.addReadout(
      'Data',
      `${parking.attribution}; ${zips.attribution}; ${terrain.attribution}`
    );
    context.controls.addNote(
      'Dot j of a (ZIP, category) slot is rejection-sampled from Philox(seed, slot, j, attempt), ' +
        'independent of the dot value, so zooming appends dots and never moves one. Dots are ' +
        'drawn in slot order, so later categories overlap earlier ones.'
    );

    /** dots per unit at the current zoom. */
    const getDotsPerUnit = (zoom: number): number =>
      2 ** (zoomCoupling * (zoom - REFERENCE_ZOOM)) / 10 ** logUnitsPerDot;

    const maskExtent = [
      terrain.bounds[0],
      terrain.bounds[1],
      terrain.cellSize[0],
      terrain.cellSize[1]
    ] as const;
    const writeParameters = (dotsPerUnit: number) => {
      parameters.write(
        getGPUDotDensityParameterValues({seed: randomSeed, dotsPerUnit, maskExtent})
      );
    };

    /** Encodes the dot graph at `dotsPerUnit` outside the frame and returns CPU copies. */
    const runDotsOutsideFrame = async (dotsPerUnit: number) => {
      parameters.write(
        getGPUDotDensityParameterValues({seed: randomSeed, dotsPerUnit, maskExtent})
      );
      const encoder = device.createCommandEncoder();
      compiledDots.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const [positions, counts, offsets] = await Promise.all([
        dotPositions.readAsync(),
        slotCounts.readAsync(),
        slotOffsets.readAsync()
      ]);
      return {
        positions: new Float32Array(
          positions.buffer,
          positions.byteOffset,
          positions.byteLength / 4
        ),
        counts: new Uint32Array(counts.buffer, counts.byteOffset, counts.byteLength / 4),
        offsets: new Uint32Array(offsets.buffer, offsets.byteOffset, offsets.byteLength / 4)
      };
    };

    /** Runs the graph at a coarse and a fine dot value and compares dots slot by slot. */
    const checkStability = async () => {
      if (checking || destroyed) return;
      checking = true;
      stabilityReadout.setValue('checking...');
      try {
        const base = getDotsPerUnit(REFERENCE_ZOOM);
        const coarse = await runDotsOutsideFrame(base);
        const fine = await runDotsOutsideFrame(base * 2.5);
        let compared = 0;
        let moved = 0;
        let removed = 0;
        for (let slot = 0; slot < slotCount; slot++) {
          if (fine.counts[slot] < coarse.counts[slot])
            removed += coarse.counts[slot] - fine.counts[slot];
          const shared = Math.min(coarse.counts[slot], fine.counts[slot]);
          for (let rank = 0; rank < shared; rank++) {
            const a = (coarse.offsets[slot] + rank) * 2;
            const b = (fine.offsets[slot] + rank) * 2;
            compared++;
            const same =
              Object.is(coarse.positions[a], fine.positions[b]) &&
              Object.is(coarse.positions[a + 1], fine.positions[b + 1]);
            if (!same) moved++;
          }
        }
        const coarseTotal = coarse.offsets[slotCount - 1] + coarse.counts[slotCount - 1];
        const fineTotal = fine.offsets[slotCount - 1] + fine.counts[slotCount - 1];
        stabilityReadout.setValue(
          `${formatCount(compared)} dots compared (${formatCount(coarseTotal)} -> ` +
            `${formatCount(fineTotal)}): ${moved} moved, ${removed} removed`
        );
      } catch (error) {
        stabilityReadout.setValue(`failed: ${(error as Error).message}`);
      } finally {
        checking = false;
      }
    };

    /** Times both graphs between frames: GPU timestamps when available, else wall clock. */
    const measureGraphs = async () => {
      if (checking || destroyed) return;
      checking = true;
      dotTimingReadout.setValue('measuring...');
      randomTimingReadout.setValue('measuring...');
      try {
        writeParameters(getDotsPerUnit(REFERENCE_ZOOM));
        randomParameters.write(getGPUDotDensityParameterValues({seed: randomSeed + 1, maskExtent}));
        const options = {
          parameters: undefined,
          completionBuffer: dotOverflow,
          signal: context.signal
        };
        const dotTiming = await measureCompiledGraph(device, compiledDots, options);
        const randomTiming = await measureCompiledGraph(device, compiledRandom, options);
        dotTimingReadout.setValue(
          `${compiledDots.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(dotTiming)}`
        );
        randomTimingReadout.setValue(
          `${compiledRandom.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(randomTiming)}`
        );
      } catch (error) {
        dotTimingReadout.setValue(`failed: ${(error as Error).message}`);
      } finally {
        checking = false;
      }
    };

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const sources = [
        [dotDrawCommands.buffer, 4, 4],
        [dotOverflow, 0, 4],
        [dotTotal, 0, 4],
        [dotFailed, 0, 4],
        [randomDrawCommands.buffer, 4, 4],
        [randomFailed, 0, 4],
        [slotCounts, 0, slotCount * 4]
      ] as const;
      let offset = 0;
      for (const [sourceBuffer, sourceOffset, size] of sources) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          sourceOffset,
          destinationBuffer: ticket.buffer,
          destinationOffset: offset,
          size
        });
        offset += size;
      }
      ticket.markEncoded({byteOffset: 0, byteLength: summaryWords * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, summaryWords);
        const totals = [0, 0, 0];
        for (let slot = 0; slot < slotCount; slot++)
          totals[slot % CATEGORY_COUNT] += words[6 + slot];
        dotsReadout.setValue(`${formatCount(words[0])} of ${formatCount(DOT_CAPACITY)}`);
        categoryReadout.setValue(totals.map(formatCount).join(' / '));
        overflowReadout.setValue(
          `${words[1] ? `YES (unclamped ${formatCount(words[2])})` : 'no'} / ${formatCount(words[3])}`
        );
        randomReadout.setValue(
          showRandomPoints
            ? `${formatCount(words[4])} placed, ${formatCount(words[5])} failed`
            : 'off'
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiledDots, compiledRandom] as CompiledGPUCommandGraph<never>[],
      encode(commandEncoder, frame) {
        // Checking owns the output buffers and the parameter words until it finishes.
        if (checking) return;
        const dotsPerUnit = getDotsPerUnit(frame.viewport.zoom);
        writeParameters(dotsPerUnit);
        dotValueReadout.setValue(
          `1 dot = ${formatUnits(1 / dotsPerUnit)} ${SOURCE_UNIT[source]} (zoom ${frame.viewport.zoom.toFixed(2)})`
        );
        compiledDots.encode(commandEncoder, {parameters: undefined});
        if (showRandomPoints) {
          randomParameters.write(
            getGPUDotDensityParameterValues({
              seed: randomSeed + 1,
              maskExtent
            })
          );
          compiledRandom.encode(commandEncoder, {parameters: undefined});
        }
        if (!readbackPending && frame.frameIndex % READBACK_INTERVAL === 1) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisSegmentLayer({
            id: 'dot-density-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: segmentCount,
            widthPixels: 1.2,
            color: [255, 255, 255, 150]
          }),
          ...(showRandomPoints
            ? []
            : [
                new SpatialAnalysisPointLayer({
                  id: 'dot-density-dots',
                  coordinateOrigin,
                  positions: dotPositions,
                  drawCommands: dotDrawCommands,
                  values: dotCategories,
                  valueFormat: 'uint32',
                  colormap: 'category',
                  palette: CATEGORY_COLORS,
                  radiusPixels: dotRadius
                })
              ])
        ];
        if (showRandomPoints) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'dot-density-random-points',
              coordinateOrigin,
              positions: randomPositions,
              drawCommands: randomDrawCommands,
              radiusPixels: dotRadius,
              color: [255, 224, 102, 235]
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

function formatUnits(value: number): string {
  return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

/** Even-odd containment of one point over all rings of one feature. */
function isInsideFeature(
  x: number,
  y: number,
  polygons: {polygonPositions: Float32Array; ringOffsets: Uint32Array; polygonOffsets: Uint32Array},
  featureRingStart: number,
  featureRingEnd: number
): boolean {
  let inside = false;
  for (let ring = featureRingStart; ring < featureRingEnd; ring++) {
    const start = polygons.ringOffsets[ring];
    const end = polygons.ringOffsets[ring + 1];
    for (let a = start, b = end - 1; a < end; b = a++) {
      const ax = polygons.polygonPositions[a * 2];
      const ay = polygons.polygonPositions[a * 2 + 1];
      const bx = polygons.polygonPositions[b * 2];
      const by = polygons.polygonPositions[b * 2 + 1];
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
    }
  }
  return inside;
}

/** Sums bike-parking spaces per ZIP into three rack-size classes (feature-major rows). */
function getBikeParkingValues(
  positions: Float32Array,
  spaces: Float32Array,
  zips: {
    polygonPositions: Float32Array;
    featureOffsets: Uint32Array;
    polygonOffsets: Uint32Array;
    ringOffsets: Uint32Array;
  }
): Float32Array {
  const featureCount = zips.featureOffsets.length - 1;
  const values = new Float32Array(featureCount * CATEGORY_COUNT);
  for (let point = 0; point < spaces.length; point++) {
    const category = spaces[point] <= 2 ? 0 : spaces[point] <= 9 ? 1 : 2;
    for (let feature = 0; feature < featureCount; feature++) {
      const ringStart = zips.polygonOffsets[zips.featureOffsets[feature]];
      const ringEnd = zips.polygonOffsets[zips.featureOffsets[feature + 1]];
      if (
        isInsideFeature(positions[point * 2], positions[point * 2 + 1], zips, ringStart, ringEnd)
      ) {
        values[feature * CATEGORY_COUNT + category] += spaces[point];
        break;
      }
    }
  }
  return values;
}

/** People per category: area times a density that varies by ZIP, split by hash-driven shares. */
function getSyntheticPopulationValues(zips: {
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
}): Float32Array {
  const featureCount = zips.featureOffsets.length - 1;
  const values = new Float32Array(featureCount * CATEGORY_COUNT);
  for (let feature = 0; feature < featureCount; feature++) {
    const ringStart = zips.polygonOffsets[zips.featureOffsets[feature]];
    const ringEnd = zips.polygonOffsets[zips.featureOffsets[feature + 1]];
    let area = 0;
    for (let ring = ringStart; ring < ringEnd; ring++) {
      const start = zips.ringOffsets[ring];
      const end = zips.ringOffsets[ring + 1];
      let signed = 0;
      for (let a = start, b = end - 1; a < end; b = a++) {
        signed +=
          zips.polygonPositions[b * 2] * zips.polygonPositions[a * 2 + 1] -
          zips.polygonPositions[a * 2] * zips.polygonPositions[b * 2 + 1];
      }
      area += ring === ringStart ? Math.abs(signed) / 2 : -Math.abs(signed) / 2;
    }
    const hash = (feature * 2654435761) >>> 0;
    const density = 4000 + (hash % 12000);
    const shares = [1 + ((hash >>> 8) % 7), 1 + ((hash >>> 12) % 7), 1 + ((hash >>> 16) % 7)];
    const shareTotal = shares[0] + shares[1] + shares[2];
    for (let category = 0; category < CATEGORY_COUNT; category++) {
      values[feature * CATEGORY_COUNT + category] =
        ((Math.max(area, 0) / 1e6) * density * shares[category]) / shareTotal;
    }
  }
  return values;
}

/** Weights in [0, 1], row 0 at the south edge: water 0, thinner on high ground. */
function getTerrainMask(terrain: {
  width: number;
  height: number;
  elevation: Float32Array;
}): Float32Array {
  const {width, height, elevation} = terrain;
  const weights = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    const sourceRow = height - 1 - row;
    for (let column = 0; column < width; column++) {
      const meters = elevation[sourceRow * width + column];
      weights[row * width + column] =
        meters > 0.5 ? Math.min(1, Math.max(0.25, 1 - meters / 320)) : 0;
    }
  }
  return weights;
}
