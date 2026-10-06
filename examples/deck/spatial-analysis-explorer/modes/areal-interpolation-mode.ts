// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Areal interpolation: San Francisco bike-parking spaces are summed per ZIP code and transferred
 * to a hexagon or square grid with the same GPU pipeline, for the two kinds of variable.
 *
 * 1. `GPUPolygonRasterization` scan-converts the ZIP polygons (source system) and the grid cells
 *    (target system) onto one shared raster. `GPUGridGenerator` makes the grid vertices, and the
 *    cell size is a per-frame buffer write.
 * 2. `GPUArealInterpolation` turns the two zone rasters into area-share weights between targets
 *    and sources (extensive and intensive normalizations from one run).
 * 3. `GPUSpatialLag` applies those cross weights (`sourceCount` != row count): extensive counts are
 *    split by area, intensive densities are area-weighted means.
 * 4. `GPUPycnophylactic` builds Tobler's smooth, volume-preserving surface from the ZIP totals. Its
 *    iteration count is compile-time, so a ladder of graphs is compiled up front and the slider
 *    picks one; a readback proves that every ZIP total is preserved.
 *
 * The graphs run only when an input changed. Small readbacks feed the conservation readouts.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
  GPUPolygonRasterization,
  getGPUPolygonRasterizationExtentValues
} from '@luma.gl/experimental/gpu-raster';
import {
  GPU_GRID_GENERATOR_PARAMETER_LENGTH,
  GPUArealInterpolation,
  GPUGridGenerator,
  GPUPycnophylactic,
  GPUSpatialLag,
  getGPUGridGeneratorParameterValues,
  getGPUGridVerticesPerCell,
  type GPUGridType
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {createZipLocator, getPolygonBounds} from './areal-interpolation-layers';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

/** Shared raster of both zone systems. Compile-time. */
const RASTER_WIDTH = 320;
const RASTER_HEIGHT = 288;
/** Maximum (edge, row) crossings per rasterization. Compile-time; overflow flags report. */
const CROSSING_CAPACITY = 1 << 18;
/** Target grid lattice. Compile-time; the cell size is a per-frame parameter. */
const GRID_COLUMNS = 48;
const GRID_ROWS = 48;
/** Slot capacity for (target, source) pairs. */
const PAIR_CAPACITY = 1 << 14;
/** Pycnophylactic iteration ladder (compile-time per rung). */
const ITERATION_LADDER = [0, 4, 16, 64, 160] as const;
const SQRT3 = Math.sqrt(3);

type TargetSystem = 'hex' | 'square';
type View = 'source' | 'target' | 'surface';
type Variable = 'extensive' | 'intensive';

/** Areal interpolation demo: ZIP-code values transferred to a grid and smoothed. */
export const arealInterpolationMode: SpatialAnalysisModeDefinition = {
  id: 'areal-interpolation',
  title: 'Areal interpolation',
  contributors: [
    'GPUArealInterpolation',
    'GPUSpatialLag',
    'GPUPycnophylactic',
    'GPUPolygonRasterization',
    'GPUGridGenerator'
  ],
  description:
    'Bike-parking spaces per ZIP code are transferred to a hexagon or square grid by area ' +
    'weights. Switch the variable between a count (extensive, split by area) and a density ' +
    '(intensive, area-weighted mean), change the grid size, or view the smooth pycnophylactic surface.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const [parking, zips] = await Promise.all([
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoZipCodes()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'areal-interpolation');
    const projection = new LocalMetricProjection(zips.origin);
    const sourceCount = zips.featureOffsets.length - 1;
    const targetCount = GRID_COLUMNS * GRID_ROWS;
    const cellCount = RASTER_WIDTH * RASTER_HEIGHT;
    const vertexCount = zips.polygonPositions.length / 2;
    const outlineSegmentCount = zips.outlineSegments.length / 4;

    // --- Raster placement: fixed over the city -----------------------------------------------
    const bounds = getPolygonBounds(zips);
    const marginX = (bounds[2] - bounds[0]) * 0.03;
    const marginY = (bounds[3] - bounds[1]) * 0.03;
    const cityBounds = [
      bounds[0] - marginX,
      bounds[1] - marginY,
      bounds[2] + marginX,
      bounds[3] + marginY
    ];
    const rasterCellSize = Math.max(
      (cityBounds[2] - cityBounds[0]) / RASTER_WIDTH,
      (cityBounds[3] - cityBounds[1]) / RASTER_HEIGHT
    );
    const rasterOriginX = (cityBounds[0] + cityBounds[2]) / 2 - (RASTER_WIDTH * rasterCellSize) / 2;
    const rasterOriginY =
      (cityBounds[1] + cityBounds[3]) / 2 - (RASTER_HEIGHT * rasterCellSize) / 2;
    const rasterBounds = [
      rasterOriginX,
      rasterOriginY,
      rasterOriginX + RASTER_WIDTH * rasterCellSize,
      rasterOriginY + RASTER_HEIGHT * rasterCellSize
    ] as const;
    const centerX = (bounds[0] + bounds[2]) / 2;
    const centerY = (bounds[1] + bounds[3]) / 2;

    // --- Source variables (CPU, setup only): parking spaces per ZIP ------------------------------
    const locate = createZipLocator(zips);
    const sourceTotals = new Float32Array(sourceCount);
    for (let point = 0; point < parking.positions.length / 2; point++) {
      const feature = locate(parking.positions[point * 2], parking.positions[point * 2 + 1]);
      if (feature >= 0) sourceTotals[feature] += parking.spaces[point];
    }
    // Zone areas from a fine CPU sampling of the same polygons (square meters).
    const sampleStep = rasterCellSize;
    const sourceAreas = new Float32Array(sourceCount);
    for (let y = rasterBounds[1] + sampleStep / 2; y < rasterBounds[3]; y += sampleStep) {
      for (let x = rasterBounds[0] + sampleStep / 2; x < rasterBounds[2]; x += sampleStep) {
        const feature = locate(x, y);
        if (feature >= 0) sourceAreas[feature] += sampleStep * sampleStep;
      }
    }
    const sourceDensities = Float32Array.from(sourceTotals, (total, row) =>
      sourceAreas[row] > 0 ? (total / sourceAreas[row]) * 1e6 : 0
    );
    const sourceTotalSum = sourceTotals.reduce((sum, value) => sum + value, 0);
    const sourceTotalMaximum = Math.max(...sourceTotals, 1);
    const sourceDensityMaximum = Math.max(...sourceDensities, 1);

    // --- Buffers ---------------------------------------------------------------------------------
    const polygonPositions = resources.createBuffer('polygon-positions', zips.polygonPositions);
    const featureOffsets = resources.createBuffer('feature-offsets', zips.featureOffsets);
    const polygonOffsets = resources.createBuffer('polygon-offsets', zips.polygonOffsets);
    const ringOffsets = resources.createBuffer('ring-offsets', zips.ringOffsets);
    const outlineSegments = resources.createBuffer('outline-segments', zips.outlineSegments);
    const sourceZones = resources.createBuffer('source-zones', cellCount * 4);
    const targetZones = resources.createBuffer('target-zones', cellCount * 4);
    const sourceRasterOverflow = resources.createBuffer('source-raster-overflow', 4);
    const targetRasterOverflow = resources.createBuffer('target-raster-overflow', 4);
    const sourceCrossingCount = resources.createBuffer('source-crossings', 4);
    const targetCrossingCount = resources.createBuffer('target-crossings', 4);
    const sourceTotalsBuffer = resources.createBuffer('source-totals', sourceTotals);
    const sourceDensitiesBuffer = resources.createBuffer('source-densities', sourceDensities);
    const targetExtensive = resources.createBuffer('target-extensive', targetCount * 4);
    const targetIntensive = resources.createBuffer('target-intensive', targetCount * 4);
    const pairOffsets = resources.createBuffer('pair-offsets', (targetCount + 1) * 4);
    const pairNeighbors = resources.createBuffer('pair-neighbors', PAIR_CAPACITY * 4);
    const pairExtensive = resources.createBuffer('pair-extensive', PAIR_CAPACITY * 4);
    const pairIntensive = resources.createBuffer('pair-intensive', PAIR_CAPACITY * 4);
    const pairOverflow = resources.createBuffer('pair-overflow', 4);
    const pairTotal = resources.createBuffer('pair-total', 4);
    const surface = resources.createBuffer('surface', cellCount * 4);
    const display = resources.createBuffer('display', cellCount * 4);
    const extentParameter = resources.createParameterBuffer(
      'extent',
      'float32',
      GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
      getGPUPolygonRasterizationExtentValues(
        rasterOriginX,
        rasterOriginY,
        rasterCellSize,
        rasterCellSize
      )
    );
    const gridParameter = resources.createParameterBuffer(
      'grid',
      'float32',
      GPU_GRID_GENERATOR_PARAMETER_LENGTH
    );
    const displayParameter = resources.createParameterBuffer('display-parameter', 'uint32', 2);

    // --- Graphs ----------------------------------------------------------------------------------
    const importPolygons = (graph: GPUCommandGraph<void>) => ({
      polygonPositions: importGraphBuffer(
        graph,
        'polygon-positions',
        polygonPositions,
        'float32x2',
        vertexCount
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

    const sourceGraph = new GPUCommandGraph<void>(device, {id: 'areal-source-raster'});
    sourceGraph.add(
      new GPUPolygonRasterization({
        id: 'source-raster',
        width: RASTER_WIDTH,
        height: RASTER_HEIGHT,
        extent: extentParameter.importToGraph(sourceGraph),
        ...importPolygons(sourceGraph),
        crossingCapacity: CROSSING_CAPACITY,
        zones: importGraphBuffer(sourceGraph, 'zones', sourceZones, 'uint32', cellCount),
        overflow: importGraphBuffer(sourceGraph, 'overflow', sourceRasterOverflow, 'uint32', 1),
        crossingCount: importGraphBuffer(sourceGraph, 'crossings', sourceCrossingCount, 'uint32', 1)
      })
    );
    const sourceCompiled = resources.track(sourceGraph.compile());

    // One target graph per grid type: grid vertices, target raster, area weights, both lags.
    const buildTargetGraph = (gridType: GPUGridType): CompiledGPUCommandGraph<void> => {
      const verticesPerCell = getGPUGridVerticesPerCell(gridType);
      const gridVertices = resources.createBuffer(
        `${gridType}-vertices`,
        targetCount * verticesPerCell * 8
      );
      const identityOffsets = Uint32Array.from({length: targetCount + 1}, (_, index) => index);
      const cellRingOffsets = Uint32Array.from(
        {length: targetCount + 1},
        (_, index) => index * verticesPerCell
      );
      const targetFeatureOffsets = resources.createBuffer(`${gridType}-features`, identityOffsets);
      const targetRingOffsets = resources.createBuffer(`${gridType}-rings`, cellRingOffsets);
      const graph = new GPUCommandGraph<void>(device, {id: `areal-target-${gridType}`});
      const vertices = importGraphBuffer(
        graph,
        'grid-vertices',
        gridVertices,
        'float32x2',
        targetCount * verticesPerCell
      );
      graph.add(
        new GPUGridGenerator({
          id: 'grid',
          gridType,
          columns: GRID_COLUMNS,
          rows: GRID_ROWS,
          parameters: gridParameter.importToGraph(graph),
          output: {positions: vertices}
        })
      );
      const targetZonesView = importGraphBuffer(
        graph,
        'target-zones',
        targetZones,
        'uint32',
        cellCount
      );
      graph.add(
        new GPUPolygonRasterization({
          id: 'target-raster',
          width: RASTER_WIDTH,
          height: RASTER_HEIGHT,
          extent: extentParameter.importToGraph(graph),
          polygonPositions: vertices,
          featureOffsets: importGraphBuffer(
            graph,
            'target-features',
            targetFeatureOffsets,
            'uint32',
            targetCount + 1
          ),
          polygonOffsets: importGraphBuffer(
            graph,
            'target-polygons',
            targetFeatureOffsets,
            'uint32',
            targetCount + 1
          ),
          ringOffsets: importGraphBuffer(
            graph,
            'target-rings',
            targetRingOffsets,
            'uint32',
            targetCount + 1
          ),
          crossingCapacity: CROSSING_CAPACITY,
          zones: targetZonesView,
          overflow: importGraphBuffer(graph, 'overflow', targetRasterOverflow, 'uint32', 1),
          crossingCount: importGraphBuffer(graph, 'crossings', targetCrossingCount, 'uint32', 1)
        })
      );
      const offsetsView = importGraphBuffer(
        graph,
        'pair-offsets',
        pairOffsets,
        'uint32',
        targetCount + 1
      );
      const neighborsView = importGraphBuffer(
        graph,
        'pair-neighbors',
        pairNeighbors,
        'uint32',
        PAIR_CAPACITY
      );
      const extensiveView = importGraphBuffer(
        graph,
        'pair-extensive',
        pairExtensive,
        'float32',
        PAIR_CAPACITY
      );
      const intensiveView = importGraphBuffer(
        graph,
        'pair-intensive',
        pairIntensive,
        'float32',
        PAIR_CAPACITY
      );
      graph.add(
        new GPUArealInterpolation({
          id: 'areal',
          sourceZones: importGraphBuffer(graph, 'source-zones', sourceZones, 'uint32', cellCount),
          targetZones: targetZonesView,
          sourceCount,
          targetCount,
          denominator: 'overlap',
          mode: 'extensive',
          weights: {offsets: offsetsView, neighbors: neighborsView, weights: extensiveView},
          alternateWeights: intensiveView,
          overflow: importGraphBuffer(graph, 'pair-overflow', pairOverflow, 'uint32', 1),
          totalPairs: importGraphBuffer(graph, 'pair-total', pairTotal, 'uint32', 1)
        })
      );
      // Cross weights: rows are targets, neighbors index the sourceCount ZIP values.
      graph.add(
        new GPUSpatialLag({
          id: 'lag-extensive',
          weights: {offsets: offsetsView, neighbors: neighborsView, weights: extensiveView},
          sourceCount,
          values: importGraphBuffer(graph, 'totals', sourceTotalsBuffer, 'float32', sourceCount),
          output: importGraphBuffer(
            graph,
            'target-extensive',
            targetExtensive,
            'float32',
            targetCount
          )
        })
      );
      graph.add(
        new GPUSpatialLag({
          id: 'lag-intensive',
          weights: {offsets: offsetsView, neighbors: neighborsView, weights: intensiveView},
          sourceCount,
          values: importGraphBuffer(
            graph,
            'densities',
            sourceDensitiesBuffer,
            'float32',
            sourceCount
          ),
          output: importGraphBuffer(
            graph,
            'target-intensive',
            targetIntensive,
            'float32',
            targetCount
          )
        })
      );
      return resources.track(graph.compile());
    };
    const targetGraphs: Record<TargetSystem, CompiledGPUCommandGraph<void>> = {
      hex: buildTargetGraph('hex'),
      square: buildTargetGraph('square')
    };

    // Pycnophylactic ladder: iterations are compile-time, the slider picks a rung.
    const pycnoGraphs = ITERATION_LADDER.map(iterations => {
      const graph = new GPUCommandGraph<void>(device, {id: `areal-pycno-${iterations}`});
      graph.add(
        new GPUPycnophylactic({
          id: 'pycnophylactic',
          width: RASTER_WIDTH,
          height: RASTER_HEIGHT,
          zones: importGraphBuffer(graph, 'source-zones', sourceZones, 'uint32', cellCount),
          zoneCount: sourceCount,
          totals: importGraphBuffer(graph, 'totals', sourceTotalsBuffer, 'float32', sourceCount),
          iterations,
          kernel: 'rook',
          output: importGraphBuffer(graph, 'surface', surface, 'float32', cellCount)
        })
      );
      return resources.track(graph.compile());
    });

    // Display: a small pass picks the variable per zone, then one value per raster cell is drawn.
    const sourceSelected = resources.createBuffer('source-selected', sourceCount * 4);
    const targetSelected = resources.createBuffer('target-selected', targetCount * 4);
    const displayGraph = new GPUCommandGraph<void>(device, {id: 'areal-display'});
    const displayParameterView = displayParameter.importToGraph(displayGraph);
    const sourceSelectedView = importGraphBuffer(
      displayGraph,
      'source-selected',
      sourceSelected,
      'float32',
      sourceCount
    );
    const targetSelectedView = importGraphBuffer(
      displayGraph,
      'target-selected',
      targetSelected,
      'float32',
      targetCount
    );
    const importFloats = (name: string, buffer: Buffer, length: number) =>
      importGraphBuffer(displayGraph, name, buffer, 'float32', length);
    addKernelPass(displayGraph, {
      id: 'areal-select',
      invocationCount: Math.max(sourceCount, targetCount),
      bindings: [
        {name: 'parameters', view: displayParameterView, type: 'u32', access: 'read'},
        {
          name: 'sourceTotals',
          view: importFloats('source-totals', sourceTotalsBuffer, sourceCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'sourceDensities',
          view: importFloats('source-densities', sourceDensitiesBuffer, sourceCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'targetExtensive',
          view: importFloats('target-extensive', targetExtensive, targetCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'targetIntensive',
          view: importFloats('target-intensive', targetIntensive, targetCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'sourceSelected', view: sourceSelectedView, type: 'f32', access: 'read_write'},
        {name: 'targetSelected', view: targetSelectedView, type: 'f32', access: 'read_write'}
      ],
      declarations: `const SOURCE_COUNT: u32 = ${sourceCount}u;\nconst TARGET_COUNT: u32 = ${targetCount}u;`,
      body: /* wgsl */ `
  let intensive = parameters[parametersOffset + 1u] == 1u;
  if (index < SOURCE_COUNT) {
    sourceSelected[sourceSelectedOffset + index] = select(
      sourceTotals[sourceTotalsOffset + index], sourceDensities[sourceDensitiesOffset + index], intensive);
  }
  if (index < TARGET_COUNT) {
    targetSelected[targetSelectedOffset + index] = select(
      targetExtensive[targetExtensiveOffset + index], targetIntensive[targetIntensiveOffset + index], intensive);
  }`
    });
    addKernelPass(displayGraph, {
      id: 'areal-display',
      invocationCount: cellCount,
      bindings: [
        {name: 'parameters', view: displayParameterView, type: 'u32', access: 'read'},
        {
          name: 'sourceZones',
          view: importGraphBuffer(displayGraph, 'source-zones', sourceZones, 'uint32', cellCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'targetZones',
          view: importGraphBuffer(displayGraph, 'target-zones', targetZones, 'uint32', cellCount),
          type: 'u32',
          access: 'read'
        },
        {name: 'sourceSelected', view: sourceSelectedView, type: 'f32', access: 'read'},
        {name: 'targetSelected', view: targetSelectedView, type: 'f32', access: 'read'},
        {
          name: 'surface',
          view: importFloats('surface', surface, cellCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'display',
          view: importFloats('display', display, cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const SOURCE_COUNT: u32 = ${sourceCount}u;\nconst TARGET_COUNT: u32 = ${targetCount}u;`,
      body: /* wgsl */ `
  let view = parameters[parametersOffset];
  let sourceZone = sourceZones[sourceZonesOffset + index];
  let targetZone = targetZones[targetZonesOffset + index];
  var value = -1.0;
  if (sourceZone < SOURCE_COUNT) {
    if (view == 0u) {
      value = sourceSelected[sourceSelectedOffset + sourceZone];
    } else if (view == 1u) {
      if (targetZone < TARGET_COUNT) {
        value = targetSelected[targetSelectedOffset + targetZone];
      }
    } else {
      value = surface[surfaceOffset + index];
    }
  }
  display[displayOffset + index] = value;`
    });
    const displayCompiled = resources.track(displayGraph.compile());

    // --- State and readbacks ---------------------------------------------------------------------
    let targetSystem: TargetSystem = 'hex';
    let view: View = 'target';
    let variable: Variable = 'extensive';
    let cellWidth = 700;
    let iterationRung = 2;
    let sourceEncoded = false;
    let targetDirty = true;
    let pycnoDirty = true;
    let displayDirty = true;
    let targetMaximum = {extensive: 1, intensive: 1};
    let surfaceMaximum = 1;
    const gridValues = new Float32Array(GPU_GRID_GENERATOR_PARAMETER_LENGTH);

    const smallReader = new SummaryReader(
      resources,
      'areal-small',
      [
        {buffer: targetExtensive, size: targetCount * 4},
        {buffer: targetIntensive, size: targetCount * 4},
        {buffer: sourceRasterOverflow, size: 4},
        {buffer: targetRasterOverflow, size: 4},
        {buffer: pairOverflow, size: 4},
        {buffer: pairTotal, size: 4},
        {buffer: sourceCrossingCount, size: 4},
        {buffer: targetCrossingCount, size: 4}
      ],
      bytes => {
        const floats = new Float32Array(bytes, 0, targetCount * 2);
        const flags = new Uint32Array(bytes, targetCount * 8, 6);
        const extensive = floats.subarray(0, targetCount);
        const intensive = floats.subarray(targetCount);
        let extensiveSum = 0;
        let extensiveMaximum = 0;
        let intensiveMinimum = Infinity;
        let intensiveMaximum = 0;
        let covered = 0;
        for (let target = 0; target < targetCount; target++) {
          extensiveSum += extensive[target];
          extensiveMaximum = Math.max(extensiveMaximum, extensive[target]);
          if (intensive[target] > 0 || extensive[target] > 0) {
            covered++;
            intensiveMinimum = Math.min(intensiveMinimum, intensive[target]);
            intensiveMaximum = Math.max(intensiveMaximum, intensive[target]);
          }
        }
        targetMaximum = {
          extensive: Math.max(extensiveMaximum, 1e-6),
          intensive: Math.max(intensiveMaximum, 1e-6)
        };
        const difference = ((extensiveSum - sourceTotalSum) / Math.max(sourceTotalSum, 1)) * 100;
        extensiveReadout.setValue(
          `${sourceTotalSum.toFixed(0)} spaces in ZIPs -> ${extensiveSum.toFixed(0)} in ` +
            `${formatCount(covered)} cells (${difference >= 0 ? '+' : ''}${difference.toFixed(3)}%)`
        );
        intensiveReadout.setValue(
          `ZIP densities ${formatDensity(Math.min(...sourceDensities))} to ` +
            `${formatDensity(sourceDensityMaximum)}; cells ${formatDensity(
              Number.isFinite(intensiveMinimum) ? intensiveMinimum : 0
            )} to ${formatDensity(intensiveMaximum)} per km2`
        );
        pairsReadout.setValue(
          `${formatCount(flags[3])} of ${formatCount(PAIR_CAPACITY)} slots` +
            (flags[2] ? ' (OVERFLOW)' : '')
        );
        rasterOverflowReadout.setValue(
          flags[0] || flags[1]
            ? `YES (${flags[4]} / ${flags[5]} crossings)`
            : `no (${formatCount(flags[4])} / ${formatCount(flags[5])} of ${formatCount(CROSSING_CAPACITY)})`
        );
        context.updateLayers();
      }
    );
    const pycnoReader = new SummaryReader(
      resources,
      'areal-pycno',
      [
        {buffer: surface, size: cellCount * 4},
        {buffer: sourceZones, size: cellCount * 4}
      ],
      bytes => {
        const mass = new Float32Array(bytes, 0, cellCount);
        const zones = new Uint32Array(bytes, cellCount * 4, cellCount);
        const sums = new Float64Array(sourceCount);
        let maximum = 0;
        for (let cell = 0; cell < cellCount; cell++) {
          if (zones[cell] < sourceCount) sums[zones[cell]] += mass[cell];
          maximum = Math.max(maximum, mass[cell]);
        }
        surfaceMaximum = Math.max(maximum, 1e-6);
        let worst = 0;
        let preserved = 0;
        for (let zone = 0; zone < sourceCount; zone++) {
          if (sourceTotals[zone] > 0) {
            worst = Math.max(worst, Math.abs(sums[zone] - sourceTotals[zone]) / sourceTotals[zone]);
            preserved++;
          }
        }
        totalsReadout.setValue(
          `${formatCount(preserved)} ZIP totals preserved, worst error ${(worst * 100).toExponential(1)}%`
        );
        context.updateLayers();
      }
    );

    const writeDisplayParameter = () => {
      displayParameter.write(
        Uint32Array.of(
          view === 'source' ? 0 : view === 'target' ? 1 : 2,
          variable === 'extensive' ? 0 : 1
        )
      );
    };
    const updateGeometryReadouts = () => {
      const widthCells = cellWidth / rasterCellSize;
      cellReadout.setValue(
        `raster ${rasterCellSize.toFixed(0)} m, target ${cellWidth.toFixed(0)} m (${widthCells.toFixed(1)} raster cells across)`
      );
      resolutionNote.setValue(
        widthCells < 4
          ? 'Resolution warning: target cells are under 4 raster cells wide, so boundary cells dominate the area error.'
          : `Cell centers decide membership, so each boundary is accurate to about one raster cell (${rasterCellSize.toFixed(0)} m); ` +
              `that is roughly ${((100 * 2 * rasterCellSize) / cellWidth).toFixed(0)}% of a target cell's width.`
      );
    };

    // --- Controls --------------------------------------------------------------------------------
    context.controls.addSelect<View>({
      label: 'Show',
      options: [
        {value: 'source', label: 'Source: ZIP codes'},
        {value: 'target', label: 'Target: interpolated grid'},
        {value: 'surface', label: 'Pycnophylactic surface (counts)'}
      ],
      value: view,
      onChange: value => {
        view = value;
        writeDisplayParameter();
        displayDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSelect<Variable>({
      label: 'Variable (per-frame lag select)',
      options: [
        {value: 'extensive', label: 'Extensive: parking spaces (split by area)'},
        {value: 'intensive', label: 'Intensive: spaces per km2 (area-weighted mean)'}
      ],
      value: variable,
      onChange: value => {
        variable = value;
        writeDisplayParameter();
        displayDirty = true;
        context.updateLayers();
      }
    });
    context.controls.addSelect<TargetSystem>({
      label: 'Target zone system (both compiled up front)',
      options: [
        {value: 'hex', label: 'Hexagon grid'},
        {value: 'square', label: 'Square grid'}
      ],
      value: targetSystem,
      onChange: value => {
        targetSystem = value;
        targetDirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Target cell width (per-frame grid parameter)',
      min: 250,
      max: 1500,
      step: 50,
      value: cellWidth,
      format: value => `${value} m`,
      onChange: value => {
        cellWidth = value;
        targetDirty = true;
        updateGeometryReadouts();
      }
    });
    context.controls.addSlider({
      label: 'Pycnophylactic iterations (compile-time ladder)',
      min: 0,
      max: ITERATION_LADDER.length - 1,
      step: 1,
      value: iterationRung,
      format: value => `${ITERATION_LADDER[value]} iterations`,
      onChange: value => {
        iterationRung = value;
        pycnoDirty = true;
      }
    });
    context.controls.addLegend({
      title: 'Selected value (range from the GPU results)',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: '0',
        maximumLabel: 'max'
      }
    });
    context.controls.addReadout('ZIP codes (source zones)', formatCount(sourceCount));
    context.controls.addReadout(
      'Target cells',
      `${GRID_COLUMNS} x ${GRID_ROWS} lattice (${formatCount(targetCount)} zones)`
    );
    const cellReadout = context.controls.addReadout('Cell sizes');
    const extensiveReadout = context.controls.addReadout('Extensive total', '...');
    const intensiveReadout = context.controls.addReadout('Intensive range', '...');
    const totalsReadout = context.controls.addReadout('Pycnophylactic', '...');
    const pairsReadout = context.controls.addReadout('Pair slots (target, source)', '...');
    const rasterOverflowReadout = context.controls.addReadout('Raster overflow / crossings', '...');
    const resolutionNote = context.controls.addNote('');
    context.controls.addNote(
      'Both systems are rasterized to one 320 x 288 grid; weights are overlap areas (denominator = overlap, so ' +
        'extensive shares of every ZIP sum to 1). Pycnophylactic smoothing keeps each ZIP total exactly; ' +
        'its iteration count is a compile-time property, so each slider step selects a precompiled graph.'
    );
    context.controls.addReadout('Data', `${parking.attribution}; ${zips.attribution}`);
    writeDisplayParameter();
    updateGeometryReadouts();

    const rowPitch = (): number => (targetSystem === 'hex' ? (cellWidth * SQRT3) / 2 : cellWidth);

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        sourceCompiled,
        targetGraphs.hex,
        targetGraphs.square,
        ...pycnoGraphs,
        displayCompiled
      ],
      encode(commandEncoder) {
        let encodedTarget = false;
        let encodedPycno = false;
        if (!sourceEncoded) {
          sourceCompiled.encode(commandEncoder, {parameters: undefined});
          sourceEncoded = true;
          targetDirty = true;
          pycnoDirty = true;
        }
        if (targetDirty) {
          getGPUGridGeneratorParameterValues(
            {
              minX: centerX - (GRID_COLUMNS * cellWidth) / 2,
              minY: centerY - (GRID_ROWS * rowPitch()) / 2,
              cellWidth,
              cellHeight: rowPitch()
            },
            gridValues
          );
          gridParameter.write(gridValues);
          targetGraphs[targetSystem].encode(commandEncoder, {parameters: undefined});
          targetDirty = false;
          encodedTarget = true;
        }
        if (pycnoDirty) {
          pycnoGraphs[iterationRung].encode(commandEncoder, {parameters: undefined});
          pycnoDirty = false;
          encodedPycno = true;
        }
        if (encodedTarget || encodedPycno || displayDirty) {
          displayCompiled.encode(commandEncoder, {parameters: undefined});
          displayDirty = false;
        }
        if (encodedTarget) smallReader.markStale();
        if (encodedPycno) pycnoReader.markStale();
        smallReader.flush(commandEncoder);
        pycnoReader.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [zips.origin[0], zips.origin[1], 0];
        const maximum =
          view === 'source'
            ? variable === 'extensive'
              ? sourceTotalMaximum
              : sourceDensityMaximum
            : view === 'target'
              ? targetMaximum[variable]
              : surfaceMaximum;
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            id: 'areal-values',
            coordinateOrigin,
            gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
            bounds: rasterBounds,
            rowOrigin: 'south',
            values: display,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [0, maximum],
            discardAtOrBelow: -0.5,
            color: [255, 255, 255, 215],
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'areal-outline',
            coordinateOrigin,
            segments: outlineSegments,
            instanceCount: outlineSegmentCount,
            widthPixels: 1.6,
            color: [255, 255, 255, 235]
          })
        ];
        return layers;
      },
      getTooltip(event: SpatialAnalysisPointerEvent) {
        if (!event.coordinate) return null;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const feature = locate(x, y);
        if (feature < 0) return null;
        return `ZIP ${zips.featureNames[feature] ?? zips.featureIds[feature]}: ${sourceTotals[feature].toFixed(0)} spaces, ${formatDensity(sourceDensities[feature])} per km2`;
      },
      destroy() {
        smallReader.stop();
        pycnoReader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};

function formatDensity(value: number): string {
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}
