// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Feature-space "find similar locations" over San Francisco terrain cells. The USGS elevation
 * raster is averaged to 128 x 128 cells; each land cell gets five attributes (elevation, slope,
 * local relief, distance to water, bike-parking spaces), derived once on the CPU. Click a cell to
 * make it the reference; `GPUSimilarLocations` standardizes the attributes, measures the weighted
 * distance of every cell to the reference (or to the mean of several references) and ranks them
 * with a stable GPU sort, so ties keep the lowest cell ID. The k best cells are highlighted and the
 * rest of the map is shaded by closeness to the reference.
 *
 * The reference selection, attribute weights, k, direction and reference exclusion are buffer
 * writes. The z-score versus rank standardization is compile-time (rank is a quadratic scan per
 * column) and rebuilds the graph. A small summary (count, top row IDs and distances) is read back
 * only when an input changes.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  GPUSimilarLocations,
  type GPUSimilarLocationsStandardization
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

const GRID_SIZE = 128;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;
const MAXIMUM_RESULT_COUNT = 32;
const MAXIMUM_REFERENCE_COUNT = 64;
const WATER_ELEVATION = 0.5;
const DISTANCE_CAP_METERS = 3000;
const ATTRIBUTES = [
  {name: 'Elevation', unit: 'm'},
  {name: 'Slope', unit: '°'},
  {name: 'Local relief', unit: 'm'},
  {name: 'Distance to water', unit: 'm'},
  {name: 'Bike parking', unit: 'ln(1 + spaces)'}
] as const;
const ATTRIBUTE_COUNT = ATTRIBUTES.length;
const OFFSCREEN = 1e9;

export const similarLocationsMode: SpatialAnalysisModeDefinition = {
  id: 'similar-locations',
  title: 'Similar locations',
  contributors: ['GPUSimilarLocations'],
  description:
    'Click a San Francisco terrain cell: the GPU ranks every other cell by how similar its ' +
    'standardized attributes are (elevation, slope, relief, distance to water, bike parking) ' +
    'and highlights the k most similar in orange. Weight or switch off attributes, flip to the ' +
    'least similar, or add several reference cells to compare against their mean.',
  initialViewState: {longitude: -122.44, latitude: 37.755, zoom: 11.6},

  async create(context) {
    const [terrain, bikeParking] = await Promise.all([
      context.data.getSanFranciscoTerrain(),
      context.data.getSanFranciscoBikeParking()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'similar-locations');
    const [minX, minY, maxX, maxY] = terrain.bounds;
    const cellWidth = (maxX - minX) / GRID_SIZE;
    const cellHeight = (maxY - minY) / GRID_SIZE;

    // --- Attributes (CPU, once) ----------------------------------------------------------------
    const attributes = new Float32Array(CELL_COUNT * ATTRIBUTE_COUNT);
    const elevation = new Float32Array(CELL_COUNT);
    const blockX = terrain.width / GRID_SIZE;
    const blockY = terrain.height / GRID_SIZE;
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let column = 0; column < GRID_SIZE; column++) {
        let sum = 0;
        let count = 0;
        for (let y = Math.floor(row * blockY); y < Math.floor((row + 1) * blockY); y++) {
          for (let x = Math.floor(column * blockX); x < Math.floor((column + 1) * blockX); x++) {
            sum += terrain.elevation[y * terrain.width + x];
            count++;
          }
        }
        elevation[row * GRID_SIZE + column] = count > 0 ? sum / count : 0;
      }
    }
    const isWater = (cell: number) => elevation[cell] <= WATER_ELEVATION;
    const distanceToWater = getDistanceToWater(isWater, cellWidth);
    const bikeCounts = new Float32Array(CELL_COUNT);
    const bikeProjection = new LocalMetricProjection(bikeParking.origin);
    for (let point = 0; point < bikeParking.spaces.length; point++) {
      const [lng, lat] = bikeProjection.unproject(
        bikeParking.positions[2 * point],
        bikeParking.positions[2 * point + 1]
      );
      const [x, y] = projection.project(lng, lat);
      const column = Math.floor((x - minX) / cellWidth);
      const row = Math.floor((maxY - y) / cellHeight);
      if (column >= 0 && column < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
        bikeCounts[row * GRID_SIZE + column] += bikeParking.spaces[point];
      }
    }
    for (let row = 0; row < GRID_SIZE; row++) {
      for (let column = 0; column < GRID_SIZE; column++) {
        const cell = row * GRID_SIZE + column;
        const target = cell * ATTRIBUTE_COUNT;
        if (isWater(cell)) {
          attributes.fill(NaN, target, target + ATTRIBUTE_COUNT);
          continue;
        }
        const at = (r: number, c: number) =>
          elevation[
            Math.min(GRID_SIZE - 1, Math.max(0, r)) * GRID_SIZE +
              Math.min(GRID_SIZE - 1, Math.max(0, c))
          ];
        const dzdx = (at(row, column + 1) - at(row, column - 1)) / (2 * cellWidth);
        const dzdy = (at(row + 1, column) - at(row - 1, column)) / (2 * cellHeight);
        let low = Infinity;
        let high = -Infinity;
        for (let dr = -2; dr <= 2; dr++) {
          for (let dc = -2; dc <= 2; dc++) {
            low = Math.min(low, at(row + dr, column + dc));
            high = Math.max(high, at(row + dr, column + dc));
          }
        }
        attributes[target] = elevation[cell];
        attributes[target + 1] = (Math.atan(Math.hypot(dzdx, dzdy)) * 180) / Math.PI;
        attributes[target + 2] = high - low;
        attributes[target + 3] = distanceToWater[cell];
        attributes[target + 4] = Math.log1p(bikeCounts[cell]);
      }
    }
    let validCellCount = 0;
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      if (!isWater(cell)) validCellCount++;
    }

    // --- Buffers -------------------------------------------------------------------------------
    const attributesBuffer = resources.createBuffer('attributes', attributes);
    const selectionValues = new Uint32Array(CELL_COUNT);
    const selectionBuffer = resources.createBuffer('selection', selectionValues);
    const ranksBuffer = resources.createBuffer('ranks', CELL_COUNT * 4);
    const distancesBuffer = resources.createBuffer('distances', CELL_COUNT * 4);
    const topIdsBuffer = resources.createBuffer('top-ids', MAXIMUM_RESULT_COUNT * 4);
    const countBuffer = resources.createBuffer('count', 4);
    const topDistancesBuffer = resources.createBuffer('top-distances', MAXIMUM_RESULT_COUNT * 4);
    const topPositionsBuffer = resources.createBuffer(
      'top-positions',
      MAXIMUM_RESULT_COUNT * 2 * 4
    );
    const displayBuffer = resources.createBuffer('display', CELL_COUNT * 4);
    const referencePositionValues = new Float32Array(MAXIMUM_REFERENCE_COUNT * 2).fill(OFFSCREEN);
    const referencePositionsBuffer = resources.createBuffer(
      'reference-positions',
      referencePositionValues
    );
    const parameters = resources.createParameterBuffer(
      'parameters',
      'float32',
      getGPUSimilarLocationsParameterLength(ATTRIBUTE_COUNT)
    );
    const geometry = resources.createParameterBuffer(
      'geometry',
      'float32',
      4,
      Float32Array.of(minX, maxY, cellWidth, cellHeight)
    );

    // --- State ---------------------------------------------------------------------------------
    let standardization: GPUSimilarLocationsStandardization = 'zscore';
    let resultCount = 12;
    let direction: 'most' | 'least' = 'most';
    let excludeReference = true;
    let addToReference = false;
    const weights = new Array<number>(ATTRIBUTE_COUNT).fill(1);
    const selectedCells: number[] = [];
    let destroyed = false;
    let measuring = false;
    let compiled: CompiledGPUCommandGraph<void> | null = null;

    // Default reference: the highest land cell.
    let highest = 0;
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      if (!isWater(cell) && elevation[cell] > elevation[highest]) highest = cell;
    }
    selectedCells.push(highest);

    function buildGraph(): CompiledGPUCommandGraph<void> {
      const graph = new GPUCommandGraph<void>(device, {id: 'similar-locations'});
      const view = <Format extends 'float32' | 'uint32'>(
        name: string,
        buffer: typeof ranksBuffer,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      const topIds = view('top-ids', topIdsBuffer, 'uint32', MAXIMUM_RESULT_COUNT);
      const count = view('count', countBuffer, 'uint32', 1);
      const distances = view('distances', distancesBuffer, 'float32', CELL_COUNT);
      const ranks = view('ranks', ranksBuffer, 'uint32', CELL_COUNT);
      const parameterView = parameters.importToGraph(graph);
      graph.add(
        new GPUSimilarLocations({
          id: 'similar',
          attributes: view('attributes', attributesBuffer, 'float32', CELL_COUNT * ATTRIBUTE_COUNT),
          attributeCount: ATTRIBUTE_COUNT,
          selection: view('selection', selectionBuffer, 'uint32', CELL_COUNT),
          parameters: parameterView,
          standardization,
          maximumResultCount: MAXIMUM_RESULT_COUNT,
          output: {ranks, distances, topIds, count}
        })
      );
      const topDistances = view(
        'top-distances',
        topDistancesBuffer,
        'float32',
        MAXIMUM_RESULT_COUNT
      );
      const topPositions = view(
        'top-positions',
        topPositionsBuffer,
        'float32',
        MAXIMUM_RESULT_COUNT * 2
      );
      const geometryView = geometry.importToGraph(graph);
      // Gather the winners' distances and cell centers (row 0 is the north edge).
      addKernelPass(graph, {
        id: 'gather-top',
        bindings: [
          {name: 'topIds', view: topIds, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'geometry', view: geometryView, type: 'f32', access: 'read'},
          {name: 'topDistances', view: topDistances, type: 'f32', access: 'read_write'},
          {name: 'topPositions', view: topPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: MAXIMUM_RESULT_COUNT,
        body: `let cell = topIds[topIdsOffset + index];
  let isValid = cell != 0xffffffffu;
  let column = f32(cell % ${GRID_SIZE}u);
  let row = f32(cell / ${GRID_SIZE}u);
  topDistances[topDistancesOffset + index] = select(0.0, distances[distancesOffset + cell], isValid);
  topPositions[topPositionsOffset + 2u * index] = select(${OFFSCREEN}.0, geometry[geometryOffset] + (column + 0.5) * geometry[geometryOffset + 2u], isValid);
  topPositions[topPositionsOffset + 2u * index + 1u] = select(${OFFSCREEN}.0, geometry[geometryOffset + 1u] - (row + 0.5) * geometry[geometryOffset + 3u], isValid);`
      });
      // Similarity shading: exp(-d / (8 * distance of the k-th match)); NaN when unranked.
      addKernelPass(graph, {
        id: 'display',
        bindings: [
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'topDistances', view: topDistances, type: 'f32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {
            name: 'display',
            view: view('display', displayBuffer, 'float32', CELL_COUNT),
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: CELL_COUNT,
        body: `let last = max(count[countOffset], 1u) - 1u;
  let scale = max(8.0 * topDistances[topDistancesOffset + last], 1e-6);
  let distance = distances[distancesOffset + index];
  let isRanked = (bitcast<u32>(distance) & 0x7fffffffu) < 0x7f800000u;
  var nan = 0x7fc00000u;
  display[displayOffset + index] = select(bitcast<f32>(nan), exp(-distance / scale), isRanked);`
      });
      return graph.compile();
    }

    function writeParameters(): void {
      parameters.write(
        getGPUSimilarLocationsParameterValues(
          {resultCount, direction, excludeReference, weights},
          ATTRIBUTE_COUNT
        )
      );
    }
    function writeSelection(): void {
      selectionValues.fill(0);
      referencePositionValues.fill(OFFSCREEN);
      selectedCells.forEach((cell, index) => {
        selectionValues[cell] = 1;
        referencePositionValues[2 * index] = minX + ((cell % GRID_SIZE) + 0.5) * cellWidth;
        referencePositionValues[2 * index + 1] =
          maxY - (Math.floor(cell / GRID_SIZE) + 0.5) * cellHeight;
      });
      selectionBuffer.write(selectionValues);
      referencePositionsBuffer.write(referencePositionValues);
      selectionChanged();
    }

    // --- Summary readback ----------------------------------------------------------------------
    const summary = new SummaryReader(
      resources,
      'similar-locations',
      [
        {buffer: countBuffer, size: 4},
        {buffer: topIdsBuffer, size: MAXIMUM_RESULT_COUNT * 4},
        {buffer: topDistancesBuffer, size: MAXIMUM_RESULT_COUNT * 4}
      ],
      bytes => {
        if (destroyed) return;
        const matchCount = new Uint32Array(bytes, 0, 1)[0];
        const ids = new Uint32Array(bytes, 4, MAXIMUM_RESULT_COUNT);
        const topDistances = new Float32Array(bytes, 4 + MAXIMUM_RESULT_COUNT * 4);
        matchesReadout.setValue(
          `${formatCount(matchCount)} (of ${formatCount(validCellCount - (excludeReference ? selectedCells.length : 0))} candidates)`
        );
        if (matchCount === 0) {
          distanceReadout.setValue('none');
          bestReadout.setValue('none');
          return;
        }
        distanceReadout.setValue(
          `${topDistances[0].toFixed(2)} (best), ${topDistances[matchCount - 1].toFixed(2)} (k-th)`
        );
        bestReadout.setValue(
          `${describeCell(ids[0])} vs ${describeReference()}`.replace(/\n/g, '; ')
        );
      }
    );
    function selectionChanged(): void {
      summary.markStale();
      context.updateLayers();
    }

    writeParameters();
    writeSelection();
    compiled = resources.track(buildGraph());

    function describeCell(cell: number): string {
      return ATTRIBUTES.map(
        (attribute, index) =>
          `${attribute.name} ${attributes[cell * ATTRIBUTE_COUNT + index].toFixed(1)}`
      ).join(', ');
    }
    function describeReference(): string {
      return `reference ${ATTRIBUTES.map((attribute, index) => {
        let sum = 0;
        for (const cell of selectedCells) sum += attributes[cell * ATTRIBUTE_COUNT + index];
        return `${(sum / Math.max(selectedCells.length, 1)).toFixed(1)}`;
      }).join(' / ')}`;
    }
    function getCellAt(coordinate: readonly [number, number] | null): number {
      if (!coordinate) return -1;
      const [x, y] = projection.project(coordinate[0], coordinate[1]);
      const column = Math.floor((x - minX) / cellWidth);
      const row = Math.floor((maxY - y) / cellHeight);
      if (column < 0 || column >= GRID_SIZE || row < 0 || row >= GRID_SIZE) return -1;
      const cell = row * GRID_SIZE + column;
      return isWater(cell) ? -1 : cell;
    }

    // --- Controls ------------------------------------------------------------------------------
    context.controls.addSelect<GPUSimilarLocationsStandardization>({
      label: 'Standardization (compile-time: rebuilds the graph)',
      options: [
        {value: 'zscore', label: 'Z-score'},
        {value: 'rank', label: 'Percentile rank (quadratic scan)'}
      ],
      value: standardization,
      onChange: value => {
        standardization = value;
        resources.release(compiled!);
        compiled = resources.track(buildGraph());
        summary.markStale();
        context.setStatus(`Standardization: ${value} (graph rebuilt)`);
      }
    });
    context.controls.addSlider({
      label: 'Matches k (per-frame)',
      min: 1,
      max: MAXIMUM_RESULT_COUNT,
      step: 1,
      value: resultCount,
      onChange: value => {
        resultCount = value;
        writeParameters();
        selectionChanged();
      }
    });
    context.controls.addSelect<'most' | 'least'>({
      label: 'Direction (per-frame)',
      options: [
        {value: 'most', label: 'Most similar'},
        {value: 'least', label: 'Least similar'}
      ],
      value: direction,
      onChange: value => {
        direction = value;
        writeParameters();
        selectionChanged();
      }
    });
    ATTRIBUTES.forEach((attribute, index) => {
      context.controls.addSlider({
        label: `Weight: ${attribute.name} (per-frame)`,
        min: 0,
        max: 3,
        step: 0.5,
        value: weights[index],
        format: value => (value === 0 ? 'off' : value.toFixed(1)),
        onChange: value => {
          weights[index] = value;
          writeParameters();
          selectionChanged();
        }
      });
    });
    context.controls.addToggle({
      label: 'Exclude reference cells from the ranking',
      value: excludeReference,
      onChange: value => {
        excludeReference = value;
        writeParameters();
        selectionChanged();
      }
    });
    context.controls.addToggle({
      label: 'Clicks add reference cells (compare to their mean)',
      value: addToReference,
      onChange: value => {
        addToReference = value;
      }
    });
    context.controls.addButton({
      label: 'Reset reference to the highest cell',
      onClick: () => {
        selectedCells.length = 0;
        selectedCells.push(highest);
        writeSelection();
      }
    });
    context.controls.addLegend({
      title: 'Similarity to the reference',
      gradient: {
        colors: [
          [68, 1, 84],
          [33, 145, 140],
          [253, 231, 37]
        ],
        minimumLabel: 'dissimilar',
        maximumLabel: 'identical'
      }
    });
    context.controls.addNote(
      'Water cells have no attributes and are never ranked. Equal distances keep the lowest cell ID. ' +
        'Orange discs are the k matches, white rings the reference cells.'
    );
    context.controls.addReadout(
      'Cells',
      `${GRID_SIZE} × ${GRID_SIZE} (${formatCount(validCellCount)} land)`
    );
    const referenceReadout = context.controls.addReadout('Reference cells');
    const matchesReadout = context.controls.addReadout('Matches returned');
    const distanceReadout = context.controls.addReadout('Distance in standardized units');
    const bestReadout = context.controls.addReadout('Best match vs reference');
    const timingReadout = context.controls.addReadout('Graph time');
    context.controls.addButton({label: 'Time the graph now', onClick: () => void measureGraph()});
    context.controls.addReadout('Data', terrain.attribution);

    async function measureGraph(): Promise<void> {
      if (measuring || destroyed || !compiled) return;
      measuring = true;
      timingReadout.setValue('measuring...');
      try {
        const timing = await measureCompiledGraph(device, compiled, {
          parameters: undefined,
          completionBuffer: countBuffer,
          signal: context.signal
        });
        if (!destroyed) timingReadout.setValue(formatCompiledGraphTiming(timing));
      } catch {
        if (!destroyed) timingReadout.setValue('interrupted');
      } finally {
        measuring = false;
      }
    }

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled!],
      encode(commandEncoder) {
        referenceReadout.setValue(`${selectedCells.length}`);
        compiled!.encode(commandEncoder, {parameters: undefined});
        summary.flush(commandEncoder);
      },
      getLayers() {
        const layers: Layer[] = [
          new SpatialAnalysisRasterLayer({
            id: 'similar-locations-display',
            coordinateOrigin: origin,
            gridSize: [GRID_SIZE, GRID_SIZE],
            bounds: terrain.bounds,
            rowOrigin: 'north',
            values: displayBuffer,
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [0, 1],
            noDataColor: [30, 30, 40, 90],
            color: [255, 255, 255, 190]
          }),
          new SpatialAnalysisPointLayer({
            id: 'similar-locations-matches-halo',
            coordinateOrigin: origin,
            positions: topPositionsBuffer,
            instanceCount: MAXIMUM_RESULT_COUNT,
            radiusPixels: 6.5,
            color: [0, 0, 0, 230]
          }),
          new SpatialAnalysisPointLayer({
            id: 'similar-locations-matches',
            coordinateOrigin: origin,
            positions: topPositionsBuffer,
            instanceCount: MAXIMUM_RESULT_COUNT,
            radiusPixels: 4.5,
            color: [255, 140, 40, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'similar-locations-reference-halo',
            coordinateOrigin: origin,
            positions: referencePositionsBuffer,
            instanceCount: MAXIMUM_REFERENCE_COUNT,
            radiusPixels: 9,
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'similar-locations-reference',
            coordinateOrigin: origin,
            positions: referencePositionsBuffer,
            instanceCount: MAXIMUM_REFERENCE_COUNT,
            radiusPixels: 6,
            color: [20, 20, 30, 255]
          })
        ];
        return layers;
      },
      onClick(event: SpatialAnalysisPointerEvent) {
        const cell = getCellAt(event.coordinate);
        if (cell < 0) return false;
        const existing = selectedCells.indexOf(cell);
        if (addToReference) {
          if (existing >= 0) {
            if (selectedCells.length > 1) selectedCells.splice(existing, 1);
          } else if (selectedCells.length < MAXIMUM_REFERENCE_COUNT) {
            selectedCells.push(cell);
          }
        } else {
          selectedCells.length = 0;
          selectedCells.push(cell);
        }
        writeSelection();
        return true;
      },
      getTooltip(event) {
        const cell = getCellAt(event.coordinate);
        return cell < 0 ? null : `Cell ${cell}: ${describeCell(cell)}`;
      },
      destroy() {
        destroyed = true;
        summary.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};

/**
 * Chamfer distance transform (3-4 weights) in meters to the nearest water cell, capped at
 * {@link DISTANCE_CAP_METERS}. With no water the cap is returned everywhere.
 */
function getDistanceToWater(isWater: (cell: number) => boolean, cellSize: number): Float32Array {
  const distance = new Float32Array(CELL_COUNT);
  for (let cell = 0; cell < CELL_COUNT; cell++) distance[cell] = isWater(cell) ? 0 : Infinity;
  const relax = (cell: number, row: number, column: number, dRow: number, dColumn: number) => {
    const neighborRow = row + dRow;
    const neighborColumn = column + dColumn;
    if (
      neighborRow < 0 ||
      neighborRow >= GRID_SIZE ||
      neighborColumn < 0 ||
      neighborColumn >= GRID_SIZE
    ) {
      return;
    }
    const weight = dRow !== 0 && dColumn !== 0 ? 1.4142 : 1;
    const candidate = distance[neighborRow * GRID_SIZE + neighborColumn] + weight * cellSize;
    if (candidate < distance[cell]) distance[cell] = candidate;
  };
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      const cell = row * GRID_SIZE + column;
      relax(cell, row, column, -1, -1);
      relax(cell, row, column, -1, 0);
      relax(cell, row, column, -1, 1);
      relax(cell, row, column, 0, -1);
    }
  }
  for (let row = GRID_SIZE - 1; row >= 0; row--) {
    for (let column = GRID_SIZE - 1; column >= 0; column--) {
      const cell = row * GRID_SIZE + column;
      relax(cell, row, column, 1, 1);
      relax(cell, row, column, 1, 0);
      relax(cell, row, column, 1, -1);
      relax(cell, row, column, 0, 1);
    }
  }
  for (let cell = 0; cell < CELL_COUNT; cell++) {
    distance[cell] = Math.min(distance[cell], DISTANCE_CAP_METERS);
  }
  return distance;
}
