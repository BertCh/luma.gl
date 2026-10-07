// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Regionalization of San Francisco ZIP codes: SKATER cuts of a minimum spanning tree.
 *
 * One graph, compiled once: `GPUContiguityWeights` (queen) gives the neighbor graph,
 * `GPUSpatialWeightsMinimumSpanningTree` builds the Borůvka forest over attribute dissimilarity,
 * `GPUSkaterRegions` makes up to `MAXIMUM_REGIONS - 1` greedy cuts, and
 * `GPURegionPartitionEvaluation` scores the labels. The region count, the minimum region size and
 * the attribute selection are buffer writes, so moving them re-encodes the same compiled graph
 * (the rebuild counter stays 0). Attribute columns that are switched off are written as constants,
 * which the tree standardises to zero.
 *
 * The map reads GPU buffers directly: the MST edges come from `treeEdgeFlags` over a static
 * one-segment-per-slot buffer, and the cuts come from the GPU cut log. Two small read-backs feed
 * the CPU-only presentation (region fill colors, readouts).
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUContiguityWeights} from '@luma.gl/experimental/gpu-spatial-analysis';
// Deep path until the main session adds the regionalization names to the entry barrel.
import {
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_SKATER_NO_CUT,
  GPU_SKATER_PARAMETER_LENGTH,
  GPURegionPartitionEvaluation,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {SpatialAnalysisPolygons} from '../spatial-analysis-data';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {triangulatePolygons, ZoneFillLayer} from './coverage-layers';
import {SummaryReader} from './summary-reader';

/** Compile-time cap on regions; `MAXIMUM_REGIONS - 1` greedy steps are unrolled. */
const MAXIMUM_REGIONS = 16;
/** Neighbor slots per polygon in the contiguity capacity. */
const NEIGHBORS_PER_POLYGON = 24;
const MAXIMUM_FLOOR = 8;
const DEFAULT_REGIONS = 6;

const ATTRIBUTES = [
  {id: 'easting', label: 'Easting'},
  {id: 'northing', label: 'Northing'},
  {id: 'area', label: 'Area'},
  {id: 'compactness', label: 'Compactness'},
  {id: 'bike-parking', label: 'Bike parking density'},
  {id: 'elevation', label: 'Mean elevation'}
] as const;

const REGION_PALETTE = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

/** Per-zone attribute table (`zoneCount * ATTRIBUTES.length`) plus centroids, computed once. */
type ZoneTable = {values: Float32Array; centroids: Float32Array};

export const regionsMode: SpatialAnalysisModeDefinition = {
  id: 'regions',
  title: 'Regions',
  contributors: [
    'GPUContiguityWeights',
    'GPUSpatialWeightsMinimumSpanningTree',
    'GPUSkaterRegions',
    'GPURegionPartitionEvaluation'
  ],
  description:
    'San Francisco ZIP codes grouped into contiguous regions of similar attributes (SKATER). ' +
    'White lines are the GPU minimum spanning tree over attribute dissimilarity; red lines are the ' +
    'greedy cuts. Move the region slider (a buffer write: the same compiled graph re-runs), ' +
    'raise the minimum size, or switch attributes on and off and watch the explained variance.',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const [zones, parking, terrain] = await Promise.all([
      context.data.getSanFranciscoZipCodes(),
      context.data.getSanFranciscoBikeParking(),
      context.data.getSanFranciscoTerrain()
    ]);
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'regions');
    const polygonCount = zones.polygonOffsets.length - 1;
    const ringCount = zones.ringOffsets.length - 1;
    const vertexCount = zones.polygonPositions.length / 2;
    const segmentCount = zones.outlineSegments.length / 4;
    const neighborCapacity = polygonCount * NEIGHBORS_PER_POLYGON;
    const columnCount = ATTRIBUTES.length;
    const maximumRegions = Math.min(MAXIMUM_REGIONS, polygonCount);
    const stepCount = maximumRegions - 1;
    const coordinateOrigin: [number, number, number] = [zones.origin[0], zones.origin[1], 0];

    const table = computeZoneTable(zones, parking, terrain);
    const triangles = triangulatePolygons(
      zones.polygonPositions,
      zones.ringOffsets,
      zones.polygonOffsets
    );

    // Static inputs and layer buffers.
    const positionsBuffer = resources.createBuffer('positions', zones.polygonPositions);
    const ringOffsetsBuffer = resources.createBuffer('ring-offsets', zones.ringOffsets);
    const polygonOffsetsBuffer = resources.createBuffer('polygon-offsets', zones.polygonOffsets);
    const triangleVertices = resources.createBuffer('triangle-vertices', triangles.corners);
    const triangleOwners = resources.createBuffer('triangle-owners', triangles.owners);
    const outlineBuffer = resources.createBuffer('outline', zones.outlineSegments);
    const centroidBuffer = resources.createBuffer('centroids', table.centroids);
    const zoneColors = resources.createBuffer('zone-colors', new Uint32Array(polygonCount));
    // Contributor inputs and outputs.
    const valuesBuffer = resources.createBuffer('values', polygonCount * columnCount * 4);
    const standardizedBuffer = resources.createBuffer(
      'standardized',
      polygonCount * columnCount * 4
    );
    const weightOffsets = resources.createBuffer('weight-offsets', (polygonCount + 1) * 4);
    const weightNeighbors = resources.createBuffer('weight-neighbors', neighborCapacity * 4);
    const weightValues = resources.createBuffer('weight-values', neighborCapacity * 4);
    const weightOverflow = resources.createBuffer('weight-overflow', 4);
    const treeFlags = resources.createBuffer('tree-flags', neighborCapacity * 4);
    const componentLabels = resources.createBuffer('component-labels', polygonCount * 4);
    const edgeIds = resources.createBuffer('edge-ids', polygonCount * 4);
    const edgeCount = resources.createBuffer('edge-count', 4);
    const edgeOverflow = resources.createBuffer('edge-overflow', 4);
    const regionLabels = resources.createBuffer('region-labels', polygonCount * 4);
    const regionCount = resources.createBuffer('region-count', 4);
    const cutEdges = resources.createBuffer('cut-edges', Math.max(stepCount, 1) * 4);
    const cutGains = resources.createBuffer('cut-gains', Math.max(stepCount, 1) * 4);
    const summary = resources.createBuffer(
      'summary',
      GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4
    );
    const regionSizes = resources.createBuffer('region-sizes', polygonCount * 4);
    const regionWithinSsd = resources.createBuffer('region-within-ssd', polygonCount * 4);
    const parameters = resources.createParameterBuffer(
      'skater-parameters',
      'uint32',
      GPU_SKATER_PARAMETER_LENGTH
    );
    // Per-slot segments (filled after the CSR read-back) and the cut mask the GPU log drives.
    const slotSegments = resources.createBuffer('slot-segments', neighborCapacity * 16);
    const cutMask = resources.createBuffer('cut-mask', neighborCapacity * 4);

    const graph = new GPUCommandGraph<void>(device, {id: 'regions'});
    const weights = {
      offsets: importGraphBuffer(
        graph,
        'weight-offsets',
        weightOffsets,
        'uint32',
        polygonCount + 1
      ),
      neighbors: importGraphBuffer(
        graph,
        'weight-neighbors',
        weightNeighbors,
        'uint32',
        neighborCapacity
      ),
      weights: importGraphBuffer(graph, 'weight-values', weightValues, 'float32', neighborCapacity)
    };
    const treeFlagsView = importGraphBuffer(
      graph,
      'tree-flags',
      treeFlags,
      'uint32',
      neighborCapacity
    );
    const componentLabelsView = importGraphBuffer(
      graph,
      'component-labels',
      componentLabels,
      'uint32',
      polygonCount
    );
    const standardizedView = importGraphBuffer(
      graph,
      'standardized',
      standardizedBuffer,
      'float32',
      polygonCount * columnCount
    );
    const regionLabelsView = importGraphBuffer(
      graph,
      'region-labels',
      regionLabels,
      'uint32',
      polygonCount
    );
    graph.add(
      new GPUContiguityWeights({
        id: 'contiguity',
        criterion: 'queen',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        ringOffsets: importGraphBuffer(
          graph,
          'ring-offsets',
          ringOffsetsBuffer,
          'uint32',
          ringCount + 1
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'polygon-offsets',
          polygonOffsetsBuffer,
          'uint32',
          polygonCount + 1
        ),
        weights,
        overflow: importGraphBuffer(graph, 'weight-overflow', weightOverflow, 'uint32', 1)
      })
    );
    graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        id: 'tree',
        weights,
        values: importGraphBuffer(
          graph,
          'values',
          valuesBuffer,
          'float32',
          polygonCount * columnCount
        ),
        columnCount,
        treeEdgeFlags: treeFlagsView,
        componentLabels: componentLabelsView,
        standardizedValues: standardizedView,
        edges: {
          ids: importGraphBuffer(graph, 'edge-ids', edgeIds, 'uint32', polygonCount),
          count: importGraphBuffer(graph, 'edge-count', edgeCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'edge-overflow', edgeOverflow, 'uint32', 1)
        }
      })
    );
    graph.add(
      new GPUSkaterRegions({
        id: 'skater',
        weights,
        treeEdgeFlags: treeFlagsView,
        componentLabels: componentLabelsView,
        values: standardizedView,
        columnCount,
        maximumRegionCount: maximumRegions,
        parameters: parameters.importToGraph(graph),
        labels: regionLabelsView,
        regionCount: importGraphBuffer(graph, 'region-count', regionCount, 'uint32', 1),
        cutEdges: importGraphBuffer(graph, 'cut-edges', cutEdges, 'uint32', Math.max(stepCount, 1)),
        cutGains: importGraphBuffer(graph, 'cut-gains', cutGains, 'float32', Math.max(stepCount, 1))
      })
    );
    graph.add(
      new GPURegionPartitionEvaluation({
        id: 'evaluation',
        values: standardizedView,
        columnCount,
        labels: regionLabelsView,
        weights,
        summary: importGraphBuffer(
          graph,
          'summary',
          summary,
          'float32',
          GPU_REGION_PARTITION_EVALUATION_LAYOUT.length
        ),
        regionSizes: importGraphBuffer(graph, 'region-sizes', regionSizes, 'uint32', polygonCount),
        regionWithinSsd: importGraphBuffer(
          graph,
          'region-within-ssd',
          regionWithinSsd,
          'float32',
          polygonCount
        )
      })
    );
    const compiled = resources.track(graph.compile());

    // Interaction state.
    let targetRegions = Math.min(DEFAULT_REGIONS, maximumRegions);
    let minimumSize = 1;
    const enabled = [true, true, false, false, true, true];
    let dirty = true;

    const writeInputs = () => {
      // Switched-off attributes become constants, which the tree standardises to zero.
      const values = new Float32Array(polygonCount * columnCount);
      for (let zone = 0; zone < polygonCount; zone++) {
        for (let column = 0; column < columnCount; column++) {
          values[zone * columnCount + column] = enabled[column]
            ? table.values[zone * columnCount + column]
            : 0;
        }
      }
      valuesBuffer.write(values);
      parameters.write(new Uint32Array([targetRegions, minimumSize]));
    };

    // One-time CSR read-back: per-slot segments for the MST layer, CPU adjacency for coloring.
    let adjacency: number[][] | null = null;
    const csrReader = new SummaryReader(
      resources,
      'regions-csr',
      [
        {buffer: weightOffsets, size: (polygonCount + 1) * 4},
        {buffer: weightNeighbors, size: neighborCapacity * 4}
      ],
      bytes => {
        const offsets = new Uint32Array(bytes, 0, polygonCount + 1);
        const neighbors = new Uint32Array(bytes, (polygonCount + 1) * 4, neighborCapacity);
        const segments = new Float32Array(neighborCapacity * 4);
        const lists: number[][] = Array.from({length: polygonCount}, () => []);
        for (let row = 0; row < polygonCount; row++) {
          for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
            const neighbor = neighbors[slot];
            lists[row].push(neighbor);
            segments[slot * 4] = table.centroids[row * 2];
            segments[slot * 4 + 1] = table.centroids[row * 2 + 1];
            segments[slot * 4 + 2] = table.centroids[neighbor * 2];
            segments[slot * 4 + 3] = table.centroids[neighbor * 2 + 1];
          }
        }
        slotSegments.write(segments);
        adjacency = lists;
        csrReady = true;
        reader.markStale();
        context.updateLayers();
      }
    );
    let csrReady = false;

    const readouts = {
      regions: context.controls.addReadout('Regions'),
      explained: context.controls.addReadout('Explained variance'),
      within: context.controls.addReadout('Within-region SSD'),
      sizes: context.controls.addReadout('Region sizes'),
      boundary: context.controls.addReadout('Boundary links'),
      tree: context.controls.addReadout('Tree edges'),
      lastCut: context.controls.addReadout('Last cut gain')
    };

    // Per-change read-back: summary words, labels, region count, the cut log and gains.
    const sources = [
      {buffer: summary, size: GPU_REGION_PARTITION_EVALUATION_LAYOUT.length * 4},
      {buffer: regionLabels, size: polygonCount * 4},
      {buffer: regionCount, size: 4},
      {buffer: cutEdges, size: Math.max(stepCount, 1) * 4},
      {buffer: cutGains, size: Math.max(stepCount, 1) * 4},
      {buffer: edgeCount, size: 4},
      {buffer: edgeOverflow, size: 4}
    ];
    const reader: SummaryReader = new SummaryReader(resources, 'regions', sources, bytes => {
      const words = GPU_REGION_PARTITION_EVALUATION_LAYOUT;
      let offset = 0;
      const summaryWords = new Float32Array(bytes, offset, words.length);
      offset += words.length * 4;
      const labels = new Uint32Array(bytes, offset, polygonCount);
      offset += polygonCount * 4;
      const regions = new Uint32Array(bytes, offset, 1)[0];
      offset += 4;
      const cuts = new Uint32Array(bytes, offset, Math.max(stepCount, 1));
      offset += Math.max(stepCount, 1) * 4;
      const gains = new Float32Array(bytes, offset, Math.max(stepCount, 1));
      offset += Math.max(stepCount, 1) * 4;
      const treeEdges = new Uint32Array(bytes, offset, 1)[0];
      const treeOverflow = new Uint32Array(bytes, offset + 4, 1)[0];

      const totalSsd = summaryWords[words.totalSsd];
      const withinSsd = summaryWords[words.withinSsd];
      readouts.regions.setValue(
        `${regions} (target ${targetRegions}${regions < targetRegions ? ', floor binds' : ''})`
      );
      readouts.explained.setValue(
        totalSsd > 0 ? `${((100 * summaryWords[words.betweenSsd]) / totalSsd).toFixed(1)}%` : 'n/a'
      );
      readouts.within.setValue(`${withinSsd.toFixed(1)} of ${totalSsd.toFixed(1)}`);
      readouts.sizes.setValue(
        `${formatCount(summaryWords[words.minimumSize])} to ${formatCount(summaryWords[words.maximumSize])} zones`
      );
      readouts.boundary.setValue(
        `${(100 * summaryWords[words.crossLinkFraction]).toFixed(1)}% of neighbor links`
      );
      readouts.tree.setValue(`${formatCount(treeEdges)}${treeOverflow ? ' (overflow)' : ''}`);
      let applied = 0;
      const mask = new Uint32Array(neighborCapacity);
      for (let step = 0; step < stepCount; step++) {
        if (cuts[step] !== GPU_SKATER_NO_CUT) {
          mask[cuts[step]] = 1;
          applied = step + 1;
        }
      }
      cutMask.write(mask);
      readouts.lastCut.setValue(
        applied > 0 ? `${gains[applied - 1].toFixed(2)} SSD (cut ${applied})` : 'no cuts'
      );
      if (adjacency) {
        zoneColors.write(assignRegionColors(labels, adjacency));
      }
    });

    context.controls.addNote(
      'Attributes (standardised before dissimilarity is taken). Switching one off writes a constant ' +
        'column; no graph is rebuilt.'
    );
    ATTRIBUTES.forEach((attribute, column) => {
      context.controls.addToggle({
        label: attribute.label,
        value: enabled[column],
        onChange: value => {
          enabled[column] = value;
          if (!enabled.some(Boolean)) {
            // A constant table would make every cut worthless.
            enabled[0] = true;
          }
          dirty = true;
        }
      });
    });
    context.controls.addSlider({
      label: 'Regions (total, islands included)',
      min: 1,
      max: maximumRegions,
      step: 1,
      value: targetRegions,
      format: value => String(Math.round(value)),
      onChange: value => {
        targetRegions = Math.round(value);
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Minimum region size (zones)',
      min: 1,
      max: MAXIMUM_FLOOR,
      step: 1,
      value: minimumSize,
      format: value => String(Math.round(value)),
      onChange: value => {
        minimumSize = Math.round(value);
        dirty = true;
      }
    });
    context.controls.addLegend({
      title: 'Map',
      entries: [
        {color: [255, 255, 255], label: 'Minimum spanning tree edge'},
        {color: [255, 60, 60], label: 'SKATER cut (tree edge removed)'}
      ]
    });
    context.setStatus(`${formatCount(polygonCount)} ZIP codes`);

    let firstEncode = true;
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder) {
        if (dirty || firstEncode) {
          writeInputs();
          compiled.encode(commandEncoder, {parameters: undefined});
          reader.request(commandEncoder);
          if (firstEncode) {
            csrReader.request(commandEncoder);
          }
          dirty = false;
          firstEncode = false;
        }
        csrReader.flush(commandEncoder);
        reader.flush(commandEncoder);
      },
      getLayers() {
        const layers: Layer[] = [
          new ZoneFillLayer({
            id: 'regions-fill',
            coordinateOrigin,
            triangleVertices,
            triangleOwners,
            zoneColors,
            triangleCount: triangles.owners.length,
            palette: REGION_PALETTE,
            fillOpacity: 0.72
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'regions-outline',
            coordinateOrigin,
            segments: outlineBuffer,
            instanceCount: segmentCount,
            widthPixels: 1,
            color: [255, 255, 255, 70]
          })
        ];
        if (csrReady) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'regions-tree',
              coordinateOrigin,
              segments: slotSegments,
              instanceCount: neighborCapacity,
              values: treeFlags,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: [255, 255, 255, 235],
              noDataColor: [0, 0, 0, 0],
              widthPixels: 2
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'regions-cuts',
              coordinateOrigin,
              segments: slotSegments,
              instanceCount: neighborCapacity,
              values: cutMask,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: [255, 60, 60, 255],
              noDataColor: [0, 0, 0, 0],
              widthPixels: 5
            }),
            new SpatialAnalysisPointLayer({
              id: 'regions-nodes',
              coordinateOrigin,
              positions: centroidBuffer,
              instanceCount: polygonCount,
              radiusPixels: 3,
              color: [20, 30, 50, 255]
            })
          );
        }
        return layers;
      },
      destroy: () => {
        csrReader.stop();
        reader.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};

/**
 * Greedy graph coloring of the regions so adjacent regions get different palette entries.
 * Largest regions first; a region takes the lowest color none of its neighbors holds.
 */
function assignRegionColors(labels: Uint32Array, adjacency: readonly number[][]): Uint32Array {
  const sizes = new Map<number, number>();
  for (const label of labels) {
    sizes.set(label, (sizes.get(label) ?? 0) + 1);
  }
  const neighborLabels = new Map<number, Set<number>>();
  for (let row = 0; row < labels.length; row++) {
    for (const neighbor of adjacency[row]) {
      if (labels[neighbor] !== labels[row]) {
        const set = neighborLabels.get(labels[row]) ?? new Set<number>();
        set.add(labels[neighbor]);
        neighborLabels.set(labels[row], set);
      }
    }
  }
  const order = [...sizes.keys()].sort((a, b) => sizes.get(b)! - sizes.get(a)! || a - b);
  const colors = new Map<number, number>();
  const paletteSize = REGION_PALETTE.length;
  for (const label of order) {
    const used = new Array<number>(paletteSize).fill(0);
    for (const other of neighborLabels.get(label) ?? []) {
      const color = colors.get(other);
      if (color !== undefined) used[color]++;
    }
    let best = 0;
    for (let color = 1; color < paletteSize; color++) {
      if (used[color] < used[best]) best = color;
    }
    colors.set(label, best);
  }
  return Uint32Array.from(labels, label => colors.get(label)!);
}

/**
 * Computes the per-zone attribute table and centroids from the zone outer rings: easting and
 * northing of the centroid, area, Polsby-Popper compactness, log bike-parking spaces per km2 and
 * mean elevation over sample points inside the zone.
 */
function computeZoneTable(
  zones: SpatialAnalysisPolygons,
  parking: {positions: Float32Array; spaces: Float32Array},
  terrain: {
    elevation: Float32Array;
    width: number;
    height: number;
    bounds: readonly [number, number, number, number];
    cellSize: readonly [number, number];
  }
): ZoneTable {
  const zoneCount = zones.polygonOffsets.length - 1;
  const columns = ATTRIBUTES.length;
  const values = new Float32Array(zoneCount * columns);
  const centroids = new Float32Array(zoneCount * 2);
  const positions = zones.polygonPositions;
  for (let zone = 0; zone < zoneCount; zone++) {
    // Outer ring of the zone's first polygon part.
    const ring = zones.polygonOffsets[zone];
    const start = zones.ringOffsets[ring];
    const end = zones.ringOffsets[ring + 1];
    let area = 0;
    let centroidX = 0;
    let centroidY = 0;
    let perimeter = 0;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let vertex = start; vertex < end; vertex++) {
      const next = vertex + 1 < end ? vertex + 1 : start;
      const x0 = positions[vertex * 2];
      const y0 = positions[vertex * 2 + 1];
      const x1 = positions[next * 2];
      const y1 = positions[next * 2 + 1];
      const cross = x0 * y1 - x1 * y0;
      area += cross;
      centroidX += (x0 + x1) * cross;
      centroidY += (y0 + y1) * cross;
      perimeter += Math.hypot(x1 - x0, y1 - y0);
      minX = Math.min(minX, x0);
      maxX = Math.max(maxX, x0);
      minY = Math.min(minY, y0);
      maxY = Math.max(maxY, y0);
    }
    area *= 0.5;
    const meanX = Math.abs(area) > 1e-6 ? centroidX / (6 * area) : (minX + maxX) / 2;
    const meanY = Math.abs(area) > 1e-6 ? centroidY / (6 * area) : (minY + maxY) / 2;
    area = Math.abs(area);
    const contains = (x: number, y: number) => {
      let inside = false;
      for (let vertex = start, previous = end - 1; vertex < end; previous = vertex++) {
        const xi = positions[vertex * 2];
        const yi = positions[vertex * 2 + 1];
        const xj = positions[previous * 2];
        const yj = positions[previous * 2 + 1];
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
          inside = !inside;
        }
      }
      return inside;
    };
    let parkingSpaces = 0;
    for (let point = 0; point < parking.spaces.length; point++) {
      const x = parking.positions[point * 2];
      const y = parking.positions[point * 2 + 1];
      if (x >= minX && x <= maxX && y >= minY && y <= maxY && contains(x, y)) {
        parkingSpaces += parking.spaces[point];
      }
    }
    let elevationSum = 0;
    let elevationCount = 0;
    for (let row = 0; row < 7; row++) {
      for (let column = 0; column < 7; column++) {
        const x = minX + ((column + 0.5) / 7) * (maxX - minX);
        const y = minY + ((row + 0.5) / 7) * (maxY - minY);
        if (contains(x, y)) {
          elevationSum += sampleTerrain(terrain, x, y);
          elevationCount++;
        }
      }
    }
    const elevation =
      elevationCount > 0 ? elevationSum / elevationCount : sampleTerrain(terrain, meanX, meanY);
    centroids[zone * 2] = meanX;
    centroids[zone * 2 + 1] = meanY;
    values[zone * columns + 0] = meanX;
    values[zone * columns + 1] = meanY;
    values[zone * columns + 2] = Math.log1p(area / 1e6);
    values[zone * columns + 3] = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0;
    values[zone * columns + 4] = Math.log1p(parkingSpaces / Math.max(area / 1e6, 1e-3));
    values[zone * columns + 5] = elevation;
  }
  return {values, centroids};
}

function sampleTerrain(
  terrain: {
    elevation: Float32Array;
    width: number;
    height: number;
    bounds: readonly [number, number, number, number];
    cellSize: readonly [number, number];
  },
  x: number,
  y: number
): number {
  const column = Math.min(
    terrain.width - 1,
    Math.max(0, Math.floor((x - terrain.bounds[0]) / terrain.cellSize[0]))
  );
  const row = Math.min(
    terrain.height - 1,
    Math.max(0, Math.floor((terrain.bounds[3] - y) / terrain.cellSize[1]))
  );
  return terrain.elevation[row * terrain.width + column];
}
