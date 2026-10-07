// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Network isochrones over the New York street graph from up to six clickable facilities. One
 * compiled graph converts the directed edge list to CSR (`GPUCOOToCSR`) and runs two
 * `GPUNetworkIsochrones` producers: a raster producer that allocates every node to its nearest
 * facility (`GPUNetworkServiceAreas` inside), splats edge-interpolated costs plus a walking buffer
 * to a viewport raster and contours it into filled bands (`GPUIsobands`), with the facility of
 * every pixel and band triangle; and a cell producer that reads the same node costs and
 * allocation, labels each reached Quadbin or H3 cell with the facility of its cheapest node and
 * outlines the cells per facility (`GPUCellSetOutline` with groups, then ring assembly), so rings
 * are colored by facility and never mix facilities. Clicking the map adds a facility (clicking a
 * facility's node removes it); the facility count, time budget, band count, walking buffer,
 * transport and raster extent are buffer writes. Only the cell family and resolution recompile
 * the graph.
 */

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import {
  createTransientView,
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUCOOToCSR,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUPointInPolygonJoin
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  getGPUNetworkIsochroneParameterValues,
  GPUNetworkIsochrones,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-network';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../spatial-analysis-resources';
import {IsobandTriangleLayer, PolylineLayer} from './contours-layers';
import {
  findNearestNode,
  sortEdgesBySource,
  WALK_SPEED,
  writeEdgeTravelSeconds,
  type Transport
} from './road-network-utils';
import {SummaryReader} from './summary-reader';

/** Search rounds of the reachability inside the producer (16 hops per round). */
const MAXIMUM_ITERATIONS = 64;
const RASTER_WIDTH = 1024;
const RASTER_HEIGHT = 640;
const RASTER_ASPECT = RASTER_WIDTH / RASTER_HEIGHT;
/** Compile-time cap on the half-window one edge sample marks, in pixels. */
const MAXIMUM_BUFFER_PIXELS = 16;
const MAXIMUM_SAMPLES_PER_EDGE = 256;
const MAXIMUM_BREAKS = 6;
const TRIANGLE_CAPACITY = 800_000;
const SEGMENT_CAPACITY = 300_000;
const PALETTE_SIZE = 64;
const RING_CAPACITY = 2048;
/** Compile-time facility capacity; the active count is a per-frame parameter. */
const MAXIMUM_FACILITIES = 6;
/** One color per facility (the layer palettes hold eight). */
const FACILITY_COLORS: readonly (readonly [number, number, number, number])[] = [
  [255, 99, 71, 200],
  [64, 170, 255, 200],
  [110, 220, 110, 200],
  [255, 200, 50, 200],
  [200, 120, 255, 200],
  [255, 140, 200, 200]
];
const RING_VERTEX_CAPACITY = 32768;
const DEMAND_INSIDE_COLOR = [90, 255, 160, 255] as const;
const DEMAND_OUTSIDE_COLOR = [150, 155, 170, 70] as const;
const RECORD_OUTLINE = 0;
const RECORD_BANDS = 1;
const RECORD_BYTE_LENGTH = 16;
/** Times Square. */
const DEFAULT_SOURCE: readonly [number, number] = [-73.9855, 40.758];
const RAMP_STOPS = [
  [252, 255, 164],
  [250, 193, 39],
  [237, 105, 37],
  [188, 55, 84],
  [120, 28, 109],
  [60, 15, 90]
] as const;
const BAND_ALPHA = 175;
const ROAD_COLOR = [150, 160, 180, 70] as const;

type View = 'bands' | 'cells' | 'both';
type BandColor = 'facility' | 'ramp';
type ClickAction = 'add' | 'replace';
type CellChoice = 'quadbin-16' | 'quadbin-17' | 'h3-8' | 'h3-9';

const CELL_CHOICES: Record<
  CellChoice,
  {family: 'quadbin' | 'h3'; resolution: number; label: string}
> = {
  'quadbin-16': {family: 'quadbin', resolution: 16, label: 'Quadbin 16 (about 470 m tiles)'},
  'quadbin-17': {family: 'quadbin', resolution: 17, label: 'Quadbin 17 (about 235 m tiles)'},
  'h3-8': {family: 'h3', resolution: 8, label: 'H3 8 (about 460 m hexagons)'},
  'h3-9': {family: 'h3', resolution: 9, label: 'H3 9 (about 175 m hexagons)'}
};

/** Packs a ramp from near (warm) to far (cool) into rgba8 words. */
function createBandPalette(): Uint32Array {
  const palette = new Uint32Array(PALETTE_SIZE);
  for (let index = 0; index < PALETTE_SIZE; index++) {
    const position = (index / (PALETTE_SIZE - 1)) * (RAMP_STOPS.length - 1);
    const lower = Math.min(Math.floor(position), RAMP_STOPS.length - 2);
    const fraction = position - lower;
    const channel = (component: number) =>
      Math.round(
        RAMP_STOPS[lower][component] * (1 - fraction) + RAMP_STOPS[lower + 1][component] * fraction
      );
    palette[index] =
      (channel(0) | (channel(1) << 8) | (channel(2) << 16) | (BAND_ALPHA << 24)) >>> 0;
  }
  return palette;
}

/** Network isochrones as GPU raster isobands or Quadbin/H3 cell outlines. */
export const isochronesMode: SpatialAnalysisModeDefinition = {
  id: 'isochrones',
  title: 'Isochrones',
  contributors: [
    'GPUNetworkIsochrones',
    'GPUNetworkServiceAreas',
    'GPUPointToCell',
    'GPUIsobands',
    'GPUCellAggregation',
    'GPUCellSetOutline',
    'GPUSegmentRingAssembly',
    'GPUPointInPolygonJoin',
    'GPUCOOToCSR'
  ],
  description:
    'Walking-time polygons on the New York street graph, built on the GPU from up to six ' +
    'facilities. Click the map to add a facility (click one to remove it); every node goes to its ' +
    'nearest facility, and bands and rings are colored by facility. Drag the time, band, and ' +
    'walking-buffer sliders, or switch to the cell-ring view.',
  initialViewState: {longitude: -73.985, latitude: 40.755, zoom: 13.3},

  async create(context) {
    const roads = await context.data.getNewYorkRoads();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(roads.origin);
    const resources = new SpatialAnalysisResources(device, 'isochrones');
    const nodeCount = roads.nodePositions.length / 2;
    const segmentCount = roads.segmentNodes.length;
    const edges = sortEdgesBySource(roads);
    const edgeCount = edges.sources.length;
    const weights = new Float32Array(edgeCount);
    const nodeLngLat = new Float32Array(nodeCount * 2);
    for (let node = 0; node < nodeCount; node++) {
      const [longitude, latitude] = projection.unproject(
        roads.nodePositions[node * 2],
        roads.nodePositions[node * 2 + 1]
      );
      nodeLngLat[node * 2] = longitude;
      nodeLngLat[node * 2 + 1] = latitude;
    }

    let transport: Transport = 'walk';
    let budgetMinutes = 15;
    let bandCount = 4;
    let bufferMeters = 40;
    let view: View = 'both';
    let bandColor: BandColor = 'facility';
    let clickAction: ClickAction = 'add';
    let cellChoice: CellChoice = 'quadbin-17';
    let showRoads = true;
    let showDemand = true;
    const facilityNodes: number[] = [
      findNearestNode(roads.nodePositions, projection.project(...DEFAULT_SOURCE)),
      findNearestNode(roads.nodePositions, projection.project(-73.9772, 40.7527)),
      findNearestNode(roads.nodePositions, projection.project(-73.9935, 40.7506))
    ].filter((node, index, nodes) => nodes.indexOf(node) === index);
    let extent: [number, number, number, number] | null = null;
    let extentKey = '';
    let dirty = true;
    let destroyed = false;

    const cooRows = resources.createBuffer('coo-rows', edges.sources);
    const cooColumns = resources.createBuffer('coo-columns', edges.targets);
    const weightsBuffer = resources.createBuffer('coo-weights', weights);
    const nodePositionsBuffer = resources.createBuffer('node-positions', roads.nodePositions);
    const nodeLngLatBuffer = resources.createBuffer('node-lnglat', nodeLngLat);
    const pois = await context.data.getNewYorkPointsOfInterest();
    context.signal.throwIfAborted();
    const poiCount = pois.positions.length / 2;
    const poiLngLat = new Float32Array(poiCount * 2);
    for (let poi = 0; poi < poiCount; poi++) {
      const [longitude, latitude] = projection.unproject(
        pois.positions[poi * 2],
        pois.positions[poi * 2 + 1]
      );
      poiLngLat[poi * 2] = longitude;
      poiLngLat[poi * 2 + 1] = latitude;
    }
    const poiLngLatBuffer = resources.createBuffer('poi-lnglat', poiLngLat);
    const poiInside = resources.createBuffer('poi-inside', poiCount * 4);
    const joinOverflow = resources.createBuffer('join-overflow', 4);
    const ringOffsets = resources.createBuffer('ring-offsets', (RING_CAPACITY + 1) * 4);
    const ringPositions = resources.createBuffer('ring-positions', RING_VERTEX_CAPACITY * 8);
    const ringIsHole = resources.createBuffer('ring-is-hole', RING_CAPACITY * 4);
    const ringShells = resources.createBuffer('ring-shells', RING_CAPACITY * 4);
    const ringCount = resources.createBuffer('ring-count', 4);
    const ringOverflow = resources.createBuffer('ring-overflow', 4);
    const ringTotal = resources.createBuffer('ring-total', 4);
    const ringOpen = resources.createBuffer('ring-open', 4);
    const ringTouching = resources.createBuffer('ring-touching', 4);
    const polygonPositions = resources.createBuffer('polygon-positions', RING_VERTEX_CAPACITY * 8);
    const polygonRingOffsets = resources.createBuffer(
      'polygon-ring-offsets',
      (RING_CAPACITY + 1) * 4
    );
    const polygonOffsets = resources.createBuffer('polygon-offsets', (RING_CAPACITY + 1) * 4);
    const polygonFeatureOffsets = resources.createBuffer('polygon-feature-offsets', 8);
    const costs = resources.createBuffer('costs', nodeCount * 4);
    const assignments = resources.createBuffer('assignments', nodeCount * 4);
    const pixelFacilities = resources.createBuffer(
      'pixel-facilities',
      RASTER_WIDTH * RASTER_HEIGHT * 4
    );
    const triangleFacilities = resources.createBuffer('triangle-facilities', TRIANGLE_CAPACITY * 4);
    const outlineGroups = resources.createBuffer('outline-groups', SEGMENT_CAPACITY * 4);
    const ringGroups = resources.createBuffer('ring-groups', RING_CAPACITY * 4);
    const facilityIdentity = resources.createBuffer(
      'facility-identity',
      Uint32Array.from({length: 8}, (_, index) => index)
    );
    const segmentsBuffer = resources.createBuffer('segments', roads.segments);
    const facilityPositions = resources.createBuffer('facility-positions', MAXIMUM_FACILITIES * 8);
    const sources = resources.createParameterBuffer('sources', 'uint32', MAXIMUM_FACILITIES);
    const sourceCount = resources.createParameterBuffer('source-count', 'uint32', 1);
    const costLimit = resources.createParameterBuffer('cost-limit', 'float32', 1);
    const breaks = resources.createParameterBuffer('breaks', 'float32', MAXIMUM_BREAKS);
    const parameters = resources.createParameterBuffer(
      'parameters',
      'float32',
      GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH
    );
    const paletteBuffer = resources.createBuffer('palette', createBandPalette());
    const triangles = resources.createBuffer('triangles', TRIANGLE_CAPACITY * 3 * 8);
    const triangleBands = resources.createBuffer('triangle-bands', TRIANGLE_CAPACITY * 4);
    const triangleCount = resources.createBuffer('triangle-count', 4);
    const triangleOverflow = resources.createBuffer('triangle-overflow', 4);
    const bandVertexCount = resources.createBuffer('band-vertex-count', 4);
    const tableCells = resources.createBuffer('table-cells', nodeCount * 8);
    const tableCounts = resources.createBuffer('table-counts', nodeCount * 4);
    const tableCount = resources.createBuffer('table-count', 4);
    const tableOverflow = resources.createBuffer('table-overflow', 4);
    const outlineRows = resources.createBuffer('outline-rows', SEGMENT_CAPACITY * 4);
    const outlineCells = resources.createBuffer('outline-cells', SEGMENT_CAPACITY * 8);
    const outlineEdges = resources.createBuffer('outline-edges', SEGMENT_CAPACITY * 4);
    const outlineEndpoints = resources.createBuffer('outline-endpoints', SEGMENT_CAPACITY * 16);
    const outlineCount = resources.createBuffer('outline-count', 4);
    const outlineOverflow = resources.createBuffer('outline-overflow', 4);
    const outlineTotal = resources.createBuffer('outline-total', 4);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: 'isochrones-draw',
        type: 'draw',
        commands: [
          {vertexCount: 6, instanceCount: 0},
          {vertexCount: 0, instanceCount: 1}
        ]
      })
    );

    let compiled: CompiledGPUCommandGraph<void> | null = null;
    function buildGraph(): void {
      if (compiled) resources.release(compiled);
      const choice = CELL_CHOICES[cellChoice];
      const graph = new GPUCommandGraph<void>(device, {id: `isochrones-${cellChoice}`});
      const csrOffsets = createTransientView(graph, 'csr-offsets', 'uint32', nodeCount + 1);
      const csrNeighbors = createTransientView(graph, 'csr-neighbors', 'uint32', edgeCount);
      const csrWeights = createTransientView(graph, 'csr-weights', 'float32', edgeCount);
      graph.add(
        new GPUCOOToCSR({
          id: 'isochrones-csr',
          rows: nodeCount,
          rowIndices: importGraphBuffer(graph, 'coo-rows', cooRows, 'uint32', edgeCount),
          columnIndices: importGraphBuffer(graph, 'coo-columns', cooColumns, 'uint32', edgeCount),
          values: importGraphBuffer(graph, 'coo-weights', weightsBuffer, 'float32', edgeCount),
          rowOffsets: csrOffsets,
          outputColumnIndices: csrNeighbors,
          outputValues: csrWeights
        })
      );
      const polygonPositionsView = importGraphBuffer(
        graph,
        'polygon-positions',
        polygonPositions,
        'float32x2',
        RING_VERTEX_CAPACITY
      );
      const polygonRingOffsetsView = importGraphBuffer(
        graph,
        'polygon-ring-offsets',
        polygonRingOffsets,
        'uint32',
        RING_CAPACITY + 1
      );
      const polygonOffsetsView = importGraphBuffer(
        graph,
        'polygon-offsets',
        polygonOffsets,
        'uint32',
        RING_CAPACITY + 1
      );
      const polygonFeatureOffsetsView = importGraphBuffer(
        graph,
        'polygon-feature-offsets',
        polygonFeatureOffsets,
        'uint32',
        2
      );
      const costsView = importGraphBuffer(graph, 'costs', costs, 'float32', nodeCount);
      const assignmentsView = importGraphBuffer(
        graph,
        'assignments',
        assignments,
        'uint32',
        nodeCount
      );
      const breaksView = breaks.importToGraph(graph);
      const parametersView = parameters.importToGraph(graph);
      graph.add(
        new GPUNetworkIsochrones({
          id: 'isochrones-raster',
          offsets: csrOffsets,
          neighbors: csrNeighbors,
          weights: csrWeights,
          nodePositions: importGraphBuffer(
            graph,
            'node-positions',
            nodePositionsBuffer,
            'float32x2',
            nodeCount
          ),
          costs: costsView,
          sources: sources.importToGraph(graph),
          sourceCount: sourceCount.importToGraph(graph),
          assignments: assignmentsView,
          costLimit: costLimit.importToGraph(graph),
          maxIterations: MAXIMUM_ITERATIONS,
          breaks: breaksView,
          parameters: parametersView,
          raster: {
            width: RASTER_WIDTH,
            height: RASTER_HEIGHT,
            mode: 'min',
            maximumBufferPixels: MAXIMUM_BUFFER_PIXELS,
            maximumSamplesPerEdge: MAXIMUM_SAMPLES_PER_EDGE,
            output: {
              pixelFacilities: importGraphBuffer(
                graph,
                'pixel-facilities',
                pixelFacilities,
                'uint32',
                RASTER_WIDTH * RASTER_HEIGHT
              ),
              triangleFacilities: importGraphBuffer(
                graph,
                'triangle-facilities',
                triangleFacilities,
                'uint32',
                TRIANGLE_CAPACITY
              ),
              triangles: importGraphBuffer(
                graph,
                'triangles',
                triangles,
                'float32x2',
                TRIANGLE_CAPACITY * 3
              ),
              triangleBands: importGraphBuffer(
                graph,
                'triangle-bands',
                triangleBands,
                'uint32',
                TRIANGLE_CAPACITY
              ),
              count: importGraphBuffer(graph, 'triangle-count', triangleCount, 'uint32', 1),
              overflow: importGraphBuffer(
                graph,
                'triangle-overflow',
                triangleOverflow,
                'uint32',
                1
              ),
              vertexCount: importGraphBuffer(
                graph,
                'band-vertex-count',
                bandVertexCount,
                'uint32',
                1
              )
            }
          }
        })
      );
      // The cell producer takes the same node costs as an input and needs longitude/latitude.
      graph.add(
        new GPUNetworkIsochrones({
          id: 'isochrones-cells',
          offsets: csrOffsets,
          neighbors: csrNeighbors,
          weights: csrWeights,
          nodePositions: importGraphBuffer(
            graph,
            'node-lnglat',
            nodeLngLatBuffer,
            'float32x2',
            nodeCount
          ),
          costs: costsView,
          assignments: assignmentsView,
          breaks: breaksView,
          parameters: parametersView,
          cellOutline: {
            family: choice.family,
            resolution: choice.resolution,
            byFacility: true,
            table: {
              cells: importGraphBuffer(graph, 'table-cells', tableCells, 'uint32x2', nodeCount),
              counts: importGraphBuffer(graph, 'table-counts', tableCounts, 'uint32', nodeCount),
              count: importGraphBuffer(graph, 'table-count', tableCount, 'uint32', 1),
              overflow: importGraphBuffer(graph, 'table-overflow', tableOverflow, 'uint32', 1)
            },
            rings: {
              normalizeWinding: true,
              output: {
                ringOffsets: importGraphBuffer(
                  graph,
                  'ring-offsets',
                  ringOffsets,
                  'uint32',
                  RING_CAPACITY + 1
                ),
                positions: importGraphBuffer(
                  graph,
                  'ring-positions',
                  ringPositions,
                  'float32x2',
                  RING_VERTEX_CAPACITY
                ),
                ringIsHole: importGraphBuffer(
                  graph,
                  'ring-is-hole',
                  ringIsHole,
                  'uint32',
                  RING_CAPACITY
                ),
                ringShells: importGraphBuffer(
                  graph,
                  'ring-shells',
                  ringShells,
                  'uint32',
                  RING_CAPACITY
                ),
                ringGroups: importGraphBuffer(
                  graph,
                  'ring-groups',
                  ringGroups,
                  'uint32',
                  RING_CAPACITY
                ),
                count: importGraphBuffer(graph, 'ring-count', ringCount, 'uint32', 1),
                overflow: importGraphBuffer(graph, 'ring-overflow', ringOverflow, 'uint32', 1),
                totalCount: importGraphBuffer(graph, 'ring-total', ringTotal, 'uint32', 1),
                openSegmentCount: importGraphBuffer(graph, 'ring-open', ringOpen, 'uint32', 1),
                touchingSegmentCount: importGraphBuffer(
                  graph,
                  'ring-touching',
                  ringTouching,
                  'uint32',
                  1
                ),
                polygons: {
                  positions: polygonPositionsView,
                  ringOffsets: polygonRingOffsetsView,
                  polygonOffsets: polygonOffsetsView,
                  featureOffsets: polygonFeatureOffsetsView
                }
              }
            },
            output: {
              rows: importGraphBuffer(
                graph,
                'outline-rows',
                outlineRows,
                'uint32',
                SEGMENT_CAPACITY
              ),
              cells: importGraphBuffer(
                graph,
                'outline-cells',
                outlineCells,
                'uint32x2',
                SEGMENT_CAPACITY
              ),
              edgeIndices: importGraphBuffer(
                graph,
                'outline-edges',
                outlineEdges,
                'uint32',
                SEGMENT_CAPACITY
              ),
              endpoints: importGraphBuffer(
                graph,
                'outline-endpoints',
                outlineEndpoints,
                'float32x4',
                SEGMENT_CAPACITY
              ),
              groups: importGraphBuffer(
                graph,
                'outline-groups',
                outlineGroups,
                'uint32',
                SEGMENT_CAPACITY
              ),
              count: importGraphBuffer(graph, 'outline-count', outlineCount, 'uint32', 1),
              overflow: importGraphBuffer(graph, 'outline-overflow', outlineOverflow, 'uint32', 1),
              totalCount: importGraphBuffer(graph, 'outline-total', outlineTotal, 'uint32', 1)
            }
          }
        })
      );
      // Points of interest inside the isochrone polygon, joined against the GPU-written rings.
      graph.add(
        new GPUPointInPolygonJoin({
          id: 'isochrones-demand-join',
          points: importGraphBuffer(graph, 'poi-lnglat', poiLngLatBuffer, 'float32x2', poiCount),
          polygonPositions: polygonPositionsView,
          featureOffsets: polygonFeatureOffsetsView,
          polygonOffsets: polygonOffsetsView,
          ringOffsets: polygonRingOffsetsView,
          candidateCapacity: Math.max(65536, poiCount * 8),
          pointFeatureIds: importGraphBuffer(graph, 'poi-inside', poiInside, 'uint32', poiCount),
          overflow: importGraphBuffer(graph, 'join-overflow', joinOverflow, 'uint32', 1)
        })
      );
      compiled = resources.track(graph.compile());
      dirty = true;
    }

    const writeWeights = () => {
      writeEdgeTravelSeconds(edges, transport, weights);
      weightsBuffer.write(weights);
    };
    const writeBudget = () => {
      const budgetSeconds = budgetMinutes * 60;
      costLimit.write(Float32Array.of(budgetSeconds));
      const values = new Float32Array(MAXIMUM_BREAKS);
      for (let band = 0; band < MAXIMUM_BREAKS; band++) {
        values[band] = (budgetSeconds * Math.min(band + 1, bandCount)) / bandCount;
      }
      breaks.write(values);
      dirty = true;
    };

    const facilityReadout = context.controls.addReadout('Facilities (click to add or remove)');
    const writeFacilities = () => {
      const words = new Uint32Array(MAXIMUM_FACILITIES);
      const positionValues = new Float32Array(MAXIMUM_FACILITIES * 2);
      facilityNodes.forEach((node, facility) => {
        words[facility] = node;
        positionValues.set(roads.nodePositions.subarray(node * 2, node * 2 + 2), facility * 2);
      });
      sources.write(words);
      sourceCount.write(Uint32Array.of(facilityNodes.length));
      facilityPositions.write(positionValues);
      facilityReadout.setValue(
        `${facilityNodes.length} of ${MAXIMUM_FACILITIES}: ${facilityNodes.map(node => `#${formatCount(node)}`).join(', ')}`
      );
      dirty = true;
      context.updateLayers();
    };

    context.controls.addSelect<View>({
      label: 'Polygons',
      options: [
        {value: 'bands', label: 'Raster isobands (GPUIsobands)'},
        {value: 'cells', label: 'Isochrone rings (cell outline + ring assembly)'},
        {value: 'both', label: 'Both'}
      ],
      value: view,
      onChange: value => {
        view = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<BandColor>({
      label: 'Band color (raster view)',
      options: [
        {value: 'facility', label: 'Facility hue, shaded by band'},
        {value: 'ramp', label: 'Cost ramp (near to far)'}
      ],
      value: bandColor,
      onChange: value => {
        bandColor = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<ClickAction>({
      label: 'Click action',
      options: [
        {value: 'add', label: 'Add a facility (click a facility to remove it)'},
        {value: 'replace', label: 'Replace all facilities with one'}
      ],
      value: clickAction,
      onChange: value => {
        clickAction = value;
      }
    });
    context.controls.addSelect<Transport>({
      label: 'Transport (rewrites edge weights)',
      options: [
        {value: 'walk', label: 'Walk (1.4 m/s)'},
        {value: 'drive', label: 'Drive (7-20 m/s by road class)'}
      ],
      value: transport,
      onChange: value => {
        transport = value;
        writeWeights();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Time budget (last threshold)',
      min: 2,
      max: 30,
      step: 1,
      value: budgetMinutes,
      format: value => `${value} min`,
      onChange: value => {
        budgetMinutes = value;
        writeBudget();
      }
    });
    context.controls.addSlider({
      label: 'Thresholds (active break count)',
      min: 1,
      max: MAXIMUM_BREAKS,
      step: 1,
      value: bandCount,
      format: value => `${value} bands`,
      onChange: value => {
        bandCount = value;
        writeBudget();
      }
    });
    const bufferReadout = context.controls.addReadout('Effective walking buffer');
    context.controls.addSlider({
      label: 'Walking buffer off the street',
      min: 0,
      max: 120,
      step: 5,
      value: bufferMeters,
      format: value => `${value} m`,
      onChange: value => {
        bufferMeters = value;
        dirty = true;
      }
    });
    context.controls.addSelect<CellChoice>({
      label: 'Cell outline (compile-time, rebuilds the graph)',
      options: (Object.keys(CELL_CHOICES) as CellChoice[]).map(value => ({
        value,
        label: CELL_CHOICES[value].label
      })),
      value: cellChoice,
      onChange: value => {
        cellChoice = value;
        buildGraph();
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Show street graph',
      value: showRoads,
      onChange: value => {
        showRoads = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Show points of interest inside the isochrone (join)',
      value: showDemand,
      onChange: value => {
        showDemand = value;
        context.updateLayers();
      }
    });
    context.controls.addLegend({
      title: 'Facilities (rings and bands share the hue of the nearest facility)',
      entries: FACILITY_COLORS.map((color, index) => ({color, label: `Facility ${index + 1}`}))
    });
    context.controls.addLegend({
      title: 'Cell view',
      entries: [{color: DEMAND_INSIDE_COLOR, label: 'Point of interest inside the isochrone'}]
    });
    context.controls.addLegend({
      title: 'Isochrone bands (near to far, within the time budget)',
      gradient: {
        colors: RAMP_STOPS,
        minimumLabel: 'source',
        maximumLabel: 'budget'
      }
    });
    context.controls.addNote(
      'Click the map to add facilities (up to six). Pixels past the last threshold stay empty; ' +
        'the raster follows the viewport, so zoom in for finer polygons. Rings never mix ' +
        'facilities: where two catchments meet, each ring runs along the shared border.'
    );
    context.controls.addReadout('Nodes', formatCount(nodeCount));
    context.controls.addReadout('Directed edges', formatCount(edgeCount));
    const rasterReadout = context.controls.addReadout('Raster');
    const triangleReadout = context.controls.addReadout('Band triangles');
    const cellReadout = context.controls.addReadout('Reached cells');
    const outlineReadout = context.controls.addReadout('Outline segments');
    const ringReadout = context.controls.addReadout('Rings (shells / holes)');
    const ringHealthReadout = context.controls.addReadout('Open / touching segments');
    const demandReadout = context.controls.addReadout('Points of interest inside');
    context.controls.addReadout('Data', roads.attribution);

    const summary = new SummaryReader(
      resources,
      'isochrones',
      [
        {buffer: triangleCount, size: 4},
        {buffer: triangleOverflow, size: 4},
        {buffer: tableCount, size: 4},
        {buffer: tableOverflow, size: 4},
        {buffer: outlineCount, size: 4},
        {buffer: outlineOverflow, size: 4},
        {buffer: outlineTotal, size: 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        triangleReadout.setValue(
          `${formatCount(words[0])} of ${formatCount(TRIANGLE_CAPACITY)}${words[1] ? ' OVERFLOW' : ''}`
        );
        cellReadout.setValue(`${formatCount(words[2])}${words[3] ? ' (table OVERFLOW)' : ''}`);
        outlineReadout.setValue(
          `${formatCount(words[4])} of ${formatCount(SEGMENT_CAPACITY)}` +
            `${words[5] ? ` OVERFLOW (needs ${formatCount(words[6])})` : ''}`
        );
      }
    );

    const ringSummary = new SummaryReader(
      resources,
      'isochrones-rings',
      [
        {buffer: ringCount, size: 4},
        {buffer: ringOverflow, size: 4},
        {buffer: ringTotal, size: 4},
        {buffer: ringOpen, size: 4},
        {buffer: ringTouching, size: 4},
        {buffer: joinOverflow, size: 4},
        {buffer: ringIsHole, size: RING_CAPACITY * 4},
        {buffer: poiInside, size: poiCount * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const count = Math.min(words[0], RING_CAPACITY);
        let holes = 0;
        for (let ring = 0; ring < count; ring++) holes += words[6 + ring] ? 1 : 0;
        let inside = 0;
        for (let poi = 0; poi < poiCount; poi++) {
          if (words[6 + RING_CAPACITY + poi] !== GPU_SPATIAL_JOIN_NO_FEATURE) inside++;
        }
        ringReadout.setValue(
          `${formatCount(count - holes)} / ${formatCount(holes)}` +
            `${words[1] ? ` OVERFLOW (${formatCount(words[2])} rings needed)` : ''}`
        );
        ringHealthReadout.setValue(`${formatCount(words[3])} / ${formatCount(words[4])}`);
        demandReadout.setValue(
          `${formatCount(inside)} of ${formatCount(poiCount)}${words[5] ? ' (join OVERFLOW)' : ''}`
        );
      }
    );

    writeWeights();
    writeBudget();
    writeFacilities();
    buildGraph();
    summary.markStale();

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => (compiled ? [compiled] : []),
      encode(commandEncoder: CommandEncoder, frame) {
        if (!compiled) return;
        // The raster follows the viewport. Quantize its size and origin so panning stays calm.
        const viewBounds = getViewportMetricBounds(frame.viewport, projection);
        const width =
          Math.max(viewBounds[2] - viewBounds[0], (viewBounds[3] - viewBounds[1]) * RASTER_ASPECT) *
          1.1;
        const step = Math.log(1.05);
        const snappedWidth = Math.exp(Math.round(Math.log(width) / step) * step);
        const pixelSize = snappedWidth / RASTER_WIDTH;
        const snap = pixelSize * 16;
        const centerX = Math.round((viewBounds[0] + viewBounds[2]) / 2 / snap) * snap;
        const centerY = Math.round((viewBounds[1] + viewBounds[3]) / 2 / snap) * snap;
        const nextExtent: [number, number, number, number] = [
          centerX - snappedWidth / 2,
          centerY - snappedWidth / RASTER_ASPECT / 2,
          centerX + snappedWidth / 2,
          centerY + snappedWidth / RASTER_ASPECT / 2
        ];
        const key = nextExtent.map(value => value.toFixed(2)).join(',');
        if (key !== extentKey) {
          extentKey = key;
          extent = nextExtent;
          dirty = true;
        }
        if (dirty && extent) {
          const maximumBuffer = MAXIMUM_BUFFER_PIXELS * pixelSize;
          const effectiveBuffer = Math.min(bufferMeters, maximumBuffer);
          parameters.write(
            getGPUNetworkIsochroneParameterValues({
              breakCount: bandCount,
              extent,
              bufferRadius: effectiveBuffer,
              walkCostPerUnit: 1 / WALK_SPEED,
              cellCostLimit: budgetMinutes * 60
            })
          );
          rasterReadout.setValue(
            `${RASTER_WIDTH} × ${RASTER_HEIGHT}, ${pixelSize.toFixed(1)} m per pixel`
          );
          bufferReadout.setValue(
            effectiveBuffer < bufferMeters
              ? `${effectiveBuffer.toFixed(0)} m (capped at ${MAXIMUM_BUFFER_PIXELS} pixels; zoom in)`
              : `${effectiveBuffer.toFixed(0)} m`
          );
          compiled.encode(commandEncoder, {parameters: undefined});
          const copyWord = (sourceBuffer: typeof outlineCount, record: number, wordIndex: number) =>
            commandEncoder.copyBufferToBuffer({
              sourceBuffer,
              sourceOffset: 0,
              destinationBuffer: drawCommands.buffer,
              destinationOffset: record * RECORD_BYTE_LENGTH + wordIndex * 4,
              size: 4
            });
          copyWord(outlineCount, RECORD_OUTLINE, 1);
          copyWord(bandVertexCount, RECORD_BANDS, 0);
          dirty = false;
          summary.markStale();
          ringSummary.markStale();
        }
        ringSummary.flush(commandEncoder);
        summary.flush(commandEncoder);
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [roads.origin[0], roads.origin[1], 0];
        const layers: Layer[] = [];
        if (showRoads) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'isochrones-roads',
              coordinateOrigin,
              segments: segmentsBuffer,
              instanceCount: segmentCount,
              widthPixels: 1,
              color: ROAD_COLOR
            })
          );
        }
        if (view !== 'cells') {
          layers.push(
            new IsobandTriangleLayer({
              id: `isochrones-bands-${bandColor}`,
              coordinateOrigin,
              gridSize: [1, 1],
              bounds: [0, 0, 1, 1],
              triangles,
              triangleBands,
              ...(bandColor === 'facility' ? {triangleFacilities, palette: FACILITY_COLORS} : {}),
              values: paletteBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              extent: parameters.buffer,
              drawCommands,
              drawCommandIndex: RECORD_BANDS
            })
          );
        }
        if (view !== 'bands') {
          layers.push(
            new PolylineLayer({
              id: `isochrones-rings-${cellChoice}`,
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              segments: ringPositions,
              polylineOffsets: ringOffsets,
              valueIndices: ringGroups,
              extent: ringCount,
              values: facilityIdentity,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: FACILITY_COLORS.map(color => [color[0], color[1], color[2], 255] as const),
              instanceCount: RING_VERTEX_CAPACITY,
              widthPixels: 3.5
            })
          );
        }
        if (showDemand) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'isochrones-demand',
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              positions: poiLngLatBuffer,
              instanceCount: poiCount,
              radiusPixels: 2.5,
              values: poiInside,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: [DEMAND_INSIDE_COLOR],
              noDataValue: GPU_SPATIAL_JOIN_NO_FEATURE,
              noDataColor: DEMAND_OUTSIDE_COLOR
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'isochrones-facility-halo',
            coordinateOrigin,
            positions: facilityPositions,
            instanceCount: facilityNodes.length,
            radiusPixels: 10,
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'isochrones-facilities',
            coordinateOrigin,
            positions: facilityPositions,
            instanceCount: facilityNodes.length,
            radiusPixels: 7,
            values: facilityIdentity,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: FACILITY_COLORS.map(color => [color[0], color[1], color[2], 255] as const)
          })
        );
        return layers;
      },
      onClick(event) {
        if (!event.coordinate) return false;
        const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
        const node = findNearestNode(roads.nodePositions, [x, y]);
        const existing = facilityNodes.indexOf(node);
        if (clickAction === 'replace') {
          facilityNodes.splice(0, facilityNodes.length, node);
        } else if (existing >= 0) {
          if (facilityNodes.length > 1) facilityNodes.splice(existing, 1);
        } else if (facilityNodes.length >= MAXIMUM_FACILITIES) {
          facilityNodes.shift();
          facilityNodes.push(node);
        } else {
          facilityNodes.push(node);
        }
        writeFacilities();
        return true;
      },
      destroy() {
        destroyed = true;
        summary.stop();
        ringSummary.stop();
        resources.destroy();
      }
    };
    return instance;
  }
};
