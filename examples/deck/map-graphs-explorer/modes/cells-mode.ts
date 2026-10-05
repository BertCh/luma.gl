// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * "Points to cells" over every New York trip vertex, in the style of Foursquare Studio's
 * "lng/lat column to H3" step but for five grid families.
 *
 * - Index graph: `GPUPointToCell` keys each vertex, `GPUCellAggregation` (H3 and Quadbin only)
 *   turns the keys into a sorted table of `(cell, count)`, and `GPUCellGeometry` decodes the
 *   table into boundary polygons that `CellBoundaryLayer` draws straight from storage buffers,
 *   colored by log count. Quadkey, geohash and S2 have no aggregation recipe, so those families
 *   draw one additive translucent cell per point instead (the density is the overlap).
 * - Selection graph (H3 and Quadbin): a click writes a lng/lat into a one-row buffer; the graph
 *   keys it, `GPUCellTopology` expands a grid disk of the maximum radius 8 and `GPUCellGeometry`
 *   decodes it. The k slider is a GPU-side filter on the distances column, so it never recompiles
 *   (the disk is always computed at k = 8: 217 H3 cells or 289 Quadbin tiles).
 * - Zones graph (H3 and Quadbin): `GPUCellCover` polyfills three hand-drawn Manhattan zones
 *   (one with a hole), `GPUCellCompaction` compacts the cover, and two more decodes draw both sets.
 *
 * Family, resolution and containment are compile-time (the output stride, kernels and key layouts
 * differ), so changing them rebuilds the graphs once after a short debounce; the footer counts
 * those rebuilds. Everything else (sampling mask, click position, k, toggles) is a buffer write
 * or a layer uniform. Each graph is encoded only when its inputs change: the points and zones are
 * static, so the index graph re-runs only when the sampling mask changes and the selection graph
 * only after a click.
 *
 * The vertex positions are stored as planar meters around the dataset origin; they are converted
 * to lng/lat once on the CPU because the cell recipes key longitude/latitude.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCellAggregation,
  GPUCellCompaction,
  GPUCellCover,
  GPUCellGeometry,
  GPUCellTopology,
  GPUPointToCell,
  GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS,
  getCellTopologyStride,
  importGraphBuffer,
  type GPUCellCoverContainment,
  type GPUCellIndexFamily
} from '@luma.gl/experimental/map-graphs';
import {LocalMetricProjection} from '../map-graphs-data';
import {MapGraphsSegmentLayer} from '../map-graphs-layers';
import type {
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsPointerEvent
} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {CellBoundaryLayer} from './cells-layers';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

type FamilySettings = {
  label: string;
  defaultResolution: number;
  minimumResolution: number;
  maximumResolution: number;
  /** `GPUCellAggregation`, `GPUCellTopology`, `GPUCellCover` and `GPUCellCompaction` support it. */
  tabular: boolean;
  maximumVertexCount: number;
  exactness: string;
};

const FAMILIES: Record<GPUCellIndexFamily, FamilySettings> = {
  h3: {
    label: 'H3',
    defaultResolution: 8,
    minimumResolution: 0,
    maximumResolution: 15,
    tabular: true,
    maximumVertexCount: GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
    exactness:
      'f32 vs h3-js: 0 mismatches at res 0-4, 4e-4 at res 8, 1e-3 at 9, 1.8% at 12, 28% at 15'
  },
  quadbin: {
    label: 'Quadbin',
    defaultResolution: 15,
    minimumResolution: 0,
    maximumResolution: 26,
    tabular: true,
    maximumVertexCount: 4,
    exactness: 'integer keys: bit-exact. Geometry f32: tile columns lose precision above zoom 24'
  },
  quadkey: {
    label: 'Quadkey',
    defaultResolution: 15,
    minimumResolution: 1,
    maximumResolution: 29,
    tabular: false,
    maximumVertexCount: 4,
    exactness: 'integer keys: bit-exact to zoom 26; 1.3% of rows are off by one tile at zoom 26'
  },
  geohash: {
    label: 'Geohash',
    defaultResolution: 6,
    minimumResolution: 1,
    maximumResolution: 12,
    tabular: false,
    maximumVertexCount: 4,
    exactness: 'integer keys: bit-exact for f32 inputs'
  },
  s2: {
    label: 'S2',
    defaultResolution: 14,
    minimumResolution: 0,
    maximumResolution: 30,
    tabular: false,
    maximumVertexCount: 4,
    exactness: 'f32 vs f64: 0 at level 0-10, 5e-4 at 12, 2.6e-3 at 15, 6.6e-2 at 20 (use <= 20)'
  }
};

const FAMILY_OPTIONS = (Object.keys(FAMILIES) as GPUCellIndexFamily[]).map(value => ({
  value,
  label: FAMILIES[value].tabular
    ? `${FAMILIES[value].label} (table, disk, cover)`
    : `${FAMILIES[value].label} (per-point cells)`
}));

/** Rows of the aggregated cell table (cells with at least one point). */
const TABLE_CAPACITY = 1 << 14;
/** Rows of the cover and compaction outputs. */
const COVER_CAPACITY = 1 << 14;
const CANDIDATE_CAPACITY = 1 << 20;
const DISK_RADIUS = GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS;
const REBUILD_DEBOUNCE_MILLISECONDS = 250;
const AUTO_MEASURE_FRAME = 45;
const SUMMARY_FIXED_WORDS = 11;
const ZONE_COLORS = [
  [78, 201, 255, 200],
  [255, 148, 72, 200],
  [189, 122, 255, 200]
] as const;

/**
 * Three illustrative Manhattan zones in lng/lat (a Central Park-like shape with a reservoir hole,
 * a Midtown block, and Lower Manhattan). They are hand drawn for this demo, not an official
 * boundary dataset.
 */
const ZONE_NAMES = ['Park with reservoir hole', 'Midtown', 'Lower Manhattan'] as const;
const ZONE_RINGS: readonly (readonly (readonly [number, number][])[])[] = [
  [
    [
      [-73.9818, 40.7681],
      [-73.9731, 40.7644],
      [-73.9493, 40.7968],
      [-73.9581, 40.8003]
    ],
    // Reservoir hole (octagon).
    Array.from({length: 8}, (_, index): [number, number] => [
      -73.9654 + 0.0075 * Math.cos((index / 8) * Math.PI * 2),
      40.785 + 0.0042 * Math.sin((index / 8) * Math.PI * 2)
    ])
  ],
  [
    [
      [-74.0023, 40.7473],
      [-73.9921, 40.7418],
      [-73.9776, 40.7436],
      [-73.9709, 40.7539],
      [-73.9786, 40.7623],
      [-73.9921, 40.7589],
      [-74.0009, 40.7564]
    ]
  ],
  [
    [
      [-74.0188, 40.7005],
      [-74.0021, 40.7003],
      [-73.9922, 40.7072],
      [-73.9948, 40.7121],
      [-74.0049, 40.7156],
      [-74.0148, 40.7107]
    ]
  ]
];

type Zones = {
  polygonPositions: Float32Array;
  featureOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  /** Ring edges `x0, y0, x1, y1` in planar meters around the dataset origin, for outlines. */
  outlineSegments: Float32Array;
};

function makeZones(projection: LocalMetricProjection): Zones {
  const positions: number[] = [];
  const outline: number[] = [];
  const ringOffsets = [0];
  const polygonOffsets = [0];
  const featureOffsets = [0];
  for (const rings of ZONE_RINGS) {
    for (const ring of rings) {
      for (const [longitude, latitude] of ring) positions.push(longitude, latitude);
      ringOffsets.push(positions.length / 2);
      ring.forEach(([longitude, latitude], index) => {
        const [nextLongitude, nextLatitude] = ring[(index + 1) % ring.length];
        outline.push(
          ...projection.project(longitude, latitude),
          ...projection.project(nextLongitude, nextLatitude)
        );
      });
    }
    polygonOffsets.push(ringOffsets.length - 1);
    featureOffsets.push(polygonOffsets.length - 1);
  }
  return {
    polygonPositions: Float32Array.from(positions),
    featureOffsets: Uint32Array.from(featureOffsets),
    polygonOffsets: Uint32Array.from(polygonOffsets),
    ringOffsets: Uint32Array.from(ringOffsets),
    outlineSegments: Float32Array.from(outline)
  };
}

/**
 * Imports a buffer under an ID prefixed so it cannot collide with the resources the recipes
 * create inside the same graph (for example a recipe's own `cover-total` transient).
 */
function importCellBuffer<Format extends Parameters<typeof importGraphBuffer>[3]>(
  graph: GPUCommandGraph<void>,
  name: string,
  buffer: Buffer,
  format: Format,
  length: number
) {
  return importGraphBuffer(graph, `import-${name}`, buffer, format, length);
}

/** Decoded output of one `GPUCellGeometry` node: fixed-stride boundaries plus vertex counts. */
type GeometryBuffers = {boundaries: Buffer; vertexCounts: Buffer; stride: number; rows: number};

/** Everything that depends on the compile-time choices; replaced on a rebuild. */
type Pipeline = {
  serial: number;
  family: GPUCellIndexFamily;
  resolution: number;
  containment: GPUCellCoverContainment;
  resources: MapGraphsResources;
  indexGraph: CompiledGPUCommandGraph<void>;
  zonesGraph: CompiledGPUCommandGraph<void> | null;
  selectionGraph: CompiledGPUCommandGraph<void> | null;
  /** Geometry of the table rows (tabular) or of every point (per-point families). */
  cells: GeometryBuffers;
  tableCounts: Buffer | null;
  tableCount: Buffer | null;
  tableOverflow: Buffer | null;
  tableTotal: Buffer | null;
  cover:
    | (GeometryBuffers & {featureIds: Buffer; count: Buffer; overflow: Buffer; total: Buffer})
    | null;
  compacted: (GeometryBuffers & {count: Buffer; overflow: Buffer}) | null;
  selection:
    | (GeometryBuffers & {distances: Buffer; counts: Buffer; cell: Buffer; stride: number})
    | null;
  readbackRing: GPUReadbackRing;
  summaryWords: number;
  /** Byte total of the buffers this pipeline owns, for the size readout. */
  byteLength: number;
  indexDirty: boolean;
  zonesDirty: boolean;
  selectionDirty: boolean;
  summaryRequested: boolean;
  /** Number of compiled graphs in this pipeline. */
  graphCount: number;
};

export const cellsMode: MapGraphsModeDefinition = {
  id: 'cells',
  title: 'Grid cells',
  recipes: [
    'GPUPointToCell',
    'GPUCellAggregation',
    'GPUCellGeometry',
    'GPUCellTopology',
    'GPUCellCover',
    'GPUCellCompaction'
  ],
  description:
    'Trip vertices keyed to H3, Quadbin, quadkey, geohash or S2 cells on the GPU, counted per ' +
    'cell and drawn as cell polygons. Click a cell for its grid disk; toggle the polyfill of ' +
    'three Manhattan zones and its compacted set.',
  initialViewState: {longitude: -73.985, latitude: 40.745, zoom: 11.7},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const projection = new LocalMetricProjection(trips.origin);
    const resources = new MapGraphsResources(device, 'cells');
    const pointCount = trips.vertexTimestamps.length;
    const zones = makeZones(projection);
    const zoneSegmentCount = zones.outlineSegments.length / 4;

    // Shared, static inputs. The cell recipes key longitude/latitude, so convert once.
    const lngLat = new Float32Array(pointCount * 2);
    for (let point = 0; point < pointCount; point++) {
      const [longitude, latitude] = projection.unproject(
        trips.vertexPositions[point * 2],
        trips.vertexPositions[point * 2 + 1]
      );
      lngLat[point * 2] = longitude;
      lngLat[point * 2 + 1] = latitude;
    }
    const positionsBuffer = resources.createBuffer('lng-lat', lngLat);
    const maskValues = new Uint32Array(pointCount).fill(1);
    const maskBuffer = resources.createBuffer('mask', maskValues);
    const polygonPositions = resources.createBuffer('zone-positions', zones.polygonPositions);
    const featureOffsets = resources.createBuffer('zone-feature-offsets', zones.featureOffsets);
    const polygonOffsets = resources.createBuffer('zone-polygon-offsets', zones.polygonOffsets);
    const ringOffsets = resources.createBuffer('zone-ring-offsets', zones.ringOffsets);
    const zoneOutline = resources.createBuffer('zone-outline', zones.outlineSegments);
    // The click position; NaN keys to the zero cell and draws nothing.
    const selectionPosition = resources.createBuffer(
      'selection-position',
      Float32Array.of(-73.985, 40.7589)
    );

    // Compile-time choices (rebuild) and live choices (buffer or uniform writes).
    let family: GPUCellIndexFamily = 'h3';
    let resolutionOffset = 0;
    let containment: GPUCellCoverContainment = 'center';
    let sampledPercent = 100;
    let diskRadius = 3;
    let showCells = true;
    let showZones = false;
    let showCompacted = false;
    let valueMaximum = 1;
    let serial = 0;
    let pipeline: Pipeline | null = null;
    let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
    let destroyed = false;
    let readbackPending = false;
    let encodedFrames = 0;
    let measuring = false;
    let autoMeasured = false;
    let indexEncodes = 0;
    const retiring = new Set<Pipeline>();

    const getResolution = (selectedFamily: GPUCellIndexFamily): number => {
      const settings = FAMILIES[selectedFamily];
      return Math.min(
        settings.maximumResolution,
        Math.max(settings.minimumResolution, settings.defaultResolution + resolutionOffset)
      );
    };

    const buildPipeline = (
      selectedFamily: GPUCellIndexFamily,
      resolution: number,
      selectedContainment: GPUCellCoverContainment
    ): Pipeline => {
      const settings = FAMILIES[selectedFamily];
      const {tabular, maximumVertexCount} = settings;
      const pipelineSerial = ++serial;
      const owned = new MapGraphsResources(device, `cells-${pipelineSerial}`);
      let byteLength = 0;
      const create = (name: string, size: number): Buffer => {
        byteLength += size;
        return owned.createBuffer(name, size);
      };
      const createGeometry = (name: string, rows: number, stride: number): GeometryBuffers => ({
        boundaries: create(`${name}-boundaries`, rows * stride * 8),
        vertexCounts: create(`${name}-vertex-counts`, rows * 4),
        stride,
        rows
      });

      // Index graph.
      const rowCount = tabular ? TABLE_CAPACITY : pointCount;
      const rawCells = create('raw-cells', pointCount * 8);
      const tableCells = tabular ? create('table-cells', TABLE_CAPACITY * 8) : null;
      const tableCounts = tabular ? create('table-counts', TABLE_CAPACITY * 4) : null;
      const tableCount = tabular ? create('table-count', 4) : null;
      const tableOverflow = tabular ? create('table-overflow', 4) : null;
      const tableTotal = tabular ? create('table-total', 4) : null;
      const cells = createGeometry('cells', rowCount, maximumVertexCount);
      const indexGraph = new GPUCommandGraph<void>(device, {id: `cells-index-${pipelineSerial}`});
      const rawCellsView = importCellBuffer(
        indexGraph,
        'raw-cells',
        rawCells,
        'uint32x2',
        pointCount
      );
      indexGraph.add(
        new GPUPointToCell({
          id: 'point-to-cell',
          family: selectedFamily,
          resolution,
          positions: importCellBuffer(
            indexGraph,
            'positions',
            positionsBuffer,
            'float32x2',
            pointCount
          ),
          mask: importCellBuffer(indexGraph, 'mask', maskBuffer, 'uint32', pointCount),
          output: {cells: rawCellsView}
        })
      );
      let geometryInput: GraphDataView<'uint32x2'> = rawCellsView;
      if (tabular && tableCells && tableCounts && tableCount && tableOverflow && tableTotal) {
        const tableCellsView = importCellBuffer(
          indexGraph,
          'table-cells',
          tableCells,
          'uint32x2',
          TABLE_CAPACITY
        );
        indexGraph.add(
          new GPUCellAggregation({
            id: 'aggregate',
            family: selectedFamily as 'h3' | 'quadbin',
            resolution,
            cells: rawCellsView,
            output: {
              cells: tableCellsView,
              counts: importCellBuffer(
                indexGraph,
                'table-counts',
                tableCounts,
                'uint32',
                TABLE_CAPACITY
              ),
              count: importCellBuffer(indexGraph, 'table-count', tableCount, 'uint32', 1),
              overflow: importCellBuffer(indexGraph, 'table-overflow', tableOverflow, 'uint32', 1),
              totalCount: importCellBuffer(indexGraph, 'table-total', tableTotal, 'uint32', 1)
            }
          })
        );
        geometryInput = tableCellsView;
      }
      indexGraph.add(
        new GPUCellGeometry({
          id: 'cell-geometry',
          family: selectedFamily,
          cells: geometryInput,
          maximumVertexCount,
          output: {
            boundaries: importCellBuffer(
              indexGraph,
              'boundaries',
              cells.boundaries,
              'float32x2',
              rowCount * maximumVertexCount
            ),
            vertexCounts: importCellBuffer(
              indexGraph,
              'vertex-counts',
              cells.vertexCounts,
              'uint32',
              rowCount
            )
          }
        })
      );
      const compiledIndex = owned.track(indexGraph.compile());

      let zonesGraph: CompiledGPUCommandGraph<void> | null = null;
      let selectionGraph: CompiledGPUCommandGraph<void> | null = null;
      let cover: Pipeline['cover'] = null;
      let compacted: Pipeline['compacted'] = null;
      let selection: Pipeline['selection'] = null;
      if (tabular) {
        const tabularFamily = selectedFamily as 'h3' | 'quadbin';
        // Zones graph: cover, compaction and two decodes.
        const coverCells = create('cover-cells', COVER_CAPACITY * 8);
        const coverFeatureIds = create('cover-feature-ids', COVER_CAPACITY * 4);
        const coverCount = create('cover-count', 4);
        const coverOverflow = create('cover-overflow', 4);
        const coverTotal = create('cover-total', 4);
        const compactCells = create('compact-cells', COVER_CAPACITY * 8);
        const compactCount = create('compact-count', 4);
        const compactOverflow = create('compact-overflow', 4);
        const coverGeometry = createGeometry('cover', COVER_CAPACITY, maximumVertexCount);
        const compactGeometry = createGeometry('compact', COVER_CAPACITY, maximumVertexCount);
        const graph = new GPUCommandGraph<void>(device, {id: `cells-zones-${pipelineSerial}`});
        const coverCellsView = importCellBuffer(
          graph,
          'cover-cells',
          coverCells,
          'uint32x2',
          COVER_CAPACITY
        );
        const coverCountView = importCellBuffer(graph, 'cover-count', coverCount, 'uint32', 1);
        const compactCellsView = importCellBuffer(
          graph,
          'compact-cells',
          compactCells,
          'uint32x2',
          COVER_CAPACITY
        );
        graph.add(
          new GPUCellCover({
            id: 'cover',
            family: tabularFamily,
            resolution,
            containment: selectedContainment,
            polygonPositions: importCellBuffer(
              graph,
              'zone-positions',
              polygonPositions,
              'float32x2',
              zones.polygonPositions.length / 2
            ),
            featureOffsets: importCellBuffer(
              graph,
              'zone-feature-offsets',
              featureOffsets,
              'uint32',
              zones.featureOffsets.length
            ),
            polygonOffsets: importCellBuffer(
              graph,
              'zone-polygon-offsets',
              polygonOffsets,
              'uint32',
              zones.polygonOffsets.length
            ),
            ringOffsets: importCellBuffer(
              graph,
              'zone-ring-offsets',
              ringOffsets,
              'uint32',
              zones.ringOffsets.length
            ),
            candidateCapacity: CANDIDATE_CAPACITY,
            output: {
              featureIds: importCellBuffer(
                graph,
                'cover-feature-ids',
                coverFeatureIds,
                'uint32',
                COVER_CAPACITY
              ),
              cells: coverCellsView,
              count: coverCountView,
              overflow: importCellBuffer(graph, 'cover-overflow', coverOverflow, 'uint32', 1),
              totalCount: importCellBuffer(graph, 'cover-total', coverTotal, 'uint32', 1)
            }
          })
        );
        graph.add(
          new GPUCellCompaction({
            id: 'compact',
            family: tabularFamily,
            operation: {type: 'compact', resolution},
            cells: coverCellsView,
            count: coverCountView,
            output: {
              cells: compactCellsView,
              count: importCellBuffer(graph, 'compact-count', compactCount, 'uint32', 1),
              overflow: importCellBuffer(graph, 'compact-overflow', compactOverflow, 'uint32', 1)
            }
          })
        );
        for (const [name, view, geometry] of [
          ['cover', coverCellsView, coverGeometry],
          ['compact', compactCellsView, compactGeometry]
        ] as const) {
          graph.add(
            new GPUCellGeometry({
              id: `${name}-geometry`,
              family: tabularFamily,
              cells: view,
              maximumVertexCount,
              output: {
                boundaries: importCellBuffer(
                  graph,
                  `${name}-boundaries`,
                  geometry.boundaries,
                  'float32x2',
                  COVER_CAPACITY * maximumVertexCount
                ),
                vertexCounts: importCellBuffer(
                  graph,
                  `${name}-vertex-counts`,
                  geometry.vertexCounts,
                  'uint32',
                  COVER_CAPACITY
                )
              }
            })
          );
        }
        zonesGraph = owned.track(graph.compile());
        cover = {
          ...coverGeometry,
          featureIds: coverFeatureIds,
          count: coverCount,
          overflow: coverOverflow,
          total: coverTotal
        };
        compacted = {...compactGeometry, count: compactCount, overflow: compactOverflow};

        // Selection graph: click -> cell -> grid disk -> decode.
        const diskStride = getCellTopologyStride(tabularFamily, {type: 'disk', k: DISK_RADIUS});
        const selectedCell = create('selected-cell', 8);
        const diskCells = create('disk-cells', diskStride * 8);
        const diskDistances = create('disk-distances', diskStride * 4);
        const diskCounts = create('disk-counts', 4);
        const diskGeometry = createGeometry('disk', diskStride, maximumVertexCount);
        const selectionBuilder = new GPUCommandGraph<void>(device, {
          id: `cells-selection-${pipelineSerial}`
        });
        const selectedCellView = importCellBuffer(
          selectionBuilder,
          'selected-cell',
          selectedCell,
          'uint32x2',
          1
        );
        const diskCellsView = importCellBuffer(
          selectionBuilder,
          'disk-cells',
          diskCells,
          'uint32x2',
          diskStride
        );
        selectionBuilder.add(
          new GPUPointToCell({
            id: 'selection-key',
            family: selectedFamily,
            resolution,
            positions: importCellBuffer(
              selectionBuilder,
              'selection-position',
              selectionPosition,
              'float32x2',
              1
            ),
            output: {cells: selectedCellView}
          })
        );
        selectionBuilder.add(
          new GPUCellTopology({
            id: 'disk',
            family: tabularFamily,
            operation: {type: 'disk', k: DISK_RADIUS},
            cells: selectedCellView,
            output: {
              cells: diskCellsView,
              distances: importCellBuffer(
                selectionBuilder,
                'disk-distances',
                diskDistances,
                'uint32',
                diskStride
              ),
              counts: importCellBuffer(selectionBuilder, 'disk-counts', diskCounts, 'uint32', 1)
            }
          })
        );
        selectionBuilder.add(
          new GPUCellGeometry({
            id: 'disk-geometry',
            family: tabularFamily,
            cells: diskCellsView,
            maximumVertexCount,
            output: {
              boundaries: importCellBuffer(
                selectionBuilder,
                'disk-boundaries',
                diskGeometry.boundaries,
                'float32x2',
                diskStride * maximumVertexCount
              ),
              vertexCounts: importCellBuffer(
                selectionBuilder,
                'disk-vertex-counts',
                diskGeometry.vertexCounts,
                'uint32',
                diskStride
              )
            }
          })
        );
        selectionGraph = owned.track(selectionBuilder.compile());
        selection = {
          ...diskGeometry,
          distances: diskDistances,
          counts: diskCounts,
          cell: selectedCell,
          stride: maximumVertexCount
        };
      }

      const summaryWords = SUMMARY_FIXED_WORDS + (tabular ? TABLE_CAPACITY : 0);
      const readbackRing = owned.track(
        new GPUReadbackRing(device, {
          id: `cells-summary-${pipelineSerial}`,
          byteLength: summaryWords * 4
        })
      );
      return {
        serial: pipelineSerial,
        family: selectedFamily,
        resolution,
        containment: selectedContainment,
        resources: owned,
        indexGraph: compiledIndex,
        zonesGraph,
        selectionGraph,
        cells,
        tableCounts,
        tableCount,
        tableOverflow,
        tableTotal,
        cover,
        compacted,
        selection,
        readbackRing,
        summaryWords,
        byteLength,
        indexDirty: true,
        zonesDirty: true,
        selectionDirty: true,
        summaryRequested: true,
        graphCount: 1 + (zonesGraph ? 1 : 0) + (selectionGraph ? 1 : 0)
      };
    };

    // Controls.
    context.controls.addSelect<GPUCellIndexFamily>({
      label: 'Cell family (compile-time: rebuilds the graphs)',
      options: FAMILY_OPTIONS,
      value: family,
      onChange: value => {
        family = value;
        resolutionOffset = 0;
        resolutionControl.setValue(0);
        updateResolutionLabel();
        scheduleRebuild();
      }
    });
    const resolutionControl = context.controls.addSlider({
      label: 'Resolution offset (compile-time: rebuilds)',
      min: -2,
      max: 2,
      step: 1,
      value: resolutionOffset,
      format: value => {
        const settings = FAMILIES[family];
        const resolution = Math.min(
          settings.maximumResolution,
          Math.max(settings.minimumResolution, settings.defaultResolution + value)
        );
        return `${value >= 0 ? '+' : ''}${value} (${settings.label} ${resolution})`;
      },
      onChange: value => {
        resolutionOffset = value;
        updateResolutionLabel();
        scheduleRebuild();
      }
    });
    const updateResolutionLabel = () => {
      resolutionControl.setValue(resolutionOffset);
    };
    const containmentControl = context.controls.addSelect<GPUCellCoverContainment>({
      label: 'Cover containment (compile-time; H3 supports center only)',
      options: [
        {value: 'center', label: 'center: cell center inside'},
        {value: 'full', label: 'full: cell inside (Quadbin)'},
        {value: 'intersects', label: 'intersects: any overlap (Quadbin)'}
      ],
      value: containment,
      onChange: value => {
        containment = value;
        scheduleRebuild();
      }
    });
    context.controls.addSlider({
      label: 'Sampled points (mask write, re-encodes the index graph)',
      min: 5,
      max: 100,
      step: 5,
      value: sampledPercent,
      format: value => `${value}%`,
      onChange: value => {
        sampledPercent = value;
        for (let point = 0; point < pointCount; point++) {
          maskValues[point] = hashToUnit(point) * 100 < sampledPercent ? 1 : 0;
        }
        maskBuffer.write(maskValues);
        if (pipeline) {
          pipeline.indexDirty = true;
          pipeline.summaryRequested = true;
        }
      }
    });
    context.controls.addSlider({
      label: 'Selected cell grid disk k (GPU distance filter, no recompile)',
      min: 0,
      max: DISK_RADIUS,
      step: 1,
      value: diskRadius,
      format: value => `k = ${value}`,
      onChange: value => {
        diskRadius = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Cells colored by point count',
      value: showCells,
      onChange: value => {
        showCells = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Zone polyfill (GPUCellCover)',
      value: showZones,
      onChange: value => {
        showZones = value;
        context.updateLayers();
      }
    });
    context.controls.addToggle({
      label: 'Compacted cover (GPUCellCompaction)',
      value: showCompacted,
      onChange: value => {
        showCompacted = value;
        context.updateLayers();
      }
    });
    context.controls.addButton({
      label: 'Measure graphs (outside frame)',
      onClick: () => void measureGraphs()
    });
    context.controls.addLegend({
      title: 'Points per cell (log scale to the busiest cell)',
      gradient: {
        colors: [
          [40, 12, 90],
          [150, 40, 100],
          [237, 105, 37],
          [252, 255, 164]
        ],
        minimumLabel: '1',
        maximumLabel: 'max'
      }
    });
    context.controls.addLegend({
      title: 'Zone cover (fill), selected disk (cyan, fades with distance)',
      entries: [
        ...ZONE_NAMES.map((name, index) => ({color: ZONE_COLORS[index], label: name})),
        {color: [40, 220, 255], label: 'Grid disk of the clicked cell'}
      ]
    });
    context.controls.addReadout('Points', formatCount(pointCount));
    const gridReadout = context.controls.addReadout('Family / resolution');
    const exactnessReadout = context.controls.addReadout('f32 accuracy');
    const cellsReadout = context.controls.addReadout('Occupied cells');
    const binnedReadout = context.controls.addReadout('Points binned');
    const maximumReadout = context.controls.addReadout('Busiest cell');
    const selectionReadout = context.controls.addReadout('Clicked cell');
    const diskReadout = context.controls.addReadout('Disk cells at k = 8');
    const coverReadout = context.controls.addReadout('Cover cells');
    const compactReadout = context.controls.addReadout('Compacted cells');
    const sizeReadout = context.controls.addReadout('GPU buffers');
    const encodeReadout = context.controls.addReadout('Encodes (index / total graphs)');
    const indexTimingReadout = context.controls.addReadout('Index graph');
    const zonesTimingReadout = context.controls.addReadout('Zones graph');
    const selectionTimingReadout = context.controls.addReadout('Selection graph');
    context.controls.addReadout('Data', `${trips.attribution}; illustrative hand-drawn zones`);
    context.controls.addNote(
      'Family, resolution and containment change kernels and output strides, so they rebuild ' +
        'the graphs (counted in the footer). The disk is always computed at k = 8 and the slider ' +
        'only filters its distances column. Quadkey, geohash and S2 have no aggregation, ' +
        'topology or cover recipe: each point draws its own cell additively. Cell keys are ' +
        'integer-exact for Quadbin, quadkey and geohash; H3 and S2 use f32 sphere math, so use ' +
        'H3 res <= 12 and S2 level <= 20.'
    );

    const scheduleRebuild = () => {
      clearTimeout(rebuildTimer);
      rebuildTimer = setTimeout(() => {
        if (!destroyed) rebuild();
      }, REBUILD_DEBOUNCE_MILLISECONDS);
    };

    const retire = (retired: Pipeline | null) => {
      if (!retired) return;
      retiring.add(retired);
      // Deck may still draw the previous layers for a frame or two; free after they are replaced.
      let frames = 0;
      const wait = () => {
        if (++frames < 4) {
          requestAnimationFrame(wait);
        } else if (retiring.delete(retired)) {
          retired.resources.destroy();
        }
      };
      requestAnimationFrame(wait);
    };

    const rebuild = () => {
      const resolution = getResolution(family);
      const effectiveContainment = family === 'h3' ? 'center' : containment;
      if (family === 'h3' && containment !== 'center') {
        containment = 'center';
        containmentControl.setValue('center');
      }
      const previous = pipeline;
      pipeline = buildPipeline(family, resolution, effectiveContainment);
      retire(previous);
      const settings = FAMILIES[family];
      gridReadout.setValue(`${settings.label} ${resolution}`);
      exactnessReadout.setValue(settings.exactness);
      containmentControl.setDisabled(family !== 'quadbin');
      sizeReadout.setValue(
        `${(pipeline.byteLength / 1e6).toFixed(1)} MB (+ ${(
          (lngLat.byteLength + maskValues.byteLength) / 1e6
        ).toFixed(1)} MB points)`
      );
      if (!settings.tabular) {
        cellsReadout.setValue('per-point cells (no aggregation recipe for this family)');
        binnedReadout.setValue('-');
        maximumReadout.setValue('-');
        selectionReadout.setValue('H3 / Quadbin only');
        diskReadout.setValue('-');
        coverReadout.setValue('-');
        compactReadout.setValue('-');
      }
      context.updateLayers();
    };

    const hexKey = (low: number, high: number): string =>
      `0x${((BigInt(high) << 32n) | BigInt(low)).toString(16)}`;

    /** Copies the small summaries into a readback ticket and updates the readouts. */
    const readSummary = async (
      active: Pipeline,
      commandEncoder: Parameters<MapGraphsModeInstance['encode']>[0]
    ) => {
      const ticket = active.readbackRing.tryAcquire();
      if (!ticket) return;
      active.summaryRequested = false;
      const sources: [Buffer | null, number][] = [
        [active.tableCount, 0],
        [active.tableOverflow, 1],
        [active.tableTotal, 2],
        [active.cover?.count ?? null, 3],
        [active.cover?.overflow ?? null, 4],
        [active.cover?.total ?? null, 5],
        [active.compacted?.count ?? null, 6],
        [active.compacted?.overflow ?? null, 7],
        [active.selection?.counts ?? null, 8]
      ];
      for (const [source, word] of sources) {
        if (!source) continue;
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: source,
          destinationBuffer: ticket.buffer,
          destinationOffset: word * 4,
          size: 4
        });
      }
      if (active.selection) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: active.selection.cell,
          destinationBuffer: ticket.buffer,
          destinationOffset: 9 * 4,
          size: 8
        });
      }
      if (active.tableCounts) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: active.tableCounts,
          destinationBuffer: ticket.buffer,
          destinationOffset: SUMMARY_FIXED_WORDS * 4,
          size: TABLE_CAPACITY * 4
        });
      }
      ticket.markEncoded({byteOffset: 0, byteLength: active.summaryWords * 4});
      readbackPending = true;
      try {
        const bytes = await ticket.read();
        if (destroyed || active !== pipeline) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, active.summaryWords);
        if (active.tableCounts) {
          const occupied = words[0];
          let sum = 0;
          let maximum = 0;
          for (let row = 0; row < occupied; row++) {
            const value = words[SUMMARY_FIXED_WORDS + row];
            sum += value;
            if (value > maximum) maximum = value;
          }
          cellsReadout.setValue(
            `${formatCount(occupied)} of ${formatCount(TABLE_CAPACITY)}` +
              (words[1] ? ` (OVERFLOW, ${formatCount(words[2])} unclamped)` : '')
          );
          binnedReadout.setValue(
            `${formatCount(sum)} (${(sum / Math.max(occupied, 1)).toFixed(1)} per cell)`
          );
          maximumReadout.setValue(`${formatCount(maximum)} points`);
          const nextMaximum = Math.max(1, maximum);
          if (nextMaximum !== valueMaximum) {
            valueMaximum = nextMaximum;
            context.updateLayers();
          }
        }
        if (active.selection) {
          selectionReadout.setValue(hexKey(words[9], words[10]));
          diskReadout.setValue(formatCount(words[8]));
        }
        if (active.cover) {
          coverReadout.setValue(
            `${formatCount(words[3])} (${formatCount(words[5])} unclamped${words[4] ? ', OVERFLOW' : ''})`
          );
          const fewer = words[3] > 0 ? (1 - words[6] / words[3]) * 100 : 0;
          compactReadout.setValue(
            `${formatCount(words[6])} (${fewer.toFixed(0)}% fewer${words[7] ? ', OVERFLOW' : ''})`
          );
        }
      } catch {
        // The ring or device was destroyed while the read was in flight.
      } finally {
        readbackPending = false;
      }
    };

    /** Times each compiled graph between frames: GPU timestamps when available. */
    const measureGraphs = async () => {
      const active = pipeline;
      if (measuring || destroyed || !active) return;
      measuring = true;
      const rows: [CompiledGPUCommandGraph<void> | null, typeof indexTimingReadout][] = [
        [active.indexGraph, indexTimingReadout],
        [active.zonesGraph, zonesTimingReadout],
        [active.selectionGraph, selectionTimingReadout]
      ];
      try {
        for (const [graph, readout] of rows) {
          if (!graph) {
            readout.setValue('n/a for this family');
            continue;
          }
          readout.setValue('measuring...');
          const timing = await measureCompiledGraph(device, graph, {
            parameters: undefined,
            completionBuffer: active.tableOverflow ?? active.cells.vertexCounts,
            signal: context.signal
          });
          if (destroyed || active !== pipeline) return;
          readout.setValue(
            `${graph.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(timing)}`
          );
        }
      } catch {
        // Interrupted by a rebuild or destroy; the next measurement replaces this one.
      } finally {
        measuring = false;
      }
    };

    rebuild();
    updateResolutionLabel();

    const instance: MapGraphsModeInstance = {
      getCompiledGraphs: () => {
        const active = pipeline;
        if (!active) return [];
        return [active.indexGraph, active.zonesGraph, active.selectionGraph].filter(
          (graph): graph is CompiledGPUCommandGraph<void> => graph !== null
        ) as CompiledGPUCommandGraph<unknown>[];
      },
      encode(commandEncoder, frame) {
        const active = pipeline;
        if (!active) return;
        encodedFrames++;
        // Inputs are static: each graph re-encodes only when its own inputs changed.
        if (active.indexDirty) {
          active.indexGraph.encode(commandEncoder, {parameters: undefined});
          active.indexDirty = false;
          indexEncodes++;
        }
        if (active.zonesGraph && active.zonesDirty) {
          active.zonesGraph.encode(commandEncoder, {parameters: undefined});
          active.zonesDirty = false;
        }
        if (active.selectionGraph && active.selectionDirty) {
          active.selectionGraph.encode(commandEncoder, {parameters: undefined});
          active.selectionDirty = false;
          active.summaryRequested = true;
        }
        if (active.summaryRequested && !readbackPending && frame.frameIndex > 0) {
          void readSummary(active, commandEncoder);
        }
        encodeReadout.setValue(`${indexEncodes} index, ${active.graphCount} graphs`);
        if (!autoMeasured && encodedFrames >= AUTO_MEASURE_FRAME) {
          autoMeasured = true;
          void measureGraphs();
        }
      },
      getLayers() {
        const active = pipeline;
        if (!active) return [];
        const tabular = FAMILIES[active.family].tabular;
        const layers: Layer[] = [];
        if (showCells) {
          layers.push(
            new CellBoundaryLayer({
              id: `cells-fill-${active.serial}`,
              boundaries: active.cells.boundaries,
              vertexCounts: active.cells.vertexCounts,
              stride: active.cells.stride,
              instanceCount: active.cells.rows,
              rowCounts: active.tableCount,
              values: active.tableCounts,
              mode: 'fill',
              colorMode: tabular ? 'count' : 'uniform',
              color: tabular ? [255, 255, 255, 190] : [255, 190, 90, 3],
              valueMaximum,
              additive: !tabular
            })
          );
          if (tabular) {
            layers.push(
              new CellBoundaryLayer({
                id: `cells-outline-${active.serial}`,
                boundaries: active.cells.boundaries,
                vertexCounts: active.cells.vertexCounts,
                stride: active.cells.stride,
                instanceCount: active.cells.rows,
                rowCounts: active.tableCount,
                mode: 'outline',
                color: [10, 10, 25, 120],
                widthPixels: 0.8
              })
            );
          }
        }
        if (tabular && showZones && active.cover && active.compacted) {
          layers.push(
            new MapGraphsSegmentLayer({
              id: `cells-zone-outline-${active.serial}`,
              coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
              segments: zoneOutline,
              instanceCount: zoneSegmentCount,
              widthPixels: 2.5,
              color: [255, 255, 255, 235]
            }),
            new CellBoundaryLayer({
              id: `cells-cover-fill-${active.serial}`,
              boundaries: active.cover.boundaries,
              vertexCounts: active.cover.vertexCounts,
              stride: active.cover.stride,
              instanceCount: COVER_CAPACITY,
              rowCounts: active.cover.count,
              values: active.cover.featureIds,
              mode: 'fill',
              colorMode: 'category',
              palette: ZONE_COLORS,
              opacity: showCompacted ? 0.45 : 1
            }),
            new CellBoundaryLayer({
              id: `cells-cover-outline-${active.serial}`,
              boundaries: active.cover.boundaries,
              vertexCounts: active.cover.vertexCounts,
              stride: active.cover.stride,
              instanceCount: COVER_CAPACITY,
              rowCounts: active.cover.count,
              mode: 'outline',
              color: [255, 255, 255, showCompacted ? 70 : 170],
              widthPixels: 1
            })
          );
          if (showCompacted) {
            layers.push(
              new CellBoundaryLayer({
                id: `cells-compact-outline-${active.serial}`,
                boundaries: active.compacted.boundaries,
                vertexCounts: active.compacted.vertexCounts,
                stride: active.compacted.stride,
                instanceCount: COVER_CAPACITY,
                rowCounts: active.compacted.count,
                mode: 'outline',
                color: [255, 244, 120, 255],
                widthPixels: 3
              })
            );
          }
        }
        if (tabular && active.selection) {
          layers.push(
            new CellBoundaryLayer({
              id: `cells-disk-fill-${active.serial}`,
              boundaries: active.selection.boundaries,
              vertexCounts: active.selection.vertexCounts,
              stride: active.selection.stride,
              instanceCount: active.selection.rows,
              values: active.selection.distances,
              mode: 'fill',
              colorMode: 'distance',
              color: [40, 220, 255, 140],
              filterMaximum: diskRadius
            }),
            new CellBoundaryLayer({
              id: `cells-disk-outline-${active.serial}`,
              boundaries: active.selection.boundaries,
              vertexCounts: active.selection.vertexCounts,
              stride: active.selection.stride,
              instanceCount: active.selection.rows,
              values: active.selection.distances,
              mode: 'outline',
              colorMode: 'distance',
              color: [255, 255, 255, 255],
              filterMaximum: diskRadius,
              widthPixels: 1.6
            })
          );
        }
        return layers;
      },
      onClick(event: MapGraphsPointerEvent) {
        const active = pipeline;
        if (!event.coordinate || !active?.selectionGraph) return false;
        selectionPosition.write(Float32Array.of(event.coordinate[0], event.coordinate[1]));
        active.selectionDirty = true;
        return true;
      },
      destroy() {
        destroyed = true;
        clearTimeout(rebuildTimer);
        pipeline?.resources.destroy();
        for (const retired of retiring) retired.resources.destroy();
        retiring.clear();
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Deterministic per-point hash in [0, 1) used to thin the mask. */
function hashToUnit(index: number): number {
  let hash = Math.imul(index ^ 0x9e3779b9, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return (hash >>> 0) / 4294967296;
}
