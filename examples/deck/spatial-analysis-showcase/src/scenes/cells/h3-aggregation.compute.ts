// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Chicago observations keyed to discrete global grid cells on the GPU.
 *
 * Three groups of compiled graphs, each keyed only by what is really compile-time:
 *
 * - Index graphs (family, resolution): `GPUPointToCell` keys every observation, `GPUCellAggregation`
 *   (H3 and Quadbin) turns the keys into a sorted `(cell, count)` table, `GPUCellGeometry` decodes
 *   the table into boundary polygons that `CellBoundaryLayer` draws from storage buffers, and two
 *   `GPUCellSetOutline` graphs trace the boundary of the occupied set (plain, with ring assembly,
 *   and with a count-tier group label). Quadkey, geohash and S2 have no aggregation contributor, so
 *   they draw one additive translucent cell per observation.
 * - Zones graph (family, resolution, containment, compaction depth): `GPUCellCover` polyfills the
 *   77 community areas, `GPUCellCompaction` compacts the cover and uncompacts it again, and
 *   `GPUCellGeometry` decodes the cover and the compacted set.
 * - Selection graph (lazily compiled per operation): a click is keyed by `GPUPointToCell`, expanded
 *   by `GPUCellTopology` (disk, ring, parent or children) and decoded by `GPUCellGeometry`.
 *
 * The observation mask (category, months, sample) is a buffer write and only re-encodes the index graph.
 */

import {COORDINATE_SYSTEM} from '@deck.gl/core';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUCellAggregation,
  GPUCellCompaction,
  GPUCellCover,
  GPUCellGeometry,
  GPUCellSetOutline,
  GPUCellTopology,
  GPUPointToCell,
  GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS,
  getCellTopologyStride,
  type GPUCellCoverContainment,
  type GPUCellIndexFamily,
  type GPUCellTopologyOperation
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {findArea, readAreaPolygons} from './b8-areas';
import type {SceneContext, SceneInstance} from '../scene';
import {CellBoundaryLayer, RingOutlineLayer} from './b8-cell-layers';
import {
  fillObservationMask,
  formatCellKey,
  formatMonthWindow,
  packKeyWords,
  readNatureCells,
  type NatureCells
} from './b8-common';

/** Option state of the h3-aggregation scene. */
export type H3AggregationOptions = {
  family: GPUCellIndexFamily;
  resolutionOffset: number;
  category: string;
  months: readonly [number, number];
  sampledPercent: number;
  ramp: RampName;
  scale: 'sqrt' | 'linear';
  ceiling: number;
  showCells: boolean;
  outline: 'off' | 'boundary' | 'rings' | 'groups';
  showSelection: boolean;
  selectionOperation: 'disk' | 'ring' | 'parent' | 'children';
  diskRadius: number;
  ringRadius: number;
  levelsUp: number;
  levelsDown: number;
  showCover: boolean;
  containment: GPUCellCoverContainment;
  coverColor: 'area' | 'core';
  showCompacted: boolean;
  compactDepth: number;
};

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

/** Per-family grid settings. Resolutions are the family's own: H3 0-15, Quadbin 0-26, and so on. */
export const FAMILIES: Record<GPUCellIndexFamily, FamilySettings> = {
  h3: {
    label: 'H3',
    defaultResolution: 8,
    minimumResolution: 0,
    maximumResolution: 15,
    tabular: true,
    maximumVertexCount: GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
    exactness: 'f32 vs h3-js: exact at res 0-4, ~4e-4 of points off by one cell at res 8, 1e-3 at 9'
  },
  quadbin: {
    label: 'Quadbin',
    defaultResolution: 15,
    minimumResolution: 0,
    maximumResolution: 26,
    tabular: true,
    maximumVertexCount: 4,
    exactness: 'integer keys: bit-exact'
  },
  quadkey: {
    label: 'Quadkey',
    defaultResolution: 15,
    minimumResolution: 1,
    maximumResolution: 29,
    tabular: false,
    maximumVertexCount: 4,
    exactness: 'integer keys: bit-exact (1.3% off by one tile at zoom 26)'
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
    exactness: 'f32 vs f64: exact to level 10, ~5e-4 at 12, 2.6e-3 at 15 (use <= 20)'
  }
};

/** Resolution of a family for a resolution offset from its default. */
export function getResolution(family: GPUCellIndexFamily, offset: number): number {
  const settings = FAMILIES[family];
  return Math.min(
    settings.maximumResolution,
    Math.max(settings.minimumResolution, settings.defaultResolution + offset)
  );
}

const H3_EDGE_KILOMETERS = [
  1107.7, 418.7, 158.2, 59.81, 22.61, 8.544, 3.229, 1.221, 0.461, 0.174, 0.0659, 0.0249, 0.0094,
  0.0036, 0.0013, 0.0005
];
const GEOHASH_SIZES = [
  '5000 x 5000 km',
  '1250 x 625 km',
  '156 x 156 km',
  '39 x 19.5 km',
  '4.9 x 4.9 km',
  '1.2 x 0.6 km',
  '153 x 153 m',
  '38 x 19 m',
  '4.8 x 4.8 m',
  '1.2 x 0.6 m',
  '15 x 15 cm',
  '4 x 2 cm'
];

/** Human readable cell size at Chicago's latitude. */
export function describeCellSize(family: GPUCellIndexFamily, resolution: number): string {
  const format = (kilometers: number) =>
    kilometers >= 1 ? `${kilometers.toFixed(2)} km` : `${Math.round(kilometers * 1000)} m`;
  if (family === 'h3') return `edge ~${format(H3_EDGE_KILOMETERS[resolution])}`;
  if (family === 'geohash') return `${GEOHASH_SIZES[resolution - 1]}`;
  if (family === 's2') {
    return `edge ~${format(Math.sqrt(510.1e6 / (6 * 4 ** resolution)))}`;
  }
  return `tile ~${format((40075.017 / 2 ** resolution) * Math.cos((41.88 * Math.PI) / 180))} wide`;
}

/** Rows of the aggregated cell table (cells with at least one observation). */
const TABLE_CAPACITY = 1 << 16;
/** Rows of the cover, compaction and uncompaction outputs. */
const COVER_CAPACITY = 1 << 16;
const CANDIDATE_CAPACITY = 1 << 20;
const DISK_RADIUS = GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS;
const REBUILD_DEBOUNCE_MILLISECONDS = 250;
/** Segments of each `GPUCellSetOutline` output. */
const OUTLINE_CAPACITY = 1 << 17;
/** Rings of the assembled outline. */
const RING_CAPACITY = 1 << 12;

/** Count tiers of the group-border outline: `min(floor(log2(count)), 4)`. */
export const TIER_COLORS = [
  [90, 110, 190, 255],
  [70, 190, 210, 255],
  [110, 220, 130, 255],
  [250, 210, 80, 255],
  [255, 110, 90, 255]
] as const;
export const TIER_LABELS = ['1 observation', '2-3', '4-7', '8-15', '16 or more'] as const;
/** Cover fill colors by `core` flag: border cells, then core cells. */
export const CORE_COLORS = [
  [255, 150, 60, 215],
  [60, 215, 140, 215]
] as const;
/** Colors of the assembled outline rings, by ring index. */
export const RING_COLORS = [
  [255, 255, 255, 255],
  [255, 214, 64, 255],
  [120, 230, 255, 255],
  [255, 150, 200, 255]
] as const;
/** Community-area cover colors (area index modulo 8). */
export const AREA_COLORS = [
  [78, 201, 255, 190],
  [255, 148, 72, 190],
  [189, 122, 255, 190],
  [87, 235, 168, 190],
  [255, 105, 168, 190],
  [245, 220, 87, 190],
  [107, 158, 255, 190],
  [255, 120, 100, 190]
] as const;
export const SELECTION_COLOR = [40, 220, 255, 150] as const;

type GeometryBuffers = {boundaries: Buffer; vertexCounts: Buffer; stride: number; rows: number};

type OutlineBuffers = {
  rings: {
    offsets: Buffer;
    counts: Buffer;
    positions: Buffer;
    count: Buffer;
    overflow: Buffer;
    openSegments: Buffer;
  } | null;
  endpoints: Buffer;
  groups: Buffer | null;
  count: Buffer;
  overflow: Buffer;
  drawCommands: DrawCommandBuffer;
};

type IndexPipeline = {
  serial: number;
  family: GPUCellIndexFamily;
  resolution: number;
  resources: SpatialAnalysisResources;
  indexGraph: CompiledGPUCommandGraph<void>;
  outlineGraphs: readonly CompiledGPUCommandGraph<void>[];
  outlineErrors: readonly string[];
  cells: GeometryBuffers;
  tableCounts: Buffer | null;
  tableCount: Buffer | null;
  outline: OutlineBuffers | null;
  groupOutline: OutlineBuffers | null;
  reader: SummaryReader | null;
  byteLength: number;
  dirty: boolean;
};

type ZonesPipeline = {
  serial: number;
  resources: SpatialAnalysisResources;
  graph: CompiledGPUCommandGraph<void>;
  cover: GeometryBuffers & {featureIds: Buffer; core: Buffer; count: Buffer};
  compacted: GeometryBuffers & {count: Buffer};
  reader: SummaryReader;
  byteLength: number;
  dirty: boolean;
};

type SelectionPipeline = {
  serial: number;
  key: string;
  operation: H3AggregationOptions['selectionOperation'];
  stride: number;
  resources: SpatialAnalysisResources;
  graph: CompiledGPUCommandGraph<void>;
  result: GeometryBuffers & {distances: Buffer | null};
  resultCells: Buffer;
  selected: GeometryBuffers;
  reader: SummaryReader;
  dirty: boolean;
};

export async function createH3Aggregation(
  ctx: SceneContext<H3AggregationOptions>
): Promise<SceneInstance<H3AggregationOptions>> {
  const {device} = ctx;
  const columns: NatureCells = readNatureCells(ctx.datasets.get('chicago-nature'));
  const areas = readAreaPolygons(ctx.datasets.get('chicago-community-areas'));
  const pointCount = columns.count;
  const resources = new SpatialAnalysisResources(device, 'h3-aggregation');

  const positionsBuffer = resources.createBuffer('lng-lat', columns.lngLat);
  const maskValues = new Uint32Array(pointCount).fill(1);
  const maskBuffer = resources.createBuffer('mask', maskValues);
  const areaPositions = resources.createBuffer('area-positions', areas.vertices);
  const areaFeatureOffsets = resources.createBuffer('area-feature-offsets', areas.featureOffsets);
  const areaPolygonOffsets = resources.createBuffer('area-polygon-offsets', areas.polygonOffsets);
  const areaRingOffsets = resources.createBuffer('area-ring-offsets', areas.ringOffsets);
  const areaOutline = resources.createBuffer('area-outline', areas.outlineSegments);
  // The clicked position; the Montrose Point cell until the first click.
  const selectionPosition = resources.createBuffer(
    'selection-position',
    Float32Array.of(-87.6325, 41.9625)
  );
  let keptCount = pointCount;
  let areaObservationCounts = new Uint32Array(areas.names.length + 1);
  let busiestCount = 1;
  let serial = 0;
  let indexPipeline: IndexPipeline | null = null;
  let zonesPipeline: ZonesPipeline | null = null;
  const selectionPipelines = new Map<string, SelectionPipeline>();
  let activeSelection: SelectionPipeline | null = null;
  let rebuildTimer: ReturnType<typeof setTimeout> | undefined;
  let needIndex = false;
  let needZones = false;
  let needSelection = false;
  let destroyed = false;
  const retiring = new Set<{resources: SpatialAnalysisResources; reader?: SummaryReader | null}>();
  /** Table rows of the last index summary, for the observation count of selected cells. */
  let tableLookup: Map<string, number> | null = null;
  let tableSnapshot: {cells: Uint32Array; counts: Uint32Array; rows: number} | null = null;
  let coverAreaCells = new Uint32Array(areas.names.length);
  let lastSelectionWords: Uint32Array | null = null;
  let lastSelectionPipeline: SelectionPipeline | null = null;

  const importCellBuffer = <Format extends Parameters<typeof importGraphBuffer>[3]>(
    graph: GPUCommandGraph<void>,
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `import-${name}`, buffer, format, length);

  const updateMask = () => {
    const options = ctx.options;
    keptCount = fillObservationMask(columns, maskValues, {
      category: options.category,
      months: options.months,
      sampledPercent: options.sampledPercent
    });
    maskBuffer.write(maskValues);
    areaObservationCounts = new Uint32Array(areas.names.length + 1);
    for (let index = 0; index < pointCount; index++) {
      if (maskValues[index]) areaObservationCounts[columns.communityArea[index]]++;
    }
    if (indexPipeline) indexPipeline.dirty = true;
    ctx.setReadout(
      'points',
      `${formatCount(keptCount)} of ${formatCount(pointCount)} (${formatMonthWindow(options.months)})`
    );
  };

  const createGeometry = (
    owned: SpatialAnalysisResources,
    track: (size: number) => void,
    name: string,
    rows: number,
    stride: number
  ): GeometryBuffers => {
    track(rows * stride * 8 + rows * 4);
    return {
      boundaries: owned.createBuffer(`${name}-boundaries`, rows * stride * 8),
      vertexCounts: owned.createBuffer(`${name}-vertex-counts`, rows * 4),
      stride,
      rows
    };
  };

  const addGeometryNode = (
    graph: GPUCommandGraph<void>,
    name: string,
    family: GPUCellIndexFamily,
    cells: GraphDataView<'uint32x2'>,
    geometry: GeometryBuffers
  ) => {
    graph.add(
      new GPUCellGeometry({
        id: `${name}-geometry`,
        family,
        cells,
        maximumVertexCount: geometry.stride,
        output: {
          boundaries: importCellBuffer(
            graph,
            `${name}-boundaries`,
            geometry.boundaries,
            'float32x2',
            geometry.rows * geometry.stride
          ),
          vertexCounts: importCellBuffer(
            graph,
            `${name}-vertex-counts`,
            geometry.vertexCounts,
            'uint32',
            geometry.rows
          )
        }
      })
    );
  };

  // --- Index pipeline ---------------------------------------------------------------------------

  const buildIndexPipeline = (family: GPUCellIndexFamily, resolution: number): IndexPipeline => {
    const settings = FAMILIES[family];
    const {tabular, maximumVertexCount} = settings;
    const pipelineSerial = ++serial;
    const owned = new SpatialAnalysisResources(device, `h3-index-${pipelineSerial}`);
    let byteLength = 0;
    const track = (size: number) => {
      byteLength += size;
    };
    const create = (name: string, size: number): Buffer => {
      byteLength += size;
      return owned.createBuffer(name, size);
    };
    const rowCount = tabular ? TABLE_CAPACITY : pointCount;
    const rawCells = create('raw-cells', pointCount * 8);
    const tableCells = tabular ? create('table-cells', TABLE_CAPACITY * 8) : null;
    const tableCounts = tabular ? create('table-counts', TABLE_CAPACITY * 4) : null;
    const tableCount = tabular ? create('table-count', 4) : null;
    const tableOverflow = tabular ? create('table-overflow', 4) : null;
    const tableTotal = tabular ? create('table-total', 4) : null;
    const cells = createGeometry(owned, track, 'cells', rowCount, maximumVertexCount);

    const indexGraph = new GPUCommandGraph<void>(device, {id: `h3-index-${pipelineSerial}`});
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
        family,
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
          family: family as 'h3' | 'quadbin',
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
    addGeometryNode(indexGraph, 'cell', family, geometryInput, cells);
    const compiledIndex = owned.track(indexGraph.compile());

    // Outline graphs (H3 and Quadbin), each compiled on its own; ring assembly falls back to plain
    // segments if it does not compile on the device.
    const outlineGraphs: CompiledGPUCommandGraph<void>[] = [];
    const outlineErrors: string[] = [];
    let outline: OutlineBuffers | null = null;
    let groupOutline: OutlineBuffers | null = null;
    if (tabular && tableCells && tableCounts && tableCount) {
      const buildOutline = (
        name: string,
        grouped: boolean,
        withRings: boolean
      ): OutlineBuffers | null => {
        try {
          const graph = new GPUCommandGraph<void>(device, {id: `h3-${name}-${pipelineSerial}`});
          const view = <Format extends Parameters<typeof importGraphBuffer>[3]>(
            suffix: string,
            buffer: Buffer,
            format: Format,
            length: number
          ) => importCellBuffer(graph, `${name}-${suffix}`, buffer, format, length);
          const tableCellsView = view('table-cells', tableCells, 'uint32x2', TABLE_CAPACITY);
          const tableCountView = view('table-count', tableCount, 'uint32', 1);
          let tableGroupsView: GraphDataView<'uint32'> | undefined;
          if (grouped) {
            // A count tier per table row: the group label whose borders are drawn.
            const tableGroups = create(`${name}-table-groups`, TABLE_CAPACITY * 4);
            tableGroupsView = view('table-groups', tableGroups, 'uint32', TABLE_CAPACITY);
            addKernelPass(graph, {
              id: `${name}-count-tiers`,
              invocationCount: TABLE_CAPACITY,
              bindings: [
                {
                  name: 'counts',
                  view: view('table-counts', tableCounts, 'uint32', TABLE_CAPACITY),
                  type: 'u32',
                  access: 'read'
                },
                {name: 'groups', view: tableGroupsView, type: 'u32', access: 'read_write'}
              ],
              body: `let value = counts[countsOffset + index];
  var tier = 0u;
  if (value > 0u) { tier = min(firstLeadingBit(value), 4u); }
  groups[groupsOffset + index] = tier;`
            });
          }
          const endpoints = create(`${name}-endpoints`, OUTLINE_CAPACITY * 16);
          const groups = grouped ? create(`${name}-groups`, OUTLINE_CAPACITY * 4) : null;
          const outlineCount = create(`${name}-count`, 4);
          const outlineOverflow = create(`${name}-overflow`, 4);
          let rings: OutlineBuffers['rings'] = null;
          let ringOptions: ConstructorParameters<typeof GPUCellSetOutline>[0]['rings'];
          if (withRings) {
            const ringOffsets = create(`${name}-ring-offsets`, (RING_CAPACITY + 1) * 4);
            const ringCounts = create(`${name}-ring-counts`, RING_CAPACITY * 4);
            rings = {
              offsets: ringOffsets,
              counts: ringCounts,
              positions: create(`${name}-ring-positions`, OUTLINE_CAPACITY * 8),
              count: create(`${name}-ring-count`, 4),
              overflow: create(`${name}-ring-overflow`, 4),
              openSegments: create(`${name}-ring-open`, 4)
            };
            const ringOffsetsView = view('ring-offsets', ringOffsets, 'uint32', RING_CAPACITY + 1);
            ringOptions = {
              normalizeWinding: true,
              output: {
                ringOffsets: ringOffsetsView,
                positions: view('ring-positions', rings.positions, 'float32x2', OUTLINE_CAPACITY),
                count: view('ring-count', rings.count, 'uint32', 1),
                overflow: view('ring-overflow', rings.overflow, 'uint32', 1),
                openSegmentCount: view('ring-open', rings.openSegments, 'uint32', 1)
              }
            };
            // Vertex count of every ring (offsets past the ring count repeat the last offset), which
            // the ring outline layer reads.
            addKernelPass(graph, {
              id: `${name}-ring-counts`,
              invocationCount: RING_CAPACITY,
              bindings: [
                {name: 'offsets', view: ringOffsetsView, type: 'u32', access: 'read'},
                {
                  name: 'counts',
                  view: view('ring-vertex-counts', ringCounts, 'uint32', RING_CAPACITY),
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              body: `counts[countsOffset + index] = offsets[offsetsOffset + index + 1u] - offsets[offsetsOffset + index];`
            });
          }
          graph.add(
            new GPUCellSetOutline({
              id: name,
              family: family as 'h3' | 'quadbin',
              cells: tableCellsView,
              count: tableCountView,
              groups: tableGroupsView,
              rings: ringOptions,
              output: {
                rows: view(
                  'rows',
                  create(`${name}-rows`, OUTLINE_CAPACITY * 4),
                  'uint32',
                  OUTLINE_CAPACITY
                ),
                cells: view(
                  'cells',
                  create(`${name}-cells`, OUTLINE_CAPACITY * 8),
                  'uint32x2',
                  OUTLINE_CAPACITY
                ),
                edgeIndices: view(
                  'edges',
                  create(`${name}-edges`, OUTLINE_CAPACITY * 4),
                  'uint32',
                  OUTLINE_CAPACITY
                ),
                endpoints: view('endpoints', endpoints, 'float32x4', OUTLINE_CAPACITY),
                groups: groups ? view('groups', groups, 'uint32', OUTLINE_CAPACITY) : undefined,
                count: view('count', outlineCount, 'uint32', 1),
                overflow: view('overflow', outlineOverflow, 'uint32', 1)
              }
            })
          );
          const compiled = owned.track(graph.compile());
          const drawCommands = owned.track(
            new DrawCommandBuffer(device, {
              id: `h3-${pipelineSerial}-${name}-draw`,
              type: 'draw',
              commands: [{vertexCount: 6, instanceCount: 0}]
            })
          );
          outlineGraphs.push(compiled);
          return {
            rings,
            endpoints,
            groups,
            count: outlineCount,
            overflow: outlineOverflow,
            drawCommands
          };
        } catch (error) {
          outlineErrors.push(
            `${name}${withRings ? ' with rings' : ''}: ${error instanceof Error ? error.message : String(error)}`
          );
          return null;
        }
      };
      outline = buildOutline('outline', false, true);
      if (!outline) outline = buildOutline('outline', false, false);
      groupOutline = buildOutline('group-outline', true, false);
    }

    // Summary: ten header words, then the table counts and keys.
    let reader: SummaryReader | null = null;
    if (tabular && tableCounts && tableCells && tableCount && tableOverflow && tableTotal) {
      const zeroWord = create('zero-word', 4);
      const header: (Buffer | null | undefined)[] = [
        tableCount,
        tableOverflow,
        tableTotal,
        outline?.count,
        outline?.overflow,
        groupOutline?.count,
        groupOutline?.overflow,
        outline?.rings?.count,
        outline?.rings?.overflow,
        outline?.rings?.openSegments
      ];
      reader = new SummaryReader(
        owned,
        `h3-index-${pipelineSerial}`,
        [
          ...header.map(buffer => ({buffer: buffer ?? zeroWord, size: 4})),
          {buffer: tableCounts, size: TABLE_CAPACITY * 4},
          {buffer: tableCells, size: TABLE_CAPACITY * 8}
        ],
        bytes => onIndexSummary(pipelineSerial, resolution, family, bytes)
      );
    }
    return {
      serial: pipelineSerial,
      family,
      resolution,
      resources: owned,
      indexGraph: compiledIndex,
      outlineGraphs,
      outlineErrors,
      cells,
      tableCounts,
      tableCount,
      outline,
      groupOutline,
      reader,
      byteLength,
      dirty: true
    };
  };

  const onIndexSummary = (
    pipelineSerial: number,
    resolution: number,
    family: GPUCellIndexFamily,
    bytes: ArrayBuffer
  ) => {
    const active = indexPipeline;
    if (!active || active.serial !== pipelineSerial || destroyed) return;
    const words = new Uint32Array(bytes);
    const occupied = Math.min(words[0], TABLE_CAPACITY);
    const counts = words.subarray(10, 10 + TABLE_CAPACITY);
    const keys = words.subarray(10 + TABLE_CAPACITY, 10 + TABLE_CAPACITY * 3);
    let sum = 0;
    let maximum = 0;
    for (let row = 0; row < occupied; row++) {
      sum += counts[row];
      if (counts[row] > maximum) maximum = counts[row];
    }
    tableSnapshot = {
      cells: keys.slice(0, occupied * 2),
      counts: counts.slice(0, occupied),
      rows: occupied
    };
    tableLookup = null;
    ctx.setReadout(
      'occupied',
      `${formatCount(occupied)} of ${formatCount(TABLE_CAPACITY)}${words[1] ? ` (OVERFLOW: ${formatCount(words[2])} cells, smallest keys kept)` : ''}`
    );
    ctx.setReadout(
      'binned',
      `${formatCount(sum)} observations (${(sum / Math.max(occupied, 1)).toFixed(1)} per cell)`
    );
    ctx.setReadout('busiest', `${formatCount(maximum)} observations in one cell`);
    if (maximum !== busiestCount) {
      busiestCount = Math.max(1, maximum);
      ctx.requestLayers();
    }
    updateLegendExtent();
    ctx.setReadout(
      'outlineSegments',
      active.outline
        ? `${formatCount(words[3])} segments${words[4] ? ' (OVERFLOW, clamped)' : ''}`
        : 'unavailable'
    );
    ctx.setReadout(
      'groupBorders',
      active.groupOutline
        ? `${formatCount(words[5])} segments (${formatCount(Math.max(0, words[5] - words[3]))} more than the plain boundary)${words[6] ? ' (OVERFLOW)' : ''}`
        : 'unavailable'
    );
    ctx.setReadout(
      'rings',
      active.outline?.rings
        ? `${formatCount(words[7])} rings, ${formatCount(words[9])} open segments${words[8] ? ' (OVERFLOW)' : ''}`
        : 'unavailable'
    );
    void resolution;
    void family;
    showSelectionSummary();
  };

  const updateLegendExtent = () => {
    ctx.setLegendExtent('cells', [
      0,
      Math.max(1, Math.round((busiestCount * ctx.options.ceiling) / 100))
    ]);
  };

  // --- Zones pipeline ---------------------------------------------------------------------------

  const buildZonesPipeline = (
    family: 'h3' | 'quadbin',
    resolution: number,
    containment: GPUCellCoverContainment,
    compactDepth: number
  ): ZonesPipeline => {
    const settings = FAMILIES[family];
    const {maximumVertexCount} = settings;
    const pipelineSerial = ++serial;
    const owned = new SpatialAnalysisResources(device, `h3-zones-${pipelineSerial}`);
    let byteLength = 0;
    const track = (size: number) => {
      byteLength += size;
    };
    const create = (name: string, size: number): Buffer => {
      byteLength += size;
      return owned.createBuffer(name, size);
    };
    const coverCells = create('cover-cells', COVER_CAPACITY * 8);
    const coverFeatureIds = create('cover-feature-ids', COVER_CAPACITY * 4);
    const coverCore = create('cover-core', COVER_CAPACITY * 4);
    const coverCount = create('cover-count', 4);
    const coverOverflow = create('cover-overflow', 4);
    const coverTotal = create('cover-total', 4);
    const compactCells = create('compact-cells', COVER_CAPACITY * 8);
    const compactCount = create('compact-count', 4);
    const compactOverflow = create('compact-overflow', 4);
    const uncompactCells = create('uncompact-cells', COVER_CAPACITY * 8);
    const uncompactCount = create('uncompact-count', 4);
    const uncompactOverflow = create('uncompact-overflow', 4);
    const uncompactDropped = create('uncompact-dropped', 4);
    const coverGeometry = createGeometry(owned, track, 'cover', COVER_CAPACITY, maximumVertexCount);
    const compactGeometry = createGeometry(
      owned,
      track,
      'compact',
      COVER_CAPACITY,
      maximumVertexCount
    );

    const graph = new GPUCommandGraph<void>(device, {id: `h3-zones-${pipelineSerial}`});
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
    const compactCountView = importCellBuffer(graph, 'compact-count', compactCount, 'uint32', 1);
    graph.add(
      new GPUCellCover({
        id: 'cover',
        family,
        resolution,
        containment,
        polygonPositions: importCellBuffer(
          graph,
          'area-positions',
          areaPositions,
          'float32x2',
          areas.vertices.length / 2
        ),
        featureOffsets: importCellBuffer(
          graph,
          'area-feature-offsets',
          areaFeatureOffsets,
          'uint32',
          areas.featureOffsets.length
        ),
        polygonOffsets: importCellBuffer(
          graph,
          'area-polygon-offsets',
          areaPolygonOffsets,
          'uint32',
          areas.polygonOffsets.length
        ),
        ringOffsets: importCellBuffer(
          graph,
          'area-ring-offsets',
          areaRingOffsets,
          'uint32',
          areas.ringOffsets.length
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
          core: importCellBuffer(graph, 'cover-core', coverCore, 'uint32', COVER_CAPACITY),
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
        family,
        operation: {
          type: 'compact',
          resolution,
          minimumResolution: Math.max(0, resolution - compactDepth)
        },
        cells: coverCellsView,
        count: coverCountView,
        output: {
          cells: compactCellsView,
          count: compactCountView,
          overflow: importCellBuffer(graph, 'compact-overflow', compactOverflow, 'uint32', 1)
        }
      })
    );
    graph.add(
      new GPUCellCompaction({
        id: 'uncompact',
        family,
        operation: {type: 'uncompact', resolution, maximumDepth: compactDepth},
        cells: compactCellsView,
        count: compactCountView,
        output: {
          cells: importCellBuffer(
            graph,
            'uncompact-cells',
            uncompactCells,
            'uint32x2',
            COVER_CAPACITY
          ),
          count: importCellBuffer(graph, 'uncompact-count', uncompactCount, 'uint32', 1),
          overflow: importCellBuffer(graph, 'uncompact-overflow', uncompactOverflow, 'uint32', 1),
          droppedCount: importCellBuffer(graph, 'uncompact-dropped', uncompactDropped, 'uint32', 1)
        }
      })
    );
    addGeometryNode(graph, 'cover', family, coverCellsView, coverGeometry);
    addGeometryNode(graph, 'compact', family, compactCellsView, compactGeometry);
    const compiled = owned.track(graph.compile());

    const reader = new SummaryReader(
      owned,
      `h3-zones-${pipelineSerial}`,
      [
        {buffer: coverCount, size: 4},
        {buffer: coverOverflow, size: 4},
        {buffer: coverTotal, size: 4},
        {buffer: compactCount, size: 4},
        {buffer: compactOverflow, size: 4},
        {buffer: uncompactCount, size: 4},
        {buffer: uncompactOverflow, size: 4},
        {buffer: uncompactDropped, size: 4},
        {buffer: coverCore, size: COVER_CAPACITY * 4},
        {buffer: coverFeatureIds, size: COVER_CAPACITY * 4}
      ],
      bytes => onZonesSummary(pipelineSerial, bytes)
    );
    return {
      serial: pipelineSerial,
      resources: owned,
      graph: compiled,
      cover: {...coverGeometry, featureIds: coverFeatureIds, core: coverCore, count: coverCount},
      compacted: {...compactGeometry, count: compactCount},
      reader,
      byteLength,
      dirty: true
    };
  };

  const onZonesSummary = (pipelineSerial: number, bytes: ArrayBuffer) => {
    const active = zonesPipeline;
    if (!active || active.serial !== pipelineSerial || destroyed) return;
    const words = new Uint32Array(bytes);
    const coverRows = Math.min(words[0], COVER_CAPACITY);
    let coreCells = 0;
    const perArea = new Uint32Array(areas.names.length);
    for (let row = 0; row < coverRows; row++) {
      if (words[8 + row] === 1) coreCells++;
      const feature = words[8 + COVER_CAPACITY + row];
      if (feature < perArea.length) perArea[feature]++;
    }
    coverAreaCells = perArea;
    const emptyAreas = perArea.reduce((total, value) => total + (value === 0 ? 1 : 0), 0);
    ctx.setReadout(
      'cover',
      `${formatCount(coverRows)} cells (${formatCount(words[2])} unclamped${words[1] ? ', OVERFLOW' : ''}); ${emptyAreas} of ${areas.names.length} areas get no cell`
    );
    ctx.setReadout(
      'core',
      `${formatCount(coreCells)} core / ${formatCount(coverRows - coreCells)} border (${coverRows > 0 ? ((100 * coreCells) / coverRows).toFixed(0) : 0}% core)`
    );
    const fewer = words[0] > 0 ? (1 - words[3] / words[0]) * 100 : 0;
    ctx.setReadout(
      'compacted',
      `${formatCount(words[3])} cells (${fewer.toFixed(0)}% fewer${words[4] ? ', OVERFLOW' : ''})`
    );
    ctx.setReadout(
      'roundTrip',
      words[5] === words[0] && words[7] === 0
        ? `uncompact gives back ${formatCount(words[5])} cells: identical count to the cover`
        : `uncompact gives ${formatCount(words[5])} cells (cover ${formatCount(words[0])}, ${formatCount(words[7])} dropped${words[6] ? ', OVERFLOW' : ''})`
    );
  };

  // --- Selection pipeline -----------------------------------------------------------------------

  const getSelectionOperation = (
    family: 'h3' | 'quadbin',
    resolution: number,
    options: Readonly<H3AggregationOptions>
  ): GPUCellTopologyOperation => {
    void family;
    if (options.selectionOperation === 'disk') return {type: 'disk', k: DISK_RADIUS};
    if (options.selectionOperation === 'ring') return {type: 'ring', k: options.ringRadius};
    if (options.selectionOperation === 'parent') {
      return {type: 'parent', resolution: Math.max(0, resolution - options.levelsUp)};
    }
    return {
      type: 'children',
      resolution: resolution + options.levelsDown,
      inputResolution: resolution
    };
  };

  const getSelectionKey = (options: Readonly<H3AggregationOptions>, resolution: number): string => {
    const {selectionOperation} = options;
    const parameter =
      selectionOperation === 'ring'
        ? options.ringRadius
        : selectionOperation === 'parent'
          ? options.levelsUp
          : selectionOperation === 'children'
            ? options.levelsDown
            : DISK_RADIUS;
    return `${options.family}|${resolution}|${selectionOperation}|${parameter}`;
  };

  const buildSelectionPipeline = (
    family: 'h3' | 'quadbin',
    resolution: number,
    options: Readonly<H3AggregationOptions>
  ): SelectionPipeline => {
    const {maximumVertexCount} = FAMILIES[family];
    const operation = getSelectionOperation(family, resolution, options);
    const stride = getCellTopologyStride(family, operation);
    const pipelineSerial = ++serial;
    const owned = new SpatialAnalysisResources(device, `h3-selection-${pipelineSerial}`);
    const track = () => {};
    const selectedCell = owned.createBuffer('selected-cell', 8);
    const resultCells = owned.createBuffer('result-cells', stride * 8);
    const resultDistances =
      operation.type === 'disk' ? owned.createBuffer('result-distances', stride * 4) : null;
    const resultCounts = owned.createBuffer('result-counts', 4);
    const result = createGeometry(owned, track, 'result', stride, maximumVertexCount);
    const selected = createGeometry(owned, track, 'selected', 1, maximumVertexCount);
    const graph = new GPUCommandGraph<void>(device, {id: `h3-selection-${pipelineSerial}`});
    const selectedCellView = importCellBuffer(graph, 'selected-cell', selectedCell, 'uint32x2', 1);
    const resultCellsView = importCellBuffer(
      graph,
      'result-cells',
      resultCells,
      'uint32x2',
      stride
    );
    graph.add(
      new GPUPointToCell({
        id: 'selection-key',
        family,
        resolution,
        positions: importCellBuffer(graph, 'selection-position', selectionPosition, 'float32x2', 1),
        output: {cells: selectedCellView}
      })
    );
    graph.add(
      new GPUCellTopology({
        id: operation.type,
        family,
        operation,
        cells: selectedCellView,
        output: {
          cells: resultCellsView,
          distances: resultDistances
            ? importCellBuffer(graph, 'result-distances', resultDistances, 'uint32', stride)
            : undefined,
          counts: importCellBuffer(graph, 'result-counts', resultCounts, 'uint32', 1)
        }
      })
    );
    addGeometryNode(graph, 'result', family, resultCellsView, result);
    addGeometryNode(graph, 'selected', family, selectedCellView, selected);
    const compiled = owned.track(graph.compile());
    const reader = new SummaryReader(
      owned,
      `h3-selection-${pipelineSerial}`,
      [
        {buffer: selectedCell, size: 8},
        {buffer: resultCounts, size: 4},
        {buffer: resultCells, size: stride * 8}
      ],
      bytes => onSelectionSummary(pipelineSerial, bytes)
    );
    return {
      serial: pipelineSerial,
      key: getSelectionKey(options, resolution),
      operation: options.selectionOperation,
      stride,
      resources: owned,
      graph: compiled,
      result: {...result, distances: resultDistances},
      resultCells,
      selected,
      reader,
      dirty: true
    };
  };

  const onSelectionSummary = (pipelineSerial: number, bytes: ArrayBuffer) => {
    const active = activeSelection;
    if (!active || active.serial !== pipelineSerial || destroyed) return;
    lastSelectionWords = new Uint32Array(bytes);
    lastSelectionPipeline = active;
    showSelectionSummary();
  };

  const getTableLookup = (): Map<string, number> | null => {
    if (!tableSnapshot) return null;
    if (!tableLookup) {
      tableLookup = new Map();
      for (let row = 0; row < tableSnapshot.rows; row++) {
        tableLookup.set(
          packKeyWords(tableSnapshot.cells[row * 2], tableSnapshot.cells[row * 2 + 1]),
          tableSnapshot.counts[row]
        );
      }
    }
    return tableLookup;
  };

  const showSelectionSummary = () => {
    const words = lastSelectionWords;
    const active = lastSelectionPipeline;
    if (!words || !active || active !== activeSelection) return;
    const lookup = getTableLookup();
    const selectedKey = packKeyWords(words[0], words[1]);
    const ownCount = lookup?.get(selectedKey);
    ctx.setReadout(
      'selected',
      `${formatCellKey(words[0], words[1])}${ownCount !== undefined ? `: ${formatCount(ownCount)} observations` : lookup ? ': no observations' : ''}`
    );
    const rows = Math.min(words[2], active.stride);
    const resultKeys = (index: number) => packKeyWords(words[3 + index * 2], words[4 + index * 2]);
    if (active.operation === 'parent' || active.operation === 'children') {
      ctx.setReadout(
        'selection',
        `${formatCount(rows)} ${active.operation === 'parent' ? 'parent cell' : 'child cells'} (a different resolution than the table, so no observation count)`
      );
    } else if (lookup) {
      let observations = 0;
      let occupiedCells = 0;
      for (let index = 0; index < rows; index++) {
        const count = lookup.get(resultKeys(index));
        if (count) {
          observations += count;
          occupiedCells++;
        }
      }
      ctx.setReadout(
        'selection',
        `${formatCount(rows)} ${active.operation === 'ring' ? 'ring' : 'disk'} cells, ${formatCount(occupiedCells)} with observations, ${formatCount(observations)} observations`
      );
    }
  };

  // --- Build orchestration ----------------------------------------------------------------------

  const retire = (
    retired: {resources: SpatialAnalysisResources; reader?: SummaryReader | null} | null
  ) => {
    if (!retired) return;
    retired.reader?.stop();
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

  const getEffectiveContainment = (options: Readonly<H3AggregationOptions>) =>
    options.family === 'h3' ? 'center' : options.containment;

  const describeGrid = () => {
    const options = ctx.options;
    const settings = FAMILIES[options.family];
    const resolution = getResolution(options.family, options.resolutionOffset);
    ctx.setReadout('grid', `${settings.label} ${resolution}`);
    ctx.setReadout('cellSize', describeCellSize(options.family, resolution));
    ctx.setReadout('exactness', settings.exactness);
    if (!settings.tabular) {
      for (const id of [
        'occupied',
        'binned',
        'busiest',
        'selected',
        'selection',
        'cover',
        'core',
        'compacted',
        'roundTrip',
        'outlineSegments',
        'rings',
        'groupBorders'
      ]) {
        ctx.setReadout(
          id,
          id === 'occupied'
            ? 'one cell per observation (no aggregation contributor for this family)'
            : 'H3 and Quadbin only'
        );
      }
      ctx.setReadout(
        'binned',
        `${formatCount(keptCount)} per-observation cells, summed by overlap`
      );
    }
  };

  const updateMemory = () => {
    const bytes =
      (indexPipeline?.byteLength ?? 0) +
      (zonesPipeline?.byteLength ?? 0) +
      pointCount * 4 * 3 +
      areas.vertices.byteLength;
    ctx.setReadout('memory', bytes);
  };

  const rebuild = (index: boolean, zones: boolean, selection: boolean) => {
    const options = ctx.options;
    const family = options.family;
    const resolution = getResolution(family, options.resolutionOffset);
    const tabular = FAMILIES[family].tabular;
    if (index) {
      const previous = indexPipeline;
      indexPipeline = buildIndexPipeline(family, resolution);
      retire(previous);
    }
    if (zones || index) {
      const previous = zonesPipeline;
      zonesPipeline = tabular
        ? buildZonesPipeline(
            family as 'h3' | 'quadbin',
            resolution,
            getEffectiveContainment(options),
            options.compactDepth
          )
        : null;
      retire(previous);
    }
    if (selection || index) {
      for (const pipeline of selectionPipelines.values()) retire(pipeline);
      selectionPipelines.clear();
      activeSelection = null;
      lastSelectionWords = null;
      if (tabular) {
        const pipeline = buildSelectionPipeline(family as 'h3' | 'quadbin', resolution, options);
        selectionPipelines.set(pipeline.key, pipeline);
        activeSelection = pipeline;
      }
    }
    describeGrid();
    updateMemory();
    ctx.requestLayers();
  };

  const flushRebuild = () => {
    rebuildTimer = undefined;
    if (destroyed) return;
    const index = needIndex;
    const zones = needZones;
    const selection = needSelection;
    needIndex = needZones = needSelection = false;
    rebuild(index, zones, selection);
  };

  const scheduleRebuild = (index: boolean, zones: boolean, selection: boolean) => {
    needIndex ||= index;
    needZones ||= zones;
    needSelection ||= selection;
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(flushRebuild, REBUILD_DEBOUNCE_MILLISECONDS);
  };

  /** Switches to the selection graph of the current operation, compiling it the first time. */
  const syncSelectionVariant = () => {
    const options = ctx.options;
    if (!FAMILIES[options.family].tabular) return;
    const resolution = getResolution(options.family, options.resolutionOffset);
    const key = getSelectionKey(options, resolution);
    let pipeline = selectionPipelines.get(key);
    if (!pipeline) {
      pipeline = buildSelectionPipeline(options.family as 'h3' | 'quadbin', resolution, options);
      selectionPipelines.set(key, pipeline);
    }
    if (pipeline !== activeSelection) {
      activeSelection = pipeline;
      pipeline.dirty = true;
      lastSelectionWords = null;
    }
    ctx.requestLayers();
  };

  updateMask();
  rebuild(true, true, true);

  const getHoverText = (longitude: number, latitude: number): string | null => {
    const area = findArea(areas, longitude, latitude);
    if (area < 0) return null;
    const name = areas.names[area] ?? `Area ${area + 1}`;
    const observations = areaObservationCounts[area + 1] ?? 0;
    const tabular = FAMILIES[ctx.options.family].tabular;
    const coverCells = coverAreaCells[area];
    return `${name}: ${formatCount(observations)} observations in the current filter${
      tabular && zonesPipeline ? `; ${formatCount(coverCells ?? 0)} cover cells` : ''
    }`;
  };

  const instance: SceneInstance<H3AggregationOptions> = {
    getCompiledGraphs: () => {
      const graphs: CompiledGPUCommandGraph<void>[] = [];
      if (indexPipeline) graphs.push(indexPipeline.indexGraph, ...indexPipeline.outlineGraphs);
      if (zonesPipeline) graphs.push(zonesPipeline.graph);
      if (activeSelection) graphs.push(activeSelection.graph);
      return graphs as CompiledGPUCommandGraph<never>[];
    },

    setOption(id) {
      switch (id) {
        case 'family':
        case 'resolutionOffset':
          scheduleRebuild(true, true, true);
          break;
        case 'containment':
        case 'compactDepth':
          scheduleRebuild(false, true, false);
          break;
        case 'selectionOperation':
          syncSelectionVariant();
          break;
        case 'ringRadius':
        case 'levelsUp':
        case 'levelsDown':
          if (
            ctx.options.selectionOperation ===
            {ringRadius: 'ring', levelsUp: 'parent', levelsDown: 'children'}[id]
          ) {
            clearTimeout(rebuildTimer);
            rebuildTimer = setTimeout(() => {
              rebuildTimer = undefined;
              if (!destroyed) syncSelectionVariant();
            }, REBUILD_DEBOUNCE_MILLISECONDS);
          }
          break;
        case 'category':
        case 'months':
        case 'sampledPercent':
          updateMask();
          break;
        case 'ceiling':
          updateLegendExtent();
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    encode(commandEncoder) {
      const active = indexPipeline;
      if (!active) return;
      if (active.dirty) {
        active.indexGraph.encode(commandEncoder, {parameters: undefined});
        for (const outlineGraph of active.outlineGraphs) {
          outlineGraph.encode(commandEncoder, {parameters: undefined});
        }
        // The outline contributors have no draw count: copy each clamped segment count into the
        // instance-count word of its indirect record.
        for (const outline of [active.outline, active.groupOutline]) {
          if (!outline) continue;
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: outline.count,
            sourceOffset: 0,
            destinationBuffer: outline.drawCommands.buffer,
            destinationOffset: 4,
            size: 4
          });
        }
        active.dirty = false;
        active.reader?.markStale();
      }
      if (zonesPipeline?.dirty) {
        zonesPipeline.graph.encode(commandEncoder, {parameters: undefined});
        zonesPipeline.dirty = false;
        zonesPipeline.reader.markStale();
      }
      if (activeSelection?.dirty) {
        activeSelection.graph.encode(commandEncoder, {parameters: undefined});
        activeSelection.dirty = false;
        activeSelection.reader.markStale();
      }
      active.reader?.flush(commandEncoder);
      zonesPipeline?.reader.flush(commandEncoder);
      activeSelection?.reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const active = indexPipeline;
      if (!active) return [];
      const tabular = FAMILIES[active.family].tabular;
      const layers: Layer[] = [];
      const ceilingCount = Math.max(1, Math.round((busiestCount * options.ceiling) / 100));
      if (options.showCells) {
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
            colorMode: tabular ? 'ramp' : 'uniform',
            ramp: options.ramp,
            sqrtScale: options.scale === 'sqrt',
            color: tabular
              ? [255, 255, 255, 215]
              : [255, 190, 90, Math.min(40, Math.round(3 * (100 / options.ceiling)))],
            valueMaximum: ceilingCount,
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
              color: [10, 10, 25, 110],
              widthPixels: 0.8
            })
          );
        }
      }
      if (tabular && active.outline && options.outline !== 'off') {
        if (options.outline === 'groups' && active.groupOutline) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `cells-group-outline-${active.serial}`,
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              segments: active.groupOutline.endpoints,
              drawCommands: active.groupOutline.drawCommands,
              values: active.groupOutline.groups,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: TIER_COLORS,
              widthPixels: 3.5
            })
          );
        }
        if (options.outline === 'rings' && active.outline.rings) {
          layers.push(
            new RingOutlineLayer({
              id: `cells-set-rings-${active.serial}`,
              ringOffsets: active.outline.rings.offsets,
              ringCounts: active.outline.rings.counts,
              ringPositions: active.outline.rings.positions,
              slotCount: OUTLINE_CAPACITY,
              ringCapacity: RING_CAPACITY,
              palette: RING_COLORS,
              widthPixels: 2.8
            })
          );
        } else if (options.outline !== 'rings') {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `cells-set-outline-${active.serial}`,
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              segments: active.outline.endpoints,
              drawCommands: active.outline.drawCommands,
              widthPixels: 2.6,
              color: [255, 255, 255, 245]
            })
          );
        }
      }
      const zones = zonesPipeline;
      if (tabular && options.showCover && zones) {
        layers.push(
          new CellBoundaryLayer({
            id: `cells-cover-fill-${zones.serial}`,
            boundaries: zones.cover.boundaries,
            vertexCounts: zones.cover.vertexCounts,
            stride: zones.cover.stride,
            instanceCount: COVER_CAPACITY,
            rowCounts: zones.cover.count,
            values: options.coverColor === 'core' ? zones.cover.core : zones.cover.featureIds,
            mode: 'fill',
            colorMode: 'category',
            palette: options.coverColor === 'core' ? CORE_COLORS : AREA_COLORS,
            opacity: options.showCompacted ? 0.4 : 0.85
          }),
          new CellBoundaryLayer({
            id: `cells-cover-outline-${zones.serial}`,
            boundaries: zones.cover.boundaries,
            vertexCounts: zones.cover.vertexCounts,
            stride: zones.cover.stride,
            instanceCount: COVER_CAPACITY,
            rowCounts: zones.cover.count,
            mode: 'outline',
            color: [255, 255, 255, options.showCompacted ? 60 : 150],
            widthPixels: 0.9
          })
        );
        if (options.showCompacted) {
          layers.push(
            new CellBoundaryLayer({
              id: `cells-compact-outline-${zones.serial}`,
              boundaries: zones.compacted.boundaries,
              vertexCounts: zones.compacted.vertexCounts,
              stride: zones.compacted.stride,
              instanceCount: COVER_CAPACITY,
              rowCounts: zones.compacted.count,
              mode: 'outline',
              color: [255, 244, 120, 255],
              widthPixels: 2.6
            })
          );
        }
      }
      if (options.showCover) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'cells-area-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: areaOutline,
            instanceCount: areas.outlineSegments.length / 4,
            widthPixels: 1.6,
            color: [255, 255, 255, 235]
          })
        );
      }
      const selection = activeSelection;
      if (tabular && options.showSelection && selection) {
        const operation = selection.operation;
        layers.push(
          new CellBoundaryLayer({
            id: `cells-selection-fill-${selection.serial}`,
            boundaries: selection.result.boundaries,
            vertexCounts: selection.result.vertexCounts,
            stride: selection.result.stride,
            instanceCount: selection.result.rows,
            values: selection.result.distances,
            mode: 'fill',
            colorMode: operation === 'disk' ? 'distance' : 'uniform',
            color:
              operation === 'parent'
                ? [255, 220, 80, 90]
                : operation === 'children'
                  ? [110, 230, 130, 150]
                  : operation === 'ring'
                    ? [255, 140, 60, 170]
                    : SELECTION_COLOR,
            filterMaximum: options.diskRadius
          }),
          new CellBoundaryLayer({
            id: `cells-selection-outline-${selection.serial}`,
            boundaries: selection.result.boundaries,
            vertexCounts: selection.result.vertexCounts,
            stride: selection.result.stride,
            instanceCount: selection.result.rows,
            values: selection.result.distances,
            mode: 'outline',
            colorMode: operation === 'disk' ? 'distance' : 'uniform',
            color: [255, 255, 255, 255],
            filterMaximum: options.diskRadius,
            widthPixels: operation === 'parent' ? 3 : 1.4
          }),
          new CellBoundaryLayer({
            id: `cells-selected-outline-${selection.serial}`,
            boundaries: selection.selected.boundaries,
            vertexCounts: selection.selected.vertexCounts,
            stride: selection.selected.stride,
            instanceCount: 1,
            mode: 'outline',
            color: [255, 40, 120, 255],
            widthPixels: 3.2
          })
        );
      }
      return layers;
    },

    onClick(event) {
      if (!event.coordinate || !activeSelection) return false;
      selectionPosition.write(Float32Array.of(event.coordinate[0], event.coordinate[1]));
      activeSelection.dirty = true;
      return true;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      return getHoverText(event.coordinate[0], event.coordinate[1]);
    },

    destroy() {
      destroyed = true;
      clearTimeout(rebuildTimer);
      indexPipeline?.reader?.stop();
      zonesPipeline?.reader.stop();
      for (const pipeline of selectionPipelines.values()) {
        pipeline.reader.stop();
        pipeline.resources.destroy();
      }
      selectionPipelines.clear();
      indexPipeline?.resources.destroy();
      zonesPipeline?.resources.destroy();
      for (const retired of retiring) retired.resources.destroy();
      retiring.clear();
      resources.destroy();
    }
  };
  return instance;
}
