// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPULatticeWeights,
  GPUSpatialWeightsAlgebra,
  type GPUNeighborSearchKernel,
  type GPUNeighborSearchWeightKind,
  type GPUSpatialWeightsCombineRule,
  type GPUSpatialWeightsKernel
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassCounts} from '../../cartography/breaks';
import {getClassIndexOf, getClassTableLayerProps} from '../../cartography/class-table';
import {CHICAGO, US} from '../../cartography/gazetteer';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import {formatCount, formatDistance, formatOrdinal, liveText} from '../../cartography/live-text';
import {getInputPolygons} from '../../cartography/picking';
import {getFeatureLabelPoint} from '../../cartography/polygon-mesh';
import {getLocalProjector, projectRingsToSegments} from '../../cartography/segments';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisStyleProps
} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {
  getVariableInfo,
  loadGeography,
  type Geography,
  type GeographyId,
  type VariableId
} from './b4-geography';
import {createGeographyBuffers, type GeographyBuffers} from './b4-layers';
import {
  createAnalysisSpace,
  NO_MODE,
  type AnalysisSpace,
  type SpaceDisplay,
  type SpaceSnapshot,
  type SummaryChoice
} from './b4-space';
import {
  createWeightsCore,
  type WeightsConfig,
  type WeightsCore,
  type WeightsSource,
  type WeightsTransform
} from './b4-weights-core';
import {FOCUS_SLOTS, type WeightsKitSummary, type WeightsMatrixChoice} from './b4-weights-kit';
import {getContextLineColor, getHairlineColor, getStateLineStyle} from './hot-spots.style';
import {
  getCardinalityChart,
  getKernelChart,
  getLagScatter,
  type KernelName
} from './spatial-weights.charts';
import {getOuterEdgeRows} from './spatial-weights.edges';
import {
  createFocusRowGatherer,
  type FocusRow,
  type FocusRowGatherer
} from './spatial-weights.focus';
import {
  getAreaLineStyle,
  getCardinalityTable,
  getCasingColor,
  getDifferenceTable,
  getEffectiveDisplay,
  getFocusPalette,
  getInkColor,
  getOneWayTable,
  getQuantileTable,
  getSummaryTable,
  getWeightTable,
  MOUNTAIN_WEST_BOUNDS,
  type Ground
} from './spatial-weights.style';

/** Combination applied on top of the chosen neighbourhood. */
export type WeightsCombine =
  | 'none'
  | 'union'
  | 'intersection'
  | 'difference'
  | 'symmetricDifference'
  | 'higherOrder'
  | 'selfWeight'
  | 'subgraph'
  | 'block';

/** What the map fill shows. */
export type SpatialWeightsDisplay =
  | 'focus'
  | 'weights'
  | 'neighbors'
  | 'oneWay'
  | 'value'
  | 'lag'
  | 'difference'
  | 'summary';

/** Option state of the spatial-weights scene. */
export type SpatialWeightsOptions = {
  geography: GeographyId;
  variable: VariableId;
  focus: 'typical' | 'corner' | 'island';
  source: WeightsSource;
  lattice: boolean;
  snapTolerance: string;
  k: number;
  bandFactor: number;
  knnCapFactor: number;
  weightKind: GPUNeighborSearchWeightKind;
  kernel: GPUNeighborSearchKernel;
  power: number;
  distanceFloorFactor: number;
  rowStandardize: boolean;
  transform: WeightsTransform;
  transformKernel: GPUSpatialWeightsKernel;
  bandwidth: string;
  doubleSum: 'one' | 'rows';
  combine: WeightsCombine;
  partnerK: number;
  weightRule: GPUSpatialWeightsCombineRule;
  order: number;
  cumulative: boolean;
  selfWeight: number;
  populationPercentile: number;
  lagNormalize: boolean;
  includeFocal: boolean;
  focalWeight: string;
  display: SpatialWeightsDisplay;
  summary: SummaryChoice;
  matrix: WeightsMatrixChoice;
  showLinks: boolean;
  showOutlines: boolean;
  showQueenOnly: boolean;
  latticeCriterion: 'rook' | 'queen';
  latticeRadius: number;
  latticeMask: boolean;
};

/** The class tables, counts and ground the legends read (stored with `ctx.setLegendData`). */
export type WeightsLegendData = {
  ground: Ground;
  tables: WeightsTables;
  counts: WeightsCounts;
};

/** The class tables of every classed display; the layer, legend and tooltip share them. */
export type WeightsTables = {
  cardinality: ClassTable;
  oneWay: ClassTable;
  weights: ClassTable;
  value: ClassTable;
  difference: ClassTable;
  summary: ClassTable | null;
};

/** Places per class of the displays, from the latest readback. */
export type WeightsCounts = {
  cardinality?: number[];
  oneWay?: number[];
  weights?: number[];
  value?: number[];
  lag?: number[];
  difference?: number[];
  summary?: number[];
};

/** Slot capacity of the analysed CSR: it must hold block weights (sum of squared group sizes). */
const ANALYSIS_SLOTS: Record<GeographyId, number> = {
  'us-counties': 320_000,
  'chicago-tracts': 48_000
};
/** Focus presets: a typical place, a corner case where queen and rook differ, and an island. */
const FOCUS_POINTS: Record<
  GeographyId,
  Record<SpatialWeightsOptions['focus'], [number, number]>
> = {
  'us-counties': {
    typical: [-87.65, 41.84],
    corner: [-108.35, 36.55],
    island: [-70.05, 41.28]
  },
  'chicago-tracts': {
    typical: [-87.63, 41.88],
    corner: [-87.6107, 41.7774],
    island: [-87.9, 41.98]
  }
};
const LATTICE_COLUMNS: Record<GeographyId, number> = {'us-counties': 110, 'chicago-tracts': 70};
const LATTICE_MAXIMUM_RADIUS = 4;
/** Segments of the focus outline and of the neighbour outlines the GPU buffers can hold. */
const FOCUS_OUTLINE_CAPACITY = 4096;
const NEIGHBOUR_OUTLINE_CAPACITY = 32768;
/** Island labels and corner labels drawn at most. */
const MAXIMUM_NAMED_ROWS = 3;
/** Rule names for the notes. */
const RULE_NAMES: Record<WeightsSource, string> = {
  queen: 'queen',
  rook: 'rook',
  knn: 'nearest',
  band: 'distance band'
};

type Lattice = {
  columns: number;
  rows: number;
  cell: number;
  /** `[minX, minY, maxX, maxY]` of the whole grid. */
  bounds: [number, number, number, number];
  featureOfCell: Int32Array;
  space: AnalysisSpace;
  structureMask: Buffer;
  values: Buffer;
  valueMask: Buffer;
  producers: Map<string, CompiledGPUCommandGraph<void>>;
  overflow: Buffer;
  total: Buffer;
  focusRow: number;
  insideMask: Uint32Array;
};

/** Queen and rook contiguity built side by side, so the corner-only links can be read off. */
type Reference = {
  queen: WeightsCore;
  rook: WeightsCore;
  reader: SummaryReader;
  /** True once both producers ran for this world. */
  encoded: boolean;
  /** CSR bytes of both producers, once read back. */
  data: {
    queenOffsets: Uint32Array;
    queenNeighbors: Uint32Array;
    rookOffsets: Uint32Array;
    rookNeighbors: Uint32Array;
  } | null;
};

type World = {
  geography: Geography;
  resources: SpatialAnalysisResources;
  core: WeightsCore;
  partner: ReturnType<WeightsCore['createPartner']>;
  space: AnalysisSpace;
  buffers: GeographyBuffers;
  values: Buffer;
  valueMask: Buffer;
  selfWeights: Buffer;
  subgraphMask: Buffer;
  groups: Buffer;
  algebra: Map<string, CompiledGPUCommandGraph<void>>;
  algebraOverflow: Buffer;
  algebraTotal: Buffer;
  lattice: Lattice | null;
  focusRow: number;
  populationSorted: Float32Array;
  snapshot: SpaceSnapshot | null;
  kitSummary: WeightsKitSummary | null;
  latticeSnapshot: SpaceSnapshot | null;
  latticeKitSummary: WeightsKitSummary | null;
  /** Lag with islands and missing values as NaN (what the lag display draws). */
  lagShown: Buffer;
  /** Lag minus value, NaN where either is missing. */
  difference: Buffer;
  /** Weight of each place in the focus row as a share of the row maximum, NaN elsewhere. */
  weightRows: Buffer;
  gatherer: FocusRowGatherer;
  focusData: FocusRow | null;
  focusOutline: Buffer;
  focusOutlineCount: number;
  neighbourOutline: Buffer;
  neighbourOutlineCount: number;
  /** Zone boundaries: state lines over counties, community-area lines over tracts. */
  zoneLines: {buffer: Buffer; count: number} | null;
  /** 1 for places with an unshared boundary edge (coast, shore, border, city limit). */
  outerEdge: Uint8Array;
  projection: LocalMetricProjection;
  reference: Reference | null;
  /** Corner-only neighbours of the focus (in queen, not in rook). */
  cornerRows: number[];
  cornerLinks: number | null;
  /** Mean and spread of the variable, for the difference classes. */
  spread: number;
};

const geographyCache = new Map<GeographyId, Promise<Geography>>();

/** Lattice cell containing the focus centroid of a world. */
function getLatticeCellOf(
  target: {geography: Geography; focusRow: number},
  minX: number,
  minY: number,
  cell: number,
  columns: number,
  rows: number
): number {
  const x = target.geography.centroids[target.focusRow * 2];
  const y = target.geography.centroids[target.focusRow * 2 + 1];
  const column = Math.min(columns - 1, Math.max(0, Math.floor((x - minX) / cell)));
  const row = Math.min(rows - 1, Math.max(0, Math.floor((y - minY) / cell)));
  return row * columns + column;
}

/** Standard deviation of the finite values. */
function getSpread(values: ArrayLike<number>): number {
  let count = 0;
  let sum = 0;
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) {
      count++;
      sum += values[index];
    }
  }
  if (count < 2) return 1;
  const mean = sum / count;
  let squares = 0;
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) squares += (values[index] - mean) ** 2;
  }
  return Math.sqrt(squares / (count - 1));
}

/**
 * Spatial weights on real polygons. One `GPUContiguityWeights` pass (queen or rook) or one
 * `GPUNeighborSearch` pass (kNN or distance band on centroids) or `GPULatticeWeights` (a raster
 * grid) writes a CSR; `GPUSpatialWeightsTransform` rewrites the weights, `GPUSpatialWeightsAlgebra`
 * combines neighbourhoods, `GPUSpatialWeightsSummary` and `GPUSpatialWeightsTranspose` diagnose
 * them, and `GPUSpatialLag` and `GPUNeighborhoodSummary` use them. The scene reads the focus row
 * back (ids and weights) to draw the bundle, the kernel weights and the notes, and builds queen
 * and rook side by side when the corner-only links are asked for. Variants compile on first use
 * and are kept, so a compile-time option rebuilds once and a parameter never does.
 */
export async function createSpatialWeights(
  ctx: SceneContext<SpatialWeightsOptions>
): Promise<SceneInstance<SpatialWeightsOptions>> {
  const {device} = ctx;
  let destroyed = false;
  let world: World | null = null;
  let switchToken = 0;
  let dirty = true;
  let readbackStale = true;
  let focusStale = true;
  let referenceStale = true;
  let tables: WeightsTables | null = null;
  let counts: WeightsCounts = {};
  let legendHighlight: number[] | null = null;

  const getGeography = (id: GeographyId) => {
    let promise = geographyCache.get(id);
    if (!promise) {
      promise = loadGeography(id, ctx.datasets, ctx.signal);
      promise.catch(() => geographyCache.delete(id));
      geographyCache.set(id, promise);
    }
    return promise;
  };

  const getConfig = (geography: Geography): WeightsConfig => {
    const o = ctx.options;
    return {
      source: o.source,
      k: o.k,
      snapTolerance: Number(o.snapTolerance),
      bandFactor: o.bandFactor,
      knnCapFactor: o.knnCapFactor,
      weightKind: o.weightKind,
      kernel: o.kernel,
      power: o.power,
      distanceFloor: o.distanceFloorFactor * geography.medianSpacing,
      rowStandardize: o.rowStandardize,
      transform: o.transform,
      transformKernel: o.transformKernel,
      bandwidthFactor: o.bandwidth === 'adaptive' ? 0 : Number(o.bandwidth),
      doubleSum: o.doubleSum
    };
  };

  const getAlgebraKey = (): string | null => {
    const o = ctx.options;
    switch (o.combine) {
      case 'none':
        return null;
      case 'union':
      case 'intersection':
      case 'difference':
      case 'symmetricDifference':
        return `${o.combine}-${o.weightRule}`;
      case 'higherOrder':
        return `higherOrder-${o.order}-${o.cumulative}`;
      default:
        return o.combine;
    }
  };

  // ---------------------------------------------------------------------------------------
  // Geometry helpers
  // ---------------------------------------------------------------------------------------

  /** The rings of one place as `[lng, lat]`, closed, for highlights and outline annotations. */
  const getRingsLngLat = (target: World, row: number): LngLat[][] => {
    const {geography, projection} = target;
    const rings: LngLat[][] = [];
    for (
      let ring = geography.featureRingOffsets[row];
      ring < geography.featureRingOffsets[row + 1];
      ring++
    ) {
      const points: LngLat[] = [];
      const first = geography.contiguityRingOffsets[ring];
      const end = geography.contiguityRingOffsets[ring + 1];
      for (let vertex = first; vertex < end; vertex++) {
        points.push(
          projection.unproject(
            geography.contiguityVertices[vertex * 2],
            geography.contiguityVertices[vertex * 2 + 1]
          ) as LngLat
        );
      }
      if (points.length) points.push(points[0]);
      rings.push(points);
    }
    return rings;
  };

  /** Centroid of one place as `[lng, lat]`. */
  const getCentroidLngLat = (target: World, row: number): LngLat =>
    target.projection.unproject(
      target.geography.centroids[row * 2],
      target.geography.centroids[row * 2 + 1]
    ) as LngLat;

  /** A label point inside one place, snapped to its geometry (falls back to the centroid). */
  const getLabelPoint = (target: World, row: number): LngLat => {
    const geojson = ctx.datasets.get(target.geography.id).geojson;
    const feature = geojson?.features[row];
    return (
      (feature ? getFeatureLabelPoint(feature.geometry) : null) ?? getCentroidLngLat(target, row)
    );
  };

  /**
   * Ground metres per planar metre at the focus. The search measures distance in metres that are
   * true at the latitude of the dataset origin; Web Mercator stretches them by cos(origin) over
   * cos(focus latitude) at the focus, so a planar radius `r` covers `r * factor` ground metres.
   */
  const getGroundFactor = (target: World): number => {
    const focusLatitude = getCentroidLngLat(target, target.focusRow)[1];
    return (
      Math.cos((focusLatitude * Math.PI) / 180) /
      Math.cos((target.geography.origin[1] * Math.PI) / 180)
    );
  };

  /** Writes ring outlines of `rows` into `buffer` (up to `capacity` segments); returns the count. */
  const writeOutlines = (
    target: World,
    buffer: Buffer,
    capacity: number,
    rows: readonly number[]
  ): number => {
    const {geography} = target;
    const data = new Float32Array(capacity * 4);
    let count = 0;
    for (const row of rows) {
      for (
        let ring = geography.featureRingOffsets[row];
        ring < geography.featureRingOffsets[row + 1];
        ring++
      ) {
        const first = geography.contiguityRingOffsets[ring];
        const end = geography.contiguityRingOffsets[ring + 1];
        for (let vertex = first; vertex < end && count < capacity; vertex++) {
          const next = vertex + 1 < end ? vertex + 1 : first;
          data[count * 4] = geography.contiguityVertices[vertex * 2];
          data[count * 4 + 1] = geography.contiguityVertices[vertex * 2 + 1];
          data[count * 4 + 2] = geography.contiguityVertices[next * 2];
          data[count * 4 + 3] = geography.contiguityVertices[next * 2 + 1];
          count++;
        }
      }
    }
    if (count > 0) buffer.write(data.subarray(0, count * 4));
    return count;
  };

  // ---------------------------------------------------------------------------------------
  // Class tables and legends
  // ---------------------------------------------------------------------------------------

  const formatVariable = (variable: VariableId) => {
    const info = getVariableInfo(variable);
    return (value: number) => (info.digits === 0 ? formatCount(value) : value.toFixed(info.digits));
  };

  /** Builds the class tables of the current variable and ground (fixed until one of them changes). */
  const buildTables = (target: World): WeightsTables => {
    const ground = ctx.ground();
    const info = getVariableInfo(ctx.options.variable);
    const format = formatVariable(ctx.options.variable);
    const values = target.geography.getVariable(ctx.options.variable);
    return {
      cardinality: getCardinalityTable(ground, `neighbours per ${target.geography.unit}`),
      oneWay: getOneWayTable(ground),
      weights: getWeightTable(ground),
      value: getQuantileTable(values, 'YlOrBr', ground, {
        unit: info.unit,
        format,
        method: 'Quantiles of the variable; the lag reuses them',
        noDataLabel: 'No data'
      }),
      difference: getDifferenceTable(target.spread, ground, {unit: info.unit, format}),
      summary: tables?.summary ?? null
    };
  };

  const publishLegendData = () => {
    if (!tables) return;
    ctx.setLegendData('weights', {
      ground: ctx.ground(),
      tables,
      counts
    } satisfies WeightsLegendData);
  };

  const refreshTables = () => {
    const target = world;
    if (!target) return;
    tables = buildTables(target);
    publishLegendData();
  };

  /** Hue, unit and number format of the neighbourhood statistic on display. */
  const getSummaryStyle = (target: World) => {
    const o = ctx.options;
    const info = getVariableInfo(o.variable);
    const format = formatVariable(o.variable);
    switch (o.summary) {
      case 'count':
        return {
          scheme: 'BuGn' as const,
          unit: 'neighbours',
          format: formatCount,
          title: 'Neighbours counted'
        };
      case 'weightSum':
        return {
          scheme: 'BuGn' as const,
          unit: 'weight',
          format: (value: number) => value.toFixed(2),
          title: 'Sum of weights'
        };
      case 'entropy':
        return {
          scheme: 'Greys' as const,
          unit: 'nats',
          format: (value: number) => value.toFixed(2),
          title: `Group diversity of the ${target.geography.unitPlural}`
        };
      default:
        return {
          scheme: 'YlOrBr' as const,
          unit: info.unit,
          format,
          title: `Neighbourhood ${o.summary}`
        };
    }
  };

  // ---------------------------------------------------------------------------------------
  // World
  // ---------------------------------------------------------------------------------------

  const buildWorld = (geography: Geography): World => {
    const resources = new SpatialAnalysisResources(device, `weights-${geography.id}`);
    const rowCount = geography.count;
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const core = createWeightsCore({device, resources, id: 'weights-core', geography});
    const analysisSlots = ANALYSIS_SLOTS[geography.id];
    const initialValues = geography.getVariable(ctx.options.variable);
    const values = create('values', Float32Array.from(initialValues));
    const valueMask = create('value-mask', new Uint32Array(rowCount));
    const displayMask = create('display-mask', new Uint32Array(rowCount).fill(1));
    const space = createAnalysisSpace({
      device,
      resources,
      id: 'polygons',
      rows: rowCount,
      slots: analysisSlots,
      unionSlots: analysisSlots,
      positions: core.positions,
      values,
      mask: valueMask,
      displayMask,
      categories: geography.groups,
      drawAllLinks: true,
      producer: {overflow: core.overflow, total: core.total}
    });
    const populationSorted = Float32Array.from(geography.population).sort();
    const focusRow = Math.max(0, geography.pick(...FOCUS_POINTS[geography.id][ctx.options.focus]));
    const projection = new LocalMetricProjection(geography.origin);

    // The zone-boundary tier: state lines over counties, community areas over tracts, projected
    // around the geography's origin so they sit exactly on the polygon edges.
    const zoneGeojson =
      geography.id === 'us-counties'
        ? ctx.datasets.get('us-states').geojson
        : ctx.datasets.get('chicago-community-areas').geojson;
    const zoneSegments = zoneGeojson
      ? projectRingsToSegments(
          getInputPolygons(zoneGeojson).flatMap(({polygon}) => polygon),
          getLocalProjector(geography.origin)
        )
      : new Float32Array(0);

    const gatherer = createFocusRowGatherer({
      device,
      resources,
      id: 'weights',
      rows: rowCount,
      slots: analysisSlots,
      csr: space.csr,
      focusRow: space.kit.focusRow,
      onRow: row => {
        const current = world;
        if (current && current.gatherer === gatherer) handleFocusRow(current, row);
      }
    });
    const created: World = {
      geography,
      resources,
      core,
      partner: core.createPartner('partner'),
      space,
      buffers: createGeographyBuffers(resources, geography, 'weights'),
      values,
      valueMask,
      selfWeights: create('self-weights', new Float32Array(rowCount)),
      subgraphMask: create('subgraph-mask', new Uint32Array(rowCount).fill(1)),
      groups: create('group-ids', geography.groups),
      algebra: new Map(),
      algebraOverflow: create('algebra-overflow', 4),
      algebraTotal: create('algebra-total', 4),
      lattice: null,
      focusRow,
      populationSorted,
      snapshot: null,
      kitSummary: null,
      latticeSnapshot: null,
      latticeKitSummary: null,
      lagShown: create('lag-shown', new Float32Array(rowCount).fill(Number.NaN)),
      difference: create('difference', new Float32Array(rowCount).fill(Number.NaN)),
      weightRows: create('weight-rows', new Float32Array(rowCount).fill(Number.NaN)),
      gatherer,
      focusData: null,
      focusOutline: create('focus-outline', FOCUS_OUTLINE_CAPACITY * 16),
      focusOutlineCount: 0,
      neighbourOutline: create('neighbour-outline', NEIGHBOUR_OUTLINE_CAPACITY * 16),
      neighbourOutlineCount: 0,
      zoneLines: zoneSegments.length
        ? {buffer: create('zone-lines', zoneSegments), count: zoneSegments.length / 4}
        : null,
      outerEdge: getOuterEdgeRows(geography),
      projection,
      reference: null,
      cornerRows: [],
      cornerLinks: null,
      spread: getSpread(initialValues)
    };
    writeValueMask(created);
    space.kit.setFocusRow(focusRow);
    return created;
  };

  const writeValueMask = (target: World) => {
    const values = target.geography.getVariable(ctx.options.variable);
    target.valueMask.write(Uint32Array.from(values, value => (Number.isFinite(value) ? 1 : 0)));
  };

  const writeVariable = (target: World) => {
    const values = target.geography.getVariable(ctx.options.variable);
    target.values.write(Float32Array.from(values));
    target.spread = getSpread(values);
    writeValueMask(target);
    if (target.lattice) writeLatticeValues(target, target.lattice);
  };

  const writeFocus = (target: World) => {
    const row = target.geography.pick(...FOCUS_POINTS[target.geography.id][ctx.options.focus]);
    if (row < 0) return;
    target.focusRow = row;
    target.space.kit.setFocusRow(row);
    if (target.lattice) {
      const [minX, minY] = target.lattice.bounds;
      target.lattice.focusRow = getLatticeCellOf(
        target,
        minX,
        minY,
        target.lattice.cell,
        target.lattice.columns,
        target.lattice.rows
      );
      target.lattice.space.kit.setFocusRow(target.lattice.focusRow);
    }
  };

  const writeSelfWeights = (target: World) => {
    target.selfWeights.write(new Float32Array(target.geography.count).fill(ctx.options.selfWeight));
  };

  const writeSubgraphMask = (target: World) => {
    const sorted = target.populationSorted;
    const threshold =
      sorted[
        Math.min(
          sorted.length - 1,
          Math.floor((ctx.options.populationPercentile / 100) * sorted.length)
        )
      ];
    target.subgraphMask.write(
      Uint32Array.from(target.geography.population, value => (value >= threshold ? 1 : 0))
    );
  };

  /** Builds the lattice the first time the lattice source is chosen. */
  const buildLattice = (target: World): Lattice => {
    const {geography, resources} = target;
    const columns = LATTICE_COLUMNS[geography.id];
    const [minX, minY, maxX, maxY] = geography.bounds;
    const cell = (maxX - minX) / columns;
    const rows = Math.ceil((maxY - minY) / cell);
    const count = columns * rows;
    const featureOfCell = new Int32Array(count).fill(-1);
    const positions = new Float32Array(count * 2);
    const categories = new Uint32Array(count);
    const insideMask = new Uint32Array(count);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        const index = row * columns + column;
        const x = minX + (column + 0.5) * cell;
        const y = minY + (row + 0.5) * cell;
        positions[index * 2] = x;
        positions[index * 2 + 1] = y;
        const feature = geography.pickMeters(x, y);
        featureOfCell[index] = feature;
        if (feature >= 0) {
          insideMask[index] = 1;
          categories[index] = geography.groups[feature];
        }
      }
    }
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(`lattice-${name}`, data);
    const slots = count * ((2 * LATTICE_MAXIMUM_RADIUS + 1) ** 2 - 1);
    const values = create('values', new Float32Array(count));
    const valueMask = create('value-mask', new Uint32Array(count));
    const structureMask = create('structure-mask', Uint32Array.from(insideMask));
    const overflow = create('overflow', 4);
    const total = create('total', 4);
    const space = createAnalysisSpace({
      device,
      resources,
      id: 'lattice',
      rows: count,
      slots,
      unionSlots: slots,
      positions: create('positions', positions),
      values,
      mask: valueMask,
      displayMask: structureMask,
      categories,
      drawAllLinks: false,
      producer: {overflow, total}
    });
    const lattice: Lattice = {
      columns,
      rows,
      cell,
      bounds: [minX, minY, minX + columns * cell, minY + rows * cell],
      featureOfCell,
      space,
      structureMask,
      values,
      valueMask,
      producers: new Map(),
      overflow,
      total,
      focusRow: getLatticeCellOf(target, minX, minY, cell, columns, rows),
      insideMask
    };
    space.kit.setFocusRow(lattice.focusRow);
    writeLatticeValues(target, lattice);
    writeLatticeStructure(lattice);
    return lattice;
  };

  const writeLatticeValues = (target: World, lattice: Lattice) => {
    const source = target.geography.getVariable(ctx.options.variable);
    const values = new Float32Array(lattice.featureOfCell.length);
    const mask = new Uint32Array(lattice.featureOfCell.length);
    for (let index = 0; index < values.length; index++) {
      const feature = lattice.featureOfCell[index];
      const value = feature >= 0 ? source[feature] : Number.NaN;
      values[index] = value;
      mask[index] = Number.isFinite(value) ? 1 : 0;
    }
    lattice.values.write(values);
    lattice.valueMask.write(mask);
  };

  const writeLatticeStructure = (lattice: Lattice) => {
    lattice.structureMask.write(
      ctx.options.latticeMask
        ? Uint32Array.from(lattice.insideMask)
        : new Uint32Array(lattice.insideMask.length).fill(1)
    );
  };

  const prepareLatticeProducer = (lattice: Lattice): boolean => {
    const {latticeCriterion: criterion, latticeRadius: radius} = ctx.options;
    const key = `${criterion}-${radius}`;
    if (lattice.producers.has(key)) return false;
    const {space} = lattice;
    const graph = new GPUCommandGraph<void>(device, {id: `lattice-${key}`});
    const count = lattice.columns * lattice.rows;
    graph.add(
      new GPULatticeWeights({
        id: 'lattice',
        width: lattice.columns,
        height: lattice.rows,
        criterion,
        radius,
        cellSize: [lattice.cell, lattice.cell],
        mask: importGraphBuffer(graph, 'mask', lattice.structureMask, 'uint32', count),
        weights: {
          offsets: importGraphBuffer(graph, 'offsets', space.csr.offsets, 'uint32', count + 1),
          neighbors: importGraphBuffer(
            graph,
            'neighbors',
            space.csr.neighbors,
            'uint32',
            space.slots
          ),
          weights: importGraphBuffer(graph, 'weights', space.csr.weights, 'float32', space.slots)
        },
        overflow: importGraphBuffer(graph, 'overflow', lattice.overflow, 'uint32', 1),
        totalNeighbors: importGraphBuffer(graph, 'total', lattice.total, 'uint32', 1)
      })
    );
    lattice.producers.set(key, world!.resources.track(graph.compile()));
    return true;
  };

  /** Compiles the algebra variant of the current options. */
  const prepareAlgebra = (target: World): boolean => {
    const key = getAlgebraKey();
    if (!key || target.algebra.has(key)) return false;
    const o = ctx.options;
    const {geography, space, core} = target;
    const rowCount = geography.count;
    const graph = new GPUCommandGraph<void>(device, {id: `weights-algebra-${key}`});
    const importCsr = (
      name: string,
      csr: {offsets: Buffer; neighbors: Buffer; weights: Buffer},
      slots: number
    ) => ({
      offsets: importGraphBuffer(graph, `${name}-offsets`, csr.offsets, 'uint32', rowCount + 1),
      neighbors: importGraphBuffer(graph, `${name}-neighbors`, csr.neighbors, 'uint32', slots),
      weights: importGraphBuffer(graph, `${name}-weights`, csr.weights, 'float32', slots)
    });
    const left = importCsr('a', core.csr, core.slots);
    const common = {
      id: 'algebra',
      output: importCsr('out', space.csr, space.slots),
      overflow: importGraphBuffer(graph, 'overflow', target.algebraOverflow, 'uint32', 1),
      totalNeighbors: importGraphBuffer(graph, 'total', target.algebraTotal, 'uint32', 1)
    };
    switch (o.combine) {
      case 'union':
      case 'intersection':
      case 'difference':
      case 'symmetricDifference':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: o.combine,
            left,
            right: importCsr('b', target.partner.csr, target.partner.slots),
            weightRule: o.weightRule
          })
        );
        break;
      case 'higherOrder':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'higherOrder',
            weights: left,
            order: o.order,
            cumulative: o.cumulative
          })
        );
        break;
      case 'selfWeight':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'selfWeight',
            weights: left,
            selfWeight: importGraphBuffer(
              graph,
              'self-weights',
              target.selfWeights,
              'float32',
              rowCount
            )
          })
        );
        break;
      case 'subgraph':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'subgraph',
            weights: left,
            mask: importGraphBuffer(graph, 'mask', target.subgraphMask, 'uint32', rowCount)
          })
        );
        break;
      case 'block':
        graph.add(
          new GPUSpatialWeightsAlgebra({
            ...common,
            operation: 'block',
            groupIds: importGraphBuffer(graph, 'groups', target.groups, 'uint32', rowCount),
            groupCount: geography.groupCount
          })
        );
        break;
      default:
        return false;
    }
    target.algebra.set(key, target.resources.track(graph.compile()));
    return true;
  };

  const isBinaryCombine = () =>
    ['union', 'intersection', 'difference', 'symmetricDifference'].includes(ctx.options.combine);

  /** The configuration of the two reference producers: exact contiguity, no transform. */
  const getReferenceConfig = (target: World, source: 'queen' | 'rook'): WeightsConfig => ({
    ...getConfig(target.geography),
    source,
    snapTolerance: 0,
    transform: 'none'
  });

  /**
   * Builds queen and rook contiguity side by side the first time the corner-only links are asked
   * for; both are compiled once and encoded once per world, and their focus rows are read back.
   */
  const ensureReference = (target: World): Reference => {
    if (target.reference) return target.reference;
    const {geography, resources} = target;
    const queen = createWeightsCore({device, resources, id: 'weights-queen', geography});
    const rook = createWeightsCore({device, resources, id: 'weights-rook', geography});
    const rows = geography.count;
    const reader = new SummaryReader(
      resources,
      'weights-reference',
      [
        {buffer: queen.csr.offsets, size: (rows + 1) * 4},
        {buffer: queen.csr.neighbors, size: queen.slots * 4},
        {buffer: rook.csr.offsets, size: (rows + 1) * 4},
        {buffer: rook.csr.neighbors, size: rook.slots * 4}
      ],
      bytes => {
        const reference = target.reference;
        if (!reference || world !== target) return;
        const offsetBytes = (rows + 1) * 4;
        const slotBytes = queen.slots * 4;
        reference.data = {
          queenOffsets: new Uint32Array(bytes, 0, rows + 1),
          queenNeighbors: new Uint32Array(bytes, offsetBytes, queen.slots),
          rookOffsets: new Uint32Array(bytes, offsetBytes + slotBytes, rows + 1),
          rookNeighbors: new Uint32Array(bytes, offsetBytes * 2 + slotBytes, rook.slots)
        };
        updateCorners(target);
      }
    );
    target.reference = {queen, rook, reader, encoded: false, data: null};
    return target.reference;
  };

  /** Compiles every variant the current options need. Called from `create` and `setOption`. */
  const prepareAll = () => {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    const focalWeight = Number(o.focalWeight);
    if (o.lattice) {
      target.lattice ??= buildLattice(target);
      prepareLatticeProducer(target.lattice);
      target.lattice.space.prepare(o.includeFocal, focalWeight);
    } else {
      target.core.prepare(getConfig(target.geography));
      if (isBinaryCombine()) target.partner.prepare(o.partnerK);
      prepareAlgebra(target);
      target.space.prepare(o.includeFocal, focalWeight);
      if (o.showQueenOnly) {
        const reference = ensureReference(target);
        reference.queen.prepare(getReferenceConfig(target, 'queen'));
        reference.rook.prepare(getReferenceConfig(target, 'rook'));
      }
    }
    const display = getEffectiveDisplay(o);
    const spaceDisplay: SpaceDisplay =
      display === 'weights' ? 'focus' : display === 'difference' ? 'lag' : display;
    target.space.setDisplayChoice(spaceDisplay, o.summary);
    target.lattice?.space.setDisplayChoice(spaceDisplay, o.summary);
  };

  const writeParameters = () => {
    const target = world;
    if (!target) return;
    target.core.writeParameters(getConfig(target.geography));
  };

  // ---------------------------------------------------------------------------------------
  // Readbacks: focus row, snapshot, corner links
  // ---------------------------------------------------------------------------------------

  /** Builds the CPU-derived buffers the lag, difference and weights displays draw. */
  const writeDerived = (target: World, snapshot: SpaceSnapshot) => {
    const {geography} = target;
    const values = geography.getVariable(ctx.options.variable);
    const lag = new Float32Array(geography.count);
    const difference = new Float32Array(geography.count);
    for (let row = 0; row < geography.count; row++) {
      const usable =
        snapshot.cardinality[row] > 0 &&
        Number.isFinite(values[row]) &&
        Number.isFinite(snapshot.lag[row]);
      lag[row] = usable ? snapshot.lag[row] : Number.NaN;
      difference[row] = usable ? snapshot.lag[row] - values[row] : Number.NaN;
    }
    target.lagShown.write(lag);
    target.difference.write(difference);
    return {lag, difference, values};
  };

  /** The snapshot of the analysed places: derived buffers, class counts, charts, readouts. */
  const handleSnapshot = (target: World, snapshot: SpaceSnapshot, isLattice: boolean) => {
    if (isLattice) {
      target.latticeSnapshot = snapshot;
      updateReadouts();
      return;
    }
    target.snapshot = snapshot;
    const {lag, difference, values} = writeDerived(target, snapshot);
    if (!tables) tables = buildTables(target);
    const o = ctx.options;
    if (getEffectiveDisplay(o) === 'summary') {
      const style = getSummaryStyle(target);
      tables = {
        ...tables,
        summary: getSummaryTable(snapshot.shown, style.scheme, ctx.ground(), style)
      };
    }
    counts = {
      cardinality: getClassCounts(snapshot.cardinality, tables.cardinality.breaks),
      oneWay: getClassCounts(snapshot.oneWay, tables.oneWay.breaks),
      value: getClassCounts(values, tables.value.breaks),
      lag: getClassCounts(lag, tables.value.breaks),
      difference: getClassCounts(difference, tables.difference.breaks),
      summary: tables.summary ? getClassCounts(snapshot.shown, tables.summary.breaks) : undefined
    };
    publishLegendData();
    ctx.setChart(
      'cardinality',
      getCardinalityChart(
        snapshot.cardinality,
        tables.cardinality,
        ctx.ground(),
        target.geography.unitPlural
      )
    );
    const scatter = getLagScatter({
      values,
      lag,
      unit: getVariableInfo(o.variable).unit
    });
    ctx.setChart('lagScatter', scatter);
    ctx.setReadout('lagSlope', Number.isFinite(scatter.slope) ? scatter.slope.toFixed(2) : null);
    updateReadouts();
    publishAnnotations();
    ctx.requestLayers();
  };

  /** The focus row arrived: draw the bundle outlines, weights, kernel chart and notes. */
  function handleFocusRow(target: World, row: FocusRow) {
    target.focusData = row;
    const {geography} = target;
    let maximum = 0;
    for (let slot = 0; slot < row.degree; slot++) {
      if (row.ids[slot] !== target.focusRow) maximum = Math.max(maximum, row.weights[slot]);
    }
    const weightRows = new Float32Array(geography.count).fill(Number.NaN);
    const neighbours: number[] = [];
    for (let slot = 0; slot < row.degree; slot++) {
      const id = row.ids[slot];
      if (id === target.focusRow || id >= geography.count) continue;
      neighbours.push(id);
      weightRows[id] = maximum > 0 ? row.weights[slot] / maximum : 0;
    }
    target.weightRows.write(weightRows);
    target.focusOutlineCount = writeOutlines(target, target.focusOutline, FOCUS_OUTLINE_CAPACITY, [
      target.focusRow
    ]);
    target.neighbourOutlineCount = writeOutlines(
      target,
      target.neighbourOutline,
      NEIGHBOUR_OUTLINE_CAPACITY,
      neighbours
    );
    publishKernelChart(target);
    updateReadouts();
    publishFurniture();
    publishAnnotations();
    ctx.requestLayers();
  }

  /** Queen and rook focus rows arrived (or the focus moved): the corner-only neighbours. */
  function updateCorners(target: World) {
    const data = target.reference?.data;
    if (!data) return;
    const focus = target.focusRow;
    const rookSet = new Set<number>();
    for (let slot = data.rookOffsets[focus]; slot < data.rookOffsets[focus + 1]; slot++) {
      rookSet.add(data.rookNeighbors[slot]);
    }
    const cornerRows: number[] = [];
    for (let slot = data.queenOffsets[focus]; slot < data.queenOffsets[focus + 1]; slot++) {
      if (!rookSet.has(data.queenNeighbors[slot])) cornerRows.push(data.queenNeighbors[slot]);
    }
    const rows = target.geography.count;
    target.cornerRows = cornerRows;
    target.cornerLinks = (data.queenOffsets[rows] - data.rookOffsets[rows]) / 2;
    updateReadouts();
    publishAnnotations();
    ctx.requestLayers();
  }

  /** The kernel curve with the focus row's neighbours on it, when the weights are a kernel. */
  function publishKernelChart(target: World) {
    const o = ctx.options;
    const row = target.focusData;
    const config = getConfig(target.geography);
    const proportional = ['none', 'row', 'double', 'kernel'].includes(o.transform);
    if (
      !row ||
      o.lattice ||
      row.degree === 0 ||
      !proportional ||
      !target.core.hasDistances(config)
    ) {
      ctx.setChart('kernelCurve', null);
      return;
    }
    const {centroids} = target.geography;
    const focus = target.focusRow;
    const distances: number[] = [];
    const weights: number[] = [];
    for (let slot = 0; slot < row.degree; slot++) {
      const id = row.ids[slot];
      if (id === focus) continue;
      distances.push(
        Math.hypot(
          centroids[id * 2] - centroids[focus * 2],
          centroids[id * 2 + 1] - centroids[focus * 2 + 1]
        )
      );
      weights.push(row.weights[slot]);
    }
    const farthest = distances.length ? Math.max(...distances) : 0;
    let kernel: KernelName | null = null;
    let bandwidth = 0;
    if (o.transform === 'kernel') {
      kernel = o.transformKernel as KernelName;
      bandwidth =
        o.bandwidth === 'adaptive'
          ? farthest
          : Number(o.bandwidth) * target.geography.medianSpacing;
    } else if (o.weightKind === 'kernel') {
      kernel = o.kernel as KernelName;
      bandwidth = o.source === 'band' ? target.core.getBandMeters(config) : farthest;
    }
    if (!kernel || !(bandwidth > 0) || distances.length === 0) {
      ctx.setChart('kernelCurve', null);
      return;
    }
    ctx.setChart(
      'kernelCurve',
      getKernelChart({
        kernel,
        bandwidth,
        groundFactor: getGroundFactor(target),
        distances,
        weights,
        focusName: target.geography.getName(focus)
      })
    );
  }

  const getFocusName = (target: World): string => {
    if (ctx.options.lattice && target.lattice) return `cell ${target.lattice.focusRow}`;
    const {geography, focusRow} = target;
    return `${geography.getName(focusRow)}, ${geography.getGroupName(focusRow)}`;
  };

  /** Readouts from the latest summaries. */
  function updateReadouts() {
    const target = world;
    if (!target) return;
    const {geography} = target;
    const o = ctx.options;
    const useLattice = Boolean(o.lattice && target.lattice);
    const snapshot = useLattice ? target.latticeSnapshot : target.snapshot;
    const kit = useLattice ? target.latticeKitSummary : target.kitSummary;
    const rows = useLattice ? target.lattice!.columns * target.lattice!.rows : geography.count;
    const insideRows = useLattice ? target.lattice!.insideMask.reduce((a, b) => a + b, 0) : rows;
    ctx.setReadout(
      'rows',
      useLattice ? `${formatCount(insideRows)} of ${formatCount(rows)} cells` : formatCount(rows)
    );
    ctx.setReadout('focusName', getFocusName(target));
    if (kit) {
      ctx.setReadout('links', formatCount(kit.slots));
      ctx.setReadout('meanNeighbors', (kit.slots / Math.max(1, rows - kit.isolates)).toFixed(1));
      ctx.setReadout('asymmetric', formatCount(kit.asymmetricSlots));
      ctx.setReadout(
        'sums',
        `${formatCount(kit.s0)} / ${formatCount(kit.s1)} / ${formatCount(kit.s2)}`
      );
      ctx.setReadout(
        'union',
        kit.unionOverflow ? 'overflow' : `${formatCount(kit.unionSlots)} slots`
      );
      const variable = getVariableInfo(o.variable);
      ctx.setReadout(
        'lagRange',
        `${kit.lagMinimum.toFixed(variable.digits)} to ${kit.lagMaximum.toFixed(variable.digits)} ${variable.unit}`
      );
      ctx.setReadout(
        'capacity',
        kit.producerOverflow ? 'producer overflow: some links dropped' : 'ok'
      );
      ctx.setReadout('islands', formatCount(kit.isolates));
      const focus = kit.focus[o.matrix];
      ctx.setReadout('focusDegree', formatCount(focus.degree));
      ctx.setReadout('weightSum', focus.weightSum.toFixed(2));
    }
    if (snapshot && !useLattice) {
      const names: string[] = [];
      let oneWayPlaces = 0;
      let edgeSum = 0;
      let edgeCount = 0;
      let interiorSum = 0;
      let interiorCount = 0;
      for (let row = 0; row < rows; row++) {
        const cardinality = snapshot.cardinality[row];
        if (cardinality === 0 && names.length < MAXIMUM_NAMED_ROWS) {
          names.push(geography.getName(row));
        }
        if (snapshot.oneWay[row] > 0) oneWayPlaces++;
        if (cardinality > 0) {
          if (target.outerEdge[row]) {
            edgeSum += cardinality;
            edgeCount++;
          } else {
            interiorSum += cardinality;
            interiorCount++;
          }
        }
      }
      ctx.setReadout('islandNames', names.length ? names.join(', ') : 'none');
      ctx.setReadout('oneWayPlaces', formatCount(oneWayPlaces));
      ctx.setReadout('edgeMean', edgeCount ? (edgeSum / edgeCount).toFixed(1) : null);
      ctx.setReadout(
        'interiorMean',
        interiorCount ? (interiorSum / interiorCount).toFixed(1) : null
      );
    }
    if (target.cornerLinks !== null) {
      ctx.setReadout('cornerLinks', formatCount(target.cornerLinks));
      ctx.setReadout(
        'cornerFocus',
        target.cornerRows.length
          ? target.cornerRows
              .slice(0, MAXIMUM_NAMED_ROWS)
              .map(row => `${geography.getName(row)}, ${geography.getGroupName(row)}`)
              .join('; ')
          : 'none'
      );
      ctx.setReadout('cornerCount', formatCount(target.cornerRows.length));
    }
    const config = getConfig(geography);
    const planar = target.core.getBandMeters(config);
    ctx.setReadout(
      'band',
      o.source === 'band' && !o.lattice ? formatDistance(planar * getGroundFactor(target)) : null
    );
    ctx.setReadout(
      'spacing',
      formatDistance(geography.medianSpacing * (o.lattice ? 1 : getGroundFactor(target)))
    );
    ctx.setReadout(
      'numerics',
      `Distances are planar metres, true at ${geography.origin[1].toFixed(1)} N; at the focus a planar metre is ${getGroundFactor(target).toFixed(2)} ground metres. Band radius ${formatDistance(planar)} planar. Weights are float32; lags sum in slot order.`
    );
    ctx.setCost({records: rows});
  }

  /** The standing sample line and the scale bar tick of the band, as runtime furniture. */
  function publishFurniture() {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    const {geography} = target;
    const sample =
      geography.id === 'us-counties'
        ? `${formatCount(geography.count)} counties of the contiguous US (Alaska and Hawaii not in the data)`
        : `${formatCount(geography.count)} census tracts of Chicago, 2020 boundaries`;
    if (o.source === 'band' && !o.lattice) {
      const factor = getGroundFactor(target);
      ctx.setFurniture({
        title: {sample},
        scaleBar: {
          latitude: getCentroidLngLat(target, target.focusRow)[1],
          ticks: [target.core.getBandMeters(getConfig(geography)) * factor]
        }
      });
    } else {
      ctx.setFurniture({title: {sample}});
    }
  }

  // ---------------------------------------------------------------------------------------
  // Annotations: the finding notes follow the data
  // ---------------------------------------------------------------------------------------

  function publishAnnotations() {
    const target = world;
    if (!target) return;
    const o = ctx.options;
    const {geography} = target;
    const display = getEffectiveDisplay(o);
    if (o.lattice) {
      ctx.setAnnotations('weights', null);
      return;
    }
    const list: MapAnnotation[] = [];
    const snapshot = target.snapshot;
    const name = (row: number) => `${geography.getName(row)}, ${geography.getGroupName(row)}`;

    if (display === 'focus' || display === 'weights') {
      const degree = target.kitSummary?.focus[o.matrix].degree;
      if (degree !== undefined) {
        list.push({
          kind: 'note',
          id: 'weights-focus-note',
          coordinate: getLabelPoint(target, target.focusRow),
          title: liveText('{n} {noun}, {rule}', {
            n: degree,
            noun: degree === 1 ? 'neighbour' : 'neighbours',
            rule: RULE_NAMES[o.source]
          }),
          text: name(target.focusRow),
          priority: 6
        });
      }
      if (o.source === 'band') {
        const radius = target.core.getBandMeters(getConfig(geography)) * getGroundFactor(target);
        list.push({
          kind: 'ring',
          id: 'weights-band-ring',
          coordinate: getCentroidLngLat(target, target.focusRow),
          radiusMeters: radius,
          text: `${formatDistance(radius)} band`,
          dashed: true,
          tone: 'ink'
        });
      }
      if (o.showQueenOnly) {
        for (const row of target.cornerRows.slice(0, MAXIMUM_NAMED_ROWS)) {
          list.push({
            kind: 'outline',
            id: `weights-corner-outline-${row}`,
            rings: getRingsLngLat(target, row),
            dashed: true
          });
          list.push({
            kind: 'point',
            id: `weights-corner-label-${row}`,
            coordinate: getLabelPoint(target, row),
            text: geography.getName(row),
            marker: 'none',
            rank: 'subject',
            priority: 5
          });
        }
      }
    }

    if (display === 'neighbors' && snapshot && geography.id === 'us-counties') {
      let named = 0;
      for (let row = 0; row < geography.count && named < MAXIMUM_NAMED_ROWS; row++) {
        if (snapshot.cardinality[row] !== 0) continue;
        named++;
        list.push({
          kind: 'point',
          id: `weights-island-${row}`,
          coordinate: getLabelPoint(target, row),
          text: geography.getName(row),
          marker: 'none',
          rank: 'subject',
          priority: 5
        });
      }
      // The edge effect, read off a county on the outer edge of the map near the Rio Grande.
      const anchor = US.places['rio-grande-valley'].lngLat;
      const [anchorX, anchorY] = geography.project(anchor[0], anchor[1]);
      let nearest = -1;
      let nearestDistance = Number.POSITIVE_INFINITY;
      for (let row = 0; row < geography.count; row++) {
        if (!target.outerEdge[row] || snapshot.cardinality[row] === 0) continue;
        const distance =
          (geography.centroids[row * 2] - anchorX) ** 2 +
          (geography.centroids[row * 2 + 1] - anchorY) ** 2;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = row;
        }
      }
      if (nearest >= 0) {
        list.push({
          kind: 'note',
          id: 'weights-edge-note',
          coordinate: getLabelPoint(target, nearest),
          title: liveText('{n} neighbours at the edge', {n: snapshot.cardinality[nearest]}),
          text: name(nearest),
          priority: 6
        });
      }
    }

    if (display === 'oneWay' && snapshot && geography.id === 'us-counties') {
      const [west, south, east, north] = MOUNTAIN_WEST_BOUNDS;
      let best = -1;
      let bestCount = 0;
      for (let row = 0; row < geography.count; row++) {
        if (snapshot.oneWay[row] <= bestCount) continue;
        const [longitude, latitude] = getCentroidLngLat(target, row);
        if (longitude < west || longitude > east || latitude < south || latitude > north) continue;
        best = row;
        bestCount = snapshot.oneWay[row];
      }
      if (best >= 0) {
        list.push({
          kind: 'note',
          id: 'weights-one-way-note',
          coordinate: getLabelPoint(target, best),
          title: liveText('{n} one-way links', {n: bestCount}),
          text: `${name(best)} lists ${formatCount(snapshot.cardinality[best])} counties, ${formatCount(bestCount)} of which do not list it back`,
          priority: 6
        });
      }
    }

    if (
      (display === 'value' || display === 'lag' || display === 'difference') &&
      snapshot &&
      geography.id === 'chicago-tracts'
    ) {
      const variable = getVariableInfo(o.variable);
      const values = geography.getVariable(o.variable);
      for (const id of ['englewood', 'lincoln-park'] as const) {
        const place = CHICAGO.places[id];
        const row = geography.pick(place.lngLat[0], place.lngLat[1]);
        if (row < 0 || !Number.isFinite(values[row]) || !Number.isFinite(snapshot.lag[row]))
          continue;
        list.push({
          kind: 'note',
          id: `weights-tract-note-${id}`,
          coordinate: place.lngLat,
          title: `${values[row].toFixed(variable.digits)} here, ${snapshot.lag[row].toFixed(variable.digits)} around`,
          text: `${place.name}: tract value and neighbourhood lag, ${variable.unit}`,
          priority: 6
        });
      }
    }
    ctx.setAnnotations('weights', list.length ? list : null);
  }

  // ---------------------------------------------------------------------------------------
  // Readback requests
  // ---------------------------------------------------------------------------------------

  const requestReadbackIfNeeded = (commandEncoder: CommandEncoder) => {
    const target = world;
    if (!target) return;
    const useLattice = Boolean(ctx.options.lattice && target.lattice);
    const space = useLattice ? target.lattice!.space : target.space;
    if (readbackStale && !space.isReading()) {
      readbackStale = false;
      space.requestReadback(
        commandEncoder,
        summary => {
          if (useLattice) target.latticeKitSummary = summary;
          else target.kitSummary = summary;
          updateReadouts();
          publishAnnotations();
        },
        snapshot => handleSnapshot(target, snapshot, useLattice)
      );
    }
    if (focusStale && !useLattice && !target.gatherer.isReading()) {
      focusStale = false;
      target.gatherer.request(commandEncoder);
    }
    const reference = target.reference;
    if (referenceStale && reference?.encoded && !reference.reader.isPending) {
      referenceStale = false;
      reference.reader.request(commandEncoder);
    }
  };

  const copyCsr = (
    commandEncoder: CommandEncoder,
    from: {offsets: Buffer; neighbors: Buffer; weights: Buffer},
    to: {offsets: Buffer; neighbors: Buffer; weights: Buffer},
    rowCount: number,
    slots: number
  ) => {
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: from.offsets,
      destinationBuffer: to.offsets,
      size: (rowCount + 1) * 4
    });
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: from.neighbors,
      destinationBuffer: to.neighbors,
      size: slots * 4
    });
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: from.weights,
      destinationBuffer: to.weights,
      size: slots * 4
    });
  };

  const stopWorld = (target: World) => {
    target.space.stop();
    target.lattice?.space.stop();
    target.gatherer.stop();
    target.reference?.reader.stop();
  };

  const startWorld = () => {
    const target = world;
    if (!target) return;
    refreshTables();
    publishFurniture();
    updateReadouts();
    ctx.setChart('cardinality', null);
    ctx.setChart('kernelCurve', null);
    ctx.setChart('lagScatter', null);
    publishAnnotations();
  };

  const switchWorld = async (id: GeographyId) => {
    const token = ++switchToken;
    ctx.setStatus(`Loading ${id === 'us-counties' ? 'counties' : 'tracts'}...`);
    const geography = await getGeography(id);
    if (token !== switchToken || destroyed) return;
    const previous = world;
    world = null;
    ctx.requestLayers();
    if (previous) {
      stopWorld(previous);
      setTimeout(() => previous.resources.destroy(), 200);
    }
    world = buildWorld(geography);
    legendHighlight = null;
    counts = {};
    tables = null;
    prepareAll();
    writeParameters();
    writeSelfWeights(world);
    writeSubgraphMask(world);
    dirty = true;
    readbackStale = true;
    focusStale = true;
    referenceStale = true;
    startWorld();
    ctx.setStatus('');
    ctx.requestLayers();
  };

  const initial = await getGeography(ctx.options.geography);
  if (ctx.signal.aborted) throw new Error('aborted');
  world = buildWorld(initial);
  prepareAll();
  writeParameters();
  writeSelfWeights(world);
  writeSubgraphMask(world);
  startWorld();
  ctx.setStatus('');

  const getPickedRow = (event: {coordinate: readonly [number, number] | null}): number => {
    const target = world;
    if (!target || !event.coordinate) return -1;
    return target.geography.pick(event.coordinate[0], event.coordinate[1]);
  };

  const getLatticeCell = (event: {coordinate: readonly [number, number] | null}): number => {
    const target = world;
    const lattice = target?.lattice;
    if (!target || !lattice || !event.coordinate) return -1;
    const [x, y] = target.geography.project(event.coordinate[0], event.coordinate[1]);
    const column = Math.floor((x - lattice.bounds[0]) / lattice.cell);
    const row = Math.floor((y - lattice.bounds[1]) / lattice.cell);
    if (column < 0 || row < 0 || column >= lattice.columns || row >= lattice.rows) return -1;
    return row * lattice.columns + column;
  };

  // ---------------------------------------------------------------------------------------
  // Layers
  // ---------------------------------------------------------------------------------------

  type FillStyle = {style: SpatialAnalysisStyleProps; side?: 'a' | 'b'};

  /** The fills of the current display: one, or two for the swipe of the variable and its lag. */
  const getFillStyles = (target: World, space: AnalysisSpace, ground: Ground): FillStyle[] => {
    const o = ctx.options;
    const display = getEffectiveDisplay(o);
    const useLattice = Boolean(o.lattice && target.lattice);
    const hidden = [0, 0, 0, 0] as const;
    const noData = NO_DATA_COLOR[ground];
    const classed = (
      table: ClassTable,
      values: Buffer,
      noDataColor: SpatialAnalysisStyleProps['noDataColor']
    ): SpatialAnalysisStyleProps => ({
      values,
      valueFormat: 'float32',
      colormap: 'greys',
      ...getClassTableLayerProps(table),
      highlightClasses: legendHighlight,
      noDataColor
    });
    const current = tables ?? buildTables(target);
    switch (display) {
      case 'weights':
        return [{style: classed(current.weights, target.weightRows, hidden)}];
      case 'neighbors':
        return [{style: classed(current.cardinality, space.shown, noData)}];
      case 'oneWay':
        return [{style: classed(current.oneWay, space.shown, noData)}];
      case 'value':
        return [
          {
            style: classed(
              current.value,
              useLattice ? target.lattice!.values : target.values,
              noData
            )
          }
        ];
      case 'lag':
        if (useLattice) return [{style: classed(current.value, space.shown, noData)}];
        return [
          {style: classed(current.value, target.values, noData), side: 'a'},
          {style: classed(current.value, target.lagShown, noData), side: 'b'}
        ];
      case 'difference':
        return [{style: classed(current.difference, target.difference, noData)}];
      case 'summary':
        return [{style: classed(current.summary ?? current.value, space.shown, noData)}];
      default:
        return [
          {
            style: {
              values: space.focusClass,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: getFocusPalette(ground),
              noDataColor: hidden
            }
          }
        ];
    }
  };

  /** Layers of the lattice raster and the polygon geometries. */
  const getLayerList = (): Layer[] => {
    const target = world;
    if (!target) return [];
    const o = ctx.options;
    const ground = ctx.ground();
    const {geography, buffers} = target;
    const coordinateOrigin: [number, number, number] = [
      geography.origin[0],
      geography.origin[1],
      0
    ];
    const useLattice = Boolean(o.lattice && target.lattice);
    const space = useLattice ? target.lattice!.space : target.space;
    const display = getEffectiveDisplay(o);
    const fillOpacity = geography.id === 'us-counties' ? 1 : 0.88;
    const hairlineWidth = geography.count > 2000 ? 0.4 : 0.5;
    const ink = getInkColor(ground);
    const casing = getCasingColor(ground);
    const layers: Layer[] = [];

    // Tier 3 context first: the polygon outlines, grey, under the fills.
    if (o.showOutlines || useLattice) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'weights-context-outline',
          coordinateOrigin,
          segments: buffers.outline,
          instanceCount: buffers.outlineSegmentCount,
          widthPixels: hairlineWidth,
          color: getContextLineColor(ground)
        })
      );
    }
    for (const {style, side} of getFillStyles(target, space, ground)) {
      if (useLattice) {
        const lattice = target.lattice!;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `weights-lattice-fill${side ? `-${side}` : ''}`,
            coordinateOrigin,
            gridSize: [lattice.columns, lattice.rows],
            bounds: lattice.bounds,
            ...style,
            compareSide: side,
            opacity: 0.88
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: `weights-fill${side ? `-${side}` : ''}`,
            coordinateOrigin,
            triangles: buffers.triangles,
            features: buffers.features,
            vertexCount: buffers.triangleVertexCount,
            ...style,
            compareSide: side,
            opacity: fillOpacity
          })
        );
      }
    }
    const fullFill = display !== 'focus' && display !== 'weights';
    if (o.showOutlines && fullFill && !useLattice) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'weights-hairline',
          coordinateOrigin,
          segments: buffers.outline,
          instanceCount: buffers.outlineSegmentCount,
          widthPixels: hairlineWidth,
          // The neutral class of the difference map is nearly the paper: grey, not white, hairlines.
          color: display === 'difference' ? getContextLineColor(ground) : getHairlineColor(ground)
        })
      );
    }
    if (o.showOutlines && target.zoneLines && !useLattice) {
      if (geography.id === 'us-counties') {
        const line = getStateLineStyle(ground);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'weights-state-lines',
            coordinateOrigin,
            segments: target.zoneLines.buffer,
            instanceCount: target.zoneLines.count,
            widthPixels: line.widthPixels,
            color: line.color,
            outlineColor: line.casing,
            outlineWidthPixels: (line.casingPixels - line.widthPixels) / 2
          })
        );
      } else {
        const line = getAreaLineStyle(ground);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'weights-area-lines',
            coordinateOrigin,
            segments: target.zoneLines.buffer,
            instanceCount: target.zoneLines.count,
            widthPixels: line.widthPixels,
            color: line.color
          })
        );
      }
    }
    // The whole link web is context: faint, thin, and invisible until the map is zoomed in.
    if (!useLattice && o.showLinks) {
      const links = target.space.kit.allSegments[o.matrix];
      if (links) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: `weights-links-${o.matrix}`,
            coordinateOrigin,
            segments: links.segments,
            weights: links.fade,
            instanceCount: links.slots,
            widthPixels: 0.5,
            color: getContextLineColor(ground),
            opacityStops:
              geography.id === 'us-counties'
                ? ([
                    [5.5, 0],
                    [6.5, 1]
                  ] as const)
                : ([
                    [8.5, 0],
                    [9.5, 1]
                  ] as const)
          })
        );
      }
    }
    // Neighbour outlines: ink 1.2 px, only where the neighbours are the subject.
    if (
      !useLattice &&
      (display === 'focus' || display === 'weights') &&
      target.neighbourOutlineCount
    ) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'weights-neighbour-outline',
          coordinateOrigin,
          segments: target.neighbourOutline,
          instanceCount: target.neighbourOutlineCount,
          widthPixels: 1.2,
          color: getInkColor(ground, 210)
        })
      );
    }
    // The focus bundle: ink over a ground casing; in the weights display the width is the weight.
    const byWeight = display === 'weights' && !useLattice && o.matrix === 'weights';
    const focusData = target.focusData;
    let maximumWeight = 0;
    if (byWeight && focusData) {
      for (let slot = 0; slot < focusData.degree; slot++) {
        maximumWeight = Math.max(maximumWeight, focusData.weights[slot]);
      }
    }
    layers.push(
      new SpatialAnalysisSegmentLayer({
        id: `weights-focus-links-${o.matrix}-${useLattice ? 'lattice' : 'polygons'}${byWeight ? '-by-weight' : ''}`,
        coordinateOrigin,
        segments: space.kit.focusSegments[o.matrix],
        instanceCount: FOCUS_SLOTS,
        widthPixels: 1.6,
        color: ink,
        outlineColor: casing,
        outlineWidthPixels: 0.8,
        ...(byWeight && maximumWeight > 0
          ? {
              instanceChannels: target.gatherer.slotWeights,
              channelStride: 1,
              channels: {width: 0},
              widthDomain: [0, maximumWeight] as const,
              widthRange: [0.8, 3.2] as const,
              widthScale: 'linear' as const
            }
          : {})
      })
    );
    // The focus place: an achromatic selection, ink core over a ground casing.
    if (!useLattice && target.focusOutlineCount) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'weights-focus-outline',
          coordinateOrigin,
          segments: target.focusOutline,
          instanceCount: target.focusOutlineCount,
          widthPixels: 2.5,
          color: ink,
          outlineColor: casing,
          outlineWidthPixels: 1
        })
      );
    }
    // Islands: a ring in ink over a ground halo, never a filled dot.
    if (!useLattice) {
      const ringProps = {
        coordinateOrigin,
        positions: target.core.positions,
        instanceCount: geography.count,
        values: target.space.isolate,
        valueFormat: 'uint32' as const,
        colormap: 'mask' as const,
        noDataColor: [0, 0, 0, 0] as const,
        shape: 'ring' as const
      };
      layers.push(
        new SpatialAnalysisPointLayer({
          ...ringProps,
          id: 'weights-island-halo',
          color: casing,
          radiusPixels: 7.5,
          outlineWidthPixels: 4
        }),
        new SpatialAnalysisPointLayer({
          ...ringProps,
          id: 'weights-island-ring',
          color: ink,
          radiusPixels: 6.5,
          outlineWidthPixels: 1.8
        })
      );
    }
    return layers;
  };

  // ---------------------------------------------------------------------------------------
  // Tooltip
  // ---------------------------------------------------------------------------------------

  const getPolygonTooltip = (target: World, row: number): TooltipContent => {
    const o = ctx.options;
    const {geography} = target;
    const display = getEffectiveDisplay(o);
    const variable = getVariableInfo(o.variable);
    const format = formatVariable(o.variable);
    const snapshot = target.snapshot;
    const current = tables ?? buildTables(target);
    const value = geography.getVariable(o.variable)[row];
    const valueText = Number.isFinite(value) ? format(value) : 'no data';
    const cardinality = snapshot?.cardinality[row];
    const lag = snapshot?.lag[row];
    const usableLag =
      snapshot !== null &&
      cardinality !== undefined &&
      cardinality > 0 &&
      Number.isFinite(lag) &&
      Number.isFinite(value);
    const tableRow = (
      label: string,
      table: ClassTable,
      shown: number,
      unit: string,
      text: string
    ): TooltipRow => {
      const index = getClassIndexOf(table, shown);
      return {
        label,
        value: text,
        unit,
        swatch: index >= 0 ? table.colors[index] : undefined,
        emphasis: true
      };
    };
    const rows: TooltipRow[] = [];
    const focusData = target.focusData;
    const weightIndex = focusData ? focusData.ids.indexOf(row) : -1;
    const isNeighbour = focusData ? weightIndex >= 0 && weightIndex < focusData.degree : false;
    const rowWeight = isNeighbour ? focusData!.weights[weightIndex] : null;
    if (row === target.focusRow) {
      rows.push({label: 'Role', value: 'The focus place', emphasis: true});
    } else if (display === 'focus' || display === 'weights') {
      rows.push({
        label: 'Role',
        value: isNeighbour ? 'Neighbour of the focus' : 'Not a neighbour',
        emphasis: true
      });
    }
    switch (display) {
      case 'neighbors':
        if (cardinality !== undefined) {
          rows.push(
            tableRow(
              `Neighbours (${RULE_NAMES[o.source]})`,
              current.cardinality,
              cardinality,
              cardinality === 1 ? 'neighbour' : 'neighbours',
              formatCount(cardinality)
            )
          );
        }
        break;
      case 'oneWay':
        if (snapshot) {
          rows.push(
            tableRow(
              'One-way links',
              current.oneWay,
              snapshot.oneWay[row],
              'not returned',
              formatCount(snapshot.oneWay[row])
            )
          );
        }
        break;
      case 'value':
        rows.push(tableRow(variable.label, current.value, value, variable.unit, valueText));
        break;
      case 'lag':
        if (usableLag) {
          rows.push(
            tableRow('Neighbourhood (lag)', current.value, lag!, variable.unit, format(lag!))
          );
        }
        break;
      case 'difference':
        if (usableLag) {
          rows.push(
            tableRow(
              'Lag minus value',
              current.difference,
              lag! - value,
              variable.unit,
              `${lag! - value >= 0 ? '+' : ''}${format(lag! - value)}`
            )
          );
        }
        break;
      default:
        break;
    }
    if (display === 'weights' && rowWeight !== null) {
      rows.push({label: 'Weight in the focus row', value: rowWeight.toFixed(3), emphasis: true});
    }
    if (display !== 'neighbors' && cardinality !== undefined) {
      rows.push({
        label: `Neighbours (${RULE_NAMES[o.source]})`,
        value: formatCount(cardinality),
        unit: cardinality === 1 ? 'neighbour' : 'neighbours'
      });
    }
    if (display !== 'value') {
      rows.push({label: variable.label, value: valueText, unit: variable.unit});
    }
    if (snapshot && display !== 'lag' && display !== 'difference' && usableLag) {
      rows.push({label: 'Neighbourhood (lag)', value: format(lag!), unit: variable.unit});
    }
    const sorted = getSortedValues(target);
    if (Number.isFinite(value) && sorted.length) {
      rows.push({
        label: 'Rank',
        value: `${formatOrdinal(getPercentileRank(sorted, value) * 100)} percentile`,
        unit: `of ${geography.unitPlural}`
      });
    }
    const mode = snapshot?.modes[row];
    return {
      title: geography.getName(row),
      subtitle: geography.getGroupName(row),
      rows,
      note:
        cardinality === 0
          ? 'Island: no neighbours under this rule'
          : mode !== undefined && mode !== NO_MODE && display === 'summary'
            ? `Dominant neighbour group: ${geography.groupNames[mode] ?? mode}`
            : undefined,
      anchor: getCentroidLngLat(target, row),
      highlight: {kind: 'polygon', rings: getRingsLngLat(target, row)}
    };
  };

  const sortedValueCache = new WeakMap<World, {variable: VariableId; sorted: Float32Array}>();
  /** Sorted finite values of the variable, for the percentile of a hovered place. */
  const getSortedValues = (target: World): Float32Array => {
    const variable = ctx.options.variable;
    const cached = sortedValueCache.get(target);
    if (cached?.variable === variable) return cached.sorted;
    const sorted = Float32Array.from(
      target.geography.getVariable(variable).filter(Number.isFinite)
    ).sort();
    sortedValueCache.set(target, {variable, sorted});
    return sorted;
  };

  /** Share of values at or below `value`, from the sorted array. */
  const getPercentileRank = (sorted: Float32Array, value: number): number => {
    let low = 0;
    let high = sorted.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (sorted[middle] <= value) low = middle + 1;
      else high = middle;
    }
    return low / sorted.length;
  };

  return {
    getCompiledGraphs() {
      const target = world;
      if (!target) return [];
      return [
        ...target.core.getGraphs(),
        ...target.algebra.values(),
        ...target.space.getGraphs(),
        target.gatherer.graph,
        ...(target.reference
          ? [...target.reference.queen.getGraphs(), ...target.reference.rook.getGraphs()]
          : []),
        ...(target.lattice
          ? [...target.lattice.producers.values(), ...target.lattice.space.getGraphs()]
          : [])
      ] as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, value) {
      const target = world;
      if (id === 'geography') {
        void switchWorld(value as GeographyId);
        return;
      }
      if (!target) return;
      if (id === 'showLinks' || id === 'showOutlines') {
        ctx.requestLayers();
        return;
      }
      if (['display', 'summary', 'source', 'lattice', 'variable'].includes(id)) {
        legendHighlight = null;
      }
      if (id === 'variable') {
        writeVariable(target);
        refreshTables();
      }
      if (id === 'focus') writeFocus(target);
      if (id === 'populationPercentile') writeSubgraphMask(target);
      if (id === 'selfWeight') writeSelfWeights(target);
      if (id === 'latticeMask' && target.lattice) writeLatticeStructure(target.lattice);
      prepareAll();
      writeParameters();
      if (id === 'focus') updateCorners(target);
      dirty = true;
      readbackStale = true;
      focusStale = true;
      publishFurniture();
      publishAnnotations();
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const target = world;
      if (!target) return;
      const o = ctx.options;
      if (dirty || frame.frameIndex < 2) {
        const focalWeight = Number(o.focalWeight);
        if (o.lattice && target.lattice) {
          const lattice = target.lattice;
          lattice.producers
            .get(`${o.latticeCriterion}-${o.latticeRadius}`)
            ?.encode(commandEncoder, {parameters: undefined});
          lattice.space.encode(commandEncoder, {
            normalizeLag: o.lagNormalize,
            matrix: o.matrix,
            display: getEffectiveDisplay(o) as SpaceDisplay,
            summary: o.summary,
            includeFocal: o.includeFocal,
            focalWeight
          });
        } else {
          target.core.encode(commandEncoder, getConfig(target.geography));
          const algebraKey = getAlgebraKey();
          if (algebraKey) {
            if (isBinaryCombine()) target.partner.encode(commandEncoder, o.partnerK);
            target.algebra.get(algebraKey)?.encode(commandEncoder, {parameters: undefined});
          } else {
            copyCsr(
              commandEncoder,
              target.core.csr,
              target.space.csr,
              target.geography.count,
              target.core.slots
            );
          }
          const display = getEffectiveDisplay(o);
          target.space.encode(commandEncoder, {
            normalizeLag: o.lagNormalize,
            matrix: o.matrix,
            display: display === 'weights' ? 'focus' : display === 'difference' ? 'lag' : display,
            summary: o.summary,
            includeFocal: o.includeFocal,
            focalWeight
          });
          target.gatherer.encode(commandEncoder);
          const reference = target.reference;
          if (o.showQueenOnly && reference && !reference.encoded) {
            reference.queen.encode(commandEncoder, getReferenceConfig(target, 'queen'));
            reference.rook.encode(commandEncoder, getReferenceConfig(target, 'rook'));
            reference.encoded = true;
            referenceStale = true;
          }
        }
        dirty = false;
        readbackStale = true;
        focusStale = true;
      }
      if (frame.frameIndex >= 1) requestReadbackIfNeeded(commandEncoder);
    },

    getLayers: getLayerList,

    // The class tables are authored per ground, so a ground flip rebuilds them.
    onGroundChange: () => {
      refreshTables();
      const target = world;
      if (target?.snapshot) handleSnapshot(target, target.snapshot, false);
      ctx.requestLayers();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes === null ? null : [...classes];
      ctx.requestLayers();
    },

    getTooltip(event) {
      const target = world;
      if (!target) return null;
      const o = ctx.options;
      if (o.lattice && target.lattice) {
        const cell = getLatticeCell(event);
        if (cell < 0) return null;
        const feature = target.lattice.featureOfCell[cell];
        if (feature < 0 && o.latticeMask) return null;
        const snapshot = target.latticeSnapshot;
        return {
          title: feature >= 0 ? target.geography.getName(feature) : 'Outside the map',
          subtitle: feature >= 0 ? target.geography.getGroupName(feature) : 'Empty grid cell',
          rows: snapshot
            ? [
                {
                  label: 'Neighbours',
                  value: formatCount(snapshot.cardinality[cell]),
                  unit: 'cells',
                  emphasis: true
                }
              ]
            : []
        };
      }
      const row = getPickedRow(event);
      if (row < 0) return null;
      return getPolygonTooltip(target, row);
    },

    onClick(event) {
      const target = world;
      if (!target) return false;
      if (ctx.options.lattice && target.lattice) {
        const cell = getLatticeCell(event);
        if (cell < 0) return false;
        target.lattice.focusRow = cell;
        target.lattice.space.kit.setFocusRow(cell);
      } else {
        const row = getPickedRow(event);
        if (row < 0) return false;
        target.focusRow = row;
        target.space.kit.setFocusRow(row);
        updateCorners(target);
      }
      dirty = true;
      readbackStale = true;
      focusStale = true;
      publishFurniture();
      return true;
    },

    destroy() {
      destroyed = true;
      switchToken++;
      const target = world;
      world = null;
      if (target) {
        stopWorld(target);
        target.resources.destroy();
      }
    }
  };
}
