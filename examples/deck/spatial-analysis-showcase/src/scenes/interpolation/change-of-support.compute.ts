// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUPolygonRasterizationExtentValues,
  GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
  GPUPolygonRasterization
} from '@luma.gl/experimental/gpu-raster';
import {
  addChangeOfSupportRecipe,
  getGPUGridGeneratorParameterValues,
  getGPUGridVerticesPerCell,
  GPU_GRID_GENERATOR_PARAMETER_LENGTH,
  GPUArealInterpolation,
  GPUGridGenerator,
  GPUPycnophylactic,
  type GPUGridType
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph
} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  fitRasterGrid,
  makeDasymetricWeights,
  rasterizePointCount,
  rasterizeStreetLength
} from './b7-ancillary';
import {
  COS_GROUPS,
  getCosVariable,
  type CosRule,
  type CosTarget,
  type CosVariableId,
  type CosView
} from './b7-cos-style';
import {formatSignificant} from './b7-format';
import {loadB7Polygons, type B7PolygonSet} from './b7-polygons';

/** Option state of the change-of-support scene. */
export type ChangeOfSupportOptions = {
  variable: CosVariableId;
  target: CosTarget;
  cellWidth: number;
  gridShift: number;
  denominator: 'overlap' | 'zone';
  ancillary: 'none' | 'streets' | 'places';
  rule: CosRule;
  view: CosView;
  iterations: number;
  kernel: 'rook' | 'box';
  ramp: RampName;
  opacity: number;
  showTractOutlines: boolean;
  showTargetOutlines: boolean;
};

/** Pycnophylactic iteration ladder; the slider picks an index. Compile-time per rung. */
export const COS_ITERATION_LADDER = [0, 2, 8, 32, 128, 512] as const;

/** Shared raster of both zone systems. Compile-time. */
const RASTER_WIDTH = 320;
const RASTER_HEIGHT = 384;
const CELL_COUNT = RASTER_WIDTH * RASTER_HEIGHT;
/** Lattice of the generated target grids. Compile-time; the cell width is a parameter. */
const GRID_COLUMNS = 56;
const GRID_ROWS = 76;
const CATEGORY_COUNT = COS_GROUPS.length;
/** Columns of the lagged source table: numerator, denominator, the tract's own rate or density. */
const COLUMN_COUNT = 3;
const PAIR_CAPACITY = 1 << 15;
const CROSSING_CAPACITY = 1 << 18;
const SETTLE_MILLISECONDS = 350;
const SQRT3 = Math.sqrt(3);

type GridTarget = Exclude<CosTarget, 'community'>;
const GRID_TYPES: Record<GridTarget, GPUGridType> = {
  hexagon: 'hex',
  square: 'square',
  triangle: 'triangle'
};

type VariableData = {
  /** Lag columns, `sourceCount * 3`, row-major. */
  table: Float32Array;
  numerator: Float32Array;
  denominator: Float32Array;
  /** Value drawn for each tract in the source view (a class index for categories). */
  display: Float32Array;
  scale: number;
  /** Per-tract class index, for the categorical shares. */
  groups: Uint32Array;
  /** Citywide numerator and denominator totals. */
  numeratorTotal: number;
  denominatorTotal: number;
};

type Variant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  targetCount: number;
  contributors: string;
  outline: {buffer: Buffer; count: number} | null;
};

type Pycno = {key: string; compiled: CompiledGPUCommandGraph<void>};

type ReadResult = {
  extensive: Float32Array;
  intensive: Float32Array;
  shares: Float32Array;
  offsets: Uint32Array;
  display: Float32Array;
  targetZones: Uint32Array;
  overflow: number;
  totalPairs: number;
  rasterOverflow: number;
  crossings: number;
  variantKey: string;
};

/**
 * Moves Chicago census-tract values to hexagons, squares, triangles or community areas with the
 * `addChangeOfSupportRecipe` chain (rasterize both systems, area-share weights, extensive and
 * intensive transfer through `GPUSpatialLag`), optionally weighted by an ancillary raster
 * (dasymetric), and builds a smooth mass-preserving surface with `GPUPycnophylactic`.
 *
 * Compile-time choices (target system, denominator, pycnophylactic iterations and kernel) select
 * a graph that is compiled the first time it is used and cached afterwards. Everything else is a
 * buffer write: variable, ancillary weights, grid cell width and offset, rule and view.
 */
export async function createChangeOfSupport(
  ctx: SceneContext<ChangeOfSupportOptions>
): Promise<SceneInstance<ChangeOfSupportOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const areas = ctx.datasets.get('chicago-community-areas');
  const roads = ctx.datasets.get('chicago-roads');
  const places = ctx.datasets.get('chicago-places');
  const {device} = ctx;
  const origin = tracts.defaultOrigin;
  const projection = tracts.getProjection(origin);
  const sources = loadB7Polygons(tracts, origin);
  const communities = loadB7Polygons(areas, origin);
  const sourceCount = sources.featureCount;
  const communityCount = communities.featureCount;
  const geoids = (tracts.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown> | null)?.GEOID ?? '')
  );
  const communityNames = (areas.geojson?.features ?? []).map(feature =>
    String((feature.properties as Record<string, unknown> | null)?.name ?? '')
  );
  const tractAreaKm2 = tracts.column<Float32Array>('areaKm2');
  const maximumTargetCount = GRID_COLUMNS * GRID_ROWS * 2;

  // --- Raster placement ----------------------------------------------------------------------
  const bounds: [number, number, number, number] = [
    Math.min(sources.bounds[0], communities.bounds[0]),
    Math.min(sources.bounds[1], communities.bounds[1]),
    Math.max(sources.bounds[2], communities.bounds[2]),
    Math.max(sources.bounds[3], communities.bounds[3])
  ];
  const grid = fitRasterGrid(bounds, RASTER_WIDTH, RASTER_HEIGHT);
  const rasterBounds: [number, number, number, number] = [
    grid.originX,
    grid.originY,
    grid.originX + RASTER_WIDTH * grid.cellSize,
    grid.originY + RASTER_HEIGHT * grid.cellSize
  ];
  const cellAreaKm2 = (grid.cellSize / 1000) ** 2;
  const centerX = (bounds[0] + bounds[2]) / 2;
  const centerY = (bounds[1] + bounds[3]) / 2;

  // --- Ancillary rasters (CPU, once) ---------------------------------------------------------
  const weightRasters = {
    none: new Float32Array(CELL_COUNT).fill(1),
    streets: makeDasymetricWeights(
      rasterizeStreetLength(roads, origin, grid),
      RASTER_WIDTH,
      RASTER_HEIGHT,
      2,
      0.04
    ),
    places: makeDasymetricWeights(
      rasterizePointCount(places, origin, grid),
      RASTER_WIDTH,
      RASTER_HEIGHT,
      3,
      0.04
    )
  };

  // --- Variables (CPU, lazily) ---------------------------------------------------------------
  const variableCache = new Map<CosVariableId, VariableData>();
  const population = tracts.column<Float32Array>('population');
  const households = tracts.column<Float32Array>('households');
  function getVariableData(id: CosVariableId): VariableData {
    const cached = variableCache.get(id);
    if (cached) return cached;
    const numerator = new Float32Array(sourceCount);
    const denominator = new Float32Array(sourceCount);
    const display = new Float32Array(sourceCount);
    const groups = new Uint32Array(sourceCount);
    const table = new Float32Array(sourceCount * COLUMN_COUNT);
    let scale = 1;
    const meta = getCosVariable(id);
    if (meta.kind === 'category') {
      const columns = COS_GROUPS.map(group => tracts.column<Float32Array>(group.id));
      for (let tract = 0; tract < sourceCount; tract++) {
        let best = 0;
        for (let group = 1; group < columns.length; group++) {
          if (columns[group][tract] > columns[best][tract]) best = group;
        }
        groups[tract] = best;
        display[tract] = best;
      }
      for (let tract = 0; tract < sourceCount; tract++) {
        table[tract * COLUMN_COUNT] = 1;
      }
      const result = {
        table,
        numerator,
        denominator,
        display,
        scale,
        groups,
        numeratorTotal: 0,
        denominatorTotal: 0
      };
      variableCache.set(id, result);
      return result;
    }
    if (meta.kind === 'count') {
      const column =
        id === 'population'
          ? population
          : tracts.column<Float32Array>(id === 'observations' ? 'natureObs2023' : 'jobsWac2021');
      for (let tract = 0; tract < sourceCount; tract++) {
        numerator[tract] = column[tract];
        table[tract * COLUMN_COUNT] = column[tract];
        table[tract * COLUMN_COUNT + 2] =
          tractAreaKm2[tract] > 0 ? column[tract] / tractAreaKm2[tract] : 0;
        display[tract] = column[tract];
      }
    } else {
      let numeratorColumn: Float32Array;
      let denominatorColumn = population;
      if (id === 'introducedShare') {
        numeratorColumn = tracts.column<Float32Array>('introducedObs2023');
        denominatorColumn = tracts.column<Float32Array>('natureObs2023');
        scale = 100;
      } else if (id === 'poverty') {
        numeratorColumn = tracts.column<Float32Array>('poverty150');
        scale = 100;
      } else if (id === 'noVehicle') {
        numeratorColumn = tracts.column<Float32Array>('noVehicle');
        denominatorColumn = households;
        scale = 100;
      } else if (id === 'age65') {
        numeratorColumn = tracts.column<Float32Array>('age65');
        scale = 100;
      } else if (id === 'diabetes') {
        const prevalence = tracts.column<Float32Array>('diabetes');
        numeratorColumn = fillFromRate(prevalence, population, 100);
        scale = 100;
      } else {
        const income = tracts.column<Float32Array>('perCapitaIncome');
        numeratorColumn = fillFromRate(income, population, 1);
        scale = 1;
      }
      let numeratorTotal = 0;
      let denominatorTotal = 0;
      for (let tract = 0; tract < sourceCount; tract++) {
        numeratorTotal += numeratorColumn[tract];
        denominatorTotal += denominatorColumn[tract];
      }
      const citywide = (scale * numeratorTotal) / Math.max(denominatorTotal, 1);
      for (let tract = 0; tract < sourceCount; tract++) {
        numerator[tract] = numeratorColumn[tract];
        denominator[tract] = denominatorColumn[tract];
        const rate =
          denominatorColumn[tract] > 0
            ? (scale * numeratorColumn[tract]) / denominatorColumn[tract]
            : citywide;
        table[tract * COLUMN_COUNT] = numeratorColumn[tract];
        table[tract * COLUMN_COUNT + 1] = denominatorColumn[tract];
        table[tract * COLUMN_COUNT + 2] = rate;
        display[tract] = rate;
      }
    }
    let numeratorTotal = 0;
    let denominatorTotal = 0;
    for (let tract = 0; tract < sourceCount; tract++) {
      numeratorTotal += numerator[tract];
      denominatorTotal += denominator[tract];
    }
    const result = {
      table,
      numerator,
      denominator,
      display,
      scale,
      groups,
      numeratorTotal,
      denominatorTotal
    };
    variableCache.set(id, result);
    return result;
  }

  /** `rate * population / scale` per tract, with missing rates filled by the citywide mean. */
  function fillFromRate(
    rate: Float32Array,
    weights: Float32Array,
    rateScale: number
  ): Float32Array {
    let sum = 0;
    let weight = 0;
    for (let tract = 0; tract < sourceCount; tract++) {
      if (Number.isFinite(rate[tract])) {
        sum += rate[tract] * weights[tract];
        weight += weights[tract];
      }
    }
    const mean = weight > 0 ? sum / weight : 0;
    return Float32Array.from(
      rate,
      (value, tract) => ((Number.isFinite(value) ? value : mean) * weights[tract]) / rateScale
    );
  }

  // --- Buffers -------------------------------------------------------------------------------
  const resources = new SpatialAnalysisResources(device, 'change-of-support');
  const tractPositions = resources.createBuffer('tract-positions', sources.polygonPositions);
  const tractFeatures = resources.createBuffer('tract-features', sources.featureOffsets);
  const tractPolygons = resources.createBuffer('tract-polygons', sources.polygonOffsets);
  const tractRings = resources.createBuffer('tract-rings', sources.ringOffsets);
  const tractOutline = resources.createBuffer('tract-outline', sources.outlineSegments);
  const communityPositions = resources.createBuffer(
    'community-positions',
    communities.polygonPositions
  );
  const communityFeatures = resources.createBuffer(
    'community-features',
    communities.featureOffsets
  );
  const communityPolygons = resources.createBuffer(
    'community-polygons',
    communities.polygonOffsets
  );
  const communityRings = resources.createBuffer('community-rings', communities.ringOffsets);
  const communityOutline = resources.createBuffer('community-outline', communities.outlineSegments);
  const sourceZones = resources.createBuffer('source-zones', CELL_COUNT * 4);
  const targetZones = resources.createBuffer('target-zones', CELL_COUNT * 4);
  const cellWeights = resources.createBuffer('cell-weights', weightRasters.none);
  const sourceTable = resources.createBuffer('source-table', sourceCount * COLUMN_COUNT * 4);
  const sourceCategories = resources.createBuffer('source-categories', sourceCount * 4);
  const sourceDisplay = resources.createBuffer('source-display', sourceCount * 4);
  const extensiveValues = resources.createBuffer(
    'extensive',
    maximumTargetCount * COLUMN_COUNT * 4
  );
  const intensiveValues = resources.createBuffer(
    'intensive',
    maximumTargetCount * COLUMN_COUNT * 4
  );
  const shares = resources.createBuffer('shares', maximumTargetCount * CATEGORY_COUNT * 4);
  const pairOffsets = resources.createBuffer('pair-offsets', (maximumTargetCount + 1) * 4);
  const pairNeighbors = resources.createBuffer('pair-neighbors', PAIR_CAPACITY * 4);
  const pairExtensive = resources.createBuffer('pair-extensive', PAIR_CAPACITY * 4);
  const pairIntensive = resources.createBuffer('pair-intensive', PAIR_CAPACITY * 4);
  const pairAreas = resources.createBuffer('pair-areas', PAIR_CAPACITY * 4);
  const overflowFlag = resources.createBuffer('overflow', 4);
  const sourceRasterOverflow = resources.createBuffer('source-raster-overflow', 4);
  const sourceCrossings = resources.createBuffer('source-crossings', 4);
  const totalPairs = resources.createBuffer('total-pairs', 4);
  const numeratorTotals = resources.createBuffer('numerator-totals', sourceCount * 4);
  const denominatorTotals = resources.createBuffer('denominator-totals', sourceCount * 4);
  const numeratorSurface = resources.createBuffer('numerator-surface', CELL_COUNT * 4);
  const denominatorSurface = resources.createBuffer('denominator-surface', CELL_COUNT * 4);
  const targetScalar = resources.createBuffer('target-scalar', maximumTargetCount * 4);
  const surfaceScalar = resources.createBuffer('surface-scalar', CELL_COUNT * 4);
  const displayValue = resources.createBuffer('display-value', CELL_COUNT * 4);
  const displayCategory = resources.createBuffer('display-category', CELL_COUNT * 4);
  const extentParameter = resources.createParameterBuffer(
    'extent',
    'float32',
    GPU_POLYGON_RASTERIZATION_EXTENT_LENGTH,
    getGPUPolygonRasterizationExtentValues(grid.originX, grid.originY, grid.cellSize, grid.cellSize)
  );
  const gridParameter = resources.createParameterBuffer(
    'grid',
    'float32',
    GPU_GRID_GENERATOR_PARAMETER_LENGTH
  );
  const displayParameter = resources.createParameterBuffer('display-parameters', 'float32', 8);

  // --- Source raster graph (once) -------------------------------------------------------------
  const sourceGraph = new GPUCommandGraph<void>(device, {id: 'cos-source-raster'});
  sourceGraph.add(
    new GPUPolygonRasterization({
      id: 'source-raster',
      width: RASTER_WIDTH,
      height: RASTER_HEIGHT,
      extent: extentParameter.importToGraph(sourceGraph),
      polygonPositions: importGraphBuffer(
        sourceGraph,
        'tract-positions',
        tractPositions,
        'float32x2',
        sources.polygonPositions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        sourceGraph,
        'tract-features',
        tractFeatures,
        'uint32',
        sources.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        sourceGraph,
        'tract-polygons',
        tractPolygons,
        'uint32',
        sources.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        sourceGraph,
        'tract-rings',
        tractRings,
        'uint32',
        sources.ringOffsets.length
      ),
      crossingCapacity: CROSSING_CAPACITY,
      zones: importGraphBuffer(sourceGraph, 'zones', sourceZones, 'uint32', CELL_COUNT),
      overflow: importGraphBuffer(sourceGraph, 'overflow', sourceRasterOverflow, 'uint32', 1),
      crossingCount: importGraphBuffer(sourceGraph, 'crossings', sourceCrossings, 'uint32', 1)
    })
  );
  const sourceCompiled = resources.track(sourceGraph.compile());

  // --- Grid geometry per type (lazily) -------------------------------------------------------
  type GridGeometry = {
    targetCount: number;
    verticesPerCell: number;
    vertices: Buffer;
    featureOffsets: Buffer;
    ringOffsets: Buffer;
    segments: Buffer;
  };
  const gridGeometries = new Map<GridTarget, GridGeometry>();
  function getGridGeometry(target: GridTarget): GridGeometry {
    let geometry = gridGeometries.get(target);
    if (!geometry) {
      const gridType = GRID_TYPES[target];
      const verticesPerCell = getGPUGridVerticesPerCell(gridType);
      const targetCount = GRID_COLUMNS * GRID_ROWS * (gridType === 'triangle' ? 2 : 1);
      geometry = {
        targetCount,
        verticesPerCell,
        vertices: resources.createBuffer(`${target}-vertices`, targetCount * verticesPerCell * 8),
        featureOffsets: resources.createBuffer(
          `${target}-features`,
          Uint32Array.from({length: targetCount + 1}, (_, index) => index)
        ),
        ringOffsets: resources.createBuffer(
          `${target}-rings`,
          Uint32Array.from({length: targetCount + 1}, (_, index) => index * verticesPerCell)
        ),
        segments: resources.createBuffer(`${target}-segments`, targetCount * verticesPerCell * 16)
      };
      gridGeometries.set(target, geometry);
    }
    return geometry;
  }

  // --- Target variants: one compiled recipe graph per (target, denominator) -------------------
  const variants = new Map<string, Variant>();
  function getVariant(target: CosTarget, denominator: 'overlap' | 'zone'): Variant {
    const key = `${target}:${denominator}`;
    let variant = variants.get(key);
    if (variant) return variant;
    const graph = new GPUCommandGraph<void>(device, {id: `cos-${target}-${denominator}`});
    const grids = target === 'community' ? null : getGridGeometry(target);
    const targetCount = grids ? grids.targetCount : communityCount;
    let targetSystem;
    let gridVerticesView: GraphDataView<'float32x2'> | null = null;
    if (grids) {
      const vertices = importGraphBuffer(
        graph,
        'grid-vertices',
        grids.vertices,
        'float32x2',
        grids.targetCount * grids.verticesPerCell
      );
      gridVerticesView = vertices;
      graph.add(
        new GPUGridGenerator({
          id: `grid-${target}`,
          gridType: GRID_TYPES[target as GridTarget],
          columns: GRID_COLUMNS,
          rows: GRID_ROWS,
          parameters: gridParameter.importToGraph(graph),
          output: {positions: vertices}
        })
      );
      const identity = importGraphBuffer(
        graph,
        'grid-features',
        grids.featureOffsets,
        'uint32',
        grids.targetCount + 1
      );
      targetSystem = {
        polygonPositions: vertices,
        featureOffsets: identity,
        polygonOffsets: identity,
        ringOffsets: importGraphBuffer(
          graph,
          'grid-rings',
          grids.ringOffsets,
          'uint32',
          grids.targetCount + 1
        ),
        crossingCapacity: CROSSING_CAPACITY
      };
    } else {
      targetSystem = {
        polygonPositions: importGraphBuffer(
          graph,
          'community-positions',
          communityPositions,
          'float32x2',
          communities.polygonPositions.length / 2
        ),
        featureOffsets: importGraphBuffer(
          graph,
          'community-features',
          communityFeatures,
          'uint32',
          communities.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'community-polygons',
          communityPolygons,
          'uint32',
          communities.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'community-rings',
          communityRings,
          'uint32',
          communities.ringOffsets.length
        ),
        crossingCapacity: CROSSING_CAPACITY
      };
    }
    const offsetsView = importGraphBuffer(
      graph,
      'pair-offsets',
      pairOffsets,
      'uint32',
      targetCount + 1
    );
    const recipe = addChangeOfSupportRecipe(graph, {
      id: 'cos',
      width: RASTER_WIDTH,
      height: RASTER_HEIGHT,
      extent: extentParameter.importToGraph(graph),
      source: {
        polygonPositions: importGraphBuffer(
          graph,
          'tract-positions',
          tractPositions,
          'float32x2',
          sources.polygonPositions.length / 2
        ),
        featureOffsets: importGraphBuffer(
          graph,
          'tract-features',
          tractFeatures,
          'uint32',
          sources.featureOffsets.length
        ),
        polygonOffsets: importGraphBuffer(
          graph,
          'tract-polygons',
          tractPolygons,
          'uint32',
          sources.polygonOffsets.length
        ),
        ringOffsets: importGraphBuffer(
          graph,
          'tract-rings',
          tractRings,
          'uint32',
          sources.ringOffsets.length
        ),
        crossingCapacity: CROSSING_CAPACITY
      },
      target: targetSystem,
      cellWeights: importGraphBuffer(graph, 'cell-weights', cellWeights, 'float32', CELL_COUNT),
      denominator,
      pairCapacity: PAIR_CAPACITY,
      sourceValues: importGraphBuffer(
        graph,
        'source-table',
        sourceTable,
        'float32',
        sourceCount * COLUMN_COUNT
      ),
      columnCount: COLUMN_COUNT,
      scratch: {
        sourceZones: importGraphBuffer(graph, 'source-zones', sourceZones, 'uint32', CELL_COUNT),
        targetZones: importGraphBuffer(graph, 'target-zones', targetZones, 'uint32', CELL_COUNT)
      },
      categories: {
        sourceCategories: importGraphBuffer(
          graph,
          'source-categories',
          sourceCategories,
          'uint32',
          sourceCount
        ),
        categoryCount: CATEGORY_COUNT,
        output: importGraphBuffer(graph, 'shares', shares, 'float32', targetCount * CATEGORY_COUNT)
      },
      outputs: {
        extensiveValues: importGraphBuffer(
          graph,
          'extensive',
          extensiveValues,
          'float32',
          targetCount * COLUMN_COUNT
        ),
        intensiveValues: importGraphBuffer(
          graph,
          'intensive',
          intensiveValues,
          'float32',
          targetCount * COLUMN_COUNT
        ),
        extensiveWeights: {
          offsets: offsetsView,
          neighbors: importGraphBuffer(
            graph,
            'pair-neighbors',
            pairNeighbors,
            'uint32',
            PAIR_CAPACITY
          ),
          weights: importGraphBuffer(
            graph,
            'pair-extensive',
            pairExtensive,
            'float32',
            PAIR_CAPACITY
          )
        },
        intensiveWeightValues: importGraphBuffer(
          graph,
          'pair-intensive',
          pairIntensive,
          'float32',
          PAIR_CAPACITY
        ),
        areas: importGraphBuffer(graph, 'pair-areas', pairAreas, 'float32', PAIR_CAPACITY),
        overflow: importGraphBuffer(graph, 'overflow', overflowFlag, 'uint32', 1),
        requiredCount: importGraphBuffer(graph, 'total-pairs', totalPairs, 'uint32', 1)
      }
    });
    let outline: Variant['outline'] = null;
    if (grids && gridVerticesView) {
      // Draw only cells that overlap some tract: a kernel turns the ring vertices into segments.
      addKernelPass(graph, {
        id: 'grid-outline',
        invocationCount: grids.targetCount * grids.verticesPerCell,
        bindings: [
          {name: 'vertices', view: gridVerticesView, type: 'f32', access: 'read'},
          {name: 'offsets', view: offsetsView, type: 'u32', access: 'read'},
          {
            name: 'segments',
            view: importGraphBuffer(
              graph,
              'grid-segments',
              grids.segments,
              'float32',
              grids.targetCount * grids.verticesPerCell * 4
            ),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const VERTICES: u32 = ${grids.verticesPerCell}u;`,
        body: /* wgsl */ `
  var nanBits: u32 = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let cell = index / VERTICES;
  let next = cell * VERTICES + (index % VERTICES + 1u) % VERTICES;
  let covered = offsets[offsetsOffset + cell + 1u] > offsets[offsetsOffset + cell];
  let a = vec2<f32>(vertices[verticesOffset + index * 2u], vertices[verticesOffset + index * 2u + 1u]);
  let b = vec2<f32>(vertices[verticesOffset + next * 2u], vertices[verticesOffset + next * 2u + 1u]);
  segments[segmentsOffset + index * 4u] = select(nan, a.x, covered);
  segments[segmentsOffset + index * 4u + 1u] = select(nan, a.y, covered);
  segments[segmentsOffset + index * 4u + 2u] = select(nan, b.x, covered);
  segments[segmentsOffset + index * 4u + 3u] = select(nan, b.y, covered);`
      });
      outline = {buffer: grids.segments, count: grids.targetCount * grids.verticesPerCell};
    }
    variant = {
      key,
      compiled: resources.track(graph.compile()),
      targetCount,
      contributors: summarizeContributors(recipe.contributors),
      outline
    };
    variants.set(key, variant);
    return variant;
  }

  // --- Pycnophylactic ladder (lazily) --------------------------------------------------------
  const pycnoGraphs = new Map<string, Pycno>();
  function getPycno(rung: number, kernel: 'rook' | 'box'): Pycno {
    const key = `${rung}:${kernel}`;
    let pycno = pycnoGraphs.get(key);
    if (pycno) return pycno;
    const graph = new GPUCommandGraph<void>(device, {id: `cos-pycno-${key}`});
    const zones = importGraphBuffer(graph, 'source-zones', sourceZones, 'uint32', CELL_COUNT);
    for (const [name, totals, output] of [
      ['numerator', numeratorTotals, numeratorSurface],
      ['denominator', denominatorTotals, denominatorSurface]
    ] as const) {
      graph.add(
        new GPUPycnophylactic({
          id: `pycno-${name}`,
          width: RASTER_WIDTH,
          height: RASTER_HEIGHT,
          zones,
          zoneCount: sourceCount,
          totals: importGraphBuffer(graph, `${name}-totals`, totals, 'float32', sourceCount),
          iterations: COS_ITERATION_LADDER[rung],
          kernel,
          output: importGraphBuffer(graph, `${name}-surface`, output, 'float32', CELL_COUNT)
        })
      );
    }
    pycno = {key, compiled: resources.track(graph.compile())};
    pycnoGraphs.set(key, pycno);
    return pycno;
  }

  // --- Display graph --------------------------------------------------------------------------
  const displayGraph = new GPUCommandGraph<void>(device, {id: 'cos-display'});
  const floats = (name: string, buffer: Buffer, length: number) =>
    importGraphBuffer(displayGraph, name, buffer, 'float32', length);
  const uints = (name: string, buffer: Buffer, length: number) =>
    importGraphBuffer(displayGraph, name, buffer, 'uint32', length);
  const paramsView = floats('params', displayParameter.buffer, 8);
  const targetScalarView = floats('target-scalar', targetScalar, maximumTargetCount);
  const surfaceScalarView = floats('surface-scalar', surfaceScalar, CELL_COUNT);
  // Pass 1: one scalar per target zone from the extensive and intensive transfers, or the class
  // with the largest area share. The rule, the variable kind and the scale are parameter words.
  addKernelPass(displayGraph, {
    id: 'cos-target-scalar',
    invocationCount: maximumTargetCount,
    bindings: [
      {name: 'params', view: paramsView, type: 'f32', access: 'read'},
      {
        name: 'extensive',
        view: floats('extensive', extensiveValues, maximumTargetCount * COLUMN_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'intensive',
        view: floats('intensive', intensiveValues, maximumTargetCount * COLUMN_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'shares',
        view: floats('shares', shares, maximumTargetCount * CATEGORY_COUNT),
        type: 'f32',
        access: 'read'
      },
      {name: 'targetScalar', view: targetScalarView, type: 'f32', access: 'read_write'}
    ],
    declarations: `const CATEGORIES: u32 = ${CATEGORY_COUNT}u;`,
    body: /* wgsl */ `
  let rule = u32(params[paramsOffset + 1u]);
  let kind = u32(params[paramsOffset + 2u]);
  let scale = params[paramsOffset + 3u];
  let targetCount = u32(params[paramsOffset + 4u]);
  var value = -1.0;
  if (index < targetCount) {
    if (kind == 2u) {
      var best = 0u;
      var bestShare = 0.0;
      for (var group = 0u; group < CATEGORIES; group = group + 1u) {
        let share = shares[sharesOffset + index * CATEGORIES + group];
        if (share > bestShare) { bestShare = share; best = group; }
      }
      if (bestShare > 0.0) { value = f32(best); }
    } else {
      let base = index * 3u;
      if (rule == 0u) {
        let numerator = extensive[extensiveOffset + base];
        let denominator = extensive[extensiveOffset + base + 1u];
        if (kind == 0u) { value = numerator; }
        else if (denominator > 0.0) { value = scale * numerator / denominator; }
      } else if (rule == 1u) {
        value = intensive[intensiveOffset + base + 2u];
      } else {
        value = extensive[extensiveOffset + base + 2u];
      }
    }
  }
  targetScalar[targetScalarOffset + index] = value;`
  });
  // Pass 2: the pycnophylactic surface as a value per cell (density, or ratio of two surfaces).
  addKernelPass(displayGraph, {
    id: 'cos-surface-scalar',
    invocationCount: CELL_COUNT,
    bindings: [
      {name: 'params', view: paramsView, type: 'f32', access: 'read'},
      {
        name: 'numeratorSurface',
        view: floats('numerator-surface', numeratorSurface, CELL_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'denominatorSurface',
        view: floats('denominator-surface', denominatorSurface, CELL_COUNT),
        type: 'f32',
        access: 'read'
      },
      {name: 'surfaceScalar', view: surfaceScalarView, type: 'f32', access: 'read_write'}
    ],
    body: /* wgsl */ `
  let kind = u32(params[paramsOffset + 2u]);
  let scale = params[paramsOffset + 3u];
  let cellArea = params[paramsOffset + 5u];
  let numerator = numeratorSurface[numeratorSurfaceOffset + index];
  let denominator = denominatorSurface[denominatorSurfaceOffset + index];
  var value = -1.0;
  if (kind == 0u) { value = numerator / cellArea; }
  else if (denominator > 0.0) { value = scale * numerator / denominator; }
  surfaceScalar[surfaceScalarOffset + index] = value;`
  });
  // Pass 3: gather one value (and class) per raster cell, clipped to the tract footprint.
  addKernelPass(displayGraph, {
    id: 'cos-display',
    invocationCount: CELL_COUNT,
    bindings: [
      {name: 'params', view: paramsView, type: 'f32', access: 'read'},
      {
        name: 'sourceZones',
        view: uints('source-zones', sourceZones, CELL_COUNT),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'targetZones',
        view: uints('target-zones', targetZones, CELL_COUNT),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'sourceDisplay',
        view: floats('source-display', sourceDisplay, sourceCount),
        type: 'f32',
        access: 'read'
      },
      {name: 'targetScalar', view: targetScalarView, type: 'f32', access: 'read'},
      {name: 'surfaceScalar', view: surfaceScalarView, type: 'f32', access: 'read'},
      {
        name: 'displayValue',
        view: floats('display-value', displayValue, CELL_COUNT),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'displayCategory',
        view: uints('display-category', displayCategory, CELL_COUNT),
        type: 'u32',
        access: 'read_write'
      }
    ],
    declarations: `const SOURCE_COUNT: u32 = ${sourceCount}u;`,
    body: /* wgsl */ `
  let view = u32(params[paramsOffset]);
  let kind = u32(params[paramsOffset + 2u]);
  let targetCount = u32(params[paramsOffset + 4u]);
  let sourceZone = sourceZones[sourceZonesOffset + index];
  let targetZone = targetZones[targetZonesOffset + index];
  var value = -1.0;
  if (sourceZone < SOURCE_COUNT) {
    if (view == 0u || (view == 2u && kind == 2u)) {
      value = sourceDisplay[sourceDisplayOffset + sourceZone];
    } else if (view == 1u) {
      if (targetZone < targetCount) { value = targetScalar[targetScalarOffset + targetZone]; }
    } else {
      value = surfaceScalar[surfaceScalarOffset + index];
    }
  }
  displayValue[displayValueOffset + index] = value;
  displayCategory[displayCategoryOffset + index] = select(255u, u32(max(value, 0.0)), value >= 0.0 && kind == 2u);`
  });
  const displayCompiled = resources.track(displayGraph.compile());

  // --- State ----------------------------------------------------------------------------------
  let destroyed = false;
  let measuring = false;
  let sourceRastered = false;
  let targetDirty = true;
  let pycnoDirty = true;
  let displayDirty = true;
  let readStale = true;
  let lastChangeTime = performance.now();
  let valueRange: [number, number] = [0, 1];
  let lastLegendKey = '';
  let activeVariant = getVariant(ctx.options.target, ctx.options.denominator);
  let activePycno: Pycno | null = null;
  let read: ReadResult | null = null;
  let pycnoCheckStale = true;
  const gridValues = new Float32Array(GPU_GRID_GENERATOR_PARAMETER_LENGTH);

  const markChanged = () => {
    lastChangeTime = performance.now();
    readStale = true;
  };

  function currentVariable(): VariableData {
    return getVariableData(ctx.options.variable);
  }

  function writeVariable(): void {
    const data = currentVariable();
    sourceTable.write(data.table);
    sourceCategories.write(data.groups);
    sourceDisplay.write(data.display);
    numeratorTotals.write(data.numerator);
    denominatorTotals.write(data.denominator);
    targetDirty = true;
    pycnoDirty = true;
    displayDirty = true;
    markChanged();
  }

  function writeDisplayParameters(): void {
    const o = ctx.options;
    const meta = getCosVariable(o.variable);
    const data = currentVariable();
    displayParameter.write(
      Float32Array.of(
        o.view === 'source' ? 0 : o.view === 'target' ? 1 : 2,
        o.rule === 'conserving' ? 0 : o.rule === 'area-mean' ? 1 : 2,
        meta.kind === 'count' ? 0 : meta.kind === 'ratio' ? 1 : 2,
        data.scale,
        activeVariant.targetCount,
        cellAreaKm2,
        0,
        0
      )
    );
  }

  function writeGridParameters(): void {
    const o = ctx.options;
    if (o.target === 'community') return;
    const width = o.cellWidth;
    const pitch = o.target === 'square' ? width : (width * SQRT3) / 2;
    const rowPitch = pitch;
    gridValues.set(
      getGPUGridGeneratorParameterValues(
        {
          minX: centerX - (GRID_COLUMNS * width) / 2 + o.gridShift * width,
          minY: centerY - (GRID_ROWS * rowPitch) / 2 + o.gridShift * rowPitch,
          cellWidth: width,
          cellHeight: rowPitch
        },
        gridValues
      )
    );
    gridParameter.write(gridValues);
  }

  /** Mirror of the display kernel for one target zone (tooltips and legends). */
  function getTargetValue(target: number): number {
    if (!read) return Number.NaN;
    const o = ctx.options;
    const meta = getCosVariable(o.variable);
    const data = currentVariable();
    const base = target * COLUMN_COUNT;
    if (o.rule === 'area-mean') return read.intensive[base + 2];
    if (o.rule === 'naive') return read.extensive[base + 2];
    if (meta.kind === 'count') return read.extensive[base];
    return read.extensive[base + 1] > 0
      ? (data.scale * read.extensive[base]) / read.extensive[base + 1]
      : Number.NaN;
  }

  // --- Read-back ------------------------------------------------------------------------------
  const maximumPairWords = maximumTargetCount + 1;
  const resultReader = new SummaryReader(
    resources,
    'cos-results',
    [
      {buffer: extensiveValues, size: maximumTargetCount * COLUMN_COUNT * 4},
      {buffer: intensiveValues, size: maximumTargetCount * COLUMN_COUNT * 4},
      {buffer: shares, size: maximumTargetCount * CATEGORY_COUNT * 4},
      {buffer: pairOffsets, size: maximumPairWords * 4},
      {buffer: displayValue, size: CELL_COUNT * 4},
      {buffer: targetZones, size: CELL_COUNT * 4},
      {buffer: overflowFlag, size: 4},
      {buffer: totalPairs, size: 4},
      {buffer: sourceRasterOverflow, size: 4},
      {buffer: sourceCrossings, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      let offset = 0;
      const take = <T extends Float32Array | Uint32Array>(
        Type: {new (buffer: ArrayBuffer, byteOffset: number, length: number): T},
        length: number
      ): T => {
        const view = new Type(bytes, offset, length);
        offset += length * 4;
        return view;
      };
      read = {
        extensive: take(Float32Array, maximumTargetCount * COLUMN_COUNT),
        intensive: take(Float32Array, maximumTargetCount * COLUMN_COUNT),
        shares: take(Float32Array, maximumTargetCount * CATEGORY_COUNT),
        offsets: take(Uint32Array, maximumPairWords),
        display: take(Float32Array, CELL_COUNT),
        targetZones: take(Uint32Array, CELL_COUNT),
        overflow: take(Uint32Array, 1)[0],
        totalPairs: take(Uint32Array, 1)[0],
        rasterOverflow: take(Uint32Array, 1)[0],
        crossings: take(Uint32Array, 1)[0],
        variantKey: activeVariant.key
      };
      summarize();
    }
  );

  function summarize(): void {
    if (!read) return;
    const o = ctx.options;
    const meta = getCosVariable(o.variable);
    const data = currentVariable();
    const targetCount = activeVariant.targetCount;
    let covered = 0;
    let extensiveNumerator = 0;
    let extensiveDenominator = 0;
    for (let target = 0; target < targetCount; target++) {
      if (read.offsets[target + 1] > read.offsets[target]) covered++;
      extensiveNumerator += read.extensive[target * COLUMN_COUNT];
      extensiveDenominator += read.extensive[target * COLUMN_COUNT + 1];
    }
    const difference = (to: number, from: number) =>
      from > 0 ? ` (${to >= from ? '+' : ''}${(((to - from) / from) * 100).toFixed(2)}%)` : '';
    if (meta.kind === 'count') {
      ctx.setReadout(
        'conservation',
        `${formatCount(data.numeratorTotal)} in tracts → ${formatCount(extensiveNumerator)} in targets${difference(extensiveNumerator, data.numeratorTotal)}`
      );
    } else if (meta.kind === 'ratio') {
      ctx.setReadout(
        'conservation',
        `numerator ${formatCount(data.numeratorTotal)} → ${formatCount(extensiveNumerator)}${difference(extensiveNumerator, data.numeratorTotal)}; denominator ${formatCount(data.denominatorTotal)} → ${formatCount(extensiveDenominator)}${difference(extensiveDenominator, data.denominatorTotal)}`
      );
    } else {
      ctx.setReadout('conservation', 'categories: shares sum to 1 in every covered target');
    }
    ctx.setReadout(
      'pairs',
      `${formatCount(read.totalPairs)} of ${formatCount(PAIR_CAPACITY)} slots${read.overflow ? ' (OVERFLOW: raise the capacity)' : ''}`
    );
    ctx.setReadout(
      'rasterCrossings',
      read.rasterOverflow
        ? 'OVERFLOW: raise the crossing capacity'
        : `${formatCount(read.crossings)} of ${formatCount(CROSSING_CAPACITY)} scanline crossings used by the tract raster`
    );
    ctx.setReadout(
      'coverage',
      `${formatCount(covered)} of ${formatCount(targetCount)} target zones overlap a tract`
    );
    if (meta.kind === 'category') {
      const dominantCounts = new Uint32Array(CATEGORY_COUNT);
      for (let target = 0; target < targetCount; target++) {
        if (read.offsets[target + 1] <= read.offsets[target]) continue;
        let dominant = 0;
        for (let group = 1; group < CATEGORY_COUNT; group++) {
          if (
            read.shares[target * CATEGORY_COUNT + group] >
            read.shares[target * CATEGORY_COUNT + dominant]
          ) {
            dominant = group;
          }
        }
        dominantCounts[dominant]++;
      }
      ctx.setChart('targetDistribution', {
        kind: 'bars',
        title: 'Target zones by largest overlap-area group',
        values: dominantCounts,
        labels: COS_GROUPS.map(group => group.label),
        colors: COS_GROUPS.map(group => group.color),
        horizontal: true,
        description:
          'Count of covered target zones assigned to each group by the largest share of overlap area; these are zone counts, not population counts.'
      });
      ctx.setChart('conservationChart', null);
    } else {
      const targetValues: number[] = [];
      for (let target = 0; target < targetCount; target++) {
        if (read.offsets[target + 1] <= read.offsets[target]) continue;
        const value = getTargetValue(target);
        if (Number.isFinite(value)) targetValues.push(value);
      }
      targetValues.sort((a, b) => a - b);
      if (targetValues.length > 0) {
        const binCount = 18;
        const minimum = targetValues[0];
        const maximum = targetValues[targetValues.length - 1];
        const span = maximum - minimum || 1;
        const counts = new Uint32Array(binCount);
        for (const value of targetValues) {
          const bin = Math.min(binCount - 1, Math.floor(((value - minimum) / span) * binCount));
          counts[bin]++;
        }
        ctx.setChart('targetDistribution', {
          kind: 'histogram',
          title: `${meta.label} across covered target zones`,
          values: counts,
          xDomain: [minimum, maximum > minimum ? maximum : minimum + 1],
          xLabel: meta.unit,
          yLabel: 'Target zones',
          markers: [
            {
              x: targetValues[Math.floor(targetValues.length / 2)],
              label: 'median'
            }
          ],
          description: `Distribution of the displayed ${meta.unit} values across ${targetValues.length.toLocaleString()} covered target zones under the selected transfer rule.`
        });
      } else {
        ctx.setChart('targetDistribution', null);
      }
      ctx.setChart('conservationChart', {
        kind: 'dumbbell',
        title: 'Extensive totals before and after transfer',
        xLabel: 'Count',
        aLabel: 'Source tracts',
        bLabel: 'Target zones',
        rows:
          meta.kind === 'count'
            ? [
                {
                  label: meta.unit,
                  a: data.numeratorTotal,
                  b: extensiveNumerator,
                  highlight: true
                }
              ]
            : [
                {
                  label: 'Numerator',
                  a: data.numeratorTotal,
                  b: extensiveNumerator,
                  highlight: true
                },
                {
                  label: 'Denominator',
                  a: data.denominatorTotal,
                  b: extensiveDenominator
                }
              ],
        formatX: formatCount,
        description:
          'Source and target totals for the extensive count columns used by the conserving transfer. A gap indicates mass lost or gained at rasterized boundaries.'
      });
    }
    // Robust color range from the displayed raster.
    const values: number[] = [];
    for (let cell = 0; cell < CELL_COUNT; cell += 3) {
      if (read.display[cell] >= 0) values.push(read.display[cell]);
    }
    if (values.length > 0 && meta.kind !== 'category') {
      values.sort((a, b) => a - b);
      const low = meta.kind === 'count' ? 0 : values[Math.floor(values.length * 0.01)];
      let high = values[Math.min(values.length - 1, Math.floor(values.length * 0.985))];
      if (!(high > low)) high = low + 1;
      valueRange = [low, high];
      ctx.setLegendExtent('values', valueRange);
      const key = `${low}:${high}`;
      if (key !== lastLegendKey) {
        lastLegendKey = key;
        ctx.requestLayers();
      }
    }
  }

  const pycnoReader = new SummaryReader(
    resources,
    'cos-pycno',
    [
      {buffer: numeratorSurface, size: CELL_COUNT * 4},
      {buffer: sourceZones, size: CELL_COUNT * 4}
    ],
    bytes => {
      if (destroyed) return;
      const mass = new Float32Array(bytes, 0, CELL_COUNT);
      const zones = new Uint32Array(bytes, CELL_COUNT * 4, CELL_COUNT);
      const data = currentVariable();
      const sums = new Float64Array(sourceCount);
      for (let cell = 0; cell < CELL_COUNT; cell++) {
        if (zones[cell] < sourceCount) sums[zones[cell]] += mass[cell];
      }
      let worst = 0;
      let preserved = 0;
      for (let tract = 0; tract < sourceCount; tract++) {
        if (data.numerator[tract] > 0) {
          worst = Math.max(
            worst,
            Math.abs(sums[tract] - data.numerator[tract]) / data.numerator[tract]
          );
          preserved++;
        }
      }
      ctx.setReadout(
        'pycnoCheck',
        preserved > 0
          ? `${formatCount(preserved)} tract totals preserved, worst error ${(worst * 100).toExponential(1)}%`
          : 'not applicable to a categorical variable'
      );
    }
  );

  // --- Fast-path timing (scratch graphs, outside the frame) -----------------------------------
  const scratchOffsets = resources.createBuffer('timing-offsets', (maximumTargetCount + 1) * 4);
  const scratchNeighbors = resources.createBuffer('timing-neighbors', PAIR_CAPACITY * 4);
  const scratchExtensive = resources.createBuffer('timing-extensive', PAIR_CAPACITY * 4);
  const scratchIntensive = resources.createBuffer('timing-intensive', PAIR_CAPACITY * 4);
  const scratchOverflow = resources.createBuffer('timing-overflow', 4);
  function buildTimingGraph(unweightedFastPath: boolean, targetCount: number) {
    const graph = new GPUCommandGraph<void>(device, {
      id: `cos-timing-${unweightedFastPath ? 'fast' : 'generic'}`
    });
    graph.add(
      new GPUArealInterpolation({
        id: 'areal-timing',
        sourceZones: importGraphBuffer(graph, 'source-zones', sourceZones, 'uint32', CELL_COUNT),
        targetZones: importGraphBuffer(graph, 'target-zones', targetZones, 'uint32', CELL_COUNT),
        sourceCount,
        targetCount,
        denominator: 'overlap',
        unweightedFastPath,
        weights: {
          offsets: importGraphBuffer(graph, 'offsets', scratchOffsets, 'uint32', targetCount + 1),
          neighbors: importGraphBuffer(
            graph,
            'neighbors',
            scratchNeighbors,
            'uint32',
            PAIR_CAPACITY
          ),
          weights: importGraphBuffer(graph, 'extensive', scratchExtensive, 'float32', PAIR_CAPACITY)
        },
        alternateWeights: importGraphBuffer(
          graph,
          'intensive',
          scratchIntensive,
          'float32',
          PAIR_CAPACITY
        ),
        overflow: importGraphBuffer(graph, 'overflow', scratchOverflow, 'uint32', 1)
      })
    );
    return graph.compile();
  }
  async function measureFastPath(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('fastPath', 'measuring...');
    const targetCount = activeVariant.targetCount;
    const fast = buildTimingGraph(true, targetCount);
    const generic = buildTimingGraph(false, targetCount);
    try {
      const options = {parameters: undefined, completionBuffer: scratchOverflow};
      const fastTiming = await measureCompiledGraph(device, fast, options);
      const genericTiming = await measureCompiledGraph(device, generic, options);
      if (destroyed) return;
      ctx.setReadout(
        'fastPath',
        `fast ${formatCompiledGraphTiming(fastTiming)} (${fast.stats.nodeOrder.length} nodes) vs generic ${formatCompiledGraphTiming(genericTiming)} (${generic.stats.nodeOrder.length} nodes): ${formatSpeedup(genericTiming.milliseconds, fastTiming.milliseconds)}`
      );
    } catch {
      if (!destroyed) ctx.setReadout('fastPath', 'interrupted');
    } finally {
      fast.destroy();
      generic.destroy();
      measuring = false;
    }
  }

  // --- Tooltip --------------------------------------------------------------------------------
  const formatValue = (value: number): string => {
    if (!Number.isFinite(value)) return 'no data';
    const meta = getCosVariable(ctx.options.variable);
    if (meta.kind === 'ratio') {
      return ctx.options.variable === 'income'
        ? `$${formatCount(value)}`
        : formatSignificant(value, 3);
    }
    return formatSignificant(value, 4);
  };
  function getTooltip(coordinate: readonly [number, number]): string | null {
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    const tract = sources.locate(x, y);
    if (tract < 0) return null;
    const o = ctx.options;
    const meta = getCosVariable(o.variable);
    const data = currentVariable();
    const lines = [`Tract ${geoids[tract] ?? tract}: ${formatCount(population[tract])} residents`];
    if (meta.kind === 'category') {
      lines.push(`largest group: ${COS_GROUPS[data.groups[tract]].label}`);
    } else {
      lines.push(
        `tract value: ${formatValue(data.display[tract])} ${meta.kind === 'count' ? meta.unit : ''}`.trim()
      );
    }
    if (read) {
      const column = Math.floor((x - grid.originX) / grid.cellSize);
      const row = Math.floor((y - grid.originY) / grid.cellSize);
      if (column >= 0 && row >= 0 && column < RASTER_WIDTH && row < RASTER_HEIGHT) {
        const target = read.targetZones[row * RASTER_WIDTH + column];
        if (target < activeVariant.targetCount) {
          const label =
            o.target === 'community'
              ? `community area ${communityNames[target] ?? target + 1}`
              : `${o.target} cell ${target}`;
          if (meta.kind === 'category') {
            const parts = COS_GROUPS.map(
              (group, index) =>
                `${group.label.split(' ')[0]} ${(read!.shares[target * CATEGORY_COUNT + index] * 100).toFixed(0)}%`
            );
            lines.push(`${label}: ${parts.join(', ')}`);
          } else {
            lines.push(`${label}: ${formatValue(getTargetValue(target))}`);
          }
        }
      }
    }
    return lines.join('\n');
  }

  // --- Initialise -----------------------------------------------------------------------------
  writeVariable();
  writeGridParameters();
  writeDisplayParameters();
  ctx.setReadout('tracts', `${formatCount(sourceCount)} census tracts`);
  ctx.setReadout(
    'raster',
    `${RASTER_WIDTH} × ${RASTER_HEIGHT} cells of ${grid.cellSize.toFixed(0)} m (one shared raster)`
  );
  ctx.setReadout('contributors', activeVariant.contributors);
  ctx.setReadout('pycnoCheck', 'switch the view to the smooth surface');
  ctx.setReadout('fastPath', 'press the button to time it');
  updateGeometryReadouts();

  function updateGeometryReadouts(): void {
    const o = ctx.options;
    if (o.target === 'community') {
      ctx.setReadout(
        'resolution',
        `boundaries accurate to about one raster cell (${grid.cellSize.toFixed(0)} m); community areas are several km wide`
      );
      return;
    }
    const share = (100 * 2 * grid.cellSize) / o.cellWidth;
    ctx.setReadout(
      'resolution',
      `${o.cellWidth / grid.cellSize < 5 ? 'Warning: target cells are under 5 raster cells wide. ' : ''}A boundary is accurate to about one raster cell (${grid.cellSize.toFixed(0)} m), roughly ${share.toFixed(0)}% of a ${o.cellWidth} m target cell's width`
    );
  }

  return {
    getCompiledGraphs: () =>
      [
        sourceCompiled,
        activeVariant.compiled,
        ...(activePycno ? [activePycno.compiled] : []),
        displayCompiled
      ] as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'variable':
          writeVariable();
          writeDisplayParameters();
          break;
        case 'ancillary':
          cellWeights.write(weightRasters[state.ancillary]);
          targetDirty = true;
          displayDirty = true;
          break;
        case 'target':
        case 'denominator': {
          activeVariant = getVariant(state.target, state.denominator);
          ctx.setReadout('contributors', activeVariant.contributors);
          writeGridParameters();
          writeDisplayParameters();
          updateGeometryReadouts();
          targetDirty = true;
          displayDirty = true;
          break;
        }
        case 'cellWidth':
        case 'gridShift':
          writeGridParameters();
          updateGeometryReadouts();
          targetDirty = true;
          displayDirty = true;
          break;
        case 'rule':
        case 'view':
          writeDisplayParameters();
          displayDirty = true;
          if (state.view === 'surface') {
            activePycno = getPycno(state.iterations, state.kernel);
            pycnoDirty = true;
          }
          break;
        case 'iterations':
        case 'kernel':
          if (state.view === 'surface') {
            activePycno = getPycno(state.iterations, state.kernel);
          } else {
            activePycno = null;
          }
          pycnoDirty = true;
          displayDirty = true;
          pycnoCheckStale = true;
          break;
        default:
          break;
      }
      markChanged();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureFastPath();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip: event => (event.coordinate ? getTooltip(event.coordinate) : null),

    encode(commandEncoder) {
      const o = ctx.options;
      if (!sourceRastered) {
        sourceCompiled.encode(commandEncoder, {parameters: undefined});
        sourceRastered = true;
      }
      if (targetDirty) {
        activeVariant.compiled.encode(commandEncoder, {parameters: undefined});
        targetDirty = false;
        displayDirty = true;
        markChanged();
      }
      if (o.view === 'surface' && pycnoDirty) {
        activePycno ??= getPycno(o.iterations, o.kernel);
        activePycno.compiled.encode(commandEncoder, {parameters: undefined});
        pycnoDirty = false;
        displayDirty = true;
        pycnoCheckStale = true;
        markChanged();
      }
      if (displayDirty) {
        displayCompiled.encode(commandEncoder, {parameters: undefined});
        displayDirty = false;
        markChanged();
      }
      if (readStale && performance.now() - lastChangeTime > SETTLE_MILLISECONDS) {
        if (!resultReader.isPending) {
          resultReader.request(commandEncoder);
          readStale = false;
        }
      } else {
        resultReader.flush(commandEncoder);
      }
      if (pycnoCheckStale && o.view === 'surface' && !pycnoDirty) {
        if (performance.now() - lastChangeTime > SETTLE_MILLISECONDS && !pycnoReader.isPending) {
          pycnoReader.request(commandEncoder);
          pycnoCheckStale = false;
        }
      } else {
        pycnoReader.flush(commandEncoder);
      }
    },

    getLayers() {
      const o = ctx.options;
      const meta = getCosVariable(o.variable);
      const category = meta.kind === 'category';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [
        new SpatialAnalysisRasterLayer({
          id: `cos-values-${category ? 'category' : 'value'}`,
          coordinateOrigin,
          gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
          bounds: rasterBounds,
          rowOrigin: 'south',
          values: category ? displayCategory : displayValue,
          valueFormat: category ? 'uint32' : 'float32',
          colormap: category ? 'category' : o.ramp,
          palette: COS_GROUPS.map(group => group.color),
          noDataValue: 255,
          valueRange,
          discardAtOrBelow: category ? undefined : -0.5,
          color: [255, 255, 255, Math.round(255 * o.opacity)]
        })
      ];
      if (o.showTractOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'cos-tract-outline',
            coordinateOrigin,
            segments: tractOutline,
            instanceCount: sources.outlineSegments.length / 4,
            widthPixels: 0.8,
            color: dark ? [255, 255, 255, 70] : [20, 20, 20, 80]
          })
        );
      }
      if (o.showTargetOutlines && o.view !== 'source') {
        if (o.target === 'community') {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'cos-target-outline-community',
              coordinateOrigin,
              segments: communityOutline,
              instanceCount: communities.outlineSegments.length / 4,
              widthPixels: 1.8,
              color: dark ? [255, 255, 255, 235] : [10, 10, 10, 220]
            })
          );
        } else if (activeVariant.outline) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `cos-target-outline-${o.target}`,
              coordinateOrigin,
              segments: activeVariant.outline.buffer,
              instanceCount: activeVariant.outline.count,
              widthPixels: 1.4,
              color: dark ? [255, 255, 255, 220] : [10, 10, 10, 200]
            })
          );
        }
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      resultReader.stop();
      pycnoReader.stop();
      resources.destroy();
    }
  };
}

/** Short description of the contributors a recipe added, for a readout. */
function summarizeContributors(contributors: readonly unknown[]): string {
  const counts = new Map<string, number>();
  for (const contributor of contributors) {
    const id = String((contributor as {id?: string}).id ?? '');
    const name = id.endsWith('-raster')
      ? 'GPUPolygonRasterization'
      : id.endsWith('-areal')
        ? 'GPUArealInterpolation'
        : id.endsWith('-lag')
          ? 'GPUSpatialLag'
          : id;
    counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return [...counts].map(([name, count]) => (count > 1 ? `${name} ×${count}` : name)).join(' → ');
}

export type {B7PolygonSet};
