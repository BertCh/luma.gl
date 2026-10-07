// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  addHotSpotAnalysisRecipe,
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUCellFamily,
  type GPUPermutationAlternative
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor,
  type SpatialAnalysisStyleProps
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {GI_COLORS, getOutlineColor, MORAN_COLORS, NEUTRAL, NO_DATA} from './b4-colors';
import {formatNumber, formatPValue, runGuarded} from './b4-format';
import {
  getVariableInfo,
  loadGeography,
  type Geography,
  type GeographyId,
  type VariableId
} from './b4-geography';
import {
  compileStatGraph,
  createStatBuffers,
  GI_BIN_COUNT,
  MASKED_CLASS,
  MORAN_CLASS_COUNT,
  type StatBuffers,
  type StatVariant
} from './b4-hotspot-stats';
import {
  B4CellLayer,
  B4PolygonFillLayer,
  createGeographyBuffers,
  type B4CellShape,
  type GeographyBuffers
} from './b4-layers';
import {
  createWeightsCore,
  type WeightsConfig,
  type WeightsCore,
  type WeightsSource
} from './b4-weights-core';

/** Where the hot spots are computed. */
export type HotSpotSource = 'nature-cells' | 'us-counties' | 'chicago-tracts';

/** Option state of the hot-spots scene. */
export type HotSpotsOptions = {
  source: HotSpotSource;
  category: string;
  hours: readonly [number, number];
  family: GPUCellFamily;
  /** Quadbin zoom level. */
  resolution: number;
  /** H3 resolution. */
  h3Resolution: number;
  radiusCells: number;
  variable: VariableId;
  weights: WeightsSource;
  k: number;
  bandFactor: number;
  weightTransform: 'binary' | 'row';
  statistic: 'gi-star' | 'local-moran';
  selfWeight: 'include' | 'exclude';
  inference: 'analytic' | 'permutation' | 'none';
  falseDiscoveryRate: boolean;
  alternative: GPUPermutationAlternative;
  maximumNeighbors: string;
  permutations: number;
  seed: number;
  significance: number;
  display: 'classes' | 'zscore' | 'values';
  ramp: RampName;
  showNotSignificant: boolean;
  showOutlines: boolean;
};

/** Rows of the cell table and slots per row of its neighbour CSR. */
const CELL_CAPACITY = 16_384;
const CELL_NEIGHBORS_PER_ROW = 40;
const HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
/** Ground metres per degree, averaged over east-west and north-south at Chicago's latitude. */
const METERS_PER_DEGREE_MEAN = 97_000;
/** Average H3 edge length in metres per resolution. */
const H3_EDGE_METERS: Record<number, number> = {
  4: 22_606,
  5: 8_544,
  6: 3_229,
  7: 1_220,
  8: 461,
  9: 174
};
const EQUATOR_CIRCUMFERENCE = 40_075_016.68;
const EARTH_RADIUS = 6_378_137;
const DEGREES = Math.PI / 180;

const geographyCache = new Map<GeographyId, Promise<Geography>>();

type StatSnapshot = {
  counts: Uint32Array;
  statistics: Float32Array;
  permutationOverflow: number;
  significantCounts: Uint32Array;
  extent: [number, number];
  weightsOverflow: number;
  tableCount: number;
  tableOverflow: number;
  zScores: Float32Array;
  pValues: Float32Array;
  pseudoPValues: Float32Array;
  classes: Uint32Array;
  centers: Float32Array | null;
  values: Float32Array | null;
};

/** What the active source (nature cells or one polygon coverage) provides to the scene. */
type HotWorld = {
  rows: number;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  snapshot: StatSnapshot | null;
  prepare: () => void;
  writeParameters: () => void;
  encode: (commandEncoder: CommandEncoder) => void;
  getLayers: () => Layer[];
  getTooltip: (event: {coordinate: readonly [number, number] | null}) => string | null;
  getGraphs: () => CompiledGPUCommandGraph<never>[];
  describe: () => string;
  /** Marks the analysis input buffers stale (the category filter or the variable changed). */
  writeInputs: () => void;
};

/**
 * Hot spots on the GPU: Getis-Ord Gi* hot and cold spots and local Moran clusters of nature observations
 * aggregated into cells (`addHotSpotAnalysisRecipe`) and of county or tract health measures
 * (`GPUHotSpotAnalysis`, `GPULocalMoran`), each confirmed by `GPULocalPermutationTest` on request.
 * Every compile-time variant is compiled on first use and cached; the radius, the seed, the
 * permutation count and the significance level are buffer writes.
 */
export async function createHotSpots(
  ctx: SceneContext<HotSpotsOptions>
): Promise<SceneInstance<HotSpotsOptions>> {
  const {device} = ctx;
  let destroyed = false;
  let world: HotWorld | null = null;
  let switchToken = 0;
  let dirty = true;
  let stale = true;

  const getGeography = (id: GeographyId) => {
    let promise = geographyCache.get(id);
    if (!promise) {
      promise = loadGeography(id, ctx.datasets, ctx.signal);
      promise.catch(() => geographyCache.delete(id));
      geographyCache.set(id, promise);
    }
    return promise;
  };

  const getStatVariant = (precomputed = false): StatVariant => {
    const o = ctx.options;
    const isGi = o.statistic === 'gi-star';
    return {
      statistic: o.statistic,
      inference: isGi && o.inference === 'none' ? 'analytic' : o.inference,
      selfWeight: isGi && o.selfWeight === 'exclude' ? 0 : 1,
      falseDiscoveryRate: o.falseDiscoveryRate && (isGi || o.inference !== 'none'),
      alternative: o.alternative,
      maximumNeighbors: Number(o.maximumNeighbors),
      precomputed
    };
  };

  const getStatKey = (variant: StatVariant) =>
    [
      variant.statistic,
      variant.inference,
      variant.selfWeight,
      variant.falseDiscoveryRate,
      variant.inference === 'permutation'
        ? `${variant.alternative}-${variant.maximumNeighbors}`
        : '',
      variant.precomputed ? 'pre' : 'full'
    ].join(':');

  const getPaletteStyle = (
    buffers: StatBuffers,
    noData: SpatialAnalysisColor
  ): SpatialAnalysisStyleProps => {
    const o = ctx.options;
    const neutral: SpatialAnalysisColor = o.showNotSignificant ? NEUTRAL : HIDDEN;
    if (o.display === 'zscore') {
      return {
        values: buffers.zScores,
        valueFormat: 'float32',
        colormap: 'diverging',
        valueRange: [-5, 5],
        noDataColor: noData
      };
    }
    const colors = o.statistic === 'gi-star' ? GI_COLORS : MORAN_COLORS;
    return {
      values: buffers.classes,
      valueFormat: 'uint32',
      colormap: 'category',
      palette: Array.from({length: 8}, (_, index) =>
        index === 0 ? neutral : (colors[index] ?? HIDDEN)
      ),
      noDataValue: MASKED_CLASS,
      noDataColor: noData
    };
  };

  const parseSnapshot = (bytes: ArrayBuffer, rows: number, cells: boolean): StatSnapshot => {
    let offset = 0;
    const floats = (count: number) => {
      const view = new Float32Array(bytes, offset, count);
      offset += count * 4;
      return view;
    };
    const words = (count: number) => {
      const view = new Uint32Array(bytes, offset, count);
      offset += count * 4;
      return view;
    };
    const counts = words(GI_BIN_COUNT);
    const statistics = floats(4);
    const permutationOverflow = words(1)[0];
    const significantCounts = words(2);
    const extent = floats(2);
    const weightsOverflow = words(1)[0];
    const zScores = floats(rows);
    const pValues = floats(rows);
    const pseudoPValues = floats(rows);
    const classes = words(rows);
    let tableCount = rows;
    let tableOverflow = 0;
    let centers: Float32Array | null = null;
    let values: Float32Array | null = null;
    if (cells) {
      tableCount = words(1)[0];
      tableOverflow = words(1)[0];
      centers = floats(rows * 2);
      values = floats(rows);
    }
    return {
      counts,
      statistics,
      permutationOverflow,
      significantCounts,
      extent: [extent[0], extent[1]],
      weightsOverflow,
      tableCount,
      tableOverflow,
      zScores,
      pValues,
      pseudoPValues,
      classes,
      centers,
      values
    };
  };

  const getReaderSources = (
    buffers: StatBuffers,
    rows: number,
    weightsOverflow: Buffer,
    cells: {tableCount: Buffer; tableOverflow: Buffer; centers: Buffer; values: Buffer} | null
  ) => [
    {buffer: buffers.counts, size: GI_BIN_COUNT * 4},
    {buffer: buffers.statistics, size: 16},
    {buffer: buffers.permutationOverflow, size: 4},
    {buffer: buffers.significantCounts, size: 8},
    {buffer: buffers.valuesExtent, size: 8},
    {buffer: weightsOverflow, size: 4},
    {buffer: buffers.zScores, size: rows * 4},
    {buffer: buffers.pValues, size: rows * 4},
    {buffer: buffers.pseudoPValues, size: rows * 4},
    {buffer: buffers.classes, size: rows * 4},
    ...(cells
      ? [
          {buffer: cells.tableCount, size: 4},
          {buffer: cells.tableOverflow, size: 4},
          {buffer: cells.centers, size: rows * 8},
          {buffer: cells.values, size: rows * 4}
        ]
      : [])
  ];

  /** Class labels of the active statistic, for tooltips. */
  const describeClass = (value: number): string => {
    const o = ctx.options;
    if (value === MASKED_CLASS) return 'no data';
    if (o.statistic === 'gi-star') {
      const bin = value > 0x7fffffff ? value - 0x100000000 : value;
      if (bin === 0) return 'not significant';
      const confidence = [90, 95, 99][Math.abs(bin) - 1] ?? 0;
      return `${bin > 0 ? 'hot spot' : 'cold spot'}, ${confidence}%`;
    }
    return (
      [
        'not significant',
        'High-High cluster',
        'Low-High outlier',
        'Low-Low cluster',
        'High-Low outlier'
      ][value] ?? 'unknown'
    );
  };

  const updateReadouts = () => {
    const target = world;
    const snapshot = target?.snapshot;
    if (!target || !snapshot) return;
    const o = ctx.options;
    const count = (index: number) => formatCount(snapshot.counts[index]);
    const rows = snapshot.tableCount;
    const analysed = Math.max(0, snapshot.statistics[0]);
    ctx.setReadout('places', `${formatCount(analysed || rows)} analysed`);
    if (o.statistic === 'gi-star') {
      ctx.setReadout('hot', `${count(6)} / ${count(5)} / ${count(4)}`);
      ctx.setReadout('cold', `${count(0)} / ${count(1)} / ${count(2)}`);
      ctx.setReadout('notSignificant', count(3));
      ctx.setReadout('outliers', 'n/a for Gi*');
    } else {
      ctx.setReadout('hot', count(1));
      ctx.setReadout('cold', count(3));
      ctx.setReadout('outliers', `${count(2)} / ${count(4)}`);
      ctx.setReadout('notSignificant', count(0));
    }
    ctx.setReadout(
      'moments',
      `${formatNumber(snapshot.statistics[1], 2)} / ${formatNumber(snapshot.statistics[3], 2)}`
    );
    if (o.inference === 'permutation') {
      ctx.setReadout(
        'permutation',
        `${formatCount(snapshot.significantCounts[1])} ${target.rows === snapshot.tableCount && !snapshot.centers ? 'places' : 'cells'} confirmed (${o.permutations} permutations)`
      );
    } else {
      ctx.setReadout(
        'permutation',
        o.inference === 'none' ? 'no test: every quadrant shown' : 'analytic p-values'
      );
    }
    const overflow =
      snapshot.weightsOverflow !== 0 || snapshot.tableOverflow !== 0
        ? 'overflow: reduce the radius or the resolution'
        : snapshot.permutationOverflow !== 0
          ? `ok; rows with more than ${o.maximumNeighbors} neighbours are skipped by the permutation test`
          : 'ok';
    ctx.setReadout('capacity', overflow);
    ctx.setReadout('scale', target.describe());
    if (Number.isFinite(snapshot.extent[0]) && Number.isFinite(snapshot.extent[1])) {
      ctx.setLegendExtent('display', snapshot.extent);
    }
  };

  const makeReader = (
    resources: SpatialAnalysisResources,
    id: string,
    sources: ReturnType<typeof getReaderSources>,
    rows: number,
    cells: boolean,
    onWorld: (snapshot: StatSnapshot) => void
  ) =>
    new SummaryReader(resources, id, sources, bytes =>
      runGuarded('hot-spot snapshot', () => onWorld(parseSnapshot(bytes, rows, cells)))
    );

  // ---------------------------------------------------------------------------------------
  // Nature cells: addHotSpotAnalysisRecipe
  // ---------------------------------------------------------------------------------------
  const buildCellsWorld = (tracts: Geography): HotWorld => {
    const observations = ctx.datasets.get('chicago-nature');
    const resources = new SpatialAnalysisResources(device, 'hotspots-cells');
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const pointCount = observations.count;
    const positionsLngLat = observations.column<Float32Array>('position');
    const timestamps = observations.column<Uint32Array>('timestamp');
    const categories = observations.column<Uint8Array>('category');
    const positionsBuffer = create('nature-positions', Float32Array.from(positionsLngLat));
    const pointMask = create('nature-mask', new Uint32Array(pointCount).fill(1));
    const hours = new Uint8Array(pointCount);
    for (let index = 0; index < pointCount; index++) {
      hours[index] = Math.floor((timestamps[index] % 86_400) / 3600);
    }
    const rows = CELL_CAPACITY;
    const slots = rows * CELL_NEIGHBORS_PER_ROW;
    const tableCells = create('table-cells', rows * 8);
    const tableCounts = create('table-counts', rows * 4);
    const tableCount = create('table-count', 4);
    const tableOverflow = create('table-overflow', 4);
    const centersDegrees = create('centers-degrees', rows * 8);
    const centersMeters = create('centers-meters', rows * 8);
    const csr = {
      offsets: create('offsets', (rows + 1) * 4),
      neighbors: create('neighbors', slots * 4),
      weights: create('weights', slots * 4)
    };
    const weightsOverflow = create('weights-overflow', 4);
    const values = create('values', rows * 4);
    const mask = create('mask', rows * 4);
    const buffers = createStatBuffers(resources, rows, 'cells');
    const parameters = resources.createParameterBuffer(
      'autocorrelation-parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const permutationParameters = resources.createParameterBuffer(
      'permutation-parameters',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const searchParameters = resources.createParameterBuffer(
      'search-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
    );
    let minimumLng = Infinity;
    let minimumLat = Infinity;
    let maximumLng = -Infinity;
    let maximumLat = -Infinity;
    for (let index = 0; index < pointCount; index++) {
      minimumLng = Math.min(minimumLng, positionsLngLat[index * 2]);
      maximumLng = Math.max(maximumLng, positionsLngLat[index * 2]);
      minimumLat = Math.min(minimumLat, positionsLngLat[index * 2 + 1]);
      maximumLat = Math.max(maximumLat, positionsLngLat[index * 2 + 1]);
    }
    const [originLng, originLat] = tracts.origin;
    const tractBuffers = createGeographyBuffers(resources, tracts, 'cells-tracts');

    const getResolution = () =>
      ctx.options.family === 'quadbin' ? ctx.options.resolution : ctx.options.h3Resolution;

    const getSpacing = (): number => {
      const o = ctx.options;
      return o.family === 'quadbin'
        ? (EQUATOR_CIRCUMFERENCE / 2 ** o.resolution) * Math.cos(originLat * DEGREES)
        : Math.sqrt(3) * (H3_EDGE_METERS[o.h3Resolution] ?? 461);
    };
    const getCellRadius = (): number => {
      const o = ctx.options;
      return o.family === 'quadbin' ? getSpacing() / 2 : (H3_EDGE_METERS[o.h3Resolution] ?? 461);
    };

    const recipes = new Map<string, CompiledGPUCommandGraph<void>>();
    const statGraphs = new Map<string, CompiledGPUCommandGraph<void>>();

    const getRecipeKey = () => {
      const o = ctx.options;
      const variant = getStatVariant();
      return [
        o.family,
        getResolution(),
        variant.selfWeight,
        variant.falseDiscoveryRate,
        o.statistic === 'gi-star' && o.inference === 'permutation'
          ? `perm-${o.alternative}-${o.maximumNeighbors}`
          : 'plain'
      ].join(':');
    };

    const buildRecipe = (): CompiledGPUCommandGraph<void> => {
      const o = ctx.options;
      const variant = getStatVariant();
      const graph = new GPUCommandGraph<void>(device, {id: `hotspots-recipe-${getRecipeKey()}`});
      const view = <Format extends 'float32' | 'uint32' | 'sint32' | 'float32x2' | 'uint32x2'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      const table = {
        cells: view('table-cells', tableCells, 'uint32x2', rows),
        counts: view('table-counts', tableCounts, 'uint32', rows),
        count: view('table-count', tableCount, 'uint32', 1),
        overflow: view('table-overflow', tableOverflow, 'uint32', 1)
      };
      const centers = view('centers-degrees', centersDegrees, 'float32x2', rows);
      const usePermutation = o.statistic === 'gi-star' && o.inference === 'permutation';
      addHotSpotAnalysisRecipe(graph, {
        id: 'hot-spot',
        source: {
          kind: 'points',
          positions: view('nature-positions', positionsBuffer, 'float32x2', pointCount),
          mask: view('nature-mask', pointMask, 'uint32', pointCount),
          family: o.family,
          resolution: getResolution(),
          tableCapacity: rows,
          neighborCapacity: slots,
          gridSize: [128, 128],
          neighborSearchParameters: searchParameters.importToGraph(graph),
          table,
          centers
        },
        parameters: parameters.importToGraph(graph),
        selfWeight: variant.selfWeight,
        falseDiscoveryRate: variant.falseDiscoveryRate,
        zScores: view('z-scores', buffers.zScores, 'float32', rows),
        bins: view('bins', buffers.base, 'sint32', rows),
        pValues: view('p-values', buffers.pValues, 'float32', rows),
        neighborCounts: view('neighbor-counts', buffers.neighborCounts, 'uint32', rows),
        globalStatistics: view('statistics', buffers.statistics, 'float32', 4),
        weights: {
          offsets: view('offsets', csr.offsets, 'uint32', rows + 1),
          neighbors: view('neighbors', csr.neighbors, 'uint32', slots),
          weights: view('weights', csr.weights, 'float32', slots)
        },
        weightsOverflow: view('weights-overflow', weightsOverflow, 'uint32', 1),
        permutation: usePermutation
          ? {
              parameters: permutationParameters.importToGraph(graph),
              maximumPermutations: 999,
              statistic: variant.selfWeight === 1 ? 'localGStar' : 'localG',
              alternative: o.alternative,
              maximumNeighbors: Number(o.maximumNeighbors),
              falseDiscoveryRate: variant.falseDiscoveryRate,
              exceedances: view('exceedances', buffers.exceedances, 'uint32', rows),
              pseudoPValues: view('pseudo-p-values', buffers.pseudoPValues, 'float32', rows),
              significant: view('significant', buffers.significant, 'uint32', rows)
            }
          : undefined
      });
      // Adapters: counts to analysis values and mask, cell centres to planar metres.
      const valuesView = view('values', values, 'float32', rows);
      const maskView = view('mask', mask, 'uint32', rows);
      addKernelPass(graph, {
        id: 'cells-values',
        invocationCount: rows,
        bindings: [
          {name: 'counts', view: table.counts, type: 'u32', access: 'read'},
          {name: 'values', view: valuesView, type: 'f32', access: 'read_write'},
          {name: 'mask', view: maskView, type: 'u32', access: 'read_write'}
        ],
        body: `let count = counts[countsOffset + index];
  values[valuesOffset + index] = f32(count);
  mask[maskOffset + index] = select(0u, 1u, count > 0u);`
      });
      addKernelPass(graph, {
        id: 'cells-centers',
        invocationCount: rows,
        bindings: [
          {name: 'degrees', view: centers, type: 'f32', access: 'read'},
          {
            name: 'meters',
            view: view('centers-meters', centersMeters, 'float32x2', rows),
            type: 'f32',
            access: 'read_write'
          }
        ],
        // Web Mercator offsets around the origin, as deck.gl's METER_OFFSETS system places them.
        body: `let longitude = degrees[degreesOffset + index * 2u];
  let latitude = degrees[degreesOffset + index * 2u + 1u];
  let scale = ${EARTH_RADIUS.toFixed(1)} * cos(${(originLat * DEGREES).toFixed(12)});
  let projected = log(tan(0.7853981633974483 + latitude * ${DEGREES.toFixed(15)} * 0.5));
  let origin = log(tan(0.7853981633974483 + ${(originLat * DEGREES).toFixed(12)} * 0.5));
  meters[metersOffset + index * 2u] = scale * (longitude - (${originLng.toFixed(9)})) * ${DEGREES.toFixed(15)};
  meters[metersOffset + index * 2u + 1u] = scale * (projected - origin);`
      });
      return resources.track(graph.compile());
    };

    const prepare = () => {
      const recipeKey = getRecipeKey();
      if (!recipes.has(recipeKey)) recipes.set(recipeKey, buildRecipe());
      const variant = getStatVariant(ctx.options.statistic === 'gi-star');
      const statKey = getStatKey(variant);
      if (!statGraphs.has(statKey)) {
        statGraphs.set(
          statKey,
          compileStatGraph({
            device,
            resources,
            id: `hotspots-cells-${statKey}`,
            rows,
            slots,
            weights: csr,
            values,
            mask,
            buffers,
            parameters,
            permutationParameters,
            variant
          })
        );
      }
    };

    const writeInputs = () => {
      const o = ctx.options;
      const wanted = o.category === 'all' ? -1 : Number(o.category);
      const [startHour, endHour] = o.hours;
      pointMask.write(
        Uint32Array.from(hours, (hour, index) =>
          (wanted < 0 || categories[index] === wanted) && hour >= startHour && hour < endHour
            ? 1
            : 0
        )
      );
    };

    const writeParameters = () => {
      const o = ctx.options;
      const padding = (getSpacing() * 4) / METERS_PER_DEGREE_MEAN;
      searchParameters.write(
        getGPUNeighborSearchParameterValues({
          bounds: [
            minimumLng - padding,
            minimumLat - padding,
            maximumLng + padding,
            maximumLat + padding
          ],
          radius: (o.radiusCells * getSpacing()) / METERS_PER_DEGREE_MEAN,
          weightKind: 'binary',
          rowStandardize: o.weightTransform === 'row'
        })
      );
      parameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: o.significance})
      );
      permutationParameters.write(
        getGPUPermutationParameterValues({
          seed: o.seed,
          permutations: o.permutations,
          significanceLevel: o.significance
        })
      );
    };

    const reader = makeReader(
      resources,
      'hotspots-cells',
      getReaderSources(buffers, rows, weightsOverflow, {
        tableCount,
        tableOverflow,
        centers: centersMeters,
        values
      }),
      rows,
      true,
      snapshot => {
        created.snapshot = snapshot;
        updateReadouts();
        ctx.requestLayers();
      }
    );

    const created: HotWorld = {
      rows,
      resources,
      reader,
      snapshot: null,
      prepare,
      writeParameters,
      writeInputs,
      describe() {
        const spacing = getSpacing();
        return `${(spacing / 1000).toFixed(2)} km cells, band ${((ctx.options.radiusCells * spacing) / 1000).toFixed(2)} km`;
      },
      encode(commandEncoder) {
        recipes.get(getRecipeKey())!.encode(commandEncoder, {parameters: undefined});
        const variant = getStatVariant(ctx.options.statistic === 'gi-star');
        statGraphs.get(getStatKey(variant))!.encode(commandEncoder, {parameters: undefined});
      },
      getGraphs: () =>
        [...recipes.values(), ...statGraphs.values()] as CompiledGPUCommandGraph<never>[],
      getLayers() {
        const o = ctx.options;
        const theme = ctx.theme();
        const tractOrigin: [number, number, number] = [originLng, originLat, 0];
        const shape: B4CellShape = o.family === 'quadbin' ? 'square' : 'hexagon';
        const style =
          o.display === 'values'
            ? {
                values,
                valueFormat: 'float32' as const,
                colormap: o.ramp,
                extent: buffers.valuesExtent,
                sqrtScale: true,
                discardAtOrBelow: 0
              }
            : getPaletteStyle(buffers, HIDDEN);
        const layers: Layer[] = [
          new B4CellLayer({
            id: `hotspot-cells-${o.family}-${getResolution()}`,
            coordinateOrigin: tractOrigin,
            positions: centersMeters,
            instanceCount: rows,
            shape,
            radiusMeters: getCellRadius() * 0.98,
            ...style,
            opacity: 0.88
          })
        ];
        if (o.showOutlines) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'hotspot-tract-outline',
              coordinateOrigin: tractOrigin,
              segments: tractBuffers.outline,
              instanceCount: tractBuffers.outlineSegmentCount,
              widthPixels: 0.7,
              color: getOutlineColor(theme)
            })
          );
        }
        return layers;
      },
      getTooltip(event) {
        const snapshot = created.snapshot;
        if (!snapshot?.centers || !snapshot.values || !event.coordinate) return null;
        const [x, y] = tracts.project(event.coordinate[0], event.coordinate[1]);
        const reach = getCellRadius() * 1.05;
        let best = -1;
        let bestDistance = reach * reach;
        for (let row = 0; row < Math.min(snapshot.tableCount, rows); row++) {
          const dx = snapshot.centers[row * 2] - x;
          const dy = snapshot.centers[row * 2 + 1] - y;
          const distance = dx * dx + dy * dy;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = row;
          }
        }
        if (best < 0) return null;
        const o = ctx.options;
        const lines = [
          `${formatCount(snapshot.values[best])} observations in this cell`,
          `${o.statistic === 'gi-star' ? 'Gi*' : 'Local Moran'} z: ${formatNumber(snapshot.zScores[best], 2)}`,
          `Analytic p: ${formatPValue(snapshot.pValues[best])}`
        ];
        if (o.inference === 'permutation') {
          lines.push(`Pseudo p: ${formatPValue(snapshot.pseudoPValues[best])}`);
        }
        lines.push(`Result: ${describeClass(snapshot.classes[best])}`);
        return lines.join('\n');
      }
    };
    return created;
  };

  // ---------------------------------------------------------------------------------------
  // Polygons: counties or tracts with a weights core
  // ---------------------------------------------------------------------------------------
  const buildPolygonWorld = (geography: Geography): HotWorld => {
    const resources = new SpatialAnalysisResources(device, `hotspots-${geography.id}`);
    const create = (name: string, data: number | Float32Array | Uint32Array) =>
      resources.createBuffer(name, data);
    const rows = geography.count;
    const core: WeightsCore = createWeightsCore({
      device,
      resources,
      id: 'hotspots-core',
      geography
    });
    const values = create('values', Float32Array.from(geography.getVariable(ctx.options.variable)));
    const mask = create('mask', new Uint32Array(rows));
    const buffers = createStatBuffers(resources, rows, 'polygons');
    const parameters = resources.createParameterBuffer(
      'autocorrelation-parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const permutationParameters = resources.createParameterBuffer(
      'permutation-parameters',
      'uint32',
      GPU_PERMUTATION_PARAMETER_LENGTH
    );
    const geometry: GeographyBuffers = createGeographyBuffers(resources, geography, 'hotspots');
    const statGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
    let weightsDirty = true;

    const getConfig = (): WeightsConfig => {
      const o = ctx.options;
      return {
        source: o.weights,
        k: o.k,
        snapTolerance: 0,
        bandFactor: o.bandFactor,
        knnCapFactor: 0,
        weightKind: 'binary',
        kernel: 'triangular',
        power: 1,
        distanceFloor: 0,
        rowStandardize: false,
        transform: o.weightTransform === 'row' ? 'row' : 'none',
        transformKernel: 'bisquare',
        bandwidthFactor: 0,
        doubleSum: 'one'
      };
    };

    const prepare = () => {
      core.prepare(getConfig());
      const variant = getStatVariant();
      const key = getStatKey(variant);
      if (!statGraphs.has(key)) {
        statGraphs.set(
          key,
          compileStatGraph({
            device,
            resources,
            id: `hotspots-${geography.id}-${key}`,
            rows,
            slots: core.slots,
            weights: core.csr,
            values,
            mask,
            buffers,
            parameters,
            permutationParameters,
            variant
          })
        );
      }
    };

    const writeInputs = () => {
      const data = geography.getVariable(ctx.options.variable);
      values.write(Float32Array.from(data));
      mask.write(Uint32Array.from(data, value => (Number.isFinite(value) ? 1 : 0)));
    };

    const writeParameters = () => {
      const o = ctx.options;
      core.writeParameters(getConfig());
      parameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: o.significance})
      );
      permutationParameters.write(
        getGPUPermutationParameterValues({
          seed: o.seed,
          permutations: o.permutations,
          significanceLevel: o.significance
        })
      );
      weightsDirty = true;
    };

    const reader = makeReader(
      resources,
      `hotspots-${geography.id}`,
      getReaderSources(buffers, rows, core.overflow, null),
      rows,
      false,
      snapshot => {
        created.snapshot = snapshot;
        updateReadouts();
        ctx.requestLayers();
      }
    );

    writeInputs();

    const created: HotWorld = {
      rows,
      resources,
      reader,
      snapshot: null,
      prepare,
      writeParameters,
      writeInputs,
      describe() {
        const o = ctx.options;
        return o.weights === 'band'
          ? `${(core.getBandMeters(getConfig()) / 1000).toFixed(1)} km band`
          : o.weights === 'knn'
            ? `${o.k} nearest centroids`
            : `${o.weights} contiguity`;
      },
      encode(commandEncoder) {
        if (weightsDirty) {
          core.encode(commandEncoder, getConfig());
          weightsDirty = false;
        }
        statGraphs
          .get(getStatKey(getStatVariant()))!
          .encode(commandEncoder, {parameters: undefined});
      },
      getGraphs: () =>
        [...core.getGraphs(), ...statGraphs.values()] as CompiledGPUCommandGraph<never>[],
      getLayers() {
        const o = ctx.options;
        const theme = ctx.theme();
        const origin: [number, number, number] = [geography.origin[0], geography.origin[1], 0];
        const style =
          o.display === 'values'
            ? {
                values,
                valueFormat: 'float32' as const,
                colormap: o.ramp,
                extent: buffers.valuesExtent,
                noDataColor: NO_DATA
              }
            : getPaletteStyle(buffers, NO_DATA);
        const layers: Layer[] = [
          new B4PolygonFillLayer({
            id: 'hotspot-fill',
            coordinateOrigin: origin,
            triangles: geometry.triangles,
            features: geometry.features,
            triangleVertexCount: geometry.triangleVertexCount,
            ...style,
            opacity: 0.92
          })
        ];
        if (o.showOutlines) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'hotspot-outline',
              coordinateOrigin: origin,
              segments: geometry.outline,
              instanceCount: geometry.outlineSegmentCount,
              widthPixels: 0.8,
              color: getOutlineColor(theme)
            })
          );
        }
        return layers;
      },
      getTooltip(event) {
        if (!event.coordinate) return null;
        const row = geography.pick(event.coordinate[0], event.coordinate[1]);
        if (row < 0) return null;
        const o = ctx.options;
        const info = getVariableInfo(o.variable);
        const value = geography.getVariable(o.variable)[row];
        const lines = [
          `${geography.getName(row)} (${geography.getGroupName(row)})`,
          `${info.label}: ${Number.isFinite(value) ? `${value.toFixed(info.digits)} ${info.unit}` : 'no data'}`
        ];
        const snapshot = created.snapshot;
        if (snapshot) {
          lines.push(
            `z: ${formatNumber(snapshot.zScores[row], 2)}, analytic p ${formatPValue(snapshot.pValues[row])}`
          );
          if (o.inference === 'permutation') {
            lines.push(`Pseudo p: ${formatPValue(snapshot.pseudoPValues[row])}`);
          }
          lines.push(`Result: ${describeClass(snapshot.classes[row])}`);
        }
        return lines.join('\n');
      }
    };
    return created;
  };

  // ---------------------------------------------------------------------------------------
  // Source switching
  // ---------------------------------------------------------------------------------------
  const buildWorldFor = async (source: HotSpotSource): Promise<HotWorld> => {
    if (source === 'nature-cells') return buildCellsWorld(await getGeography('chicago-tracts'));
    return buildPolygonWorld(await getGeography(source));
  };

  const activate = async (source: HotSpotSource) => {
    const token = ++switchToken;
    ctx.setStatus('Loading...');
    const next = await buildWorldFor(source);
    if (token !== switchToken || destroyed) {
      next.reader.stop();
      next.resources.destroy();
      return;
    }
    const previous = world;
    world = null;
    ctx.requestLayers();
    if (previous) {
      previous.reader.stop();
      setTimeout(() => previous.resources.destroy(), 200);
    }
    world = next;
    next.writeInputs();
    next.prepare();
    next.writeParameters();
    dirty = true;
    stale = true;
    ctx.setStatus('');
    ctx.requestLayers();
  };

  world = await buildWorldFor(ctx.options.source);
  if (ctx.signal.aborted) throw new Error('aborted');
  world.writeInputs();
  world.prepare();
  world.writeParameters();
  ctx.setStatus('');

  return {
    getCompiledGraphs: () => world?.getGraphs() ?? [],

    setOption(id, value) {
      if (id === 'source') {
        void activate(value as HotSpotSource);
        return;
      }
      const target = world;
      if (!target) return;
      if (['category', 'hours', 'variable'].includes(id)) target.writeInputs();
      target.prepare();
      target.writeParameters();
      dirty = true;
      stale = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const target = world;
      if (!target) return;
      if (dirty || frame.frameIndex < 2) {
        target.encode(commandEncoder);
        dirty = false;
        stale = true;
      }
      if (stale && frame.frameIndex >= 1 && !target.reader.isPending) {
        stale = false;
        target.reader.request(commandEncoder);
      }
    },

    getLayers: () => world?.getLayers() ?? [],

    getTooltip: event => world?.getTooltip(event) ?? null,

    destroy() {
      destroyed = true;
      switchToken++;
      const target = world;
      world = null;
      if (target) {
        target.reader.stop();
        target.resources.destroy();
      }
    }
  };
}

void MORAN_CLASS_COUNT;
