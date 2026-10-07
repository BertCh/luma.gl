// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, Viewport} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  evaluateVariogramModel,
  fitVariogramModel,
  getGPUVariogramParameterValues,
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPUVariogram,
  type VariogramModel,
  type VariogramModelType,
  type VariogramModelWeighting
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUFocalStatisticsParameterValues,
  getGPUInverseDistanceWeightingParameterValues,
  getGPUKrigingParameterValues,
  GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  GPU_KRIGING_PARAMETER_LENGTH,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting,
  GPUKriging,
  type GPUFocalStatisticsShape
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {fetchText, getDataFileUrl, parseCsv} from '../../data/loaders';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {formatSignificant, formatSparkline, hashInteger} from './b7-format';
import {
  getRainfallSurfaceStyle,
  RAINFALL_VARIABLES,
  type RainfallFocalStatistic,
  type RainfallSurface,
  type RainfallVariable
} from './b7-rainfall-style';
import {B7TessellatedRasterLayer} from './b7-tessellated-raster-layer';

/** Option state of the rainfall scene. */
export type RainfallOptions = {
  variable: RainfallVariable;
  holdOut: boolean;
  surface: RainfallSurface;
  power: number;
  neighborCount: number;
  krigingNeighborCount: number;
  searchRadiusKm: number;
  minimumStations: number;
  variogramModel: VariogramModelType;
  weighting: VariogramModelWeighting;
  robust: boolean;
  maximumLagKm: number;
  direction: 'all' | '0' | '1' | '2' | '3';
  azimuth: number;
  focalStatistic: RainfallFocalStatistic;
  focalRadius: number;
  focalShape: GPUFocalStatisticsShape;
  focalMinimumCount: number;
  keepHoles: boolean;
  ramp: RampName;
  opacity: number;
  showStations: boolean;
};

/** Raster that follows the camera. Compile-time. */
const RASTER_WIDTH = 384;
const RASTER_HEIGHT = 240;
const CELL_COUNT = RASTER_WIDTH * RASTER_HEIGHT;
/** Compile-time capacities of the per-frame neighbor limits. */
const MAXIMUM_IDW_NEIGHBORS = 32;
const MAXIMUM_KRIGING_NEIGHBORS = 16;
const MAXIMUM_FOCAL_RADIUS = 8;
const INDEX_GRID_SIZE: readonly [number, number] = [64, 64];
/** Variogram lags and direction sectors. Compile-time. */
const LAG_COUNT = 16;
const SECTOR_COUNT = 4;
const BIN_COUNT = LAG_COUNT * SECTOR_COUNT;
const VARIOGRAM_GRID_SIZE: readonly [number, number] = [32, 32];
const HOLD_OUT_MODULUS = 10;
/** Milliseconds the camera and controls stay still before the surfaces are read back. */
const SETTLE_MILLISECONDS = 450;
const STATION_NAME_LIMIT_PIXELS = 11;

type Bins = {distances: number[]; gammas: number[]; robust: number[]; pairs: number[]};

/**
 * Interpolates GHCN-Daily gauge readings onto a raster that follows the camera.
 *
 * One compiled graph holds `GPUInverseDistanceWeighting`, `GPUKriging` (variance output), a
 * kernel pass that picks the displayed surface and `GPUFocalStatistics`. A second small graph runs
 * `GPUVariogram` (4 direction sectors, 16 lags) whenever the stations, hold-out mask or maximum lag
 * change; the CPU fits a model to the read-back bins and writes it into the kriging parameters.
 * Every control is a parameter or buffer write. Only the stations that are not held out are used,
 * so the surfaces can be validated against the held-out gauges from one read-back.
 */
export async function createRainfallInterpolation(
  ctx: SceneContext<RainfallOptions>
): Promise<SceneInstance<RainfallOptions>> {
  const stations = ctx.datasets.get('ghcn-stations');
  const {device} = ctx;
  const origin = stations.defaultOrigin;
  const projection = stations.getProjection(origin);
  const positions = stations.projectColumn('position', origin);
  const stationCount = positions.length / 2;
  const valuesByVariable: Record<RainfallVariable, Float32Array> = {
    prcp: stations.column('prcp'),
    prcpPrevDay: stations.column('prcpPrevDay'),
    tmax: stations.column('tmax')
  };
  const elevation = stations.column('elevation');
  const stationTable = parseCsv(
    await fetchText(getDataFileUrl('ghcn-stations', 'stations.csv'), ctx.signal)
  );
  ctx.signal.throwIfAborted();
  const stationNames = stationTable.rows.map(row => row[1] ?? '');
  const stationStates = stationTable.rows.map(row => row[2] ?? '');

  const stationBounds: [number, number, number, number] = [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ];
  for (let station = 0; station < stationCount; station++) {
    stationBounds[0] = Math.min(stationBounds[0], positions[station * 2]);
    stationBounds[1] = Math.min(stationBounds[1], positions[station * 2 + 1]);
    stationBounds[2] = Math.max(stationBounds[2], positions[station * 2]);
    stationBounds[3] = Math.max(stationBounds[3], positions[station * 2 + 1]);
  }
  const indexBounds: [number, number, number, number] = [
    stationBounds[0] - 1000,
    stationBounds[1] - 1000,
    stationBounds[2] + 1000,
    stationBounds[3] + 1000
  ];

  const resources = new SpatialAnalysisResources(device, 'rainfall');
  const positionsBuffer = resources.createBuffer('positions', positions);
  const valuesBuffer = resources.createBuffer('values', valuesByVariable[ctx.options.variable]);
  const maskBuffer = resources.createBuffer('mask', new Uint32Array(stationCount).fill(1));
  const heldOutIds = resources.createBuffer('held-out-ids', new Uint32Array(stationCount));
  const idwSurface = resources.createBuffer('idw-surface', CELL_COUNT * 4);
  const idwCounts = resources.createBuffer('idw-counts', CELL_COUNT * 4);
  const krigingSurface = resources.createBuffer('kriging-surface', CELL_COUNT * 4);
  const krigingVariance = resources.createBuffer('kriging-variance', CELL_COUNT * 4);
  const selectedSurface = resources.createBuffer('selected-surface', CELL_COUNT * 4);
  const auxiliarySurface = resources.createBuffer('auxiliary-surface', CELL_COUNT * 4);
  const focalStatistics = {
    mean: resources.createBuffer('focal-mean', CELL_COUNT * 4),
    min: resources.createBuffer('focal-min', CELL_COUNT * 4),
    max: resources.createBuffer('focal-max', CELL_COUNT * 4),
    range: resources.createBuffer('focal-range', CELL_COUNT * 4),
    standardDeviation: resources.createBuffer('focal-standard-deviation', CELL_COUNT * 4)
  } satisfies Record<RainfallFocalStatistic, Buffer>;
  const semivarianceBuffer = resources.createBuffer('semivariances', BIN_COUNT * 4);
  const robustBuffer = resources.createBuffer('robust-semivariances', BIN_COUNT * 4);
  const pairCountBuffer = resources.createBuffer('pair-counts', BIN_COUNT * 4);
  const distanceBuffer = resources.createBuffer('mean-distances', BIN_COUNT * 4);
  const statisticsBuffer = resources.createBuffer('variogram-statistics', 5 * 4);
  const idwParameters = resources.createParameterBuffer(
    'idw-parameters',
    'float32',
    GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH
  );
  const krigingParameters = resources.createParameterBuffer(
    'kriging-parameters',
    'float32',
    GPU_KRIGING_PARAMETER_LENGTH
  );
  const focalParameters = resources.createParameterBuffer(
    'focal-parameters',
    'float32',
    GPU_FOCAL_STATISTICS_PARAMETER_LENGTH
  );
  const variogramParameters = resources.createParameterBuffer(
    'variogram-parameters',
    'float32',
    GPU_VARIOGRAM_PARAMETER_LENGTH
  );
  const modeParameters = resources.createParameterBuffer('mode-parameters', 'float32', 4);

  // --- Main graph -----------------------------------------------------------------------------
  const graph = new GPUCommandGraph<void>(device, {id: 'rainfall'});
  const positionsView = importGraphBuffer(
    graph,
    'positions',
    positionsBuffer,
    'float32x2',
    stationCount
  );
  const valuesView = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', stationCount);
  const maskView = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', stationCount);
  const idwSurfaceView = importGraphBuffer(graph, 'idw-surface', idwSurface, 'float32', CELL_COUNT);
  const krigingSurfaceView = importGraphBuffer(
    graph,
    'kriging-surface',
    krigingSurface,
    'float32',
    CELL_COUNT
  );
  const krigingVarianceView = importGraphBuffer(
    graph,
    'kriging-variance',
    krigingVariance,
    'float32',
    CELL_COUNT
  );
  const idwCountsView = importGraphBuffer(graph, 'idw-counts', idwCounts, 'uint32', CELL_COUNT);
  const selectedView = importGraphBuffer(graph, 'selected', selectedSurface, 'float32', CELL_COUNT);
  graph.add(
    new GPUInverseDistanceWeighting({
      id: 'idw',
      positions: positionsView,
      values: valuesView,
      mask: maskView,
      parameters: idwParameters.importToGraph(graph),
      width: RASTER_WIDTH,
      height: RASTER_HEIGHT,
      indexGridSize: INDEX_GRID_SIZE,
      indexBounds,
      maximumNeighborCount: MAXIMUM_IDW_NEIGHBORS,
      output: {values: idwSurfaceView, counts: idwCountsView}
    })
  );
  graph.add(
    new GPUKriging({
      id: 'kriging',
      positions: positionsView,
      values: valuesView,
      mask: maskView,
      parameters: krigingParameters.importToGraph(graph),
      width: RASTER_WIDTH,
      height: RASTER_HEIGHT,
      indexGridSize: INDEX_GRID_SIZE,
      indexBounds,
      maximumNeighborCount: MAXIMUM_KRIGING_NEIGHBORS,
      output: {values: krigingSurfaceView, variance: krigingVarianceView}
    })
  );
  // Surface selector. A parameter word chooses what feeds the focal statistics (IDW, kriging, or
  // their difference) and what the auxiliary map shows (kriging standard error or IDW support).
  addKernelPass(graph, {
    id: 'select-surface',
    invocationCount: CELL_COUNT,
    bindings: [
      {name: 'idw', view: idwSurfaceView, type: 'f32', access: 'read'},
      {name: 'kriging', view: krigingSurfaceView, type: 'f32', access: 'read'},
      {name: 'variance', view: krigingVarianceView, type: 'f32', access: 'read'},
      {name: 'counts', view: idwCountsView, type: 'u32', access: 'read'},
      {name: 'mode', view: modeParameters.importToGraph(graph), type: 'f32', access: 'read'},
      {name: 'selected', view: selectedView, type: 'f32', access: 'read_write'},
      {
        name: 'auxiliary',
        view: importGraphBuffer(graph, 'auxiliary', auxiliarySurface, 'float32', CELL_COUNT),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: /* wgsl */ `
  var nanBits: u32 = 0x7fc00000u;
  let nan = bitcast<f32>(nanBits);
  let a = idw[idwOffset + index];
  let b = kriging[krigingOffset + index];
  let v = variance[varianceOffset + index];
  let aInvalid = (bitcast<u32>(a) & 0x7fffffffu) >= 0x7f800000u;
  let bInvalid = (bitcast<u32>(b) & 0x7fffffffu) >= 0x7f800000u;
  let vInvalid = (bitcast<u32>(v) & 0x7fffffffu) >= 0x7f800000u;
  let surface = u32(mode[modeOffset]);
  var selectedValue = a;
  var invalid = aInvalid;
  if (surface == 1u) {
    selectedValue = b;
    invalid = bInvalid;
  } else if (surface == 2u) {
    selectedValue = b - a;
    invalid = aInvalid || bInvalid;
  }
  selected[selectedOffset + index] = select(selectedValue, nan, invalid);
  if (u32(mode[modeOffset + 1u]) == 1u) {
    auxiliary[auxiliaryOffset + index] = f32(counts[countsOffset + index]);
  } else {
    auxiliary[auxiliaryOffset + index] = select(sqrt(max(v, 0.0)), nan, vInvalid);
  }`
  });
  graph.add(
    new GPUFocalStatistics({
      id: 'focal',
      values: selectedView,
      width: RASTER_WIDTH,
      height: RASTER_HEIGHT,
      maximumRadius: MAXIMUM_FOCAL_RADIUS,
      parameters: focalParameters.importToGraph(graph),
      output: {
        mean: importGraphBuffer(graph, 'focal-mean', focalStatistics.mean, 'float32', CELL_COUNT),
        min: importGraphBuffer(graph, 'focal-min', focalStatistics.min, 'float32', CELL_COUNT),
        max: importGraphBuffer(graph, 'focal-max', focalStatistics.max, 'float32', CELL_COUNT),
        range: importGraphBuffer(
          graph,
          'focal-range',
          focalStatistics.range,
          'float32',
          CELL_COUNT
        ),
        standardDeviation: importGraphBuffer(
          graph,
          'focal-standard-deviation',
          focalStatistics.standardDeviation,
          'float32',
          CELL_COUNT
        )
      }
    })
  );
  const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

  // --- Variogram graph ------------------------------------------------------------------------
  const variogramGraph = new GPUCommandGraph<void>(device, {id: 'rainfall-variogram'});
  variogramGraph.add(
    new GPUVariogram({
      id: 'variogram',
      positions: importGraphBuffer(
        variogramGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        stationCount
      ),
      values: importGraphBuffer(variogramGraph, 'values', valuesBuffer, 'float32', stationCount),
      mask: importGraphBuffer(variogramGraph, 'mask', maskBuffer, 'uint32', stationCount),
      parameters: variogramParameters.importToGraph(variogramGraph),
      gridSize: VARIOGRAM_GRID_SIZE,
      lagCount: LAG_COUNT,
      directionCount: SECTOR_COUNT,
      semivariances: importGraphBuffer(
        variogramGraph,
        'semivariances',
        semivarianceBuffer,
        'float32',
        BIN_COUNT
      ),
      robustSemivariances: importGraphBuffer(
        variogramGraph,
        'robust',
        robustBuffer,
        'float32',
        BIN_COUNT
      ),
      pairCounts: importGraphBuffer(
        variogramGraph,
        'pair-counts',
        pairCountBuffer,
        'uint32',
        BIN_COUNT
      ),
      meanDistances: importGraphBuffer(
        variogramGraph,
        'mean-distances',
        distanceBuffer,
        'float32',
        BIN_COUNT
      ),
      statistics: importGraphBuffer(variogramGraph, 'statistics', statisticsBuffer, 'float32', 5)
    })
  );
  const variogramCompiled: CompiledGPUCommandGraph<void> = resources.track(
    variogramGraph.compile()
  );

  // --- State ----------------------------------------------------------------------------------
  let destroyed = false;
  let measuring = false;
  let dirtyRaster = true;
  let variogramStale = true;
  let splitSeed = 1;
  let heldOutCount = 0;
  let trainingCount = 0;
  let lastChangeTime = performance.now();
  let readStale = true;
  let viewBounds: [number, number, number, number] = [...stationBounds];
  let encodedBounds: [number, number, number, number] = [...stationBounds];
  let bins: {sectors: Bins[]; omni: Bins; statistics: Float32Array} | null = null;
  let fit: VariogramModel | null = null;
  let heldOut = new Uint8Array(stationCount);
  let heldOutList: number[] = [];
  let surfaceRead: {
    extent: [number, number, number, number];
    idw: Float32Array;
    kriging: Float32Array;
    variance: Float32Array;
  } | null = null;
  let requestedExtent: [number, number, number, number] = [...stationBounds];
  let lastLegendKey = '';

  const markChanged = () => {
    lastChangeTime = performance.now();
    readStale = true;
  };

  function getValues(): Float32Array {
    return valuesByVariable[ctx.options.variable];
  }

  function writeSplit(): void {
    heldOut = new Uint8Array(stationCount);
    heldOutList = [];
    const values = getValues();
    trainingCount = 0;
    for (let station = 0; station < stationCount; station++) {
      const finite = Number.isFinite(values[station]);
      const out =
        ctx.options.holdOut &&
        finite &&
        hashInteger(station * 31 + splitSeed) % HOLD_OUT_MODULUS === 0;
      if (out) {
        heldOut[station] = 1;
        heldOutList.push(station);
      } else if (finite) {
        trainingCount++;
      }
    }
    heldOutCount = heldOutList.length;
    maskBuffer.write(Uint32Array.from(heldOut, flag => (flag ? 0 : 1)));
    heldOutIds.write(Uint32Array.from(heldOutList));
    ctx.setReadout('stations', `${formatCount(stationCount)} gauges`);
    ctx.setReadout('training', formatCount(trainingCount));
    ctx.setReadout('heldOut', ctx.options.holdOut ? formatCount(heldOutCount) : 'none');
  }

  function getSampleVariance(): number {
    const variance = bins?.statistics[2];
    return variance && variance > 0 ? variance : 1;
  }

  function writeParameters(): void {
    const o = ctx.options;
    const radius = o.searchRadiusKm * 1000;
    idwParameters.write(
      getGPUInverseDistanceWeightingParameterValues({
        extent: viewBounds,
        searchRadius: radius,
        power: o.power,
        neighborCount: o.neighborCount,
        minimumNeighborCount: o.minimumStations
      })
    );
    const total = getSampleVariance();
    const model = fit ?? {
      model: o.variogramModel,
      nugget: 0.1 * total,
      sill: 0.9 * total,
      range: 150_000
    };
    krigingParameters.write(
      getGPUKrigingParameterValues({
        extent: viewBounds,
        searchRadius: radius,
        neighborCount: o.krigingNeighborCount,
        minimumNeighborCount: Math.max(o.minimumStations, 3),
        variogram: {
          model: model.model,
          nugget: Math.max(model.nugget, 1e-4 * total),
          sill: Math.max(model.sill, 1e-4 * total),
          range: Math.max(model.range, 1000)
        }
      })
    );
    focalParameters.write(
      getGPUFocalStatisticsParameterValues({
        radius: o.focalRadius,
        shape: o.focalShape,
        minimumCount: o.focalMinimumCount,
        propagateCenterNoData: o.keepHoles
      })
    );
    modeParameters.write(
      Float32Array.of(
        o.surface === 'kriging' ? 1 : o.surface === 'difference' ? 2 : 0,
        o.surface === 'support' ? 1 : 0,
        0,
        0
      )
    );
  }

  function writeVariogramParameters(): void {
    variogramParameters.write(
      getGPUVariogramParameterValues({
        bounds: indexBounds,
        maximumDistance: ctx.options.maximumLagKm * 1000,
        azimuthOffset: (ctx.options.azimuth * Math.PI) / 180
      })
    );
  }

  // --- Variogram read-back and CPU fit --------------------------------------------------------
  const variogramReader = new SummaryReader(
    resources,
    'rainfall-variogram',
    [
      {buffer: semivarianceBuffer, size: BIN_COUNT * 4},
      {buffer: robustBuffer, size: BIN_COUNT * 4},
      {buffer: pairCountBuffer, size: BIN_COUNT * 4},
      {buffer: distanceBuffer, size: BIN_COUNT * 4},
      {buffer: statisticsBuffer, size: 5 * 4}
    ],
    bytes => {
      const floats = new Float32Array(bytes);
      const counts = new Uint32Array(bytes);
      const read = (sector: number): Bins => {
        const result: Bins = {distances: [], gammas: [], robust: [], pairs: []};
        for (let lag = 0; lag < LAG_COUNT; lag++) {
          const bin = sector * LAG_COUNT + lag;
          result.gammas.push(floats[bin]);
          result.robust.push(floats[BIN_COUNT + bin]);
          result.pairs.push(counts[2 * BIN_COUNT + bin]);
          result.distances.push(floats[3 * BIN_COUNT + bin]);
        }
        return result;
      };
      const sectors = Array.from({length: SECTOR_COUNT}, (_, sector) => read(sector));
      const omni: Bins = {distances: [], gammas: [], robust: [], pairs: []};
      for (let lag = 0; lag < LAG_COUNT; lag++) {
        let pairs = 0;
        let gamma = 0;
        let robust = 0;
        let distance = 0;
        for (const sector of sectors) {
          const n = sector.pairs[lag];
          if (n > 0 && Number.isFinite(sector.gammas[lag])) {
            pairs += n;
            gamma += n * sector.gammas[lag];
            robust += n * sector.robust[lag];
            distance += n * sector.distances[lag];
          }
        }
        omni.pairs.push(pairs);
        omni.gammas.push(pairs > 0 ? gamma / pairs : Number.NaN);
        omni.robust.push(pairs > 0 ? robust / pairs : Number.NaN);
        omni.distances.push(pairs > 0 ? distance / pairs : Number.NaN);
      }
      bins = {sectors, omni, statistics: floats.slice(4 * BIN_COUNT, 4 * BIN_COUNT + 5)};
      refit();
    }
  );

  function fitBins(source: Bins): VariogramModel | null {
    try {
      return fitVariogramModel(
        {
          distances: source.distances,
          semivariances: ctx.options.robust ? source.robust : source.gammas,
          pairCounts: source.pairs
        },
        {model: ctx.options.variogramModel, weighting: ctx.options.weighting}
      );
    } catch {
      return null;
    }
  }

  /** Fits the model on the CPU to the omnidirectional bins and refreshes the variogram readouts. */
  function refit(): void {
    if (!bins || destroyed) return;
    const o = ctx.options;
    const unit = RAINFALL_VARIABLES[o.variable].squaredUnit;
    fit = fitBins(bins.omni);
    const lags = o.robust ? bins.omni.robust : bins.omni.gammas;
    const modelValues = bins.omni.distances.map(distance =>
      fit ? evaluateVariogramModel(fit, distance) : Number.NaN
    );
    const scale = Math.max(
      ...lags.filter(Number.isFinite),
      ...modelValues.filter(Number.isFinite),
      1e-9
    );
    ctx.setReadout('empirical', formatSparkline(lags, scale));
    ctx.setReadout('fitted', formatSparkline(modelValues, scale));
    ctx.setReadout(
      'lagAxis',
      `0 to ${o.maximumLagKm} km in ${LAG_COUNT} lags; top of the chart ${formatSignificant(scale)} ${unit}`
    );
    ctx.setReadout(
      'variogramModel',
      fit
        ? `nugget ${formatSignificant(fit.nugget)} · partial sill ${formatSignificant(fit.sill)} ${unit} · range ${formatSignificant(fit.range / 1000)} km`
        : 'not fitted'
    );
    ctx.setReadout('sampleVariance', `${formatSignificant(bins.statistics[2])} ${unit}`);
    updateDirectionReadout();
    dirtyRaster = true;
    markChanged();
    ctx.requestLayers();
  }

  function updateDirectionReadout(): void {
    if (!bins) return;
    const o = ctx.options;
    if (o.direction === 'all') {
      ctx.setReadout('sectorRange', 'omnidirectional (choose a sector to compare directions)');
      return;
    }
    const sector = Number(o.direction);
    const from = (o.azimuth + (sector * 180) / SECTOR_COUNT) % 180;
    const to = from + 180 / SECTOR_COUNT;
    const sectorFit = fitBins(bins.sectors[sector]);
    ctx.setReadout(
      'sectorRange',
      sectorFit
        ? `${from.toFixed(0)}° to ${to.toFixed(0)}° from east: range ${formatSignificant(sectorFit.range / 1000)} km, sill ${formatSignificant(sectorFit.nugget + sectorFit.sill)} (omnidirectional ${formatSignificant((fit?.range ?? Number.NaN) / 1000)} km)`
        : `${from.toFixed(0)}° to ${to.toFixed(0)}° from east: too few pairs`
    );
  }

  // --- Surface read-back: hold-out validation and tooltips ------------------------------------
  const surfaceReader = new SummaryReader(
    resources,
    'rainfall-surfaces',
    [
      {buffer: idwSurface, size: CELL_COUNT * 4},
      {buffer: krigingSurface, size: CELL_COUNT * 4},
      {buffer: krigingVariance, size: CELL_COUNT * 4}
    ],
    bytes => {
      surfaceRead = {
        extent: requestedExtent,
        idw: new Float32Array(bytes, 0, CELL_COUNT),
        kriging: new Float32Array(bytes, CELL_COUNT * 4, CELL_COUNT),
        variance: new Float32Array(bytes, CELL_COUNT * 8, CELL_COUNT)
      };
      validateHeldOut();
    }
  );

  /** Bilinear sample of a read-back raster at planar meters; NaN when outside or nodata. */
  function sampleRaster(
    raster: Float32Array,
    extent: readonly [number, number, number, number],
    x: number,
    y: number
  ): number {
    const cellWidth = (extent[2] - extent[0]) / RASTER_WIDTH;
    const cellHeight = (extent[3] - extent[1]) / RASTER_HEIGHT;
    const column = (x - extent[0]) / cellWidth - 0.5;
    const row = (y - extent[1]) / cellHeight - 0.5;
    if (column < -0.5 || row < -0.5 || column > RASTER_WIDTH - 0.5 || row > RASTER_HEIGHT - 0.5) {
      return Number.NaN;
    }
    const column0 = Math.max(0, Math.min(RASTER_WIDTH - 2, Math.floor(column)));
    const row0 = Math.max(0, Math.min(RASTER_HEIGHT - 2, Math.floor(row)));
    const fx = Math.max(0, Math.min(1, column - column0));
    const fy = Math.max(0, Math.min(1, row - row0));
    let sum = 0;
    let weight = 0;
    for (const [dx, dy, w] of [
      [0, 0, (1 - fx) * (1 - fy)],
      [1, 0, fx * (1 - fy)],
      [0, 1, (1 - fx) * fy],
      [1, 1, fx * fy]
    ] as const) {
      const value = raster[(row0 + dy) * RASTER_WIDTH + column0 + dx];
      if (Number.isFinite(value) && w > 0) {
        sum += w * value;
        weight += w;
      }
    }
    return weight > 0.5 ? sum / weight : Number.NaN;
  }

  function validateHeldOut(): void {
    if (!surfaceRead || !ctx.options.holdOut) {
      for (const id of ['validationIdw', 'validationKriging', 'validationVerdict'])
        ctx.setReadout(id, ctx.options.holdOut ? '...' : 'turn on "Hold out 10% of gauges"');
      return;
    }
    const values = getValues();
    const unit = RAINFALL_VARIABLES[ctx.options.variable].unit;
    const scores = {idw: {sum: 0, abs: 0, bias: 0, n: 0}, kriging: {sum: 0, abs: 0, bias: 0, n: 0}};
    for (const station of heldOutList) {
      const x = positions[station * 2];
      const y = positions[station * 2 + 1];
      for (const [name, raster] of [
        ['idw', surfaceRead.idw],
        ['kriging', surfaceRead.kriging]
      ] as const) {
        const predicted = sampleRaster(raster, surfaceRead.extent, x, y);
        if (!Number.isFinite(predicted)) continue;
        const error = predicted - values[station];
        scores[name].sum += error * error;
        scores[name].abs += Math.abs(error);
        scores[name].bias += error;
        scores[name].n++;
      }
    }
    const describe = (score: typeof scores.idw) =>
      score.n > 0
        ? `RMSE ${formatSignificant(Math.sqrt(score.sum / score.n))} ${unit} · MAE ${formatSignificant(score.abs / score.n)} · bias ${formatSignificant(score.bias / score.n)} (n=${score.n})`
        : 'no held-out gauge on screen';
    ctx.setReadout('validationIdw', describe(scores.idw));
    ctx.setReadout('validationKriging', describe(scores.kriging));
    if (scores.idw.n > 5 && scores.kriging.n > 5) {
      const idwRmse = Math.sqrt(scores.idw.sum / scores.idw.n);
      const krigingRmse = Math.sqrt(scores.kriging.sum / scores.kriging.n);
      const change = ((krigingRmse - idwRmse) / idwRmse) * 100;
      ctx.setReadout(
        'validationVerdict',
        `kriging RMSE is ${Math.abs(change).toFixed(1)}% ${change <= 0 ? 'lower' : 'higher'} than IDW at the held-out gauges on screen`
      );
    } else {
      ctx.setReadout('validationVerdict', 'zoom out so more held-out gauges are visible');
    }
  }

  // --- Timing ---------------------------------------------------------------------------------
  async function measureGraphs(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('graphTime', 'measuring...');
    try {
      const options = {
        parameters: undefined,
        completionBuffer: auxiliarySurface,
        signal: ctx.signal
      };
      const main = await measureCompiledGraph(device, compiled, options);
      const variogram = await measureCompiledGraph(device, variogramCompiled, {
        ...options,
        completionBuffer: statisticsBuffer
      });
      if (destroyed) return;
      ctx.setReadout(
        'graphTime',
        `${compiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(main)}`
      );
      ctx.setReadout(
        'variogramTime',
        `${variogramCompiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(variogram)}`
      );
    } catch {
      if (!destroyed) ctx.setReadout('graphTime', 'interrupted');
    } finally {
      measuring = false;
    }
  }

  writeSplit();
  writeVariogramParameters();
  ctx.setReadout('empirical', '...');
  ctx.setReadout('fitted', '...');

  // --- Tooltip --------------------------------------------------------------------------------
  function getTooltip(coordinate: readonly [number, number]): string | null {
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    const zoom = ctx.getViewport()?.zoom ?? 5;
    const metersPerPixel = (78271.51696 * Math.cos((coordinate[1] * Math.PI) / 180)) / 2 ** zoom;
    const limit = STATION_NAME_LIMIT_PIXELS * metersPerPixel;
    const values = getValues();
    const meta = RAINFALL_VARIABLES[ctx.options.variable];
    let nearest = -1;
    let nearestDistance = limit * limit;
    for (let station = 0; station < stationCount; station++) {
      if (!Number.isFinite(values[station])) continue;
      const dx = positions[station * 2] - x;
      const dy = positions[station * 2 + 1] - y;
      const distance = dx * dx + dy * dy;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = station;
      }
    }
    if (nearest >= 0) {
      const state = stationStates[nearest] ? `, ${stationStates[nearest]}` : '';
      return `${stationNames[nearest]}${state}\n${formatSignificant(values[nearest])} ${meta.unit} observed · elevation ${formatSignificant(elevation[nearest], 4)} m${heldOut[nearest] ? '\nheld out: not used by the surfaces' : ''}`;
    }
    if (!surfaceRead) return null;
    const idw = sampleRaster(surfaceRead.idw, surfaceRead.extent, x, y);
    const kriging = sampleRaster(surfaceRead.kriging, surfaceRead.extent, x, y);
    const variance = sampleRaster(surfaceRead.variance, surfaceRead.extent, x, y);
    if (!Number.isFinite(idw) && !Number.isFinite(kriging)) return null;
    const parts = [`IDW ${formatSignificant(idw)} ${meta.unit}`];
    if (Number.isFinite(kriging)) {
      parts.push(
        `kriging ${formatSignificant(kriging)} ± ${formatSignificant(Math.sqrt(Math.max(variance, 0)))} ${meta.unit}`
      );
    }
    return parts.join('\n');
  }

  return {
    getCompiledGraphs: () => [compiled, variogramCompiled] as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'variable':
          valuesBuffer.write(getValues());
          writeSplit();
          variogramStale = true;
          dirtyRaster = true;
          break;
        case 'holdOut':
          writeSplit();
          variogramStale = true;
          dirtyRaster = true;
          break;
        case 'variogramModel':
        case 'weighting':
        case 'robust':
          refit();
          break;
        case 'maximumLagKm':
        case 'azimuth':
          writeVariogramParameters();
          variogramStale = true;
          break;
        case 'direction':
          updateDirectionReadout();
          break;
        case 'ramp':
        case 'opacity':
        case 'showStations':
        case 'focalStatistic':
          break;
        default:
          dirtyRaster = true;
      }
      void state;
      markChanged();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'reseed') {
        splitSeed++;
        writeSplit();
        variogramStale = true;
        dirtyRaster = true;
        markChanged();
        ctx.requestLayers();
      } else if (id === 'measure') {
        void measureGraphs();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip: event => (event.coordinate ? getTooltip(event.coordinate) : null),

    encode(commandEncoder, frame) {
      encodeFrame(commandEncoder, frame.viewport);
    },

    getLayers() {
      return buildLayers();
    },

    destroy() {
      destroyed = true;
      variogramReader.stop();
      surfaceReader.stop();
      resources.destroy();
    }
  };

  function encodeFrame(
    commandEncoder: Parameters<SceneInstance<RainfallOptions>['encode']>[0],
    viewport: Viewport
  ) {
    const next = getViewportMetricBounds(viewport, projection);
    if (next.some((value, index) => value !== viewBounds[index])) {
      viewBounds = next;
      dirtyRaster = true;
      markChanged();
      ctx.setReadout(
        'cellSize',
        `${formatSignificant((viewBounds[2] - viewBounds[0]) / RASTER_WIDTH / 1000)} km × ${formatSignificant((viewBounds[3] - viewBounds[1]) / RASTER_HEIGHT / 1000)} km`
      );
    }
    if (variogramStale) {
      variogramStale = false;
      variogramCompiled.encode(commandEncoder, {parameters: undefined});
      variogramReader.request(commandEncoder);
    }
    variogramReader.flush(commandEncoder);
    if (dirtyRaster) {
      dirtyRaster = false;
      writeParameters();
      compiled.encode(commandEncoder, {parameters: undefined});
      encodedBounds = [...viewBounds];
    }
    if (readStale && performance.now() - lastChangeTime > SETTLE_MILLISECONDS) {
      if (!surfaceReader.isPending) {
        requestedExtent = [...encodedBounds];
        surfaceReader.request(commandEncoder);
        readStale = false;
      }
    } else {
      surfaceReader.flush(commandEncoder);
    }
  }

  function buildLayers(): Layer[] {
    const o = ctx.options;
    const style = getRainfallSurfaceStyle(o);
    const meta = RAINFALL_VARIABLES[o.variable];
    const dark = ctx.theme() === 'dark';
    const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
    const isAuxiliary = o.surface === 'error' || o.surface === 'support';
    let range = style.range;
    if (!range) {
      const total = fit ? fit.nugget + fit.sill : getSampleVariance();
      range = [0, Math.sqrt(Math.max(total, 1e-9))];
    }
    const legendKey = `${range[0]}:${range[1]}`;
    if (legendKey !== lastLegendKey) {
      lastLegendKey = legendKey;
      ctx.setLegendExtent('surface', range);
    }
    const statistic = o.focalRadius > 0 ? o.focalStatistic : 'mean';
    const layers: Layer[] = [
      new B7TessellatedRasterLayer({
        id: `rainfall-${o.surface}-${statistic}`,
        coordinateOrigin,
        gridSize: [RASTER_WIDTH, RASTER_HEIGHT],
        bounds: idwParameters.buffer,
        rowOrigin: 'south',
        values: isAuxiliary ? auxiliarySurface : focalStatistics[statistic],
        valueFormat: 'float32',
        colormap: style.ramp,
        valueRange: range,
        noDataColor: [0, 0, 0, 0],
        color: [255, 255, 255, Math.round(255 * o.opacity)]
      })
    ];
    if (o.showStations) {
      const stationRamp = o.ramp;
      const stationRange = meta.range;
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'rainfall-station-halo',
          coordinateOrigin,
          positions: positionsBuffer,
          values: valuesBuffer,
          valueFormat: 'float32',
          // A one-color ramp: stations without a value (NaN) stay invisible.
          colormap: 'grayscale',
          valueRange: dark ? [-1e30, -1e29] : [1e30, 1e31],
          noDataColor: [0, 0, 0, 0],
          instanceCount: stationCount,
          radiusPixels: 2.6,
          color: [255, 255, 255, dark ? 130 : 190]
        }),
        new SpatialAnalysisPointLayer({
          id: 'rainfall-stations',
          coordinateOrigin,
          positions: positionsBuffer,
          values: valuesBuffer,
          valueFormat: 'float32',
          colormap: stationRamp,
          valueRange: stationRange,
          noDataColor: [0, 0, 0, 0],
          instanceCount: stationCount,
          radiusPixels: 1.7,
          color: [255, 255, 255, 255]
        })
      );
      if (o.holdOut && heldOutCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'rainfall-held-out-ring',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: heldOutIds,
            instanceCount: heldOutCount,
            radiusPixels: 5.4,
            color: dark ? [255, 255, 255, 255] : [20, 20, 20, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'rainfall-held-out',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: heldOutIds,
            values: valuesBuffer,
            valueFormat: 'float32',
            colormap: stationRamp,
            valueRange: stationRange,
            noDataColor: [0, 0, 0, 0],
            instanceCount: heldOutCount,
            radiusPixels: 3.6,
            color: [255, 255, 255, 255]
          })
        );
      }
    }
    return layers;
  }
}
