// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection,
  GPU_CHANGE_DETECTION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-raster';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {createSeededRandom} from '../../engine/projection';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {computeTrendReference} from './b16-stats';
import {
  createGraphViewer,
  createUtmFrame,
  formatFixed,
  formatPercent,
  UtmRasterLayer
} from './b16-common';
import {fetchBytes, fetchJson, getDataFileUrl} from '../../data/loaders';

/** Option state of the vegetation-trends scene. */
export type VegetationTrendsOptions = {
  layer:
    | 'significance'
    | 'senSlope'
    | 'mannKendallZ'
    | 'tStatistic'
    | 'tPValue'
    | 'mannKendallP'
    | 'difference'
    | 'percentChange'
    | 'logRatio'
    | 'ndviBefore'
    | 'ndviAfter';
  ramp: RampName;
  opacity: number;
  colorScale: number;
  significanceSource: 't-test' | 'mann-kendall';
  alpha: number;
  splitSlice: number;
  beforeSlice: number;
  afterSlice: number;
  dateWindow: readonly [number, number];
  minimumValid: number;
  epsilon: number;
};

const SETTLE_MILLISECONDS = 350;
const SPARK = ['.', ':', '-', '=', '+', '*', '#', '@'];
const CHECK_PIXELS = 400;

type OutputName =
  | 'difference'
  | 'logRatio'
  | 'percentChange'
  | 'tStatistic'
  | 'tDegreesOfFreedom'
  | 'tPValue'
  | 'senSlope'
  | 'mannKendallZ'
  | 'mannKendallP';

const FLOAT_OUTPUTS: readonly OutputName[] = [
  'difference',
  'logRatio',
  'percentChange',
  'tStatistic',
  'tDegreesOfFreedom',
  'tPValue',
  'senSlope',
  'mannKendallZ',
  'mannKendallP'
];

/** The `ndvi-timeseries` dataset as float32 NDVI per cell and date. */
type NdviStack = {
  width: number;
  height: number;
  cellCount: number;
  sliceCount: number;
  dates: string[];
  meanNdvi: number[];
  /** Cell-major: `(cell * sliceCount + slice)`; NaN where the pixel was cloud or no data. */
  cellMajor: Float32Array;
};

const DATASET_ID = 'ndvi-timeseries';

type NdviManifest = {
  bbox: [number, number, number, number];
  raster: {
    width: number;
    height: number;
    depth: number;
    boundsProjected: [number, number, number, number];
    cellSizeM: number;
  };
  properties: {dates: string[]; scenes: {meanNDVI: number; validFraction: number}[]};
};

/**
 * Loads the stack with the scene's own fetches: the shared catalog decodes every `manifest.raster`
 * file as a PNG, which fails for this dataset's raw `uint8-bin` (see the handoff).
 */
async function loadNdviStack(signal: AbortSignal): Promise<NdviStack & {manifest: NdviManifest}> {
  const manifest = await fetchJson<NdviManifest>(
    getDataFileUrl(DATASET_ID, 'manifest.json'),
    signal
  );
  const bytes = new Uint8Array(await fetchBytes(getDataFileUrl(DATASET_ID, 'ndvi.bin'), signal));
  const {width, height, depth: sliceCount} = manifest.raster;
  const cellCount = width * height;
  const cellMajor = new Float32Array(cellCount * sliceCount);
  for (let slice = 0; slice < sliceCount; slice++) {
    const base = slice * cellCount;
    for (let cell = 0; cell < cellCount; cell++) {
      const value = bytes[base + cell];
      cellMajor[cell * sliceCount + slice] = value === 0 ? Number.NaN : value / 127.5 - 1;
    }
  }
  return {
    manifest,
    width,
    height,
    cellCount,
    sliceCount,
    dates: manifest.properties.dates,
    meanNdvi: manifest.properties.scenes.map(scene => scene.meanNDVI),
    cellMajor
  };
}

/**
 * Per-pixel trends of 16 summers of NDVI around the Dixie Fire. `GPUChangeDetection` is compiled
 * twice (Welch t-test and Mann-Kendall as the significance source, which is compile-time) and
 * every other control is a parameter write or a small buffer write: the date window rewrites the
 * stack and the cell mask, the before and after dates and alpha go into the five-float parameter
 * buffer. Results are read back once the controls settle for tooltips and the CPU check.
 */
export async function createVegetationTrends(
  ctx: SceneContext<VegetationTrendsOptions>
): Promise<SceneInstance<VegetationTrendsOptions>> {
  const {device} = ctx;
  ctx.setStatus('Loading the NDVI stack');
  const ndvi = await loadNdviStack(ctx.signal);
  ctx.signal.throwIfAborted();
  const {width, height, cellCount, sliceCount, manifest} = ndvi;
  const frame = createUtmFrame(
    [(manifest.bbox[0] + manifest.bbox[2]) / 2, (manifest.bbox[1] + manifest.bbox[3]) / 2],
    {
      boundsProjected: manifest.raster.boundsProjected,
      cellSizeM: manifest.raster.cellSizeM,
      width,
      height
    }
  );
  const resources = new SpatialAnalysisResources(device, 'trend');

  // The stack the contributor reads: the date window rewrites it (dates outside become NaN).
  const stack = new Float32Array(ndvi.cellMajor);
  const stackBuffer = resources.createBuffer('slices', stack);
  const maskBuffer = resources.createBuffer('mask', new Uint32Array(cellCount).fill(1));
  const parameters = resources.createParameterBuffer(
    'parameters',
    'float32',
    GPU_CHANGE_DETECTION_PARAMETER_LENGTH
  );
  const outputs = {} as Record<OutputName, ReturnType<typeof resources.createBuffer>>;
  for (const name of FLOAT_OUTPUTS) outputs[name] = resources.createBuffer(name, cellCount * 4);
  const mannKendallS = resources.createBuffer('mann-kendall-s', cellCount * 4);
  const significanceTTest = resources.createBuffer('significance-t', cellCount * 4);
  const significanceMannKendall = resources.createBuffer('significance-mk', cellCount * 4);
  const slice = {
    before: resources.createBuffer('ndvi-before', cellCount * 4),
    after: resources.createBuffer('ndvi-after', cellCount * 4)
  };
  const markerBuffer = resources.createBuffer('marker', new Float32Array([0, 0]));

  // Two graphs: the t-test one carries every statistic; the second only switches the significance
  // source (compile-time) to Mann-Kendall.
  const buildGraph = (source: 't-test' | 'mann-kendall'): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: `trend-${source}`});
    const view = createGraphViewer(graph);
    const f = (
      name: string,
      buffer: ReturnType<typeof resources.createBuffer>,
      length = cellCount
    ) => view(name, buffer, 'float32', length);
    graph.add(
      new GPUChangeDetection({
        id: `change-${source}`,
        slices: f('slices', stackBuffer, cellCount * sliceCount),
        mask: view('mask', maskBuffer, 'uint32', cellCount),
        parameters: parameters.importToGraph(graph),
        cellCount,
        sliceCount,
        significanceSource: source,
        output:
          source === 't-test'
            ? {
                difference: f('difference', outputs.difference),
                logRatio: f('log-ratio', outputs.logRatio),
                percentChange: f('percent-change', outputs.percentChange),
                tStatistic: f('t-statistic', outputs.tStatistic),
                tDegreesOfFreedom: f('t-df', outputs.tDegreesOfFreedom),
                tPValue: f('t-p', outputs.tPValue),
                senSlope: f('sen-slope', outputs.senSlope),
                mannKendallS: view('mk-s', mannKendallS, 'sint32', cellCount),
                mannKendallZ: f('mk-z', outputs.mannKendallZ),
                mannKendallP: f('mk-p', outputs.mannKendallP),
                significance: view('significance', significanceTTest, 'uint32', cellCount)
              }
            : {significance: view('significance', significanceMannKendall, 'uint32', cellCount)}
      })
    );
    return resources.track(graph.compile());
  };
  const compiled = {'t-test': buildGraph('t-test'), 'mann-kendall': buildGraph('mann-kendall')};

  // --- State -----------------------------------------------------------------------------------
  let destroyed = false;
  let dirty = true;
  let lastChange = performance.now();
  let readStale = true;
  let appliedWindow = '';
  let appliedMinimum = -1;
  let appliedSlices = '';
  let selected = -1;
  const readArrays: Partial<
    Record<OutputName | 'significance' | 'mannKendallS', Float32Array | Uint32Array | Int32Array>
  > = {};
  let readSnapshot: {split: number; window: string} | null = null;

  const markDirty = () => {
    dirty = true;
    lastChange = performance.now();
  };

  const countValid = new Uint8Array(cellCount);

  /** Rewrites the stack for the date window and the mask for the minimum valid dates. */
  const applyWindow = (state: VegetationTrendsOptions) => {
    const [first, last] = [Math.min(...state.dateWindow), Math.max(...state.dateWindow)];
    const key = `${first}-${last}`;
    if (key !== appliedWindow) {
      appliedWindow = key;
      for (let cell = 0; cell < cellCount; cell++) {
        const base = cell * sliceCount;
        let valid = 0;
        for (let t = 0; t < sliceCount; t++) {
          const inside = t >= first && t <= last;
          const value = inside ? ndvi.cellMajor[base + t] : Number.NaN;
          stack[base + t] = value;
          if (value === value) valid++;
        }
        countValid[cell] = valid;
      }
      stackBuffer.write(stack);
      appliedMinimum = -1;
    }
    if (state.minimumValid !== appliedMinimum) {
      appliedMinimum = state.minimumValid;
      const mask = new Uint32Array(cellCount);
      let analysed = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        if (countValid[cell] >= state.minimumValid) {
          mask[cell] = 1;
          analysed++;
        }
      }
      maskBuffer.write(mask);
      ctx.setReadout(
        'analysed',
        `${formatCount(analysed)} of ${formatCount(cellCount)} cells (at least ${state.minimumValid} valid dates)`
      );
    }
    const sliceKey = `${state.beforeSlice}-${state.afterSlice}-${key}`;
    if (sliceKey !== appliedSlices) {
      appliedSlices = sliceKey;
      for (const [buffer, index] of [
        [slice.before, state.beforeSlice],
        [slice.after, state.afterSlice]
      ] as const) {
        const values = new Float32Array(cellCount);
        for (let cell = 0; cell < cellCount; cell++)
          values[cell] = stack[cell * sliceCount + index];
        buffer.write(values);
      }
    }
  };

  const writeParameters = (state: VegetationTrendsOptions) => {
    parameters.write(
      getGPUChangeDetectionParameterValues({
        beforeSlice: state.beforeSlice,
        afterSlice: state.afterSlice,
        epsilon: state.epsilon,
        alpha: state.alpha,
        splitSlice: Math.min(Math.max(state.splitSlice, 1), sliceCount - 1)
      })
    );
  };

  // --- Readback --------------------------------------------------------------------------------
  const readNames = [
    'senSlope',
    'mannKendallZ',
    'mannKendallP',
    'tStatistic',
    'tPValue',
    'difference'
  ] as const;
  const reader = new SummaryReader(
    resources,
    'trend-read',
    [
      ...readNames.map(name => ({buffer: outputs[name], size: cellCount * 4})),
      {buffer: mannKendallS, size: cellCount * 4},
      {buffer: significanceTTest, size: cellCount * 4},
      {buffer: significanceMannKendall, size: cellCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      let offset = 0;
      for (const name of readNames) {
        readArrays[name] = new Float32Array(bytes, offset, cellCount);
        offset += cellCount * 4;
      }
      readArrays.mannKendallS = new Int32Array(bytes, offset, cellCount);
      offset += cellCount * 4;
      readArrays.significance = new Uint32Array(bytes, offset, cellCount);
      offset += cellCount * 4;
      (readArrays as Record<string, unknown>).significanceMannKendall = new Uint32Array(
        bytes,
        offset,
        cellCount
      );
      readSnapshot = {split: ctx.options.splitSlice, window: appliedWindow};
      publishReadouts();
    }
  );

  const getSeries = (cell: number): Float32Array =>
    stack.subarray(cell * sliceCount, (cell + 1) * sliceCount);

  const sparkline = (series: ArrayLike<number>): string =>
    Array.from(series, value =>
      Number.isFinite(value)
        ? SPARK[
            Math.max(
              0,
              Math.min(SPARK.length - 1, Math.floor(((value + 0.1) / 1.0) * SPARK.length))
            )
          ]
        : '·'
    ).join('');

  function publishReadouts(): void {
    const state = ctx.options;
    const significance = (
      state.significanceSource === 'mann-kendall'
        ? (readArrays as Record<string, unknown>).significanceMannKendall
        : readArrays.significance
    ) as Uint32Array | undefined;
    if (significance) {
      let increase = 0;
      let decrease = 0;
      let analysed = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        if (maskValue(cell)) analysed++;
        if (significance[cell] === 1) increase++;
        else if (significance[cell] === 2) decrease++;
      }
      ctx.setReadout(
        'shares',
        `${formatPercent(analysed ? decrease / analysed : 0, 1)} declining · ${formatPercent(analysed ? increase / analysed : 0, 1)} increasing · ${formatPercent(analysed ? 1 - (increase + decrease) / analysed : 0, 1)} no significant trend (alpha ${state.alpha})`
      );
      ctx.setReadout(
        'falsePositives',
        `about ${formatPercent(state.alpha, 1)} of cells (${formatCount(Math.round(analysed * state.alpha))}) would pass even for pure noise: alpha is not corrected for the number of cells`
      );
    }
    runParityCheck();
    publishPixel();
  }

  const maskValue = (cell: number) => countValid[cell] >= ctx.options.minimumValid;

  function runParityCheck(): void {
    const sen = readArrays.senSlope as Float32Array | undefined;
    const z = readArrays.mannKendallZ as Float32Array | undefined;
    const s = readArrays.mannKendallS as Int32Array | undefined;
    const t = readArrays.tStatistic as Float32Array | undefined;
    if (!sen || !z || !s || !t) return;
    const random = createSeededRandom(2021);
    let compared = 0;
    let mismatchS = 0;
    let maximumZ = 0;
    let maximumSen = 0;
    let maximumT = 0;
    const split = readSnapshot?.split ?? ctx.options.splitSlice;
    for (let index = 0; index < CHECK_PIXELS; index++) {
      const cell = Math.floor(random() * cellCount);
      if (!maskValue(cell)) continue;
      const reference = computeTrendReference(getSeries(cell), split);
      compared++;
      if (reference.mannKendallS !== s[cell]) mismatchS++;
      if (Number.isFinite(reference.mannKendallZ) && Number.isFinite(z[cell])) {
        maximumZ = Math.max(maximumZ, Math.abs(reference.mannKendallZ - z[cell]));
      }
      if (Number.isFinite(reference.senSlope) && Number.isFinite(sen[cell])) {
        maximumSen = Math.max(maximumSen, Math.abs(reference.senSlope - sen[cell]));
      }
      if (Number.isFinite(reference.tStatistic) && Number.isFinite(t[cell])) {
        maximumT = Math.max(maximumT, Math.abs(reference.tStatistic - t[cell]));
      }
    }
    ctx.setReadout(
      'parity',
      `${compared} random pixels: Mann-Kendall S differs in ${mismatchS}; largest |dZ| ${maximumZ.toExponential(1)}, |dSen| ${maximumSen.toExponential(1)}, |dt| ${maximumT.toExponential(1)}`
    );
  }

  function publishPixel(): void {
    if (selected < 0) {
      ctx.setReadout('pixel', 'click the map');
      ctx.setReadout('pixelSeries', 'click the map');
      return;
    }
    const series = getSeries(selected);
    const reference = computeTrendReference(series, readSnapshot?.split ?? ctx.options.splitSlice);
    const column = selected % width;
    const row = Math.floor(selected / width);
    const [longitude, latitude] = frame.cellToLngLat(column, row);
    ctx.setReadout(
      'pixel',
      `column ${column}, row ${row} (${latitude.toFixed(4)}, ${longitude.toFixed(4)})`
    );
    const gpu = (name: 'senSlope' | 'mannKendallZ' | 'mannKendallP' | 'tStatistic' | 'tPValue') =>
      (readArrays[name] as Float32Array | undefined)?.[selected] ?? Number.NaN;
    ctx.setReadout(
      'pixelSeries',
      `${sparkline(series)}  ·  Sen ${formatFixed(gpu('senSlope'), 4)} (CPU ${formatFixed(reference.senSlope, 4)}) · MK Z ${formatFixed(gpu('mannKendallZ'), 2)} (CPU ${formatFixed(reference.mannKendallZ, 2)}) · t ${formatFixed(gpu('tStatistic'), 2)} (CPU ${formatFixed(reference.tStatistic, 2)}) · p(t) ${formatFixed(gpu('tPValue'), 4)} (CPU ${formatFixed(reference.tPValue, 4)})`
    );
  }

  ctx.setReadout(
    'meanSeries',
    sparkline(ndvi.meanNdvi) +
      `  (${ndvi.meanNdvi[0].toFixed(2)} in ${ndvi.dates[0].slice(0, 4)}, ${Math.min(...ndvi.meanNdvi).toFixed(2)} lowest, ${ndvi.meanNdvi[ndvi.meanNdvi.length - 1].toFixed(2)} in ${ndvi.dates[ndvi.dates.length - 1].slice(0, 4)})`
  );
  const validFractions = manifest.properties.scenes.map(scene => scene.validFraction);
  ctx.setReadout(
    'acquisitionContract',
    `${sliceCount} irregular summer acquisitions · ${ndvi.dates[0]} to ${ndvi.dates[sliceCount - 1]} · ${(Math.min(...validFractions) * 100).toFixed(1)}–${(Math.max(...validFractions) * 100).toFixed(1)}% valid pixels per scene`
  );
  ctx.setReadout('pixel', 'click the map');
  ctx.setReadout('pixelSeries', 'click the map');
  ctx.setStatus('');

  const getCell = (coordinate: readonly [number, number] | null): number => {
    if (!coordinate) return -1;
    const [column, row] = frame.lngLatToCell(coordinate[0], coordinate[1]);
    const c = Math.floor(column);
    const r = Math.floor(row);
    if (c < 0 || r < 0 || c >= width || r >= height) return -1;
    return r * width + c;
  };

  // --- Instance --------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [compiled['t-test'], compiled['mann-kendall']],

    setOption() {
      markDirty();
      readStale = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frameInfo) {
      const state = ctx.options;
      if (dirty || frameInfo.frameIndex < 2) {
        applyWindow(state);
        writeParameters(state);
        compiled['t-test'].encode(commandEncoder, {parameters: undefined});
        if (state.significanceSource === 'mann-kendall') {
          compiled['mann-kendall'].encode(commandEncoder, {parameters: undefined});
        }
        dirty = false;
        readStale = true;
      }
      if (readStale && performance.now() - lastChange > SETTLE_MILLISECONDS && !reader.isPending) {
        reader.request(commandEncoder);
        readStale = false;
      }
      reader.flush(commandEncoder);
    },

    getLayers() {
      const state = ctx.options;
      const base = {
        id: `trend-${state.layer}`,
        frame,
        opacity: state.opacity,
        noDataColor: [0, 0, 0, 0] as const
      };
      let layer: Layer;
      if (state.layer === 'significance') {
        layer = new UtmRasterLayer({
          ...base,
          values:
            state.significanceSource === 'mann-kendall'
              ? significanceMannKendall
              : significanceTTest,
          valueFormat: 'uint32',
          colormap: 'category',
          palette: [
            [140, 140, 140, 70],
            [38, 166, 91, 235],
            [214, 69, 65, 235]
          ]
        });
      } else if (state.layer === 'ndviBefore' || state.layer === 'ndviAfter') {
        layer = new UtmRasterLayer({
          ...base,
          values: state.layer === 'ndviBefore' ? slice.before : slice.after,
          valueFormat: 'float32',
          colormap: state.ramp,
          valueRange: [-0.1, 0.9]
        });
      } else if (state.layer === 'tPValue' || state.layer === 'mannKendallP') {
        layer = new UtmRasterLayer({
          ...base,
          values: outputs[state.layer],
          valueFormat: 'float32',
          colormap: state.ramp,
          valueRange: [0, 0.2 * state.colorScale]
        });
      } else {
        const range = getSignedRange(state.layer) * state.colorScale;
        layer = new UtmRasterLayer({
          ...base,
          values: outputs[state.layer],
          valueFormat: 'float32',
          colormap: 'diverging',
          valueRange: [-range, range]
        });
      }
      const layers: Layer[] = [layer];
      if (selected >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'trend-marker',
            coordinateOrigin: [frame.origin[0], frame.origin[1], 0],
            positions: markerBuffer,
            instanceCount: 1,
            radiusPixels: 7,
            color: ctx.theme() === 'dark' ? [255, 255, 255, 255] : [20, 20, 20, 255]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return null;
      const series = getSeries(cell);
      const lines = [sparkline(series)];
      const sen = readArrays.senSlope as Float32Array | undefined;
      const z = readArrays.mannKendallZ as Float32Array | undefined;
      if (sen && z) {
        lines.push(
          `Sen ${formatFixed(sen[cell], 4)} NDVI / date · MK Z ${formatFixed(z[cell], 2)}`
        );
      }
      const before = series[ctx.options.beforeSlice];
      const after = series[ctx.options.afterSlice];
      lines.push(
        `NDVI ${formatFixed(before, 2)} (${ndvi.dates[ctx.options.beforeSlice]}) → ${formatFixed(after, 2)} (${ndvi.dates[ctx.options.afterSlice]})`
      );
      return lines.join('\n');
    },

    onClick(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return false;
      selected = cell;
      const [meterX, meterY] = frame.cellToMeters(cell % width, Math.floor(cell / width));
      markerBuffer.write(Float32Array.of(meterX, meterY));
      ctx.requestLayers();
      publishPixel();
      return true;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}

/** Half-width of the symmetric color range of the signed layers at colorScale 1. */
function getSignedRange(layer: VegetationTrendsOptions['layer']): number {
  switch (layer) {
    case 'senSlope':
      return 0.04;
    case 'mannKendallZ':
      return 4;
    case 'tStatistic':
      return 8;
    case 'difference':
      return 0.6;
    case 'percentChange':
      return 100;
    case 'logRatio':
      return 1;
    default:
      return 1;
  }
}
