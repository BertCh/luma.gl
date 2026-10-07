// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTemporalReductionParameterValues,
  GPUTemporalReduction
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  DAYS_IN_YEAR,
  formatYearDay,
  loadMigrationTracks,
  MIGRATION_DATASET_ID,
  MIGRATION_SPECIES_COLORS,
  MONTH_START_DAYS,
  SECONDS_PER_DAY
} from './migration-shared';

/** Option state of the migration timing scene. */
export type MigrationTimingOptions = {
  species: 'all' | 'marsh' | 'montagu' | 'spoonbill';
  cellValue: 'share' | 'count' | 'speed';
  bucketDays: number;
  liftLow: boolean;
  probeLatitude: number;
  showProbe: boolean;
  showTracks: boolean;
  trackOpacity: number;
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  play: boolean;
  day: number;
  playSpeed: number;
  loop: boolean;
};

/** Latitude bands of 3 degrees from 6 N to 66 N. */
export const LATITUDE_START = 6;
export const BAND_DEGREES = 3;
export const BAND_COUNT = 20;
const BUCKET_COUNT = 128;
const SPECIES_COUNT = 3;
const CELL_COUNT = SPECIES_COUNT * BAND_COUNT;
const SLOT_COUNT = CELL_COUNT * BUCKET_COUNT;
const MATRIX_LENGTH = BAND_COUNT * BUCKET_COUNT;
/** The week-by-latitude chart is drawn on the map at sea west of Africa: longitudes -52 to -22. */
const CHART_ORIGIN_LONGITUDE = -52;
const CHART_WIDTH_DEGREES = 30;
const SPECIES_MASKS = {all: 7, marsh: 1, montagu: 2, spoonbill: 4} as const;
const MODE_INDEXES = {share: 0, count: 1, speed: 2} as const;
const SETTLE_MILLISECONDS = 200;
const OVERLAY_SEGMENTS = 24;
/** Latitude groups for the line chart: four bands (12 degrees) each, north to south. */
const GROUP_BANDS = 4;

/**
 * Migration timing: `GPUTemporalReduction` reduces every fix to one slot per (species, 3-degree
 * latitude band, time bucket): a count and the fastest ground speed. A kernel folds the species
 * you pick into a band-by-week matrix that a raster layer draws in map space (rows sit at their true
 * latitudes, west of Africa, with the week of the year along x), and the CPU turns the same counts
 * into the charts. A playback clock sweeps a cursor across the year.
 */
export async function createMigrationTiming(
  ctx: SceneContext<MigrationTimingOptions>
): Promise<SceneInstance<MigrationTimingOptions>> {
  const tracks = loadMigrationTracks(ctx.datasets.get(MIGRATION_DATASET_ID));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const resources = new SpatialAnalysisResources(device, 'timing');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const view = <Format extends 'float32' | 'uint32'>(
    graph: GPUCommandGraph<void>,
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);

  // ---- Per-fix inputs: cell (species and band), time, ground speed ------------------------------
  const cellIds = new Uint32Array(vertexCount);
  const speeds = new Float32Array(vertexCount);
  const R = Math.PI / 180;
  for (let track = 0; track < trackCount; track++) {
    for (let vertex = tracks.offsets[track]; vertex < tracks.offsets[track + 1]; vertex++) {
      const latitude = tracks.lngLat[vertex * 2 + 1];
      const band = Math.min(
        BAND_COUNT - 1,
        Math.max(0, Math.floor((latitude - LATITUDE_START) / BAND_DEGREES))
      );
      cellIds[vertex] = tracks.species[track] * BAND_COUNT + band;
      if (vertex > tracks.offsets[track]) {
        // Ground speed from consecutive fixes in km/h (haversine), computed once at load.
        const lat0 = tracks.lngLat[vertex * 2 - 1] * R;
        const lat1 = latitude * R;
        const h =
          Math.sin((lat1 - lat0) / 2) ** 2 +
          Math.cos(lat0) *
            Math.cos(lat1) *
            Math.sin(((tracks.lngLat[vertex * 2] - tracks.lngLat[vertex * 2 - 2]) * R) / 2) ** 2;
        const kilometers = 12742 * Math.asin(Math.sqrt(h));
        const hours = (tracks.timestamps[vertex] - tracks.timestamps[vertex - 1]) / 3600;
        speeds[vertex] = hours > 0 ? kilometers / hours : 0;
      }
    }
  }
  const cellIdsBuffer = resources.createBuffer('cell-ids', cellIds);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const speedsBuffer = resources.createBuffer('speeds', speeds);
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const speciesBuffer = resources.createBuffer('species', Uint32Array.from(tracks.species));
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);

  // ---- Temporal reduction graph --------------------------------------------------------------------
  const bucketParameters = resources.createParameterBuffer(
    'bucket',
    'float32',
    2,
    getGPUTemporalReductionParameterValues(0, 7 * SECONDS_PER_DAY)
  );
  const counts = resources.createBuffer('counts', SLOT_COUNT * 4);
  const minima = resources.createBuffer('min', SLOT_COUNT * 4);
  const maxima = resources.createBuffer('max', SLOT_COUNT * 4);
  const firsts = resources.createBuffer('first', SLOT_COUNT * 4);
  const lasts = resources.createBuffer('last', SLOT_COUNT * 4);
  const occupiedIds = resources.createBuffer('occupied-ids', SLOT_COUNT * 4);
  const occupiedCount = resources.createBuffer('occupied-count', 4);
  const occupiedOverflow = resources.createBuffer('occupied-overflow', 4);
  const temporalGraph = new GPUCommandGraph<void>(device, {id: 'timing-temporal'});
  temporalGraph.add(
    new GPUTemporalReduction({
      id: 'reduce-fixes',
      cellIds: view(temporalGraph, 'cell-ids', cellIdsBuffer, 'uint32', vertexCount),
      timestamps: view(temporalGraph, 'timestamps', timestampsBuffer, 'float32', vertexCount),
      values: view(temporalGraph, 'speeds', speedsBuffer, 'float32', vertexCount),
      parameters: bucketParameters.importToGraph(temporalGraph),
      cellCount: CELL_COUNT,
      bucketCount: BUCKET_COUNT,
      output: {
        counts: view(temporalGraph, 'counts', counts, 'uint32', SLOT_COUNT),
        min: view(temporalGraph, 'min', minima, 'float32', SLOT_COUNT),
        max: view(temporalGraph, 'max', maxima, 'float32', SLOT_COUNT),
        first: view(temporalGraph, 'first', firsts, 'float32', SLOT_COUNT),
        last: view(temporalGraph, 'last', lasts, 'float32', SLOT_COUNT),
        occupiedSlots: {
          ids: view(temporalGraph, 'occupied', occupiedIds, 'uint32', SLOT_COUNT),
          count: view(temporalGraph, 'occupied-count', occupiedCount, 'uint32', 1),
          overflow: view(temporalGraph, 'occupied-overflow', occupiedOverflow, 'uint32', 1)
        }
      }
    })
  );
  const temporalCompiled = resources.track(temporalGraph.compile());

  // ---- Display graph: the species you pick, folded into a band-by-bucket matrix --------------------
  const matrix = resources.createBuffer('matrix', MATRIX_LENGTH * 4);
  // [species mask, mode (0 share, 1 count, 2 speed), valid buckets, unused]
  const displayParameters = resources.createParameterBuffer('display', 'float32', 4);
  const displayGraph = new GPUCommandGraph<void>(device, {id: 'timing-display'});
  addKernelPass(displayGraph, {
    id: 'band-matrix',
    invocationCount: MATRIX_LENGTH,
    declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;
const BANDS: u32 = ${BAND_COUNT}u;
const SPECIES: u32 = ${SPECIES_COUNT}u;`,
    bindings: [
      {
        name: 'counts',
        view: view(displayGraph, 'counts', counts, 'uint32', SLOT_COUNT),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'fastest',
        view: view(displayGraph, 'max', maxima, 'float32', SLOT_COUNT),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'settings',
        view: displayParameters.importToGraph(displayGraph),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'matrix',
        view: view(displayGraph, 'matrix', matrix, 'float32', MATRIX_LENGTH),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let row = index / BUCKETS;
  let bucket = index % BUCKETS;
  let selected = u32(settings[settingsOffset]);
  let mode = u32(settings[settingsOffset + 1u]);
  let inRange = f32(bucket) < settings[settingsOffset + 2u];
  var inRow = 0.0;
  var inColumn = 0.0;
  var fastestHere = 0.0;
  for (var species = 0u; species < SPECIES; species++) {
    if (((selected >> species) & 1u) == 0u) {
      continue;
    }
    for (var band = 0u; band < BANDS; band++) {
      let slot = (species * BANDS + band) * BUCKETS + bucket;
      let slotCount = f32(counts[countsOffset + slot]);
      inColumn += slotCount;
      if (band == row) {
        inRow += slotCount;
        if (slotCount > 0.0) {
          fastestHere = max(fastestHere, fastest[fastestOffset + slot]);
        }
      }
    }
  }
  var result = nan;
  if (inRange && inRow > 0.0) {
    if (mode == 0u) {
      result = inRow / inColumn;
    } else if (mode == 1u) {
      result = inRow;
    } else {
      result = fastestHere;
    }
  }
  matrix[matrixOffset + index] = result;`
  });
  const displayCompiled = resources.track(displayGraph.compile());

  // ---- Overlay segments: frame, ticks, probe and the playback cursor -------------------------------
  const overlaySegments = resources.createBuffer('overlay-segments', OVERLAY_SEGMENTS * 16);

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let reductionDirty = true;
  let displayDirty = true;
  let settleStale = true;
  let lastChange = performance.now();
  let snapshot: {counts: Uint32Array; fastest: Float32Array; occupied: number} | null = null;
  let legendRange = 1;
  let profileBucket = -1;
  const clock = createPlaybackClock(
    ctx,
    {time: 'day', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, DAYS_IN_YEAR - 1], rate: 1, step: 1}
  );

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };
  const validBuckets = () => Math.ceil(DAYS_IN_YEAR / ctx.options.bucketDays);
  const bucketDegrees = () => (CHART_WIDTH_DEGREES * ctx.options.bucketDays) / DAYS_IN_YEAR;
  const bucketOfDay = (day: number) =>
    Math.min(validBuckets() - 1, Math.floor(day / ctx.options.bucketDays));

  function writeBucketParameters(): void {
    bucketParameters.write(
      getGPUTemporalReductionParameterValues(0, ctx.options.bucketDays * SECONDS_PER_DAY)
    );
    reductionDirty = true;
    displayDirty = true;
    profileBucket = -1;
    markChanged();
    writeOverlay();
  }

  function writeDisplayParameters(): void {
    displayParameters.write(
      Float32Array.of(
        SPECIES_MASKS[ctx.options.species],
        MODE_INDEXES[ctx.options.cellValue],
        validBuckets(),
        0
      )
    );
    displayDirty = true;
    profileBucket = -1;
    markChanged();
  }

  /** Map position of a (day, latitude) pair of the chart drawn at sea. */
  const chartX = (day: number) =>
    CHART_ORIGIN_LONGITUDE + (CHART_WIDTH_DEGREES * day) / DAYS_IN_YEAR;
  const LATITUDE_END = LATITUDE_START + BAND_COUNT * BAND_DEGREES;

  function writeOverlay(): void {
    const rows: number[] = [];
    const add = (x0: number, y0: number, x1: number, y1: number) => rows.push(x0, y0, x1, y1);
    const left = chartX(0);
    const right = chartX(DAYS_IN_YEAR);
    // Frame.
    add(left, LATITUDE_START, right, LATITUDE_START);
    add(right, LATITUDE_START, right, LATITUDE_END);
    add(right, LATITUDE_END, left, LATITUDE_END);
    add(left, LATITUDE_END, left, LATITUDE_START);
    // Latitude ticks every 12 degrees (the line chart's groups).
    for (let latitude = LATITUDE_START + 12; latitude < LATITUDE_END; latitude += 12) {
      add(left, latitude, right, latitude);
    }
    // Month ticks.
    for (let month = 1; month < 12; month++) {
      const x = chartX(MONTH_START_DAYS[month]);
      add(x, LATITUDE_START, x, LATITUDE_START + 2);
    }
    // Probe latitude across the whole map.
    if (ctx.options.showProbe) {
      add(-62, ctx.options.probeLatitude, 62, ctx.options.probeLatitude);
    } else {
      add(Number.NaN, Number.NaN, Number.NaN, Number.NaN);
    }
    // Cursor.
    const x = chartX(clock.time + ctx.options.bucketDays / 2);
    add(x, LATITUDE_START - 1.5, x, LATITUDE_END + 1.5);
    const data = new Float32Array(OVERLAY_SEGMENTS * 4).fill(Number.NaN);
    data.set(rows.slice(0, OVERLAY_SEGMENTS * 4));
    overlaySegments.write(data);
  }

  // ---- Summaries --------------------------------------------------------------------------------
  /** CPU copy of the display matrix for one bucket: `[row]` values or NaN. */
  function columnValues(bucket: number): Float64Array {
    const result = new Float64Array(BAND_COUNT).fill(Number.NaN);
    if (!snapshot) return result;
    const mask = SPECIES_MASKS[ctx.options.species];
    let total = 0;
    const perBand = new Float64Array(BAND_COUNT);
    for (let species = 0; species < SPECIES_COUNT; species++) {
      if (!((mask >> species) & 1)) continue;
      for (let band = 0; band < BAND_COUNT; band++) {
        const count = snapshot.counts[(species * BAND_COUNT + band) * BUCKET_COUNT + bucket];
        perBand[band] += count;
        total += count;
      }
    }
    for (let band = 0; band < BAND_COUNT; band++) {
      if (perBand[band] > 0) result[band] = total > 0 ? perBand[band] / total : Number.NaN;
    }
    return result;
  }

  function summarize(): void {
    if (!snapshot) return;
    const options = ctx.options;
    const buckets = validBuckets();
    const width = options.bucketDays;
    // Legend range: 99th percentile of the displayed values.
    const values: number[] = [];
    const mask = SPECIES_MASKS[options.species];
    for (let band = 0; band < BAND_COUNT; band++) {
      for (let bucket = 0; bucket < buckets; bucket++) {
        let count = 0;
        let fastest = 0;
        for (let species = 0; species < SPECIES_COUNT; species++) {
          if (!((mask >> species) & 1)) continue;
          const slot = (species * BAND_COUNT + band) * BUCKET_COUNT + bucket;
          count += snapshot.counts[slot];
          if (snapshot.counts[slot] > 0) fastest = Math.max(fastest, snapshot.fastest[slot]);
        }
        if (count === 0) continue;
        values.push(
          options.cellValue === 'count'
            ? count
            : options.cellValue === 'speed'
              ? fastest
              : columnValues(bucket)[band]
        );
      }
    }
    values.sort((a, b) => a - b);
    legendRange = values.length
      ? values[Math.min(values.length - 1, Math.floor(values.length * 0.99))]
      : 1;
    const scale = options.cellValue === 'share' ? 100 : 1;
    ctx.setLegendExtent('matrix', [0, legendRange * scale]);

    // Line chart: share of the fixes in each 12-degree latitude group, per bucket.
    const centers: number[] = [];
    const groups: number[][] = Array.from({length: BAND_COUNT / GROUP_BANDS}, () => []);
    for (let bucket = 0; bucket < buckets; bucket++) {
      const column = columnValues(bucket);
      if (column.every(Number.isNaN)) continue;
      centers.push(bucket * width + width / 2);
      for (let group = 0; group < groups.length; group++) {
        let sum = 0;
        for (let band = group * GROUP_BANDS; band < (group + 1) * GROUP_BANDS; band++) {
          if (Number.isFinite(column[band])) sum += column[band];
        }
        // Groups run south to north in the matrix.
        groups[group].push(sum * 100);
      }
    }
    const groupLabel = (group: number) =>
      `${LATITUDE_START + group * 12} to ${LATITUDE_START + (group + 1) * 12} N`;
    bandSeries = groups
      .map((y, group) => ({label: groupLabel(group), x: centers, y, color: group}))
      .reverse();
    setBandChart();
    ctx.setReadout(
      'reduction',
      `${formatCount(vertexCount)} fixes into ${formatCount(snapshot.occupied)} (species, band, ${width}-day) slots of ${formatCount(CELL_COUNT * buckets)}`
    );
    updateCursor(true);
    summarizeProbe();
    ctx.requestLayers();
  }

  function summarizeProbe(): void {
    if (!snapshot) return;
    const options = ctx.options;
    const band = Math.min(
      BAND_COUNT - 1,
      Math.max(0, Math.floor((options.probeLatitude - LATITUDE_START) / BAND_DEGREES))
    );
    const mask = SPECIES_MASKS[options.species];
    const weeks: number[] = [];
    for (let bucket = 0; bucket < validBuckets(); bucket++) {
      let count = 0;
      for (let species = 0; species < SPECIES_COUNT; species++) {
        if (!((mask >> species) & 1)) continue;
        count += snapshot.counts[(species * BAND_COUNT + band) * BUCKET_COUNT + bucket];
      }
      weeks.push(count);
    }
    const total = weeks.reduce((a, b) => a + b, 0);
    const lower = LATITUDE_START + band * BAND_DEGREES;
    if (total === 0) {
      ctx.setReadout('probe', `${lower} to ${lower + BAND_DEGREES} N: no fixes`);
      return;
    }
    // Busiest bucket in the second half of the year (southbound) and the first half (northbound).
    const width = options.bucketDays;
    const half = Math.floor(validBuckets() / 2);
    const peak = (from: number, to: number) => {
      let best = -1;
      let bestCount = 0;
      for (let bucket = from; bucket < to; bucket++) {
        if (weeks[bucket] > bestCount) {
          bestCount = weeks[bucket];
          best = bucket;
        }
      }
      return best < 0 ? 'none' : `${formatYearDay(best * width)} (${formatCount(bestCount)} fixes)`;
    };
    ctx.setReadout(
      'probe',
      `${lower} to ${lower + BAND_DEGREES} N, ${formatCount(total)} fixes. Busiest ${width}-day bucket from January: ${peak(0, half)}; from July: ${peak(half, validBuckets())}`
    );
  }

  function updateCursor(force: boolean): void {
    const options = ctx.options;
    const bucket = bucketOfDay(clock.time);
    writeOverlay();
    if (!snapshot || (!force && bucket === profileBucket)) return;
    profileBucket = bucket;
    const column = columnValues(bucket);
    const width = options.bucketDays;
    let weighted = 0;
    let total = 0;
    column.forEach((share, band) => {
      if (Number.isFinite(share)) {
        weighted += share * (LATITUDE_START + (band + 0.5) * BAND_DEGREES);
        total += share;
      }
    });
    const mask = SPECIES_MASKS[options.species];
    let fixes = 0;
    for (let species = 0; species < SPECIES_COUNT; species++) {
      if (!((mask >> species) & 1)) continue;
      for (let band = 0; band < BAND_COUNT; band++) {
        fixes += snapshot.counts[(species * BAND_COUNT + band) * BUCKET_COUNT + bucket];
      }
    }
    ctx.setReadout(
      'cursor',
      `${formatYearDay(bucket * width)} to ${formatYearDay(Math.min(365, (bucket + 1) * width - 1))}: ${formatCount(fixes)} fixes${total > 0 ? `, mean latitude ${(weighted / total).toFixed(1)} N` : ''}`
    );
    const labels = Array.from(
      {length: BAND_COUNT},
      (_, band) => `${LATITUDE_START + band * BAND_DEGREES}`
    );
    let top = 0;
    column.forEach((share, band) => {
      if (Number.isFinite(share) && share > (Number.isFinite(column[top]) ? column[top] : -1))
        top = band;
    });
    ctx.setChart('profileChart', {
      kind: 'bars',
      values: Array.from(column, share => (Number.isFinite(share) ? share * 100 : 0)),
      labels,
      highlight: total > 0 ? [top] : [],
      height: 110,
      yLabel: '% of fixes',
      formatY: value => `${Math.round(value)}`,
      description:
        'Where the selected species are at the cursor: the share of fixes in each 3-degree latitude band (labelled by its southern edge) in the time bucket of the cursor.'
    });
    setBandChart();
  }

  let bandSeries: {label: string; x: number[]; y: number[]; color: number}[] = [];
  function setBandChart(): void {
    if (!bandSeries.length) return;
    ctx.setChart('bandChart', {
      kind: 'line',
      series: bandSeries,
      xDomain: [0, DAYS_IN_YEAR],
      yDomain: [0, 100],
      markers: [{x: clock.time, label: 'now'}],
      xLabel: 'day of the year',
      yLabel: '% of fixes',
      height: 140,
      formatX: value => formatYearDay(Math.min(365, value)),
      formatY: value => `${Math.round(value)}`,
      description:
        "For each time bucket, the share of the selected species' fixes in each 12-degree latitude group. The groups sum to 100 percent: a hand-over from one line to the next is the population moving through."
    });
  }

  const reader = new SummaryReader(
    resources,
    'timing-summary',
    [
      {buffer: counts, size: SLOT_COUNT * 4},
      {buffer: maxima, size: SLOT_COUNT * 4},
      {buffer: occupiedCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      snapshot = {
        counts: new Uint32Array(bytes, 0, SLOT_COUNT).slice(),
        fastest: new Float32Array(bytes, SLOT_COUNT * 4, SLOT_COUNT).slice(),
        occupied: new Uint32Array(bytes, SLOT_COUNT * 8, 1)[0]
      };
      summarize();
    }
  );

  writeBucketParameters();
  writeDisplayParameters();
  ctx.setReadout(
    'tracks',
    `${trackCount} animal-years, ${formatCount(vertexCount)} fixes, ${tracks.individualIds.length} birds`
  );

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [temporalCompiled, displayCompiled],

    setOption(id) {
      switch (id) {
        case 'bucketDays':
          writeBucketParameters();
          writeDisplayParameters();
          ctx.requestLayers();
          break;
        case 'species':
        case 'cellValue':
          writeDisplayParameters();
          ctx.requestLayers();
          break;
        case 'probeLatitude':
        case 'showProbe':
          writeOverlay();
          summarizeProbe();
          break;
        case 'play':
        case 'day':
        case 'playSpeed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      clock.advance(frame);
      if (clock.moved) updateCursor(false);
      if (reductionDirty) {
        temporalCompiled.encode(commandEncoder, {parameters: undefined});
        reductionDirty = false;
      }
      if (displayDirty) {
        displayCompiled.encode(commandEncoder, {parameters: undefined});
        displayDirty = false;
        settleStale = true;
      }
      if (settleStale && performance.now() - lastChange > SETTLE_MILLISECONDS) {
        reader.markStale();
        settleStale = false;
      }
      reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showTracks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'timing-tracks',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            values: speciesBuffer,
            valueFormat: 'uint32',
            valueIndices: segmentTracksBuffer,
            colormap: 'category',
            palette: MIGRATION_SPECIES_COLORS,
            widthPixels: 1,
            opacity: options.trackOpacity
          })
        );
      }
      const columnWidth = bucketDegrees();
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: 'timing-matrix',
          ...drawProps,
          gridSize: [BUCKET_COUNT, BAND_COUNT],
          bounds: [
            CHART_ORIGIN_LONGITUDE,
            LATITUDE_START,
            CHART_ORIGIN_LONGITUDE + BUCKET_COUNT * columnWidth,
            LATITUDE_END
          ],
          tessellation: 24,
          values: matrix,
          valueFormat: 'float32',
          colormap: options.ramp,
          valueRange: [0, legendRange],
          sqrtScale: options.liftLow,
          noDataColor: dark ? [255, 255, 255, 14] : [0, 0, 0, 14],
          opacity: 0.95
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'timing-overlay',
          ...drawProps,
          segments: overlaySegments,
          instanceCount: OVERLAY_SEGMENTS,
          widthPixels: 1.4,
          color: dark ? [235, 238, 245, 200] : [30, 36, 50, 200]
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
