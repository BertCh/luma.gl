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
import {getClassIndexOf, getClassTableLayerProps} from '../../cartography/class-table';
import {formatCount, formatPercent} from '../../cartography/live-text';
import type {ClassTable, MapAnnotation, MapHighlight} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {
  DAYS_IN_YEAR,
  loadMigrationTracks,
  MIGRATION_DATASET_ID,
  SECONDS_PER_DAY
} from './migration-shared';
import {
  buildBandChart,
  buildFixesChart,
  buildOccupancyChart,
  buildProfileChart
} from './migration-timing-charts';
import {
  CHART_WEST,
  dayToLongitude,
  formatBucketRange,
  getAxisAnnotations,
  getBucketColumnRows,
  getBucketEdgeRows,
  getColumnDegrees,
  getCursorRows,
  getGroupColors,
  getGroupKeyRows,
  getHairlineRows,
  getMatrixBounds,
  getOverlayInks,
  getPanelBounds,
  getPanelFill,
  getPanelFrameRows,
  getProbeAnnotation,
  getProbeRowRows,
  getTimingTable,
  getTraceRows,
  isInsideMatrix,
  longitudeToDay,
  type SegmentRows,
  type TimingMode
} from './migration-timing-panel';
import {
  BAND_DEGREES,
  BUCKET_COUNT,
  BUCKET_DAYS_RANGE,
  findCrossings,
  getBucketOfDay,
  getCell,
  getClassOf,
  getColumn,
  getCountBreaks,
  getFastestCell,
  getFixesPerBucket,
  getGroupLabel,
  getGroupShares,
  getMedianLatitudes,
  getOccupiedCells,
  getRowLatitude,
  getRowOfLatitude,
  getValidBuckets,
  LATITUDE_START,
  ROWS_PER_GROUP,
  SPECIES_COUNT,
  SPECIES_MASKS,
  type SpeciesChoice,
  THIN_FIXES,
  type TimingCube
} from './migration-timing-stats';
import {formatFoldedDay, getSpeciesInk, SPECIES_NAMES} from './movement-style';

/** Option state of the migration timing scene. */
export type MigrationTimingOptions = {
  species: SpeciesChoice;
  cellValue: TimingMode;
  bucketDays: number;
  probeLatitude: number;
  showProbe: boolean;
  showTracks: boolean;
  trackOpacity: number;
  showBucketEdges: boolean;
  showGroups: boolean;
  showTrace: boolean;
  showCrossings: boolean;
  showFixes: boolean;
  play: boolean;
  day: number;
  playSpeed: number;
  loop: boolean;
};

const MODE_INDEXES: Readonly<Record<TimingMode, number>> = {share: 0, count: 1, speed: 2};
const SETTLE_MILLISECONDS = 200;
/** Widths of the bucket sweep that fills the occupancy curve: every width the slider offers. */
const SWEEP_WIDTHS = Array.from(
  {length: BUCKET_DAYS_RANGE[1] - BUCKET_DAYS_RANGE[0] + 1},
  (_, index) => BUCKET_DAYS_RANGE[0] + index
);
/** The widest bucket holds the most fixes; this many dots are enough for any bucket. */
const WIDEST_BUCKET_DAYS = BUCKET_DAYS_RANGE[1];
/** Most rows a latitude axis can hold: 20 bands from 6 N. */
const MAXIMUM_ROWS = 20;

/** A buffer of line segments `x0, y0, x1, y1` with a count, so only real segments are drawn. */
type SegmentSet = {buffer: Buffer; count: number; write: (rows: SegmentRows) => void};

/**
 * Migration timing: `GPUTemporalReduction` reduces every GPS fix to one slot per (species,
 * 3-degree latitude band, time bucket): a count and the fastest step ending there. A display
 * kernel folds the chosen species into a band-by-bucket matrix that a raster layer draws on a
 * paper panel at sea west of Africa, with the bucket along longitude and the band at its true
 * latitude (a Hovmöller diagram). The CPU turns the same read-back cube into the charts, the
 * median-latitude trace and the crossing dates. A playback clock sweeps a cursor across the year.
 */
export async function createMigrationTiming(
  ctx: SceneContext<MigrationTimingOptions>
): Promise<SceneInstance<MigrationTimingOptions>> {
  const dataset = ctx.datasets.get(MIGRATION_DATASET_ID);
  const tracks = loadMigrationTracks(dataset);
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

  // The rows the data needs: the northernmost fix sets the top of the matrix (no empty rows).
  const rowCount = Math.min(
    MAXIMUM_ROWS,
    Math.max(1, Math.ceil((dataset.manifest.bbox[3] - LATITUDE_START) / BAND_DEGREES))
  );
  const cellCount = SPECIES_COUNT * rowCount;
  const slotCount = cellCount * BUCKET_COUNT;
  const matrixLength = rowCount * BUCKET_COUNT;

  // ---- Per-fix inputs: cell (species and band), ground speed, species, day ------------------------
  const cellIds = new Uint32Array(vertexCount);
  const speeds = new Float32Array(vertexCount);
  const fixSpecies = new Uint32Array(vertexCount);
  const fixesPerDay = Array.from({length: SPECIES_COUNT}, () => new Float64Array(DAYS_IN_YEAR));
  const intervals: number[] = [];
  const radians = Math.PI / 180;
  for (let track = 0; track < trackCount; track++) {
    const species = tracks.species[track];
    for (let vertex = tracks.offsets[track]; vertex < tracks.offsets[track + 1]; vertex++) {
      const latitude = tracks.lngLat[vertex * 2 + 1];
      const band = Math.min(
        rowCount - 1,
        Math.max(0, Math.floor((latitude - LATITUDE_START) / BAND_DEGREES))
      );
      cellIds[vertex] = species * rowCount + band;
      fixSpecies[vertex] = species;
      const day = Math.min(
        DAYS_IN_YEAR - 1,
        Math.max(0, Math.floor(tracks.timestamps[vertex] / SECONDS_PER_DAY))
      );
      fixesPerDay[species][day]++;
      if (vertex > tracks.offsets[track]) {
        // Ground speed of the step that ends at this fix, in km/h (haversine), computed once at load.
        const lat0 = tracks.lngLat[vertex * 2 - 1] * radians;
        const lat1 = latitude * radians;
        const h =
          Math.sin((lat1 - lat0) / 2) ** 2 +
          Math.cos(lat0) *
            Math.cos(lat1) *
            Math.sin(((tracks.lngLat[vertex * 2] - tracks.lngLat[vertex * 2 - 2]) * radians) / 2) **
              2;
        const kilometers = 12742 * Math.asin(Math.sqrt(h));
        const seconds = tracks.timestamps[vertex] - tracks.timestamps[vertex - 1];
        speeds[vertex] = seconds > 0 ? (kilometers * 3600) / seconds : 0;
        intervals.push(seconds);
      }
    }
  }
  intervals.sort((a, b) => a - b);
  const medianInterval = intervals[Math.floor(intervals.length / 2)] ?? 0;
  const shortestInterval = intervals[0] ?? 0;
  const cellIdsBuffer = resources.createBuffer('cell-ids', cellIds);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const speedsBuffer = resources.createBuffer('speeds', speeds);
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const speciesBuffer = resources.createBuffer('species', Uint32Array.from(tracks.species));
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);

  // ---- Temporal reduction graph ---------------------------------------------------------------------
  const bucketParameters = resources.createParameterBuffer(
    'bucket',
    'float32',
    2,
    getGPUTemporalReductionParameterValues(0, 7 * SECONDS_PER_DAY)
  );
  const counts = resources.createBuffer('counts', slotCount * 4);
  const minima = resources.createBuffer('min', slotCount * 4);
  const maxima = resources.createBuffer('max', slotCount * 4);
  const firsts = resources.createBuffer('first', slotCount * 4);
  const lasts = resources.createBuffer('last', slotCount * 4);
  const occupiedIds = resources.createBuffer('occupied-ids', slotCount * 4);
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
      cellCount,
      bucketCount: BUCKET_COUNT,
      output: {
        counts: view(temporalGraph, 'counts', counts, 'uint32', slotCount),
        min: view(temporalGraph, 'min', minima, 'float32', slotCount),
        max: view(temporalGraph, 'max', maxima, 'float32', slotCount),
        first: view(temporalGraph, 'first', firsts, 'float32', slotCount),
        last: view(temporalGraph, 'last', lasts, 'float32', slotCount),
        occupiedSlots: {
          ids: view(temporalGraph, 'occupied', occupiedIds, 'uint32', slotCount),
          count: view(temporalGraph, 'occupied-count', occupiedCount, 'uint32', 1),
          overflow: view(temporalGraph, 'occupied-overflow', occupiedOverflow, 'uint32', 1)
        }
      }
    })
  );
  const temporalCompiled = resources.track(temporalGraph.compile());

  // ---- Display graph: the species you pick, folded into a band-by-bucket matrix --------------------
  // The matrix is laid out with the valid bucket count as its row stride, so the raster layer's grid
  // is exactly the drawn columns and nothing is tinted beyond the end of the year.
  const matrix = resources.createBuffer('matrix', matrixLength * 4);
  const thinCells = resources.createBuffer('thin-cells', matrixLength * 4);
  // [species mask, mode (0 share, 1 count, 2 fastest step), valid buckets, thin limit in fixes]
  const displayParameters = resources.createParameterBuffer('display', 'float32', 4);
  const displayGraph = new GPUCommandGraph<void>(device, {id: 'timing-display'});
  addKernelPass(displayGraph, {
    id: 'band-matrix',
    invocationCount: matrixLength,
    declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;
const BANDS: u32 = ${rowCount}u;
const SPECIES: u32 = ${SPECIES_COUNT}u;`,
    bindings: [
      {
        name: 'counts',
        view: view(displayGraph, 'counts', counts, 'uint32', slotCount),
        type: 'u32',
        access: 'read'
      },
      {
        name: 'fastest',
        view: view(displayGraph, 'max', maxima, 'float32', slotCount),
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
        view: view(displayGraph, 'matrix', matrix, 'float32', matrixLength),
        type: 'f32',
        access: 'read_write'
      },
      {
        name: 'thinCells',
        view: view(displayGraph, 'thin-cells', thinCells, 'float32', matrixLength),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let nan = bitcast<f32>(0x7fc00000u | (index & 0u));
  let selected = u32(settings[settingsOffset]);
  let mode = u32(settings[settingsOffset + 1u]);
  let valid = max(1u, u32(settings[settingsOffset + 2u]));
  let thinLimit = settings[settingsOffset + 3u];
  let row = index / valid;
  let bucket = index % valid;
  if (row >= BANDS) {
    matrix[matrixOffset + index] = nan;
    thinCells[thinCellsOffset + index] = nan;
    return;
  }
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
  var isThin = nan;
  if (inRow > 0.0) {
    if (mode == 0u) {
      result = inRow / inColumn;
    } else if (mode == 1u) {
      result = inRow;
    } else {
      result = fastestHere;
    }
    if (inRow < thinLimit) {
      isThin = 1.0;
    }
  }
  matrix[matrixOffset + index] = result;
  thinCells[thinCellsOffset + index] = isThin;`
  });
  const displayCompiled = resources.track(displayGraph.compile());

  // ---- Overlay segment sets ------------------------------------------------------------------------
  const createSegmentSet = (name: string, capacity: number): SegmentSet => {
    const buffer = resources.createBuffer(name, capacity * 16);
    const set: SegmentSet = {
      buffer,
      count: 0,
      write(rows) {
        const count = Math.min(capacity, Math.floor(rows.length / 4));
        set.count = count;
        if (count > 0) buffer.write(Float32Array.from(rows.slice(0, count * 4)));
      }
    };
    return set;
  };
  const frameRows = createSegmentSet('panel-frame', 4);
  const hairlineRows = createSegmentSet('hairlines', 12);
  const bucketEdgeRows = createSegmentSet('bucket-edges', BUCKET_COUNT);
  const probeRowRows = createSegmentSet('probe-row', 4);
  const bucketColumnRows = createSegmentSet('bucket-column', 4);
  const cursorRows = createSegmentSet('cursor', 1);
  const traceRows = createSegmentSet('trace', BUCKET_COUNT);
  const groupKeyRows = createSegmentSet('group-key', MAXIMUM_ROWS);
  const groupKeyIds = resources.createBuffer(
    'group-key-ids',
    Uint32Array.from({length: MAXIMUM_ROWS}, (_, index) => index)
  );

  // ---- Fix dots of the current bucket (rebuilt only when the bucket changes) -----------------------
  const widestBucketCounts = new Uint32Array(getValidBuckets(WIDEST_BUCKET_DAYS));
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    widestBucketCounts[
      Math.min(
        widestBucketCounts.length - 1,
        Math.floor(tracks.timestamps[vertex] / (WIDEST_BUCKET_DAYS * SECONDS_PER_DAY))
      )
    ]++;
  }
  const dotCapacity = Math.max(1, ...widestBucketCounts);
  const dotPositions = resources.createBuffer('dot-positions', dotCapacity * 8);
  const dotSpecies = resources.createBuffer('dot-species', dotCapacity * 4);
  let dotCount = 0;
  let dotKey = '';
  let dotIndex: {width: number; starts: Uint32Array; order: Uint32Array} | null = null;

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let reductionDirty = true;
  let displayDirty = true;
  let settleStale = true;
  let lastChange = performance.now();
  /** Counts up whenever the bucket width changes: a read-back of an older width is discarded. */
  let generation = 0;
  let inflightGeneration = -1;
  let cube: TimingCube | null = null;
  let countBreaks: number[] = [];
  let tables: Record<TimingMode, ClassTable> = {
    share: getTimingTable('share', ctx.ground(), []),
    count: getTimingTable('count', ctx.ground(), []),
    speed: getTimingTable('speed', ctx.ground(), [])
  };
  let lastBucket = -1;
  let medians: Float64Array = new Float64Array(0);
  let bandInput: {groups: number[][]; days: number[]; subject: string} | null = null;
  let fixesInput: {days: number[]; fixes: number[]; subject: string} | null = null;
  const clock = createPlaybackClock(
    ctx,
    {time: 'day', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [0, DAYS_IN_YEAR - 1], rate: 1, step: 1}
  );
  const sweep = {
    active: false,
    done: false,
    awaiting: false,
    index: 0,
    occupied: [] as number[]
  };

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };
  const getMask = () => SPECIES_MASKS[ctx.options.species];
  const getSubject = () =>
    ctx.options.species === 'all'
      ? 'fixes of all three species'
      : `fixes of the ${SPECIES_NAMES[Math.log2(getMask())].toLowerCase()}`;
  const getBucketWord = (width: number) => (width === 7 ? 'week' : `${width}-day bucket`);
  const getSpeciesLabel = () =>
    ctx.options.species === 'all' ? 'All three species' : SPECIES_NAMES[Math.log2(getMask())];

  // ---- Parameter writes ---------------------------------------------------------------------------
  function writeBucketParameters(): void {
    generation++;
    // A sweep owns the parameter buffer until it ends; it restores this width itself.
    if (sweep.active) return;
    bucketParameters.write(
      getGPUTemporalReductionParameterValues(0, ctx.options.bucketDays * SECONDS_PER_DAY)
    );
    reductionDirty = true;
    displayDirty = true;
    markChanged();
  }

  function writeDisplayParameters(): void {
    displayParameters.write(
      Float32Array.of(
        getMask(),
        MODE_INDEXES[ctx.options.cellValue],
        getValidBuckets(ctx.options.bucketDays),
        THIN_FIXES
      )
    );
    displayDirty = true;
    markChanged();
  }

  // ---- Overlays and annotations -------------------------------------------------------------------
  function writeStaticOverlays(): void {
    frameRows.write(getPanelFrameRows());
    groupKeyRows.write(getGroupKeyRows(rowCount));
  }

  /** Everything that depends on the bucket width, the probe, the trace and the playhead. */
  function writeOverlays(): void {
    const options = ctx.options;
    const width = options.bucketDays;
    hairlineRows.write(getHairlineRows(width, rowCount));
    bucketEdgeRows.write(options.showBucketEdges ? getBucketEdgeRows(width, rowCount) : []);
    probeRowRows.write(
      options.showProbe
        ? getProbeRowRows(width, rowCount, getRowOfLatitude(options.probeLatitude, rowCount))
        : []
    );
    traceRows.write(
      options.showTrace && cube && cube.bucketDays === width ? getTraceRows(medians, width) : []
    );
    writeCursor();
  }

  function writeCursor(): void {
    const width = ctx.options.bucketDays;
    cursorRows.write(getCursorRows(clock.time, rowCount));
    bucketColumnRows.write(getBucketColumnRows(width, rowCount, getBucketOfDay(clock.time, width)));
  }

  function publishAxes(): void {
    const width = ctx.options.bucketDays;
    const mode = ctx.options.cellValue;
    const word = getBucketWord(width);
    const title =
      mode === 'share'
        ? `Share of the ${word}'s fixes by 3° band`
        : mode === 'count'
          ? `Fixes by 3° band and ${word}`
          : `Fastest step by 3° band and ${word}`;
    ctx.setAnnotations('timing-axes', getAxisAnnotations(title, rowCount));
  }

  function publishProbe(): void {
    ctx.setAnnotations(
      'timing-probe',
      ctx.options.showProbe ? [getProbeAnnotation(ctx.options.probeLatitude)] : null
    );
  }

  /** Finding notes read from the cube: crossing dates at the probe, the fastest step. */
  function publishNotes(): void {
    const options = ctx.options;
    const notes: MapAnnotation[] = [];
    if (cube && cube.bucketDays === options.bucketDays) {
      if (options.showCrossings) {
        const crossings = findCrossings(medians, cube.bucketDays, options.probeLatitude);
        const latitude = `${options.probeLatitude}° N`;
        if (crossings.northbound !== null) {
          notes.push({
            kind: 'note',
            id: 'timing-northbound',
            coordinate: [dayToLongitude(crossings.northbound), options.probeLatitude],
            title: `Northbound ${formatFoldedDay(crossings.northbound)}`,
            text: `The median bird passes ${latitude}`
          });
        }
        if (crossings.southbound !== null) {
          notes.push({
            kind: 'note',
            id: 'timing-southbound',
            coordinate: [dayToLongitude(crossings.southbound), options.probeLatitude],
            title: `Southbound ${formatFoldedDay(crossings.southbound)}`,
            text: `The median bird passes ${latitude}`
          });
        }
      }
      if (options.cellValue === 'speed') {
        const fastest = getFastestCell(cube, getMask());
        if (fastest) {
          notes.push({
            kind: 'note',
            id: 'timing-fastest',
            coordinate: [
              CHART_WEST + (fastest.bucket + 0.5) * getColumnDegrees(cube.bucketDays),
              getRowLatitude(fastest.row) + BAND_DEGREES / 2
            ],
            title: `${Math.round(fastest.speed)} km/h`,
            text: 'The fastest step in the sample'
          });
        }
      }
    }
    ctx.setAnnotations('timing-notes', notes.length ? notes : null);
  }

  // ---- Summaries --------------------------------------------------------------------------------
  /**
   * Rebuilds the three class tables for the current ground and publishes them with the cells per
   * class (when the cells are known), the species counts and the group key for the legends.
   */
  function publishLegend(cells?: readonly {count: number; fastest: number; share: number}[]): void {
    const ground = ctx.ground();
    tables = {
      share: getTimingTable('share', ground, []),
      count: getTimingTable('count', ground, countBreaks),
      speed: getTimingTable('speed', ground, [])
    };
    let classCounts: Record<TimingMode, number[]> | null = null;
    if (cells) {
      classCounts = {
        share: new Array(tables.share.breaks.length + 1).fill(0),
        count: new Array(tables.count.breaks.length + 1).fill(0),
        speed: new Array(tables.speed.breaks.length + 1).fill(0)
      };
      for (const cell of cells) {
        classCounts.share[getClassOf(cell.share, tables.share.breaks)]++;
        classCounts.count[getClassOf(cell.count, tables.count.breaks)]++;
        classCounts.speed[getClassOf(cell.fastest, tables.speed.breaks)]++;
      }
    }
    const speciesCounts = new Array<number>(SPECIES_COUNT).fill(0);
    for (let track = 0; track < trackCount; track++) speciesCounts[tracks.species[track]]++;
    ctx.setLegendData('timing', {
      ground,
      tables,
      classCounts,
      speciesCounts,
      groupLabels: Array.from({length: Math.ceil(rowCount / ROWS_PER_GROUP)}, (_, group) =>
        getGroupLabel(group, rowCount)
      ),
      groupColors: getGroupColors(ground)
    });
  }

  function summarize(): void {
    if (!cube) return;
    const mask = getMask();
    const width = cube.bucketDays;
    const buckets = getValidBuckets(width);
    const subject = getSubject();

    // Class tables and the cells per class (the legend counts).
    const occupied = getOccupiedCells(cube, mask);
    countBreaks = getCountBreaks(occupied.values.map(cell => cell.count));
    publishLegend(occupied.values);
    ctx.setReadout(
      'thinCells',
      `${formatCount(occupied.thin)} of ${formatCount(occupied.values.length)} occupied cells`
    );

    // The bucket centres, the stacked band chart, the denominator strip.
    const centres = Array.from({length: buckets}, (_, bucket) => (bucket + 0.5) * width);
    const groups = getGroupShares(cube, mask);
    const fixes = getFixesPerBucket(cube, mask);
    // The last bucket is shorter than the others when the width does not divide the year.
    const fullBuckets = Math.floor(DAYS_IN_YEAR / width);
    bandInput = {groups, days: centres, subject};
    fixesInput = {
      days: centres.slice(0, fullBuckets),
      fixes: Array.from(fixes.slice(0, fullBuckets)),
      subject
    };
    const fullFixes = Array.from(fixes.slice(0, fullBuckets)).filter(value => value > 0);
    ctx.setReadout(
      'weeklyFixes',
      fullFixes.length
        ? `${formatCount(Math.min(...fullFixes))} to ${formatCount(Math.max(...fullFixes))} fixes per ${getBucketWord(width)}`
        : null
    );
    ctx.setTimelineData({
      domain: [0, DAYS_IN_YEAR],
      histogram: Array.from(fixesPerDaySelected(mask))
    });

    // Median latitude per bucket: the trace, the crossings and the speed of the migration.
    medians = getMedianLatitudes(cube, mask);
    const crossings20 = findCrossings(medians, width, 20);
    const crossings50 = findCrossings(medians, width, 50);
    const weeks = (from: number | null, to: number | null) =>
      from !== null && to !== null && to > from ? (to - from) / 7 : null;
    const spring = weeks(crossings20.northbound, crossings50.northbound);
    const autumn = weeks(crossings50.southbound, crossings20.southbound);
    ctx.setReadout(
      'springWeeks',
      spring === null
        ? 'The median does not climb from 20 to 50° N'
        : `${spring.toFixed(1)} weeks (${formatFoldedDay(crossings20.northbound!)} to ${formatFoldedDay(crossings50.northbound!)})`
    );
    ctx.setReadout(
      'autumnWeeks',
      autumn === null
        ? 'The median does not fall from 50 to 20° N'
        : `${autumn.toFixed(1)} weeks (${formatFoldedDay(crossings50.southbound!)} to ${formatFoldedDay(crossings20.southbound!)})`
    );
    const fastest = getFastestCell(cube, mask);
    ctx.setReadout(
      'fastest',
      fastest
        ? `${Math.round(fastest.speed)} km/h (${formatBucketRange(fastest.bucket, width)}, ${getRowLatitude(fastest.row)}-${getRowLatitude(fastest.row + 1)}° N)`
        : null
    );
    ctx.setReadout(
      'reduction',
      `${formatCount(cube.occupied)} of ${formatCount(cellCount * buckets)} slots occupied`
    );
    ctx.setReadout('buckets', `${buckets} buckets of ${width} days`);

    publishAxes();
    publishNotes();
    writeOverlays();
    summarizeProbe();
    updateCursor(true);
    ctx.requestLayers();
  }

  function fixesPerDaySelected(mask: number): Float64Array {
    const total = new Float64Array(DAYS_IN_YEAR);
    for (let species = 0; species < SPECIES_COUNT; species++) {
      if (!((mask >> species) & 1)) continue;
      for (let day = 0; day < DAYS_IN_YEAR; day++) total[day] += fixesPerDay[species][day];
    }
    return total;
  }

  function summarizeProbe(): void {
    if (!cube) return;
    const options = ctx.options;
    const mask = getMask();
    const row = getRowOfLatitude(options.probeLatitude, rowCount);
    const south = getRowLatitude(row);
    const buckets = getValidBuckets(cube.bucketDays);
    const perBucket: number[] = [];
    for (let bucket = 0; bucket < buckets; bucket++) {
      perBucket.push(getCell(cube, mask, row, bucket).count);
    }
    const total = perBucket.reduce((sum, count) => sum + count, 0);
    const crossings = findCrossings(medians, cube.bucketDays, options.probeLatitude);
    ctx.setReadout(
      'crossings',
      crossings.northbound === null && crossings.southbound === null
        ? `The median bird never crosses ${options.probeLatitude}° N`
        : `North ${crossings.northbound === null ? 'none' : formatFoldedDay(crossings.northbound)}, south ${crossings.southbound === null ? 'none' : formatFoldedDay(crossings.southbound)}`
    );
    if (total === 0) {
      ctx.setReadout('probe', `${south}-${south + BAND_DEGREES}° N: no fixes`);
      return;
    }
    // Busiest bucket in the first and in the second half of the year.
    const half = Math.floor(buckets / 2);
    const peak = (from: number, to: number) => {
      let best = -1;
      let bestCount = 0;
      for (let bucket = from; bucket < to; bucket++) {
        if (perBucket[bucket] > bestCount) {
          bestCount = perBucket[bucket];
          best = bucket;
        }
      }
      return best < 0
        ? 'none'
        : `${formatBucketRange(best, cube!.bucketDays)} (${formatCount(bestCount)} fixes)`;
    };
    ctx.setReadout(
      'probe',
      `${south}-${south + BAND_DEGREES}° N, ${formatCount(total)} fixes. Busiest bucket in the first half: ${peak(0, half)}; in the second: ${peak(half, buckets)}`
    );
  }

  /** The playhead moved: the cursor, the bucket column, the profile, the dots and the strips. */
  function updateCursor(force: boolean): void {
    const options = ctx.options;
    const width = options.bucketDays;
    const bucket = getBucketOfDay(clock.time, width);
    writeCursor();
    if (!cube || cube.bucketDays !== width) return;
    if (!force && bucket === lastBucket) return;
    lastBucket = bucket;
    const mask = getMask();
    const {perRow, total} = getColumn(cube, mask, bucket);
    const shares = Array.from(perRow, count => (total > 0 ? count / total : 0));
    let top = -1;
    perRow.forEach((count, row) => {
      if (count > 0 && (top < 0 || count > perRow[top])) top = row;
    });
    ctx.setChart('profileChart', buildProfileChart(shares, top));
    const median = medians[bucket];
    ctx.setReadout(
      'cursor',
      `${formatBucketRange(bucket, width)}: ${formatCount(total)} fixes${Number.isFinite(median) ? `, median latitude ${median.toFixed(1)}° N` : ''}`
    );
    const playhead = bucket * width + width / 2 - 0.5;
    if (bandInput) {
      ctx.setChart(
        'bandChart',
        buildBandChart({
          groups: bandInput.groups,
          days: bandInput.days,
          playhead,
          rowCount,
          ground: ctx.ground(),
          subject: bandInput.subject
        })
      );
    }
    if (fixesInput) {
      ctx.setChart(
        'fixesChart',
        buildFixesChart(fixesInput.days, fixesInput.fixes, playhead + 0.5, fixesInput.subject)
      );
    }
    updateDots();
  }

  // ---- Fix dots -----------------------------------------------------------------------------------
  function ensureDotIndex(width: number): NonNullable<typeof dotIndex> {
    if (dotIndex && dotIndex.width === width) return dotIndex;
    const buckets = getValidBuckets(width);
    const starts = new Uint32Array(buckets + 1);
    const bucketOf = new Uint16Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const bucket = Math.min(
        buckets - 1,
        Math.floor(tracks.timestamps[vertex] / (width * SECONDS_PER_DAY))
      );
      bucketOf[vertex] = bucket;
      starts[bucket + 1]++;
    }
    for (let bucket = 0; bucket < buckets; bucket++) starts[bucket + 1] += starts[bucket];
    const cursors = starts.slice(0, buckets);
    const order = new Uint32Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++)
      order[cursors[bucketOf[vertex]]++] = vertex;
    dotIndex = {width, starts, order};
    return dotIndex;
  }

  function updateDots(): void {
    const options = ctx.options;
    if (!options.showFixes) {
      dotCount = 0;
      dotKey = '';
      return;
    }
    const width = options.bucketDays;
    const bucket = getBucketOfDay(clock.time, width);
    const key = `${width}:${bucket}:${options.species}`;
    if (key === dotKey) return;
    dotKey = key;
    const index = ensureDotIndex(width);
    const mask = getMask();
    const positions = new Float32Array(dotCapacity * 2);
    const species = new Uint32Array(dotCapacity);
    let count = 0;
    for (
      let slot = index.starts[bucket];
      slot < index.starts[bucket + 1] && count < dotCapacity;
      slot++
    ) {
      const vertex = index.order[slot];
      if (!((mask >> fixSpecies[vertex]) & 1)) continue;
      positions[count * 2] = tracks.lngLat[vertex * 2];
      positions[count * 2 + 1] = tracks.lngLat[vertex * 2 + 1];
      species[count] = fixSpecies[vertex];
      count++;
    }
    dotCount = count;
    if (count > 0) {
      dotPositions.write(positions.slice(0, count * 2));
      dotSpecies.write(species.slice(0, count));
    }
    ctx.requestLayers();
  }

  // ---- Read-back of the cube and the width sweep ------------------------------------------------
  const reader = new SummaryReader(
    resources,
    'timing-summary',
    [
      {buffer: counts, size: slotCount * 4},
      {buffer: maxima, size: slotCount * 4},
      {buffer: occupiedCount, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      if (inflightGeneration !== generation) {
        // The width changed while this read was in flight: read again.
        settleStale = true;
        return;
      }
      cube = {
        counts: new Uint32Array(bytes, 0, slotCount).slice(),
        fastest: new Float32Array(bytes, slotCount * 4, slotCount).slice(),
        occupied: new Uint32Array(bytes, slotCount * 8, 1)[0],
        rowCount,
        bucketDays: ctx.options.bucketDays
      };
      summarize();
      // The first cube is on screen: sweep every width once for the occupancy curve.
      if (!sweep.done && !sweep.active) {
        sweep.active = true;
        sweep.index = 0;
        sweep.occupied = [];
      }
    }
  );

  // One slot is occupied when at least one fix fell in it. The sweep writes each width into the
  // parameter buffer of the same compiled graph (no recompile), encodes the reduction and reads
  // back only the occupied-slot count.
  const sweepReader = new SummaryReader(
    resources,
    'timing-sweep',
    [{buffer: occupiedCount, size: 4}],
    bytes => {
      if (destroyed || !sweep.active) return;
      sweep.occupied.push(new Uint32Array(bytes, 0, 1)[0]);
      sweep.index++;
      sweep.awaiting = false;
      if (sweep.index >= SWEEP_WIDTHS.length) finishSweep();
    }
  );

  function finishSweep(): void {
    sweep.active = false;
    sweep.done = true;
    const share = SWEEP_WIDTHS.map(
      (width, index) => (100 * sweep.occupied[index]) / (cellCount * getValidBuckets(width))
    );
    ctx.setChart('occupancyChart', buildOccupancyChart(SWEEP_WIDTHS, share));
    // Put the reduction back to the reader's width.
    writeBucketParameters();
    writeDisplayParameters();
  }

  // ---- Hover and click --------------------------------------------------------------------------
  /** The cell under a longitude and latitude, or `null` outside the matrix or while it is stale. */
  function getCellAt(longitude: number, latitude: number) {
    const options = ctx.options;
    if (!cube || cube.bucketDays !== options.bucketDays) return null;
    if (!isInsideMatrix(longitude, latitude, options.bucketDays, rowCount)) return null;
    const bucket = getBucketOfDay(longitudeToDay(longitude), options.bucketDays);
    const row = Math.min(rowCount - 1, Math.floor((latitude - LATITUDE_START) / BAND_DEGREES));
    return {row, bucket};
  }

  function describeValue(mode: TimingMode, value: number): string {
    if (!Number.isFinite(value)) return 'No fixes';
    if (mode === 'share') return formatPercent(value, value < 0.1 ? 1 : 0);
    if (mode === 'count') return `${formatCount(value)} fixes`;
    return `${Math.round(value)} km/h`;
  }

  // ---- Initial state ------------------------------------------------------------------------------
  writeBucketParameters();
  writeDisplayParameters();
  writeStaticOverlays();
  writeOverlays();
  publishAxes();
  publishProbe();
  publishLegend();
  ctx.setReadout(
    'tracks',
    `${tracks.individualIds.length} birds, ${trackCount} animal-years, ${formatCount(vertexCount)} fixes`
  );
  ctx.setReadout(
    'stepLength',
    `${(medianInterval / 3600).toFixed(1)} h between fixes (the closest pair is ${(shortestInterval / 3600).toFixed(1)} h apart)`
  );
  ctx.setFurniture({
    title: {
      sample: `${tracks.individualIds.length} birds, ${trackCount} animal-years, ${formatCount(vertexCount)} GPS fixes`
    }
  });
  ctx.setCost({records: vertexCount, passes: 2});

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [temporalCompiled, displayCompiled],

    setOption(id) {
      switch (id) {
        case 'bucketDays':
          writeBucketParameters();
          writeDisplayParameters();
          dotKey = '';
          publishAxes();
          writeOverlays();
          ctx.requestLayers();
          break;
        case 'species':
        case 'cellValue':
          writeDisplayParameters();
          dotKey = '';
          if (cube) summarize();
          else ctx.requestLayers();
          break;
        case 'probeLatitude':
        case 'showProbe':
          publishProbe();
          writeOverlays();
          summarizeProbe();
          publishNotes();
          ctx.requestLayers();
          break;
        case 'showBucketEdges':
        case 'showTrace':
          writeOverlays();
          ctx.requestLayers();
          break;
        case 'showCrossings':
          publishNotes();
          break;
        case 'showFixes':
          dotKey = '';
          updateDots();
          ctx.requestLayers();
          break;
        case 'showGroups':
        case 'showTracks':
        case 'trackOpacity':
          ctx.requestLayers();
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

    // The class tables and the chart colours are authored per ground.
    onGroundChange() {
      if (cube) {
        lastBucket = -1;
        summarize();
      } else {
        publishLegend();
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      clock.advance(frame);
      if (clock.moved) updateCursor(false);
      if (sweep.active) {
        if (!sweep.awaiting) {
          bucketParameters.write(
            getGPUTemporalReductionParameterValues(0, SWEEP_WIDTHS[sweep.index] * SECONDS_PER_DAY)
          );
          temporalCompiled.encode(commandEncoder, {parameters: undefined});
          sweep.awaiting = true;
          sweepReader.request(commandEncoder);
        } else {
          sweepReader.flush(commandEncoder);
        }
        return;
      }
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
      const wasPending = reader.isPending;
      reader.flush(commandEncoder);
      if (!wasPending && reader.isPending) inflightGeneration = generation;
    },

    getLayers() {
      const options = ctx.options;
      const ground = ctx.ground();
      const inks = getOverlayInks(ground);
      const width = options.bucketDays;
      const valid = getValidBuckets(width);
      const table = tables[options.cellValue];
      const layers: Layer[] = [];
      if (options.showTracks) {
        // Species ink: faint on paper; with one species chosen the others fall back to ghosts.
        const mask = getMask();
        const palette = getSpeciesInk(ground).map((color, species) => {
          const selected = (mask >> species) & 1;
          return [color[0], color[1], color[2], selected ? 255 : 51] as [
            number,
            number,
            number,
            number
          ];
        });
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
            palette,
            widthPixels: 0.9,
            opacity: options.trackOpacity
          })
        );
      }
      if (options.showFixes && dotCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'timing-fixes',
            ...drawProps,
            positions: dotPositions,
            instanceCount: dotCount,
            values: dotSpecies,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: getSpeciesInk(ground).map(
              color => [...color] as [number, number, number, number]
            ),
            radiusPixels: 3.4,
            outlineColor: inks.halo,
            outlineWidthPixels: 1,
            opacity: 1
          })
        );
      }
      // The paper panel, then the figure on it.
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: 'timing-panel',
          ...drawProps,
          gridSize: [1, 1],
          bounds: getPanelBounds(),
          colormap: 'uniform',
          color: getPanelFill(ground),
          opacity: 1
        })
      );
      if (hairlineRows.count > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'timing-hairlines',
            ...drawProps,
            segments: hairlineRows.buffer,
            instanceCount: hairlineRows.count,
            widthPixels: 0.5,
            cap: 'butt',
            color: inks.hairline
          })
        );
      }
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: 'timing-matrix',
          ...drawProps,
          gridSize: [valid, rowCount],
          bounds: getMatrixBounds(width, rowCount),
          tessellation: rowCount,
          values: matrix,
          valueFormat: 'float32',
          colormap: 'uniform',
          ...getClassTableLayerProps(table),
          // NaN is an empty cell: nothing is drawn, the paper shows (and nothing past the year).
          noDataColor: [0, 0, 0, 0],
          opacity: 1
        }),
        // Cells under the thin-cell limit get stripes over their colour (a class of one value).
        new SpatialAnalysisRasterLayer({
          id: 'timing-thin-cells',
          ...drawProps,
          gridSize: [valid, rowCount],
          bounds: getMatrixBounds(width, rowCount),
          tessellation: rowCount,
          values: thinCells,
          valueFormat: 'float32',
          colormap: 'uniform',
          classBreaks: [0.5],
          classColors: [
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ],
          hatchClasses: [1],
          hatchColor: inks.hatch,
          hatchSpacingPixels: 3,
          hatchWidthPixels: 1,
          noDataColor: [0, 0, 0, 0],
          opacity: 1
        })
      );
      if (options.showGroups && groupKeyRows.count > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'timing-group-key',
            ...drawProps,
            segments: groupKeyRows.buffer,
            instanceCount: groupKeyRows.count,
            values: groupKeyIds,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: getGroupColors(ground),
            widthPixels: 5,
            cap: 'butt'
          })
        );
      }
      if (options.showBucketEdges && bucketEdgeRows.count > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'timing-bucket-edges',
            ...drawProps,
            segments: bucketEdgeRows.buffer,
            instanceCount: bucketEdgeRows.count,
            widthPixels: 0.6,
            cap: 'butt',
            color: [inks.ink[0], inks.ink[1], inks.ink[2], 90]
          })
        );
      }
      if (options.showTrace && traceRows.count > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'timing-trace',
            ...drawProps,
            segments: traceRows.buffer,
            instanceCount: traceRows.count,
            widthPixels: 1.4,
            cap: 'round',
            color: inks.ink,
            outlineColor: inks.halo,
            outlineWidthPixels: 1.4
          })
        );
      }
      if (options.showProbe && probeRowRows.count > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'timing-probe-row',
            ...drawProps,
            segments: probeRowRows.buffer,
            instanceCount: probeRowRows.count,
            widthPixels: 1.4,
            cap: 'square',
            color: inks.signal
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'timing-bucket-column',
          ...drawProps,
          segments: bucketColumnRows.buffer,
          instanceCount: bucketColumnRows.count,
          widthPixels: 0.9,
          cap: 'square',
          color: [inks.ink[0], inks.ink[1], inks.ink[2], 150]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'timing-cursor',
          ...drawProps,
          segments: cursorRows.buffer,
          instanceCount: cursorRows.count,
          widthPixels: 1.6,
          cap: 'butt',
          color: inks.ink,
          outlineColor: inks.halo,
          outlineWidthPixels: 1.6
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'timing-panel-frame',
          ...drawProps,
          segments: frameRows.buffer,
          instanceCount: frameRows.count,
          widthPixels: 0.8,
          cap: 'square',
          color: inks.rule
        })
      );
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const options = ctx.options;
      const found = getCellAt(event.coordinate[0], event.coordinate[1]);
      if (!found || !cube) return null;
      const {row, bucket} = found;
      const mask = getMask();
      const cell = getCell(cube, mask, row, bucket);
      const {total} = getColumn(cube, mask, bucket);
      const mode = options.cellValue;
      const share = total > 0 ? cell.count / total : Number.NaN;
      const value =
        cell.count === 0
          ? Number.NaN
          : mode === 'share'
            ? share
            : mode === 'count'
              ? cell.count
              : cell.fastest;
      const table = tables[mode];
      const south = getRowLatitude(row);
      const west = CHART_WEST + bucket * getColumnDegrees(cube.bucketDays);
      const highlight: MapHighlight = {
        kind: 'box',
        bounds: [west, south, west + getColumnDegrees(cube.bucketDays), south + BAND_DEGREES]
      };
      const rows: TooltipContent['rows'] = [
        {
          label:
            mode === 'share' ? 'Share of the fixes' : mode === 'count' ? 'Fixes' : 'Fastest step',
          value: describeValue(mode, value),
          swatch: cell.count > 0 ? table.colors[getClassIndexOf(table, value)] : undefined,
          emphasis: true
        },
        ...(cell.count > 0 && mode !== 'count'
          ? [{label: 'Fixes in the cell', value: formatCount(cell.count)}]
          : []),
        ...(cell.count > 0 && mode !== 'share'
          ? [{label: 'Share of the bucket', value: formatPercent(share, share < 0.1 ? 1 : 0)}]
          : []),
        ...(cell.count > 0 && mode !== 'speed'
          ? [{label: 'Fastest step ending here', value: `${Math.round(cell.fastest)} km/h`}]
          : [])
      ];
      return {
        title: getSpeciesLabel(),
        subtitle: `${south}-${south + BAND_DEGREES}° N · ${formatBucketRange(bucket, cube.bucketDays)}`,
        rows,
        note:
          cell.count > 0 && cell.count < THIN_FIXES
            ? `Fewer than ${THIN_FIXES} fixes: read with care`
            : cell.count === 0
              ? 'No fixes in this cell'
              : undefined,
        highlight
      };
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const found = getCellAt(event.coordinate[0], event.coordinate[1]);
      if (!found) return false;
      const latitude = Math.round(event.coordinate[1] * 2) / 2;
      ctx.setOptions(
        {
          day: Math.min(
            DAYS_IN_YEAR - 1,
            Math.max(0, Math.floor(longitudeToDay(event.coordinate[0])))
          ),
          probeLatitude: Math.min(
            getRowLatitude(rowCount) - 0.5,
            Math.max(LATITUDE_START, latitude)
          )
        },
        {notify: true}
      );
      return true;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      sweepReader.stop();
      resources.destroy();
    }
  };
}
