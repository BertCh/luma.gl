// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPULineDensityParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPULineDensity,
  GPUTrackSimilarity,
  GPUTrajectoryResample
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {findNearestTrack} from './b12-tracks';
import {
  buildTrackSubset,
  describeMigrationTrack,
  loadMigrationTracks,
  MIGRATION_DATASET_ID,
  MIGRATION_SEASONS,
  MIGRATION_SPECIES_COLORS,
  SECONDS_PER_DAY,
  type MigrationSeason,
  type TrackSubset
} from './migration-shared';

/** Option state of the migration flyways scene. */
export type MigrationFlywaysOptions = {
  species: 'all' | 'marsh' | 'montagu' | 'spoonbill';
  season: MigrationSeason;
  cellSize: number;
  cellValue: 'density' | 'length';
  liftLow: boolean;
  probeLatitude: number;
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  showTracks: boolean;
  colorBy: 'species' | 'date' | 'similarity' | 'plain';
  trackOpacity: number;
  routeSpacing: 'arc-length' | 'time';
  similarityMetric: 'frechet' | 'hausdorff';
  minCoverageDays: number;
  similarityRangeKm: number;
};

/** Grid of the density raster: longitude -20 to 60 and latitude 3 to 73 at the finest cell size. */
const COLUMNS = 320;
const ROWS = 280;
const CELL_COUNT = COLUMNS * ROWS;
const GRID_ORIGIN: readonly [number, number] = [-20, 3];
const SIMILARITY_SAMPLES = 96;
const SETTLE_MILLISECONDS = 200;
const NO_TRACK = 0xffffffff;
const SPECIES_VALUES = {all: null, marsh: 0, montagu: 1, spoonbill: 2} as const;
const SIMILARITY_BIN_KILOMETERS = 100;
const SIMILARITY_BINS = 30;
/** Density in 1/m of the sphere, shown as kilometers of track per 1,000 square kilometers. */
const DENSITY_DISPLAY_SCALE = 1e6;

type DensityVariant = {
  subset: TrackSubset;
  positions: Buffer | null;
  offsets: Buffer | null;
  compiled: CompiledGPUCommandGraph<void> | null;
};

type DensitySnapshot = {
  lengths: Float32Array;
  densities: Float32Array;
  overflow: boolean;
  totalRecords: number;
};

/**
 * Migration flyways: `GPULineDensity` clips every track to a longitude/latitude grid and sums the
 * great-circle length per cell, so the busiest corridors glow; a species and a season restrict which
 * tracks are summed (each combination is its own compiled graph, built when first picked). The tracks
 * are also resampled to 96 points (`GPUTrajectoryResample`) and compared pairwise by Frechet or
 * Hausdorff distance (`GPUTrackSimilarity`), to ask whether a bird repeats its own route.
 */
export async function createMigrationFlyways(
  ctx: SceneContext<MigrationFlywaysOptions>
): Promise<SceneInstance<MigrationFlywaysOptions>> {
  const tracks = loadMigrationTracks(ctx.datasets.get(MIGRATION_DATASET_ID));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const resources = new SpatialAnalysisResources(device, 'flyways');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    graph: GPUCommandGraph<void>,
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);

  // ---- Static drawing inputs --------------------------------------------------------------------
  const segmentsBuffer = resources.createBuffer('segments', tracks.segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', tracks.segmentTracks);
  const segmentEndsBuffer = resources.createBuffer('segment-ends', tracks.segmentEndVertices);
  const timestampsBuffer = resources.createBuffer('timestamps', tracks.timestamps);
  const speciesBuffer = resources.createBuffer('species', Uint32Array.from(tracks.species));
  const distanceBuffer = resources.createBuffer('selected-distance', trackCount * 4);
  const selectedSegments = resources.createBuffer('selected-segments', tracks.longestTrack * 16);
  const siblingCapacity = 8;
  const siblingSegments = resources.createBuffer(
    'sibling-segments',
    siblingCapacity * tracks.longestTrack * 16
  );

  // ---- Density graph variants (species x season), compiled when first used ----------------------
  const densityParameters = resources.createParameterBuffer(
    'density-parameters',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH
  );
  const lengths = resources.createBuffer('cell-lengths', CELL_COUNT * 4);
  const densities = resources.createBuffer('cell-densities', CELL_COUNT * 4);
  const densityOverflow = resources.createBuffer('density-overflow', 4);
  const totalRecords = resources.createBuffer('total-records', 4);
  const variants = new Map<string, DensityVariant>();
  const emptyCells = new Float32Array(CELL_COUNT);

  function getVariant(species: MigrationFlywaysOptions['species'], season: MigrationSeason) {
    const key = `${species}|${season}`;
    let variant = variants.get(key);
    if (variant) return variant;
    const [fromDay, toDay] = MIGRATION_SEASONS[season].days;
    const subset = buildTrackSubset(
      tracks,
      SPECIES_VALUES[species],
      fromDay * SECONDS_PER_DAY,
      toDay * SECONDS_PER_DAY
    );
    variant = {subset, positions: null, offsets: null, compiled: null};
    if (subset.trackCount > 0) {
      variant.positions = resources.createBuffer(`${key}-positions`, subset.positions);
      variant.offsets = resources.createBuffer(`${key}-offsets`, subset.offsets);
      const graph = new GPUCommandGraph<void>(device, {id: `flyways-density-${key}`});
      graph.add(
        new GPULineDensity({
          id: 'line-density',
          positions: view(graph, 'positions', variant.positions, 'float32x2', subset.vertexCount),
          pathOffsets: view(graph, 'offsets', variant.offsets, 'uint32', subset.trackCount + 1),
          columns: COLUMNS,
          rows: ROWS,
          coordinateSystem: 'spherical',
          maximumRecords: Math.max(1024, 6 * subset.vertexCount),
          parameters: densityParameters.importToGraph(graph),
          output: {
            lengths: view(graph, 'lengths', lengths, 'float32', CELL_COUNT),
            densities: view(graph, 'densities', densities, 'float32', CELL_COUNT),
            overflow: view(graph, 'overflow', densityOverflow, 'uint32', 1),
            totalRecords: view(graph, 'total-records', totalRecords, 'uint32', 1)
          }
        })
      );
      variant.compiled = resources.track(graph.compile());
    }
    variants.set(key, variant);
    return variant;
  }

  // ---- Route similarity: resample, then every pair ----------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', tracks.positions);
  const offsetsBuffer = resources.createBuffer('offsets', tracks.offsets);
  const routesBuffer = resources.createBuffer('routes', trackCount * SIMILARITY_SAMPLES * 8);
  const routeOffsets = Uint32Array.from({length: trackCount + 1}, (_, i) => i * SIMILARITY_SAMPLES);
  const routeOffsetsBuffer = resources.createBuffer('route-offsets', routeOffsets);
  const pairCount = (trackCount * (trackCount - 1)) / 2;
  const pairA = new Uint32Array(pairCount);
  const pairB = new Uint32Array(pairCount);
  {
    let index = 0;
    for (let a = 0; a < trackCount; a++) {
      for (let b = a + 1; b < trackCount; b++) {
        pairA[index] = a;
        pairB[index] = b;
        index++;
      }
    }
  }
  const pairABuffer = resources.createBuffer('pair-a', pairA);
  const pairBBuffer = resources.createBuffer('pair-b', pairB);
  const pairHausdorff = resources.createBuffer('pair-hausdorff', pairCount * 4);
  const pairFrechet = resources.createBuffer('pair-frechet', pairCount * 4);
  const pairStatus = resources.createBuffer('pair-status', pairCount * 4);

  function buildResample(spacing: 'arc-length' | 'time'): CompiledGPUCommandGraph<void> {
    const graph = new GPUCommandGraph<void>(device, {id: `flyways-routes-${spacing}`});
    graph.add(
      new GPUTrajectoryResample({
        id: `routes-${spacing}`,
        positions: view(graph, 'positions', positionsBuffer, 'float32x2', vertexCount),
        timestamps: view(graph, 'timestamps', timestampsBuffer, 'float32', vertexCount),
        trackOffsets: view(graph, 'offsets', offsetsBuffer, 'uint32', trackCount + 1),
        sampleCount: SIMILARITY_SAMPLES,
        spacing,
        samples: view(graph, 'routes', routesBuffer, 'float32x2', trackCount * SIMILARITY_SAMPLES)
      })
    );
    return resources.track(graph.compile());
  }
  const resampleGraphs = {'arc-length': buildResample('arc-length'), time: buildResample('time')};
  const similarityGraph = new GPUCommandGraph<void>(device, {id: 'flyways-similarity'});
  similarityGraph.add(
    new GPUTrackSimilarity({
      id: 'route-similarity',
      positionsA: view(
        similarityGraph,
        'routes',
        routesBuffer,
        'float32x2',
        trackCount * SIMILARITY_SAMPLES
      ),
      offsetsA: view(
        similarityGraph,
        'route-offsets',
        routeOffsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      pairA: view(similarityGraph, 'pair-a', pairABuffer, 'uint32', pairCount),
      pairB: view(similarityGraph, 'pair-b', pairBBuffer, 'uint32', pairCount),
      hausdorff: view(similarityGraph, 'pair-hausdorff', pairHausdorff, 'float32', pairCount),
      frechet: view(similarityGraph, 'pair-frechet', pairFrechet, 'float32', pairCount),
      status: view(similarityGraph, 'pair-status', pairStatus, 'uint32', pairCount),
      maxFrechetVertices: SIMILARITY_SAMPLES
    })
  );
  const similarityCompiled = resources.track(similarityGraph.compile());

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let densityDirty = true;
  let similarityDirty = true;
  let settleStale = true;
  let lastChange = performance.now();
  let currentSpacing = ctx.options.routeSpacing;
  let active = getVariant(ctx.options.species, ctx.options.season);
  let densitySnapshot: DensitySnapshot | null = null;
  let densityRange = {density: 1, length: 1};
  let matrix: Float32Array | null = null;
  let selectedTrack = NO_TRACK;
  const individualTracks = new Map<number, number[]>();
  for (let track = 0; track < trackCount; track++) {
    const list = individualTracks.get(tracks.individual[track]) ?? [];
    list.push(track);
    individualTracks.set(tracks.individual[track], list);
  }

  const markChanged = () => {
    lastChange = performance.now();
    settleStale = true;
  };

  function writeDensityParameters(): void {
    const size = ctx.options.cellSize;
    densityParameters.write(
      getGPULineDensityParameterValues({
        minX: GRID_ORIGIN[0],
        minY: GRID_ORIGIN[1],
        cellWidth: size,
        cellHeight: size
      })
    );
    densityDirty = true;
    markChanged();
  }

  function isEligible(track: number): boolean {
    return tracks.coverageDays[track] >= ctx.options.minCoverageDays;
  }

  function writeSelection(): void {
    const selected = new Float32Array(tracks.longestTrack * 4).fill(Number.NaN);
    const siblings = new Float32Array(siblingCapacity * tracks.longestTrack * 4).fill(Number.NaN);
    if (selectedTrack !== NO_TRACK) {
      const writeTrack = (track: number, target: Float32Array, rowOffset: number) => {
        const first = tracks.offsets[track];
        const last = tracks.offsets[track + 1] - 1;
        for (let vertex = first; vertex < last; vertex++) {
          target.set(
            tracks.lngLat.subarray(vertex * 2, vertex * 2 + 4),
            (rowOffset + vertex - first) * 4
          );
        }
      };
      writeTrack(selectedTrack, selected, 0);
      const others = (individualTracks.get(tracks.individual[selectedTrack]) ?? []).filter(
        track => track !== selectedTrack
      );
      others.slice(0, siblingCapacity).forEach((track, index) => {
        writeTrack(track, siblings, index * tracks.longestTrack);
      });
    }
    selectedSegments.write(selected);
    siblingSegments.write(siblings);
  }

  function writeSelectedDistances(): void {
    const distances = new Float32Array(trackCount).fill(Number.NaN);
    if (matrix && selectedTrack !== NO_TRACK) {
      for (let track = 0; track < trackCount; track++) {
        if (!isEligible(track)) continue;
        distances[track] = track === selectedTrack ? 0 : matrix[selectedTrack * trackCount + track];
      }
    }
    distanceBuffer.write(distances);
  }

  function describeSelected(): void {
    if (selectedTrack === NO_TRACK) {
      ctx.setReadout('selected', 'click a track');
      return;
    }
    const siblings = (individualTracks.get(tracks.individual[selectedTrack]) ?? []).length - 1;
    ctx.setReadout(
      'selected',
      `${describeMigrationTrack(tracks, selectedTrack)}: ${tracks.coverageDays[selectedTrack].toFixed(0)} days, ${formatCount(tracks.pathMeters[selectedTrack] / 1000)} km, ${siblings} other tagged year${siblings === 1 ? '' : 's'} of this bird`
    );
  }

  // ---- Density summary ---------------------------------------------------------------------------
  function summarizeDensity(): void {
    const snapshot = densitySnapshot;
    if (!snapshot) return;
    const size = ctx.options.cellSize;
    let total = 0;
    let occupied = 0;
    const positiveDensities: number[] = [];
    const positiveLengths: number[] = [];
    for (let cell = 0; cell < CELL_COUNT; cell++) {
      const length = snapshot.lengths[cell];
      if (length > 0) {
        total += length;
        occupied++;
        positiveLengths.push(length);
        positiveDensities.push(snapshot.densities[cell]);
      }
    }
    positiveDensities.sort((a, b) => a - b);
    positiveLengths.sort((a, b) => a - b);
    const percentile = (values: number[]) =>
      values.length ? values[Math.min(values.length - 1, Math.floor(values.length * 0.995))] : 1;
    densityRange = {
      density: percentile(positiveDensities) * DENSITY_DISPLAY_SCALE,
      length: percentile(positiveLengths) / 1000
    };
    ctx.setLegendExtent('density', [0, densityRange.density]);
    ctx.setLegendExtent('length', [0, densityRange.length]);
    ctx.requestLayers();

    const subset = active.subset;
    ctx.setReadout(
      'flyway',
      `${formatCount(subset.trackCount)} tracks, ${formatCount(subset.vertexCount)} fixes, ${formatCount(total / 1000)} km of track in ${formatCount(occupied)} cells of ${size.toFixed(2)} degrees${snapshot.overflow ? ' (piece capacity overflowed: lengths are low)' : ''}`
    );
    const busiest = positiveLengths.length ? positiveLengths[positiveLengths.length - 1] : 0;
    ctx.setReadout('busiest', `${formatCount(busiest / 1000)} km of track in the busiest cell`);

    // Corridor: central 80% of the track length of each latitude row, as longitudes.
    const latitudes: number[] = [];
    const centers: number[] = [];
    const lows: number[] = [];
    const highs: number[] = [];
    const rowTotals = new Float64Array(ROWS);
    let maxRow = 0;
    for (let row = 0; row < ROWS; row++) {
      let sum = 0;
      for (let column = 0; column < COLUMNS; column++)
        sum += snapshot.lengths[row * COLUMNS + column];
      rowTotals[row] = sum;
      maxRow = Math.max(maxRow, sum);
    }
    const quantiles = new Map<number, [number, number, number]>();
    for (let row = 0; row < ROWS; row++) {
      if (rowTotals[row] < maxRow * 0.01) continue;
      const targets = [0.1, 0.5, 0.9];
      const found: number[] = [];
      let running = 0;
      let next = 0;
      for (let column = 0; column < COLUMNS && next < targets.length; column++) {
        running += snapshot.lengths[row * COLUMNS + column];
        while (next < targets.length && running >= targets[next] * rowTotals[row]) {
          found.push(GRID_ORIGIN[0] + (column + 0.5) * size);
          next++;
        }
      }
      if (found.length === 3) {
        quantiles.set(row, [found[0], found[1], found[2]]);
        latitudes.push(GRID_ORIGIN[1] + (row + 0.5) * size);
        lows.push(found[0]);
        centers.push(found[1]);
        highs.push(found[2]);
      }
    }
    ctx.setChart(
      'corridorChart',
      latitudes.length
        ? {
            kind: 'line',
            series: [{label: 'median longitude of the track length', x: latitudes, y: centers}],
            band: {x: latitudes, low: lows, high: highs, label: '10th to 90th percentile'},
            markers: [{x: ctx.options.probeLatitude, label: 'probe'}],
            xLabel: 'latitude (N)',
            yLabel: 'longitude (E)',
            height: 130,
            formatX: value => `${Math.round(value)}`,
            formatY: value => `${Math.round(value)}`,
            description:
              'For each latitude row, the longitude of the median track length and the 10th to 90th percentile band. A narrow band is a bottleneck.'
          }
        : null
    );
    // Probe: the row nearest the probe latitude.
    const probeRow = Math.round((ctx.options.probeLatitude - GRID_ORIGIN[1]) / size - 0.5);
    const probe = quantiles.get(probeRow);
    if (probe) {
      const latitude = GRID_ORIGIN[1] + (probeRow + 0.5) * size;
      const widthKm = (probe[2] - probe[0]) * 111.32 * Math.cos((latitude * Math.PI) / 180);
      ctx.setReadout(
        'probe',
        `${latitude.toFixed(1)} N: central 80% of track length within ${widthKm.toFixed(0)} km (${probe[0].toFixed(1)} to ${probe[2].toFixed(1)} E), median ${probe[1].toFixed(1)} E`
      );
    } else {
      ctx.setReadout(
        'probe',
        `${ctx.options.probeLatitude.toFixed(1)} N: no track crosses this row`
      );
    }
  }

  // ---- Similarity summary -----------------------------------------------------------------------
  let similaritySnapshot: {
    hausdorff: Float32Array;
    frechet: Float32Array;
    status: Uint32Array;
  } | null = null;

  function buildMatrix(): void {
    if (!similaritySnapshot) return;
    const source =
      ctx.options.similarityMetric === 'frechet'
        ? similaritySnapshot.frechet
        : similaritySnapshot.hausdorff;
    matrix = new Float32Array(trackCount * trackCount);
    for (let pair = 0; pair < pairCount; pair++) {
      matrix[pairA[pair] * trackCount + pairB[pair]] = source[pair];
      matrix[pairB[pair] * trackCount + pairA[pair]] = source[pair];
    }
    writeSelectedDistances();
    summarizeSimilarity();
  }

  function summarizeSimilarity(): void {
    if (!matrix) return;
    const groups: number[][] = [[], [], []];
    let eligible = 0;
    for (let track = 0; track < trackCount; track++) if (isEligible(track)) eligible++;
    for (let pair = 0; pair < pairCount; pair++) {
      const a = pairA[pair];
      const b = pairB[pair];
      if (!isEligible(a) || !isEligible(b)) continue;
      const kilometers = matrix[a * trackCount + b] / 1000;
      if (tracks.individual[a] === tracks.individual[b]) groups[0].push(kilometers);
      else if (tracks.species[a] === tracks.species[b]) groups[1].push(kilometers);
      else groups[2].push(kilometers);
    }
    const median = (values: number[]) =>
      values.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : NaN;
    const names = ['same bird, other year', 'other bird, same species', 'other species'];
    const centers = Array.from(
      {length: SIMILARITY_BINS},
      (_, bin) => (bin + 0.5) * SIMILARITY_BIN_KILOMETERS
    );
    const series = groups
      .map((values, index) => {
        if (values.length === 0) return null;
        const share = new Float64Array(SIMILARITY_BINS);
        for (const value of values) {
          share[Math.min(SIMILARITY_BINS - 1, Math.floor(value / SIMILARITY_BIN_KILOMETERS))]++;
        }
        return {
          label: `${names[index]} (${values.length})`,
          x: centers,
          y: Array.from(share, count => (100 * count) / values.length),
          color: index === 0 ? 1 : index === 1 ? 0 : 5,
          area: index === 0
        };
      })
      .filter(entry => entry !== null);
    ctx.setChart(
      'fidelityChart',
      series.length
        ? {
            kind: 'line',
            series,
            xLabel: `${ctx.options.similarityMetric === 'frechet' ? 'Frechet' : 'Hausdorff'} distance between routes (km)`,
            yLabel: '% of pairs',
            xDomain: [0, SIMILARITY_BINS * SIMILARITY_BIN_KILOMETERS],
            height: 130,
            formatX: value => `${Math.round(value)}`,
            formatY: value => `${Math.round(value)}`,
            description:
              'Distribution of the distance between two annual routes, for the same bird in different years, for different birds of one species and for different species. Only tracks covering the minimum number of days are compared.'
          }
        : null
    );
    ctx.setReadout(
      'fidelity',
      `${eligible} tracks, median distance: same bird ${formatMedian(median(groups[0]), groups[0].length)}, other bird of the species ${formatMedian(median(groups[1]), groups[1].length)}, other species ${formatMedian(median(groups[2]), groups[2].length)}`
    );
    ctx.setReadout(
      'routes',
      `${trackCount} routes x ${SIMILARITY_SAMPLES} samples, ${formatCount(pairCount)} pairs scored`
    );
  }

  function formatMedian(value: number, count: number): string {
    return Number.isFinite(value) ? `${value.toFixed(0)} km (${count} pairs)` : 'n/a';
  }

  // ---- Readers ----------------------------------------------------------------------------------
  const densityReader = new SummaryReader(
    resources,
    'flyways-density',
    [
      {buffer: lengths, size: CELL_COUNT * 4},
      {buffer: densities, size: CELL_COUNT * 4},
      {buffer: densityOverflow, size: 4},
      {buffer: totalRecords, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      densitySnapshot = {
        lengths: new Float32Array(bytes, 0, CELL_COUNT).slice(),
        densities: new Float32Array(bytes, CELL_COUNT * 4, CELL_COUNT).slice(),
        overflow: new Uint32Array(bytes, CELL_COUNT * 8, 1)[0] !== 0,
        totalRecords: new Uint32Array(bytes, CELL_COUNT * 8 + 4, 1)[0]
      };
      summarizeDensity();
    }
  );
  const similarityReader = new SummaryReader(
    resources,
    'flyways-similarity',
    [
      {buffer: pairHausdorff, size: pairCount * 4},
      {buffer: pairFrechet, size: pairCount * 4},
      {buffer: pairStatus, size: pairCount * 4}
    ],
    bytes => {
      if (destroyed) return;
      similaritySnapshot = {
        hausdorff: new Float32Array(bytes, 0, pairCount).slice(),
        frechet: new Float32Array(bytes, pairCount * 4, pairCount).slice(),
        status: new Uint32Array(bytes, pairCount * 8, pairCount).slice()
      };
      buildMatrix();
    }
  );

  // Default selection: an eligible track of a bird with other tagged years, the longest such path.
  {
    let best = NO_TRACK;
    for (let track = 0; track < trackCount; track++) {
      const siblings = (individualTracks.get(tracks.individual[track]) ?? []).length;
      if (siblings < 2 || tracks.coverageDays[track] < 270) continue;
      if (best === NO_TRACK || tracks.pathMeters[track] > tracks.pathMeters[best]) best = track;
    }
    selectedTrack = best;
  }
  writeSelection();
  writeDensityParameters();
  describeSelected();
  ctx.setReadout(
    'birds',
    `${trackCount} animal-years of ${individualTracks.size} birds, ${formatCount(vertexCount)} fixes`
  );

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      ...(active.compiled ? [active.compiled] : []),
      resampleGraphs[currentSpacing],
      similarityCompiled
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'species':
        case 'season':
          active = getVariant(state.species, state.season);
          densityDirty = true;
          markChanged();
          ctx.requestLayers();
          break;
        case 'cellSize':
          writeDensityParameters();
          ctx.requestLayers();
          break;
        case 'cellValue':
        case 'liftLow':
          ctx.requestLayers();
          break;
        case 'probeLatitude':
          summarizeDensity();
          break;
        case 'routeSpacing':
          currentSpacing = state.routeSpacing;
          similarityDirty = true;
          break;
        case 'similarityMetric':
          buildMatrix();
          ctx.requestLayers();
          break;
        case 'minCoverageDays':
          writeSelectedDistances();
          summarizeSimilarity();
          ctx.requestLayers();
          break;
        case 'similarityRangeKm':
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (densityDirty) {
        if (active.compiled) {
          active.compiled.encode(commandEncoder, {parameters: undefined});
        } else {
          // No track in this species and season: zero the outputs with queue writes.
          lengths.write(emptyCells);
          densities.write(emptyCells);
          densityOverflow.write(new Uint32Array(1));
          totalRecords.write(new Uint32Array(1));
        }
        densityDirty = false;
        settleStale = true;
      }
      if (settleStale && performance.now() - lastChange > SETTLE_MILLISECONDS) {
        densityReader.markStale();
        settleStale = false;
      }
      densityReader.flush(commandEncoder);
      if (similarityDirty) {
        resampleGraphs[currentSpacing].encode(commandEncoder, {parameters: undefined});
        similarityCompiled.encode(commandEncoder, {parameters: undefined});
        similarityDirty = false;
        similarityReader.markStale();
      }
      similarityReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const size = options.cellSize;
      const byDensity = options.cellValue === 'density';
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: 'flyway-density',
          ...drawProps,
          gridSize: [COLUMNS, ROWS],
          bounds: [
            GRID_ORIGIN[0],
            GRID_ORIGIN[1],
            GRID_ORIGIN[0] + COLUMNS * size,
            GRID_ORIGIN[1] + ROWS * size
          ],
          tessellation: 64,
          values: byDensity ? densities : lengths,
          valueFormat: 'float32',
          colormap: options.ramp,
          valueScale: byDensity ? DENSITY_DISPLAY_SCALE : 0.001,
          valueRange: [0, byDensity ? densityRange.density : densityRange.length],
          sqrtScale: options.liftLow,
          discardAtOrBelow: 0,
          opacity: 0.9
        })
      );
      if (options.showTracks) {
        let colorProps: Record<string, unknown>;
        switch (options.colorBy) {
          case 'species':
            colorProps = {
              values: speciesBuffer,
              valueFormat: 'uint32',
              valueIndices: segmentTracksBuffer,
              colormap: 'category',
              palette: MIGRATION_SPECIES_COLORS
            };
            break;
          case 'date':
            colorProps = {
              values: timestampsBuffer,
              valueFormat: 'float32',
              valueIndices: segmentEndsBuffer,
              valueScale: 1 / SECONDS_PER_DAY,
              valueRange: [0, 366],
              colormap: 'cividis'
            };
            break;
          case 'similarity':
            colorProps = {
              values: distanceBuffer,
              valueFormat: 'float32',
              valueIndices: segmentTracksBuffer,
              valueScale: 0.001,
              valueRange: [0, options.similarityRangeKm],
              colormap: 'cividis',
              noDataColor: [140, 146, 160, 40]
            };
            break;
          default:
            colorProps = {color: dark ? [225, 230, 240, 255] : [50, 60, 80, 255]};
        }
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'flyway-tracks',
            ...drawProps,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 1.1,
            opacity: options.trackOpacity,
            ...colorProps
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'flyway-siblings',
          ...drawProps,
          segments: siblingSegments,
          instanceCount: siblingCapacity * tracks.longestTrack,
          widthPixels: 2.2,
          color: [255, 214, 64, 235]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'flyway-selected',
          ...drawProps,
          segments: selectedSegments,
          instanceCount: tracks.longestTrack,
          widthPixels: 3,
          color: dark ? [255, 255, 255, 240] : [15, 20, 30, 240]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const track = pickTrack(event.coordinate);
      if (track < 0) return null;
      return `${describeMigrationTrack(tracks, track)}: ${tracks.coverageDays[track].toFixed(0)} days of data, ${formatCount(tracks.pathMeters[track] / 1000)} km`;
    },

    onClick(event) {
      const track = pickTrack(event.coordinate);
      if (track < 0) return false;
      selectedTrack = track;
      writeSelection();
      writeSelectedDistances();
      describeSelected();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      destroyed = true;
      densityReader.stop();
      similarityReader.stop();
      resources.destroy();
    }
  };

  /** Nearest track within 200 km of a longitude/latitude, or -1. */
  function pickTrack(coordinate: readonly [number, number] | null): number {
    if (!coordinate) return -1;
    const [x, y] = tracks.project(coordinate[0], coordinate[1]);
    const {track, distance} = findNearestTrack(tracks, x, y);
    return track >= 0 && distance < 200000 ? track : -1;
  }
}
