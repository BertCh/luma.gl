// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUOutlineGeometryParameterValues,
  getGPUOutlineGeometryVerticesPerInput,
  GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH,
  GPUOutlineGeometry
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {SatelliteCellLayer, SatelliteSwathLayer} from './satellite-layers';
import {
  createTrackSubset,
  formatSatelliteClock,
  loadSatelliteTracks,
  SATELLITE_DATASET_ID,
  SATELLITE_GROUP_COLORS,
  selectTracksByGroup,
  type SatelliteInfo,
  type TrackSubset
} from './satellite-tracks';

/** Sensors with a nominal swath, in the order of the select option. */
export const SWATH_SENSORS = [
  'OLI',
  'C-SAR IW',
  'MSI',
  'OLCI',
  'MODIS',
  'TROPOMI',
  'AVHRR',
  'VIIRS'
] as const;
type Sensor = (typeof SWATH_SENSORS)[number];

/** Option state of the swath coverage scene. */
export type SatelliteSwathOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  satellites: string;
  swathMode: 'nominal' | 'custom';
  customWidth: number;
  showCoverage: boolean;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  showSwaths: boolean;
  swathOpacity: number;
};

/** Coverage grid: 1 degree cells from 180 W to 180 E and 85 S to 85 N. */
export const COVERAGE_GRID = {
  minLongitude: -180,
  minLatitude: -85,
  cell: 1,
  columns: 360,
  rows: 170
};
const JOIN_SEGMENTS = 16;
const KILOMETERS_PER_DEGREE = 111.195;
const TIME_BIN_SECONDS = 150;

type SensorClass = {
  sensor: Sensor;
  subset: TrackSubset;
  swathKm: number;
  outline: Buffer;
  times: Buffer;
  satellites: Buffer;
};

/**
 * Swath coverage of the Earth observation satellites. `GPUOutlineGeometry` buffers every ground
 * track by half the swath width (one node per instrument, because the contributor has one distance
 * per node) and a layer draws the triangles up to the playhead. A bespoke compute pass then finds,
 * for every 1-degree cell, the first time any swath reached it, from which the covered area over time
 * and the time to 50, 90 and 99 percent follow.
 */
export async function createSatelliteSwath(
  ctx: SceneContext<SatelliteSwathOptions>
): Promise<SceneInstance<SatelliteSwathOptions>> {
  const tracks = loadSatelliteTracks(ctx.datasets.get(SATELLITE_DATASET_ID));
  const resources = new SpatialAnalysisResources(ctx.device, 'satellite-swath');
  const rowsPerInput = getGPUOutlineGeometryVerticesPerInput(JOIN_SEGMENTS);
  const cellCount = COVERAGE_GRID.columns * COVERAGE_GRID.rows;
  let destroyed = false;

  const earthObservationTracks = selectTracksByGroup(tracks, [4]);
  const all = createTrackSubset(tracks, earthObservationTracks);
  const satelliteVisible = resources.createBuffer(
    'satellite-visible',
    new Uint32Array(tracks.satelliteCount).fill(1)
  );
  const infoOf = (satellite: number): SatelliteInfo => tracks.satellites[satellite];

  // ---- Outline graphs ------------------------------------------------------------------------------
  function createOutlineBuffers(name: string, subset: TrackSubset) {
    return {
      positions: resources.createBuffer(`${name}-positions`, subset.positions),
      offsets: resources.createBuffer(`${name}-offsets`, subset.offsets),
      outline: resources.createBuffer(`${name}-outline`, subset.vertexCount * rowsPerInput * 8),
      times: resources.createBuffer(`${name}-times`, subset.timestamps),
      satellites: resources.createBuffer(`${name}-satellites`, subset.vertexSatellite)
    };
  }

  const nominalGraph = new GPUCommandGraph<void>(ctx.device, {id: 'satellite-swath-nominal'});
  const classes: SensorClass[] = [];
  for (const sensor of SWATH_SENSORS) {
    const trackList = earthObservationTracks.filter(
      track => infoOf(tracks.satelliteIndex[track]).sensor === sensor
    );
    if (!trackList.length) continue;
    const subset = createTrackSubset(tracks, trackList);
    const swathKm = subset.swathKm[0];
    const name = sensor.replace(/[^A-Za-z0-9]/g, '-');
    const buffers = createOutlineBuffers(`nominal-${name}`, subset);
    const parameters = resources.createParameterBuffer(
      `nominal-${name}-distance`,
      'float32',
      GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH,
      getGPUOutlineGeometryParameterValues({distance: (swathKm * 1000) / 2})
    );
    nominalGraph.add(
      new GPUOutlineGeometry({
        id: `outline-${name}`,
        positions: importGraphBuffer(
          nominalGraph,
          `${name}-positions`,
          buffers.positions,
          'float32x2',
          subset.vertexCount
        ),
        geometryType: 'lines',
        pathOffsets: importGraphBuffer(
          nominalGraph,
          `${name}-offsets`,
          buffers.offsets,
          'uint32',
          subset.trackCount + 1
        ),
        coordinateSystem: 'spherical',
        joinSegments: JOIN_SEGMENTS,
        parameters: parameters.importToGraph(nominalGraph),
        output: {
          positions: importGraphBuffer(
            nominalGraph,
            `${name}-outline`,
            buffers.outline,
            'float32x2',
            subset.vertexCount * rowsPerInput
          )
        }
      })
    );
    classes.push({sensor, subset, swathKm, ...buffers});
  }
  const nominalCompiled = resources.track(nominalGraph.compile());

  const customBuffers = createOutlineBuffers('custom', all);
  const customParameters = resources.createParameterBuffer(
    'custom-distance',
    'float32',
    GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH,
    getGPUOutlineGeometryParameterValues({distance: (ctx.options.customWidth * 1000) / 2})
  );
  const customGraph = new GPUCommandGraph<void>(ctx.device, {id: 'satellite-swath-custom'});
  customGraph.add(
    new GPUOutlineGeometry({
      id: 'outline-custom',
      positions: importGraphBuffer(
        customGraph,
        'positions',
        customBuffers.positions,
        'float32x2',
        all.vertexCount
      ),
      geometryType: 'lines',
      pathOffsets: importGraphBuffer(
        customGraph,
        'offsets',
        customBuffers.offsets,
        'uint32',
        all.trackCount + 1
      ),
      coordinateSystem: 'spherical',
      joinSegments: JOIN_SEGMENTS,
      parameters: customParameters.importToGraph(customGraph),
      output: {
        positions: importGraphBuffer(
          customGraph,
          'outline',
          customBuffers.outline,
          'float32x2',
          all.vertexCount * rowsPerInput
        )
      }
    })
  );
  const customCompiled = resources.track(customGraph.compile());

  // ---- Coverage kernel -----------------------------------------------------------------------------
  // One row per consecutive vertex pair of the Earth observation tracks.
  let segmentCount = 0;
  for (let track = 0; track < all.trackCount; track++) {
    segmentCount += Math.max(0, all.offsets[track + 1] - all.offsets[track] - 1);
  }
  const segmentGeometry = new Float32Array(segmentCount * 4);
  const segmentTimes = new Float32Array(segmentCount * 2);
  const segmentSatellite = new Uint32Array(segmentCount);
  let segmentRow = 0;
  for (let track = 0; track < all.trackCount; track++) {
    for (let vertex = all.offsets[track]; vertex < all.offsets[track + 1] - 1; vertex++) {
      segmentGeometry.set(all.positions.subarray(vertex * 2, vertex * 2 + 4), segmentRow * 4);
      segmentTimes[segmentRow * 2] = all.timestamps[vertex];
      segmentTimes[segmentRow * 2 + 1] = all.timestamps[vertex + 1];
      segmentSatellite[segmentRow] = all.satelliteIndex[track];
      segmentRow++;
    }
  }
  const segmentBuffer = resources.createBuffer('coverage-segments', segmentGeometry);
  const segmentTimeBuffer = resources.createBuffer('coverage-segment-times', segmentTimes);
  const halfWidthBuffer = resources.createBuffer('coverage-half-widths', segmentCount * 4);
  const firstCoverBuffer = resources.createBuffer('first-cover', cellCount * 4);
  const coverageGraph = new GPUCommandGraph<void>(ctx.device, {id: 'satellite-coverage'});
  addKernelPass(coverageGraph, {
    id: 'first-cover',
    invocationCount: cellCount,
    declarations: `const COLUMNS: u32 = ${COVERAGE_GRID.columns}u;
const SEGMENTS: u32 = ${segmentCount}u;
const CELL: f32 = ${COVERAGE_GRID.cell.toFixed(4)};
const MIN_LONGITUDE: f32 = ${COVERAGE_GRID.minLongitude.toFixed(4)};
const MIN_LATITUDE: f32 = ${COVERAGE_GRID.minLatitude.toFixed(4)};
const KILOMETERS_PER_DEGREE: f32 = ${KILOMETERS_PER_DEGREE};
const DEGREES_TO_RADIANS: f32 = 0.017453292;`,
    bindings: [
      {
        name: 'segments',
        view: importGraphBuffer(
          coverageGraph,
          'segments',
          segmentBuffer,
          'float32',
          segmentCount * 4
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'segmentTimes',
        view: importGraphBuffer(
          coverageGraph,
          'segment-times',
          segmentTimeBuffer,
          'float32',
          segmentCount * 2
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'halfWidths',
        view: importGraphBuffer(
          coverageGraph,
          'half-widths',
          halfWidthBuffer,
          'float32',
          segmentCount
        ),
        type: 'f32',
        access: 'read'
      },
      {
        name: 'firstCover',
        view: importGraphBuffer(
          coverageGraph,
          'first-cover',
          firstCoverBuffer,
          'float32',
          cellCount
        ),
        type: 'f32',
        access: 'read_write'
      }
    ],
    // For the cell center, the earliest time any segment passes within its half swath. Distances are
    // measured in the local east/north kilometres around the cell, as GPUOutlineGeometry draws them.
    body: `let column = index % COLUMNS;
  let row = index / COLUMNS;
  let cellLongitude = MIN_LONGITUDE + (f32(column) + 0.5) * CELL;
  let cellLatitude = MIN_LATITUDE + (f32(row) + 0.5) * CELL;
  var bestTime = 1.0e30;
  for (var segment = 0u; segment < SEGMENTS; segment = segment + 1u) {
    let halfWidth = halfWidths[halfWidthsOffset + segment];
    if (halfWidth <= 0.0) {
      continue;
    }
    let base = segmentsOffset + segment * 4u;
    let startLongitude = segments[base];
    let startLatitude = segments[base + 1u];
    let endLongitude = segments[base + 2u];
    let endLatitude = segments[base + 3u];
    let southLatitude = min(startLatitude, endLatitude);
    let northLatitude = max(startLatitude, endLatitude);
    let reach = halfWidth / KILOMETERS_PER_DEGREE;
    if (cellLatitude < southLatitude - reach || cellLatitude > northLatitude + reach) {
      continue;
    }
    let timeBase = segmentTimesOffset + segment * 2u;
    let startTime = segmentTimes[timeBase];
    let endTime = segmentTimes[timeBase + 1u];
    if (startTime >= bestTime) {
      continue;
    }
    let nearLatitude = clamp(cellLatitude, southLatitude, northLatitude);
    let eastScale = max(cos(0.5 * (cellLatitude + nearLatitude) * DEGREES_TO_RADIANS), 0.02) * KILOMETERS_PER_DEGREE;
    var deltaStart = startLongitude - cellLongitude;
    var deltaEnd = endLongitude - cellLongitude;
    deltaStart = deltaStart - 360.0 * round(deltaStart / 360.0);
    deltaEnd = deltaEnd - 360.0 * round(deltaEnd / 360.0);
    let fromCell = vec2<f32>(deltaStart * eastScale, (startLatitude - cellLatitude) * KILOMETERS_PER_DEGREE);
    let toCell = vec2<f32>(deltaEnd * eastScale, (endLatitude - cellLatitude) * KILOMETERS_PER_DEGREE);
    let along = toCell - fromCell;
    let lengthSquared = dot(along, along);
    var fraction = 0.0;
    if (lengthSquared > 1.0e-9) {
      fraction = clamp(-dot(fromCell, along) / lengthSquared, 0.0, 1.0);
    }
    if (length(fromCell + along * fraction) <= halfWidth) {
      bestTime = min(bestTime, startTime + fraction * (endTime - startTime));
    }
  }
  firstCover[firstCoverOffset + index] = select(-1.0, bestTime, bestTime < 1.0e29);`
  });
  const coverageCompiled = resources.track(coverageGraph.compile());

  // ---- Selection and widths -----------------------------------------------------------------------
  const nominalWidthOf = (satellite: number) => infoOf(satellite).swathKm;

  function isSelected(satellite: number, selection: string): boolean {
    const info = infoOf(satellite);
    if (info.group !== 4) return false;
    if (selection === 'all') return true;
    if (selection === 'narrow') return info.swathKm < 300;
    if (selection === 'wide') return info.swathKm > 1000;
    return info.sensor === selection;
  }

  function widthOf(satellite: number): number {
    return ctx.options.swathMode === 'custom' ? ctx.options.customWidth : nominalWidthOf(satellite);
  }

  let selectedCount = 0;
  function writeSelection(): void {
    const selection = ctx.options.satellites;
    const visible = new Uint32Array(tracks.satelliteCount);
    selectedCount = 0;
    for (let satellite = 0; satellite < tracks.satelliteCount; satellite++) {
      visible[satellite] = isSelected(satellite, selection) ? 1 : 0;
      selectedCount += visible[satellite];
    }
    satelliteVisible.write(visible);
    const halfWidths = new Float32Array(segmentCount);
    for (let segment = 0; segment < segmentCount; segment++) {
      const satellite = segmentSatellite[segment];
      halfWidths[segment] = visible[satellite] ? widthOf(satellite) / 2 : 0;
    }
    halfWidthBuffer.write(halfWidths);
    coverageDirty = true;
  }

  let coverageDirty = true;
  let customDirty = true;
  let nominalDone = false;
  let result: {
    firstCover: Float32Array;
    cumulative: Float64Array;
    covered: number;
  } | null = null;

  // Area weight of each cell row: cos(latitude), the area of a longitude/latitude cell.
  const rowWeights = new Float64Array(COVERAGE_GRID.rows);
  let totalWeight = 0;
  for (let row = 0; row < COVERAGE_GRID.rows; row++) {
    const latitude = COVERAGE_GRID.minLatitude + (row + 0.5) * COVERAGE_GRID.cell;
    rowWeights[row] = Math.cos((latitude * Math.PI) / 180);
    totalWeight += rowWeights[row] * COVERAGE_GRID.columns;
  }
  const binCount = Math.ceil(tracks.durationSeconds / TIME_BIN_SECONDS);

  const reader = new SummaryReader(
    resources,
    'first-cover',
    [{buffer: firstCoverBuffer, size: cellCount * 4}],
    bytes => {
      if (destroyed) return;
      const firstCover = new Float32Array(bytes);
      const perBin = new Float64Array(binCount + 1);
      let covered = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        const time = firstCover[cell];
        if (time < 0) continue;
        const weight = rowWeights[Math.floor(cell / COVERAGE_GRID.columns)];
        perBin[Math.min(binCount, Math.floor(time / TIME_BIN_SECONDS))] += weight;
        covered += weight;
      }
      const cumulative = new Float64Array(binCount + 1);
      let running = 0;
      for (let bin = 0; bin <= binCount; bin++) {
        running += perBin[bin];
        cumulative[bin] = running / totalWeight;
      }
      result = {firstCover, cumulative, covered: covered / totalWeight};
      describeResult();
    }
  );

  function timeToFraction(target: number): string {
    if (!result) return '...';
    for (let bin = 0; bin <= binCount; bin++) {
      if (result.cumulative[bin] >= target) {
        const seconds = (bin + 1) * TIME_BIN_SECONDS;
        return `${Math.floor(seconds / 3600)} h ${String(Math.round((seconds % 3600) / 60)).padStart(2, '0')} min`;
      }
    }
    return 'not within 3 h';
  }

  function describeResult(): void {
    if (!result) return;
    const hours = Array.from({length: binCount + 1}, (_, bin) => (bin * TIME_BIN_SECONDS) / 3600);
    const percent = Float64Array.from(result.cumulative, value => value * 100);
    ctx.setChart('coverageChart', {
      kind: 'line',
      xLabel: 'hours since 00:00 UTC',
      yLabel: '% of Earth area covered',
      xDomain: [0, tracks.durationSeconds / 3600],
      yDomain: [0, 100],
      height: 130,
      series: [{label: 'covered', x: hours, y: percent, area: true}],
      markers: [{x: clock.time / 3600, label: 'now'}],
      formatX: value => value.toFixed(1),
      formatY: value => `${Math.round(value)}`,
      guides: [
        {y: 50, label: '50%'},
        {y: 90, label: '90%'}
      ],
      description:
        'Share of the Earth between 85 S and 85 N inside at least one swath, against time. Cells are 1 degree and weighted by area.'
    });
    ctx.setReadout(
      'covered3h',
      `${(result.covered * 100).toFixed(1)}% of the Earth (85 S to 85 N)`
    );
    ctx.setReadout('t50', timeToFraction(0.5));
    ctx.setReadout('t90', timeToFraction(0.9));
    ctx.setReadout('t99', timeToFraction(0.99));
    updateNow();
  }

  function updateNow(): void {
    if (!result) return;
    const bin = Math.min(binCount, Math.floor(clock.time / TIME_BIN_SECONDS));
    ctx.setReadout('coveredNow', `${(result.cumulative[bin] * 100).toFixed(1)}%`);
  }

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, tracks.durationSeconds], rate: 1, step: 30}
  );
  writeSelection();
  ctx.setReadout(
    'satelliteCount',
    `${selectedCount} satellites, ${formatCount(segmentCount)} track segments`
  );
  let lastChartBin = -1;

  return {
    getCompiledGraphs: () => [nominalCompiled, customCompiled, coverageCompiled],

    setOption(id) {
      switch (id) {
        case 'satellites':
        case 'swathMode':
          writeSelection();
          customDirty = true;
          ctx.setReadout(
            'satelliteCount',
            `${selectedCount} satellites, ${formatCount(segmentCount)} track segments`
          );
          ctx.requestLayers();
          break;
        case 'customWidth':
          customParameters.write(
            getGPUOutlineGeometryParameterValues({distance: (ctx.options.customWidth * 1000) / 2})
          );
          customDirty = true;
          writeSelection();
          ctx.requestLayers();
          break;
        case 'time':
        case 'play':
        case 'speed':
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
      const playhead = clock.advance(frame);
      ctx.setReadout('clock', formatSatelliteClock(tracks, playhead));
      if (!nominalDone) {
        nominalCompiled.encode(commandEncoder, {parameters: undefined});
        nominalDone = true;
      }
      if (customDirty && ctx.options.swathMode === 'custom') {
        customCompiled.encode(commandEncoder, {parameters: undefined});
        customDirty = false;
      }
      if (coverageDirty) {
        coverageCompiled.encode(commandEncoder, {parameters: undefined});
        coverageDirty = false;
        reader.request(commandEncoder);
      } else {
        reader.flush(commandEncoder);
      }
      const bin = Math.floor(playhead / TIME_BIN_SECONDS);
      if (bin !== lastChartBin) {
        lastChartBin = bin;
        describeResult();
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showCoverage) {
        layers.push(
          new SatelliteCellLayer({
            id: 'satellite-coverage-cells',
            values: firstCoverBuffer,
            gridSize: [COVERAGE_GRID.columns, COVERAGE_GRID.rows],
            grid: [
              COVERAGE_GRID.minLongitude,
              COVERAGE_GRID.minLatitude,
              COVERAGE_GRID.cell,
              COVERAGE_GRID.cell
            ],
            colormap: options.ramp,
            valueRange: [0, tracks.durationSeconds],
            discardAtOrBelow: -0.5,
            discardAbove: clock.time,
            opacity: 0.8
          })
        );
      }
      if (options.showSwaths) {
        const color = SATELLITE_GROUP_COLORS[4];
        const swathColor: [number, number, number, number] = dark
          ? [color[0], color[1], color[2], 255]
          : [120, 60, 200, 255];
        const drawn =
          options.swathMode === 'custom'
            ? [
                {
                  id: 'custom',
                  subset: all,
                  outline: customBuffers.outline,
                  times: customBuffers.times,
                  satellites: customBuffers.satellites
                }
              ]
            : classes.map(item => ({
                id: item.sensor,
                subset: item.subset,
                outline: item.outline,
                times: item.times,
                satellites: item.satellites
              }));
        for (const item of drawn) {
          layers.push(
            new SatelliteSwathLayer({
              id: `satellite-swath-${item.id}`,
              positions: item.outline,
              times: item.times,
              satellites: item.satellites,
              visibleSatellites: satelliteVisible,
              rowsPerInput,
              vertexCount: item.subset.vertexCount * rowsPerInput,
              timeLimit: clock.time,
              color: swathColor,
              opacity: options.swathOpacity
            })
          );
        }
      }
      return layers;
    },

    getTooltip(event) {
      if (!result || !event.coordinate) return null;
      const [longitude, latitude] = event.coordinate;
      const column = Math.floor((longitude - COVERAGE_GRID.minLongitude) / COVERAGE_GRID.cell);
      const row = Math.floor((latitude - COVERAGE_GRID.minLatitude) / COVERAGE_GRID.cell);
      if (column < 0 || column >= COVERAGE_GRID.columns || row < 0 || row >= COVERAGE_GRID.rows) {
        return null;
      }
      const time = result.firstCover[row * COVERAGE_GRID.columns + column];
      if (time < 0) return 'not reached by any selected swath in 3 h';
      return `first covered at ${formatSatelliteClock(tracks, time)}`;
    },

    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}

export type {SatelliteInfo};
