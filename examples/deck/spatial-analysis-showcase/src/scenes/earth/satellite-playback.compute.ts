// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {GPU_TRAJECTORY_PLAYHEAD_STATUS} from '@luma.gl/experimental/gpu-spatial-analysis';
import {createPlaybackClock} from '../../engine/playback';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {SatelliteMarkerLayer, SatelliteTrailLayer} from './satellite-layers';
import {
  createSatellitePlaybackCore,
  type SatellitePlaybackSnapshot
} from './satellite-playback-core';
import {
  formatElapsed,
  formatSatelliteClock,
  loadSatelliteTracks,
  SATELLITE_DATASET_ID,
  SATELLITE_GROUP_COLORS,
  SATELLITE_GROUPS
} from './satellite-tracks';

/** Option state of the satellite playback scene. */
export type SatellitePlaybackOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  group: string;
  colorBy: 'group' | 'altitude';
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  altitudeDisplay: 'compressed' | 'linear';
  altitudeScale: number;
  showStems: boolean;
  markerSize: number;
  showTrails: boolean;
  trailMinutes: number;
  tailFade: number;
};

/** Altitude bands of the chart: `[low, high)` in km. */
export const ALTITUDE_BANDS_KM = [0, 450, 600, 1000, 2000, 10000, 100000] as const;
export const ALTITUDE_BAND_LABELS = ['<450', '450-600', '600-1k', '1-2k', '2-10k', '>10k'] as const;
const NO_TRACK = 0xffffffff;

/** Mirrors the shader's altitude display (meters in, meters out). */
export function getDisplayAltitude(
  altitude: number,
  scale: number,
  display: 'compressed' | 'linear'
): number {
  if (display === 'compressed')
    return scale * 400000 * Math.log2(1 + Math.max(altitude, 0) / 400000);
  return scale * altitude;
}

/**
 * Satellite playback: `GPUTrajectoryPlayhead` interpolates every ground track at the clock and
 * writes the altitude as the elevation of each marker; `GPUTimeWindowFilter` selects the trail
 * segments behind it. Both read only parameter buffers per frame.
 */
export async function createSatellitePlayback(
  ctx: SceneContext<SatellitePlaybackOptions>
): Promise<SceneInstance<SatellitePlaybackOptions>> {
  const tracks = loadSatelliteTracks(ctx.datasets.get(SATELLITE_DATASET_ID));
  const resources = new SpatialAnalysisResources(ctx.device, 'satellite-playback');
  let destroyed = false;
  let selectedSatellite = NO_TRACK;

  const core = createSatellitePlaybackCore(resources, tracks, tracks.group, snapshot => {
    if (destroyed) return;
    describeSnapshot(snapshot);
  });
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, tracks.durationSeconds], rate: 1, step: 30}
  );

  const groupFilter = (value: string) => (value === 'all' ? null : Number(value));
  core.setGroupFilter(groupFilter(ctx.options.group));
  ctx.setReadout(
    'satellites',
    `${formatCount(tracks.satelliteCount)} satellites, ${formatCount(tracks.trackCount)} tracks`
  );
  ctx.setReadout('samples', `${formatCount(tracks.vertexCount)} SGP4 positions`);
  let oldest = 0;
  for (const satellite of tracks.satellites) oldest = Math.max(oldest, satellite.tleAgeDays);
  ctx.setReadout('oldestElements', `${oldest.toFixed(1)} days`);

  function inFilter(track: number): boolean {
    const filter = groupFilter(ctx.options.group);
    return filter === null || tracks.group[track] === filter;
  }

  function describeSnapshot(snapshot: SatellitePlaybackSnapshot): void {
    const counts = new Float64Array(ALTITUDE_BAND_LABELS.length);
    let active = 0;
    let highest = 0;
    let lowest = Infinity;
    for (let track = 0; track < tracks.trackCount; track++) {
      if (snapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active) continue;
      if (!inFilter(track)) continue;
      active++;
      const km = snapshot.elevations[track] / 1000;
      highest = Math.max(highest, km);
      lowest = Math.min(lowest, km);
      for (let band = 0; band < counts.length; band++) {
        if (km >= ALTITUDE_BANDS_KM[band] && km < ALTITUDE_BANDS_KM[band + 1]) {
          counts[band]++;
          break;
        }
      }
    }
    ctx.setReadout('active', active);
    ctx.setReadout('trailSegments', snapshot.trailCount);
    ctx.setReadout('highest', active ? `${formatCount(Math.round(highest))} km` : 'none');
    ctx.setReadout('lowest', active ? `${formatCount(Math.round(lowest))} km` : 'none');
    let peak = 0;
    for (let band = 1; band < counts.length; band++) if (counts[band] > counts[peak]) peak = band;
    ctx.setChart('altitudeChart', {
      kind: 'bars',
      values: counts,
      labels: ALTITUDE_BAND_LABELS,
      highlight: active ? [peak] : [],
      height: 120,
      xLabel: 'altitude band (km)',
      yLabel: 'satellites',
      formatY: value => `${Math.round(value)}`,
      description:
        'Satellites above the ground track at the playhead, by altitude band. The tallest bar is highlighted.'
    });
    describeSelection();
  }

  function describeSatellite(satellite: number): string {
    const info = tracks.satellites[satellite];
    return `${info.name} (${SATELLITE_GROUPS[info.group]}, ${info.inclination.toFixed(1)} deg, ${info.periodMinutes.toFixed(0)} min orbit, elements ${info.tleAgeDays.toFixed(1)} d old)`;
  }

  function describeSelection(): void {
    if (selectedSatellite === NO_TRACK) {
      ctx.setReadout('selected', 'click a satellite');
      return;
    }
    const snapshot = core.getSnapshot();
    let altitude = '';
    if (snapshot) {
      for (let track = 0; track < tracks.trackCount; track++) {
        if (
          tracks.satelliteIndex[track] === selectedSatellite &&
          snapshot.status[track] === GPU_TRAJECTORY_PLAYHEAD_STATUS.active
        ) {
          altitude = ` at ${Math.round(snapshot.elevations[track] / 1000)} km`;
          break;
        }
      }
    }
    ctx.setReadout('selected', `${describeSatellite(selectedSatellite)}${altitude}`);
  }
  describeSelection();

  /** Nearest active satellite within 14 CSS pixels of the pointer, or -1. */
  function pickSatellite(pixel: readonly [number, number]): number {
    const viewport = ctx.getViewport();
    const snapshot = core.getSnapshot();
    if (!viewport || !snapshot) return -1;
    const options = ctx.options;
    let best = -1;
    let bestDistance = 14 * 14;
    for (let track = 0; track < tracks.trackCount; track++) {
      if (snapshot.status[track] !== GPU_TRAJECTORY_PLAYHEAD_STATUS.active || !inFilter(track)) {
        continue;
      }
      const height = getDisplayAltitude(
        snapshot.elevations[track],
        options.altitudeScale,
        options.altitudeDisplay
      );
      const [x, y] = viewport.project([
        snapshot.positions[track * 2],
        snapshot.positions[track * 2 + 1],
        height
      ]);
      const squared = (x - pixel[0]) ** 2 + (y - pixel[1]) ** 2;
      if (squared < bestDistance) {
        bestDistance = squared;
        best = track;
      }
    }
    return best;
  }

  return {
    getCompiledGraphs: () => [core.playheadCompiled, core.trailCompiled],

    setOption(id, _value, state) {
      switch (id) {
        case 'group':
          core.setGroupFilter(groupFilter(state.group));
          core.reader.markStale();
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
      const options = ctx.options;
      const playhead = clock.advance(frame);
      ctx.setReadout(
        'clock',
        `${formatSatelliteClock(tracks, playhead)} (+${formatElapsed(playhead)})`
      );
      core.encodePlayhead(commandEncoder, playhead);
      if (options.showTrails) {
        const trailSeconds = options.trailMinutes * 60;
        core.encodeTrails(commandEncoder, playhead, trailSeconds, trailSeconds * options.tailFade);
      }
      if (frame.frameIndex % 10 === 0 || clock.moved) core.reader.markStale();
      core.reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const filter = groupFilter(options.group);
      const colorMode = options.colorBy === 'altitude' ? 'altitude' : 'palette';
      const layers: Layer[] = [];
      if (options.showTrails) {
        layers.push(
          new SatelliteTrailLayer({
            id: 'satellite-trails',
            positions: core.positionsBuffer,
            altitudes: core.altitudesBuffer,
            segmentEnds: core.segmentEndsBuffer,
            segmentTracks: core.segmentTracksBuffer,
            ids: core.trailIds,
            weights: core.fadeWeights,
            clipFractions: core.clipFractions,
            drawCommands: core.trailDraw,
            colors: core.colorsBuffer,
            groups: core.groupsBuffer,
            palette: SATELLITE_GROUP_COLORS,
            colorMode,
            ramp: options.ramp as RampName,
            widthPixels: 1.6,
            opacity: 0.6,
            altitudeScale: options.altitudeScale,
            altitudeDisplay: options.altitudeDisplay
          })
        );
      }
      const markerProps = {
        ids: core.activeIds,
        positions: core.currentPositions,
        elevations: core.currentElevations,
        colors: core.colorsBuffer,
        groups: core.groupsBuffer,
        satellites: core.satellitesBuffer,
        drawCommands: core.markerDraw,
        palette: SATELLITE_GROUP_COLORS,
        colorMode: colorMode as 'palette' | 'altitude',
        ramp: options.ramp as RampName,
        altitudeScale: options.altitudeScale,
        altitudeDisplay: options.altitudeDisplay,
        groupFilter: filter
      };
      if (options.showStems) {
        layers.push(
          new SatelliteMarkerLayer({
            ...markerProps,
            id: 'satellite-stems',
            stems: true,
            stemWidthPixels: 1,
            opacity: 0.8
          })
        );
      }
      layers.push(
        new SatelliteMarkerLayer({
          ...markerProps,
          id: 'satellite-markers',
          sizePixels: options.markerSize,
          selectedSatellite: selectedSatellite === NO_TRACK ? null : selectedSatellite,
          outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
        })
      );
      return layers;
    },

    getTooltip(event) {
      const track = pickSatellite(event.pixel);
      if (track < 0) return null;
      const snapshot = core.getSnapshot();
      const altitude = snapshot ? `, ${Math.round(snapshot.elevations[track] / 1000)} km up` : '';
      return `${describeSatellite(tracks.satelliteIndex[track])}${altitude}`;
    },

    onClick(event) {
      const track = pickSatellite(event.pixel);
      const satellite = track < 0 ? NO_TRACK : tracks.satelliteIndex[track];
      selectedSatellite = satellite === selectedSatellite ? NO_TRACK : satellite;
      describeSelection();
      ctx.requestLayers();
      return track >= 0;
    },

    destroy() {
      destroyed = true;
      core.stop();
      resources.destroy();
    }
  };
}
