// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUKMeans} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import {SatelliteMarkerLayer, SatelliteTrailLayer} from './satellite-layers';
import {createSatellitePlaybackCore} from './satellite-playback-core';
import {
  formatSatelliteClock,
  loadSatelliteTracks,
  SATELLITE_DATASET_ID,
  type SatelliteTrackSet
} from './satellite-tracks';

/** Option state of the constellation scene. */
export type SatelliteConstellationsOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  family: 'both' | 'starlink' | 'gps' | 'all';
  k: number;
  seed: string;
  altitudeWeight: number;
  altitudeDisplay: 'compressed' | 'linear';
  altitudeScale: number;
  showStems: boolean;
  markerSize: number;
  showTrails: boolean;
  trailMinutes: number;
};

/** One color per shell, up to eight (Okabe-Ito plus two), readable on light and dark basemaps. */
export const SHELL_COLORS: readonly (readonly [number, number, number, number])[] = [
  [86, 180, 233, 255],
  [240, 150, 30, 255],
  [0, 190, 140, 255],
  [220, 80, 70, 255],
  [200, 140, 245, 255],
  [240, 228, 66, 255],
  [255, 130, 190, 255],
  [170, 175, 185, 255]
];

const ITERATIONS = 24;
const HIDDEN = 0xffffffff;
const FAMILY_GROUPS: Record<SatelliteConstellationsOptions['family'], readonly number[] | null> = {
  both: [1, 2],
  starlink: [1],
  gps: [2],
  all: null
};

type Shell = {
  /** Rank by mean altitude, the palette row. */
  rank: number;
  inclination: number;
  altitudeKm: number;
  size: number;
};

type KMeansGraph = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  labels: Buffer;
  centers: Buffer;
  sizes: Buffer;
  convergenceBuffer: Buffer;
  convergenceResult: readonly [number, number] | null;
  reader: SummaryReader;
  k: number;
  satellites: Uint32Array;
  features: Buffer;
  dirty: boolean;
  shells: Shell[] | null;
  labelBySatellite: Map<number, number> | null;
};

/**
 * Constellation geometry: `GPUKMeans` groups the satellites of a family by orbital inclination and
 * altitude (on the GPU, deterministic per seed), and the playhead scene colors every satellite by its
 * shell. The k-means graphs depend on the family, k and seed (compile time), so each combination
 * compiles on first use; the altitude weight only rewrites the feature buffer.
 */
export async function createSatelliteConstellations(
  ctx: SceneContext<SatelliteConstellationsOptions>
): Promise<SceneInstance<SatelliteConstellationsOptions>> {
  const tracks = loadSatelliteTracks(ctx.datasets.get(SATELLITE_DATASET_ID));
  const resources = new SpatialAnalysisResources(ctx.device, 'satellite-constellations');
  let destroyed = false;

  const colorRows = new Uint32Array(tracks.trackCount).fill(HIDDEN);
  const core = createSatellitePlaybackCore(resources, tracks, colorRows, () => {
    if (destroyed) return;
    const snapshot = core.getSnapshot();
    if (snapshot) ctx.setReadout('trailSegments', snapshot.trailCount);
  });
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, tracks.durationSeconds], rate: 1, step: 30}
  );

  const familySatellites = (family: SatelliteConstellationsOptions['family']): Uint32Array => {
    const groups = FAMILY_GROUPS[family];
    const list: number[] = [];
    tracks.satellites.forEach((satellite, index) => {
      if (!groups || groups.includes(satellite.group)) list.push(index);
    });
    return Uint32Array.from(list);
  };

  function writeFeatures(graph: KMeansGraph): void {
    const weight = ctx.options.altitudeWeight;
    const features = new Float32Array(graph.satellites.length * 2);
    graph.satellites.forEach((satellite, row) => {
      const info = tracks.satellites[satellite];
      features[row * 2] = info.inclination;
      features[row * 2 + 1] = weight * Math.log10(Math.max(info.meanAltitudeKm, 1));
    });
    graph.features.write(features);
  }

  // ---- k-means graphs, compiled on first use ----------------------------------------------------
  const graphs = new Map<string, KMeansGraph>();
  let active: KMeansGraph;

  function getGraph(): KMeansGraph {
    const {family, k, seed} = ctx.options;
    const key = `${family}-${k}-${seed}`;
    const existing = graphs.get(key);
    if (existing) return existing;
    const satellites = familySatellites(family);
    const count = satellites.length;
    const effectiveK = Math.min(k, count);
    const features = resources.createBuffer(`${key}-features`, count * 8);
    const labels = resources.createBuffer(`${key}-labels`, count * 4);
    const centers = resources.createBuffer(`${key}-centers`, effectiveK * 8);
    const sizes = resources.createBuffer(`${key}-sizes`, effectiveK * 4);
    const convergence = resources.createBuffer(`${key}-convergence`, 8);
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `satellite-kmeans-${key}`});
    graph.add(
      new GPUKMeans({
        id: 'shells',
        positions: importGraphBuffer(graph, 'features', features, 'float32x2', count),
        k: effectiveK,
        iterations: ITERATIONS,
        initialization: 'kmeans++',
        seed: Number(seed),
        labels: importGraphBuffer(graph, 'labels', labels, 'uint32', count),
        centers: importGraphBuffer(graph, 'centers', centers, 'float32x2', effectiveK),
        sizes: importGraphBuffer(graph, 'sizes', sizes, 'uint32', effectiveK),
        convergence: importGraphBuffer(graph, 'convergence', convergence, 'uint32', 2)
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      `kmeans-${key}`,
      [
        {buffer: convergence, size: 8},
        {buffer: sizes, size: effectiveK * 4},
        {buffer: centers, size: effectiveK * 8},
        {buffer: labels, size: count * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        const sizeStart = 2;
        const centerStart = sizeStart + effectiveK;
        const labelStart = centerStart + effectiveK * 2;
        const weight = ctx.options.altitudeWeight;
        const shellList: Shell[] = [];
        for (let cluster = 0; cluster < effectiveK; cluster++) {
          shellList.push({
            rank: cluster,
            inclination: floats[centerStart + cluster * 2],
            altitudeKm: 10 ** (floats[centerStart + cluster * 2 + 1] / weight),
            size: words[sizeStart + cluster]
          });
        }
        // Palette order: lowest shell first, so the same shell keeps its color across settings.
        const order = shellList
          .map((shell, cluster) => ({shell, cluster}))
          .filter(entry => entry.shell.size > 0)
          .sort(
            (a, b) =>
              a.shell.altitudeKm - b.shell.altitudeKm || a.shell.inclination - b.shell.inclination
          );
        const rankOfCluster = new Map<number, number>();
        order.forEach((entry, rank) => {
          entry.shell.rank = rank;
          rankOfCluster.set(entry.cluster, rank);
        });
        const labelBySatellite = new Map<number, number>();
        satellites.forEach((satellite, row) => {
          labelBySatellite.set(satellite, rankOfCluster.get(words[labelStart + row]) ?? 0);
        });
        graph_.shells = order.map(entry => entry.shell);
        graph_.labelBySatellite = labelBySatellite;
        graph_.convergenceResult = [words[0], words[1]];
        if (graph_ === active) applyShells();
      }
    );
    const graph_: KMeansGraph = {
      key,
      compiled,
      labels,
      centers,
      sizes,
      convergenceBuffer: convergence,
      convergenceResult: null,
      reader,
      k: effectiveK,
      satellites,
      features,
      dirty: true,
      shells: null,
      labelBySatellite: null
    };
    graphs.set(key, graph_);
    writeFeatures(graph_);
    return graph_;
  }

  function describeShell(shell: Shell): string {
    return `${shell.inclination.toFixed(1)}°, ${formatCount(Math.round(shell.altitudeKm))} km`;
  }

  function applyShells(): void {
    const graph = active;
    if (!graph.shells || !graph.labelBySatellite) return;
    const labelBySatellite = graph.labelBySatellite;
    for (let track = 0; track < tracks.trackCount; track++) {
      const label = labelBySatellite.get(tracks.satelliteIndex[track]);
      colorRows[track] = label === undefined ? HIDDEN : label;
    }
    core.colorsBuffer.write(colorRows);
    const options = ctx.options;
    const groups = FAMILY_GROUPS[options.family];
    core.setTrackFilter(track => (groups ? groups.includes(tracks.group[track]) : true));
    ctx.setLegendData(
      'shells',
      graph.shells.map(shell => ({
        color: SHELL_COLORS[shell.rank % SHELL_COLORS.length],
        label: `${describeShell(shell)} (${shell.size})`
      }))
    );
    ctx.setReadout(
      'shells',
      graph.shells
        .map(
          (shell, index) =>
            `${index + 1}. ${describeShell(shell)}: ${formatCount(shell.size)} satellites`
        )
        .join('\n')
    );
    const convergence = graph.convergenceResult;
    if (convergence) {
      ctx.setReadout(
        'converged',
        convergence[1]
          ? `converged after ${convergence[0]} of ${ITERATIONS} iterations`
          : `not converged in ${ITERATIONS} iterations`
      );
    }
    ctx.setChart('shellChart', {
      kind: 'bars',
      values: graph.shells.map(shell => shell.size),
      labels: graph.shells.map(
        shell => `${shell.inclination.toFixed(0)}°/${Math.round(shell.altitudeKm)}`
      ),
      height: 120,
      xLabel: 'inclination (degrees) / altitude (km)',
      yLabel: 'satellites',
      formatY: value => `${Math.round(value)}`,
      description:
        'Satellites in each shell found by GPUKMeans, labeled by the center of the shell.'
    });
    const inclinations = Float32Array.from(
      graph.satellites,
      satellite => tracks.satellites[satellite].inclination
    );
    ctx.setChart(
      'inclinationChart',
      histogramChart(binValues(inclinations, 0, 100, 50), 0, 100, {
        xLabel: 'inclination (degrees)',
        yLabel: 'satellites',
        height: 110,
        markers: graph.shells.map(shell => ({x: shell.inclination})),
        formatX: value => `${Math.round(value)}`,
        description:
          'Histogram of orbital inclination of the family in 2-degree bins; rules mark the k-means centers.'
      })
    );
    ctx.requestLayers();
  }

  function activate(): void {
    active = getGraph();
    writeFeatures(active);
    active.dirty = true;
    if (active.shells) applyShells();
  }
  activate();

  ctx.setReadout('satellites', `${formatCount(tracks.satelliteCount)} satellites in the dataset`);

  return {
    getCompiledGraphs: () => [core.playheadCompiled, core.trailCompiled, active.compiled],

    setOption(id) {
      switch (id) {
        case 'family':
        case 'k':
        case 'seed':
          activate();
          break;
        case 'altitudeWeight':
          writeFeatures(active);
          active.dirty = true;
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
      ctx.setReadout('clock', formatSatelliteClock(tracks, playhead));
      if (active.dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        active.dirty = false;
        active.reader.request(commandEncoder);
      } else {
        active.reader.flush(commandEncoder);
      }
      core.encodePlayhead(commandEncoder, playhead);
      if (options.showTrails) {
        const trailSeconds = options.trailMinutes * 60;
        core.encodeTrails(commandEncoder, playhead, trailSeconds, trailSeconds);
      }
      if (frame.frameIndex % 30 === 0) core.reader.markStale();
      core.reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showTrails) {
        layers.push(
          new SatelliteTrailLayer({
            id: 'constellation-trails',
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
            palette: SHELL_COLORS,
            widthPixels: 1.4,
            opacity: 0.55,
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
        palette: SHELL_COLORS,
        altitudeScale: options.altitudeScale,
        altitudeDisplay: options.altitudeDisplay
      };
      if (options.showStems) {
        layers.push(
          new SatelliteMarkerLayer({
            ...markerProps,
            id: 'constellation-stems',
            stems: true,
            stemWidthPixels: 1,
            opacity: 0.8
          })
        );
      }
      layers.push(
        new SatelliteMarkerLayer({
          ...markerProps,
          id: 'constellation-markers',
          sizePixels: options.markerSize,
          outlineColor: dark ? [8, 10, 16, 235] : [20, 24, 32, 215]
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      core.stop();
      for (const graph of graphs.values()) graph.reader.stop();
      resources.destroy();
    }
  };
}

export type {SatelliteTrackSet};
