// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH,
  GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH,
  GPUParticleAdvection
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPULineDensityParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPUGeodesicPairs,
  GPULineDensity
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {
  FIELD,
  FIELD_CELLS,
  formatDriftDate,
  formatKilometers,
  formatPosition,
  getFieldCell,
  getGreatCircleMeters,
  getSortedQuantile,
  LEAD_DAYS,
  LEAD_SLOTS,
  loadCurrentField,
  loadDrifters,
  OCEAN_REGIONS
} from './ocean-drifters-data';

/** Option state of the ocean-drifters-vs-model scene. */
export type OceanDriftersVsModelOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  speedScale: number;
  coast: 'remove' | 'stick';
  distanceModel: 'sphere' | 'wgs84' | 'rhumb';
  releaseSet: 'all' | 'deployed' | 'atSea';
  region: string;
  showReleases: boolean;
  showLinks: boolean;
  showReal: boolean;
  showVirtual: boolean;
  showTrails: boolean;
  pointSize: number;
  ramp: 'viridis' | 'magma' | 'inferno' | 'cividis';
  sepMax: number;
  background: 'none' | 'speed' | 'realDensity' | 'modelDensity' | 'densityDiff';
  speedMax: number;
  densityMax: number;
  densityMinKm: number;
  backgroundOpacity: number;
};

/** Where lost particles are parked (outside the field, so they stay lost). */
const PARKING: readonly [number, number] = [0, 89.95];
/** Scales swept by the "Find the best speed scale" button. */
const SWEEP_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3];
const CHART_INTERVAL_SECONDS = 0.25;

const REAL_COLORS = {dark: [100, 180, 255, 235], light: [10, 84, 196, 235]} as const;
const MODEL_COLORS = {dark: [255, 168, 64, 235], light: [200, 92, 0, 235]} as const;

type Viewer = <Format extends GPUVectorFormat>(
  name: string,
  buffer: Buffer,
  format: Format,
  length?: number
) => GraphDataView<Format>;

/** Imports each buffer once per graph and returns typed views of it. */
function createViewer(graph: GPUCommandGraph<void>): Viewer {
  const handles = new Map<Buffer, ReturnType<GPUCommandGraph<void>['importBuffer']>>();
  return (name, buffer, format, length) => {
    let handle = handles.get(buffer);
    if (!handle) {
      handle = graph.importBuffer(
        {id: name, byteLength: buffer.byteLength, usage: buffer.usage},
        buffer
      );
      handles.set(buffer, handle);
    }
    const rowBytes = format === 'float32x2' ? 8 : 4;
    return graph.createDataView(handle, {
      format,
      length: length ?? Math.floor(buffer.byteLength / rowBytes)
    });
  };
}

/** Statistics per lead day over the selected drifters. */
type LeadStatistics = {
  count: Float64Array;
  median: Float64Array;
  p10: Float64Array;
  p25: Float64Array;
  p75: Float64Array;
  p90: Float64Array;
  stayMedian: Float64Array;
  lostFraction: Float64Array;
};

/** One finished model run read back to the CPU. */
type ModelRun = {
  distances: Float32Array;
  alive: Uint32Array;
  realLengths: Float32Array;
  modelLengths: Float32Array;
  scale: number;
};

/**
 * Ocean drifters against the ECCO model. A virtual particle is released at the first fix of every
 * 2017 drifter and advected for 30 days through the annual-mean ECCO current field by
 * `GPUParticleAdvection` (one RK2 step per day, a snapshot of every particle per day); then
 * `GPUGeodesicPairs` measures the distance between each virtual particle and the real drifter on
 * every lead day, and `GPULineDensity` turns real and virtual tracks into per-cell track length so
 * the two can be compared on a map. Everything per frame (the lead time shown) is a parameter
 * write; the model scale, coast handling and distance model are parameter writes or prebuilt
 * graph switches. Only the run itself (30 submits plus two small readbacks) happens outside
 * `encode`.
 */
export async function createOceanDriftersVsModel(
  ctx: SceneContext<OceanDriftersVsModelOptions>
): Promise<SceneInstance<OceanDriftersVsModelOptions>> {
  const {device} = ctx;
  const drifters = loadDrifters(ctx.datasets.get('poopdeck-drifters'));
  const field = loadCurrentField(ctx.datasets.get('poopdeck-ecco-currents'));
  const count = drifters.releaseCount;
  const slots = LEAD_SLOTS;
  const trackRows = count * slots;
  const segmentRows = count * LEAD_DAYS;
  if (ctx.limits.maxStorageBuffersPerShaderStage < 13) {
    throw new Error('This scene needs 13 storage buffers per shader stage; the GPU grants fewer');
  }
  const resources = new SpatialAnalysisResources(device, 'drifters');
  const drawProps = {coordinateSystem: COORDINATE_SYSTEM.LNGLAT} as const;

  // Releases the model cannot start: no current data in the release cell (polar ice, shelf).
  const releaseValid = new Uint8Array(count);
  let validReleases = 0;
  for (let track = 0; track < count; track++) {
    const cell = getFieldCell(drifters.release[track * 2], drifters.release[track * 2 + 1]);
    if (cell >= 0 && Number.isFinite(field.holes[cell * 2])) {
      releaseValid[track] = 1;
      validReleases++;
    }
  }

  // ---- Static buffers -------------------------------------------------------------------------
  const velocityBuffer = resources.createBuffer('velocity', field.holes);
  const speedBuffer = resources.createBuffer('speed', field.speed);
  const releaseBuffer = resources.createBuffer('release', drifters.release);
  const dailyBuffer = resources.createBuffer('daily', drifters.daily);
  const realSegments = resources.createBuffer('real-segments', drifters.dailySegments);
  const realSegmentWeights = resources.createBuffer('real-segment-weights', segmentRows * 4);
  const virtualSegments = resources.createBuffer('virtual-segments', segmentRows * 16);
  const virtualSegmentWeights = resources.createBuffer('virtual-segment-weights', segmentRows * 4);
  const maskBuffer = resources.createBuffer('mask', count * 4);
  const aliveBuffer = resources.createBuffer('alive', count * 4);
  const firstMonthPositions = resources.createBuffer('first-month', drifters.firstMonthPositions);
  const firstMonthOffsets = resources.createBuffer(
    'first-month-offsets',
    drifters.firstMonthOffsets
  );
  const modelOffsetValues = new Uint32Array(count + 1);
  for (let track = 0; track <= count; track++) modelOffsetValues[track] = track * slots;
  const modelOffsets = resources.createBuffer('model-offsets', modelOffsetValues);

  // ---- Model state ------------------------------------------------------------------------------
  const particlePositions = resources.createBuffer('particle-positions', count * 8);
  const particleAges = resources.createBuffer('particle-ages', count * 4);
  const particleGenerations = resources.createBuffer('particle-generations', count * 4);
  const snapshotBuffer = resources.createBuffer('snapshots', trackRows * 8);
  const virtualTrackBuffer = resources.createBuffer('virtual-track', trackRows * 8);
  const distanceBuffer = resources.createBuffer('distances', trackRows * 4);
  const advectionParameters = resources.createParameterBuffer(
    'advection-parameters',
    'float32',
    GPU_PARTICLE_ADVECTION_PARAMETER_LENGTH
  );
  const advectionWords = resources.createParameterBuffer(
    'advection-words',
    'uint32',
    GPU_PARTICLE_ADVECTION_WORD_PARAMETER_LENGTH
  );

  // ---- Density ----------------------------------------------------------------------------------
  const realLengths = resources.createBuffer('real-lengths', FIELD_CELLS * 4);
  const realDensities = resources.createBuffer('real-densities', FIELD_CELLS * 4);
  const realOverflow = resources.createBuffer('real-overflow', 4);
  const modelLengths = resources.createBuffer('model-lengths', FIELD_CELLS * 4);
  const modelDensities = resources.createBuffer('model-densities', FIELD_CELLS * 4);
  const modelOverflow = resources.createBuffer('model-overflow', 4);
  const differenceBuffer = resources.createBuffer('difference', FIELD_CELLS * 4);
  const densityParameters = resources.createParameterBuffer(
    'density-parameters',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH,
    getGPULineDensityParameterValues({
      minX: FIELD.west,
      minY: FIELD.south,
      cellWidth: FIELD.cellDegrees,
      cellHeight: FIELD.cellDegrees
    })
  );
  const differenceParameters = resources.createParameterBuffer(
    'difference-parameters',
    'float32',
    4
  );

  // ---- Per-frame outputs ------------------------------------------------------------------------
  const leadParameters = resources.createParameterBuffer('lead-parameters', 'float32', 4);
  const virtualNow = resources.createBuffer('virtual-now', count * 8);
  const realNow = resources.createBuffer('real-now', count * 8);
  const separationNow = resources.createBuffer('separation-now', count * 4);
  const pairFlag = resources.createBuffer('pair-flag', count * 4);
  const linkSegments = resources.createBuffer('links', count * 16);
  const headVirtual = resources.createBuffer('head-virtual', count * 16);
  const headReal = resources.createBuffer('head-real', count * 16);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'drifters-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );

  // ---- Graph: advection (one daily step) + snapshot ---------------------------------------------
  const advectionGraph = new GPUCommandGraph<void>(device, {id: 'drifters-advection'});
  {
    const view = createViewer(advectionGraph);
    const positions = view('positions', particlePositions, 'float32x2', count);
    const words = advectionWords.importToGraph(advectionGraph);
    advectionGraph.add(
      new GPUParticleAdvection({
        id: 'advect',
        velocities: view('velocity', velocityBuffer, 'float32x2', FIELD_CELLS),
        fieldWidth: FIELD.width,
        fieldHeight: FIELD.height,
        parameters: advectionParameters.importToGraph(advectionGraph),
        wordParameters: words,
        state: {
          positions,
          ages: view('ages', particleAges, 'uint32', count),
          generations: view('generations', particleGenerations, 'uint32', count)
        }
      })
    );
    // Copies every particle into its (track, day) slot; the frame word is the lead day.
    addKernelPass(advectionGraph, {
      id: 'drifters-snapshot',
      invocationCount: count,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'words', view: words, type: 'u32', access: 'read'},
        {
          name: 'snapshots',
          view: view('snapshots', snapshotBuffer, 'float32x2', trackRows),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let day = min(words[wordsOffset + 1u], ${LEAD_DAYS}u);
  let slot = (index * ${slots}u + day) * 2u;
  snapshots[snapshotsOffset + slot] = positions[positionsOffset + 2u * index];
  snapshots[snapshotsOffset + slot + 1u] = positions[positionsOffset + 2u * index + 1u];`
    });
  }
  const advectionCompiled = resources.track(advectionGraph.compile());

  // ---- Graphs: geodesic pairs, one per distance model (compiled up front, switched by option) ----
  const pairsCompiled = {} as Record<
    OceanDriftersVsModelOptions['distanceModel'],
    CompiledGPUCommandGraph<void>
  >;
  for (const model of ['sphere', 'wgs84', 'rhumb'] as const) {
    const graph = new GPUCommandGraph<void>(device, {id: `drifters-pairs-${model}`});
    const view = createViewer(graph);
    graph.add(
      new GPUGeodesicPairs({
        id: `pairs-${model}`,
        origins: view('virtual-track', virtualTrackBuffer, 'float32x2', trackRows),
        targets: view('daily', dailyBuffer, 'float32x2', trackRows),
        model,
        output: {distances: view('distances', distanceBuffer, 'float32', trackRows)}
      })
    );
    pairsCompiled[model] = resources.track(graph.compile());
  }

  // ---- Graph: line density of real and virtual tracks ---------------------------------------------
  const densityGraph = new GPUCommandGraph<void>(device, {id: 'drifters-density'});
  {
    const view = createViewer(densityGraph);
    const parameters = densityParameters.importToGraph(densityGraph);
    densityGraph.add(
      new GPULineDensity({
        id: 'real-density',
        positions: view('first-month', firstMonthPositions, 'float32x2'),
        pathOffsets: view('first-month-offsets', firstMonthOffsets, 'uint32'),
        columns: FIELD.width,
        rows: FIELD.height,
        coordinateSystem: 'spherical',
        parameters,
        output: {
          lengths: view('real-lengths', realLengths, 'float32', FIELD_CELLS),
          densities: view('real-densities', realDensities, 'float32', FIELD_CELLS),
          overflow: view('real-overflow', realOverflow, 'uint32', 1)
        }
      })
    );
    densityGraph.add(
      new GPULineDensity({
        id: 'model-density',
        positions: view('virtual-track', virtualTrackBuffer, 'float32x2', trackRows),
        pathOffsets: view('model-offsets', modelOffsets, 'uint32', count + 1),
        columns: FIELD.width,
        rows: FIELD.height,
        coordinateSystem: 'spherical',
        parameters,
        output: {
          lengths: view('model-lengths', modelLengths, 'float32', FIELD_CELLS),
          densities: view('model-densities', modelDensities, 'float32', FIELD_CELLS),
          overflow: view('model-overflow', modelOverflow, 'uint32', 1)
        }
      })
    );
  }
  const densityCompiled = resources.track(densityGraph.compile());

  // ---- Graph: normalized difference of the two densities -----------------------------------------
  const differenceGraph = new GPUCommandGraph<void>(device, {id: 'drifters-difference'});
  {
    const view = createViewer(differenceGraph);
    addKernelPass(differenceGraph, {
      id: 'drifters-difference',
      invocationCount: FIELD_CELLS,
      declarations: /* wgsl */ `fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }`,
      bindings: [
        {
          name: 'real',
          view: view('real-lengths', realLengths, 'float32', FIELD_CELLS),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'model',
          view: view('model-lengths', modelLengths, 'float32', FIELD_CELLS),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'settings',
          view: differenceParameters.importToGraph(differenceGraph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'difference',
          view: view('difference', differenceBuffer, 'float32', FIELD_CELLS),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let total = real[realOffset + index] + model[modelOffset + index];
  var result = getQuietNaN();
  if (total >= settings[settingsOffset]) {
    result = (model[modelOffset + index] - real[realOffset + index]) / total;
  }
  difference[differenceOffset + index] = result;`
    });
  }
  const differenceCompiled = resources.track(differenceGraph.compile());

  // ---- Graph: the drifters at the lead time ---------------------------------------------------------
  const nowGraph = new GPUCommandGraph<void>(device, {id: 'drifters-now'});
  {
    const view = createViewer(nowGraph);
    addKernelPass(nowGraph, {
      id: 'drifters-now',
      invocationCount: count,
      declarations: /* wgsl */ `
fn getQuietNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
fn isNonFinite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) >= 0x7f800000u; }`,
      bindings: [
        {
          name: 'virtualTrack',
          view: view('virtual-track', virtualTrackBuffer, 'float32x2', trackRows),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'daily',
          view: view('daily', dailyBuffer, 'float32x2', trackRows),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'distances',
          view: view('distances', distanceBuffer, 'float32', trackRows),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'alive',
          view: view('alive', aliveBuffer, 'uint32', count),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'mask',
          view: view('mask', maskBuffer, 'float32', count),
          type: 'f32',
          access: 'read'
        },
        {name: 'lead', view: leadParameters.importToGraph(nowGraph), type: 'f32', access: 'read'},
        {
          name: 'virtualNow',
          view: view('virtual-now', virtualNow, 'float32x2', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'realNow',
          view: view('real-now', realNow, 'float32x2', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'separation',
          view: view('separation-now', separationNow, 'float32', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'flag',
          view: view('pair-flag', pairFlag, 'float32', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'links',
          view: view('links', linkSegments, 'float32x2', count * 2),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'headVirtual',
          view: view('head-virtual', headVirtual, 'float32x2', count * 2),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'headReal',
          view: view('head-real', headReal, 'float32x2', count * 2),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let nan = getQuietNaN();
  let leadDays = clamp(lead[leadOffset], 0.0, ${LEAD_DAYS}.0);
  let day = min(u32(floor(leadDays)), ${LEAD_DAYS - 1}u);
  let fraction = leadDays - f32(day);
  let slotA = (index * ${slots}u + day) * 2u;
  let slotB = slotA + 2u;
  let virtualA = vec2<f32>(virtualTrack[virtualTrackOffset + slotA], virtualTrack[virtualTrackOffset + slotA + 1u]);
  let virtualB = vec2<f32>(virtualTrack[virtualTrackOffset + slotB], virtualTrack[virtualTrackOffset + slotB + 1u]);
  let realA = vec2<f32>(daily[dailyOffset + slotA], daily[dailyOffset + slotA + 1u]);
  let realB = vec2<f32>(daily[dailyOffset + slotB], daily[dailyOffset + slotB + 1u]);
  let enabled = mask[maskOffset + index] > 0.5;
  let lastDay = select(day, day + 1u, fraction > 0.0);
  let virtualOk = enabled && lastDay < alive[aliveOffset + index];
  let realOk = enabled && !isNonFinite(realA.x) && !isNonFinite(realA.y) &&
    (fraction == 0.0 || (!isNonFinite(realB.x) && !isNonFinite(realB.y)));
  var virtualPosition = vec2<f32>(nan, nan);
  var realPosition = vec2<f32>(nan, nan);
  var separationValue = nan;
  var headVirtualA = vec2<f32>(nan, nan);
  var headRealA = vec2<f32>(nan, nan);
  if (virtualOk) {
    virtualPosition = mix(virtualA, virtualB, fraction);
    headVirtualA = virtualA;
  }
  if (realOk) {
    // Do not blend across the antimeridian: take the nearer fix.
    if (abs(realB.x - realA.x) > 180.0) {
      realPosition = select(realA, realB, fraction >= 0.5);
    } else {
      realPosition = mix(realA, realB, fraction);
    }
    headRealA = realA;
  }
  if (virtualOk && realOk) {
    let distanceA = distances[distancesOffset + index * ${slots}u + day];
    let distanceB = distances[distancesOffset + index * ${slots}u + min(day + 1u, ${LEAD_DAYS}u)];
    separationValue = mix(distanceA, distanceB, fraction);
  }
  virtualNow[virtualNowOffset + 2u * index] = virtualPosition.x;
  virtualNow[virtualNowOffset + 2u * index + 1u] = virtualPosition.y;
  realNow[realNowOffset + 2u * index] = realPosition.x;
  realNow[realNowOffset + 2u * index + 1u] = realPosition.y;
  separation[separationOffset + index] = separationValue;
  flag[flagOffset + index] = select(0.0, 1.0, virtualOk && realOk);
  let linkBase = linksOffset + 4u * index;
  links[linkBase] = realPosition.x;
  links[linkBase + 1u] = realPosition.y;
  links[linkBase + 2u] = virtualPosition.x;
  links[linkBase + 3u] = virtualPosition.y;
  let headVirtualBase = headVirtualOffset + 4u * index;
  headVirtual[headVirtualBase] = headVirtualA.x;
  headVirtual[headVirtualBase + 1u] = headVirtualA.y;
  headVirtual[headVirtualBase + 2u] = virtualPosition.x;
  headVirtual[headVirtualBase + 3u] = virtualPosition.y;
  let headRealBase = headRealOffset + 4u * index;
  headReal[headRealBase] = headRealA.x;
  headReal[headRealBase + 1u] = headRealA.y;
  headReal[headRealBase + 2u] = realPosition.x;
  headReal[headRealBase + 3u] = realPosition.y;`
    });
  }
  const nowCompiled = resources.track(nowGraph.compile());

  // ---- Readback -----------------------------------------------------------------------------------
  const snapshotRing = resources.track(
    new GPUReadbackRing(device, {
      id: 'drifters-snapshot-ring',
      byteLength: trackRows * 8,
      slotCount: 2
    })
  );
  const analysisBytes = trackRows * 4 + FIELD_CELLS * 8;
  const analysisRing = resources.track(
    new GPUReadbackRing(device, {
      id: 'drifters-analysis-ring',
      byteLength: analysisBytes,
      slotCount: 2
    })
  );

  // ---- CPU state ------------------------------------------------------------------------------------
  let destroyed = false;
  let runCounter = 0;
  let running = false;
  let rerunRequested = false;
  let sweeping = false;
  let lastRun: ModelRun | null = null;
  let statistics: LeadStatistics | null = null;
  let mask = new Float32Array(count);
  let selectedCount = 0;
  let differenceDirty = true;
  let lastChartSeconds = -Infinity;
  let lastChartDay = -1;
  const regionOf = (id: string) =>
    OCEAN_REGIONS.find(region => region.id === id) ?? OCEAN_REGIONS[0];

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, LEAD_DAYS], rate: 1, step: 0.25, loop: true}
  );

  /** Writes the mask of drifters in the comparison and the weights of the trail segments. */
  const updateSelection = () => {
    const options = ctx.options;
    const region = regionOf(options.region);
    mask = new Float32Array(count);
    selectedCount = 0;
    for (let track = 0; track < count; track++) {
      const longitude = drifters.release[track * 2];
      const latitude = drifters.release[track * 2 + 1];
      const setOk =
        options.releaseSet === 'all' ||
        (options.releaseSet === 'deployed'
          ? drifters.deployed[track] === 1
          : drifters.deployed[track] === 0);
      if (releaseValid[track] && setOk && region.contains(longitude, latitude)) {
        mask[track] = 1;
        selectedCount++;
      }
    }
    maskBuffer.write(mask);
    const realWeights = new Float32Array(segmentRows);
    for (let day = 0; day < LEAD_DAYS; day++) {
      for (let track = 0; track < count; track++) {
        const row = day * count + track;
        realWeights[row] = drifters.dailySegmentValid[row] ? mask[track] : 0;
      }
    }
    realSegmentWeights.write(realWeights);
    if (lastRun) writeVirtualSegments(lastRun);
  };

  /** Rebuilds the virtual daily segments and their weights from a sanitized run. */
  let sanitizedTrack = new Float32Array(trackRows * 2);
  const writeVirtualSegments = (run: ModelRun) => {
    const segments = new Float32Array(segmentRows * 4);
    const weights = new Float32Array(segmentRows);
    for (let day = 0; day < LEAD_DAYS; day++) {
      for (let track = 0; track < count; track++) {
        const row = day * count + track;
        const a = (track * slots + day) * 2;
        segments.set(sanitizedTrack.subarray(a, a + 4), row * 4);
        weights[row] = day + 1 < run.alive[track] ? mask[track] : 0;
      }
    }
    virtualSegments.write(segments);
    virtualSegmentWeights.write(weights);
  };

  /** Median, percentiles and the staying-put baseline for each lead day over `selection`. */
  const computeStatistics = (
    run: ModelRun,
    selection: (track: number) => boolean
  ): LeadStatistics => {
    const result: LeadStatistics = {
      count: new Float64Array(slots),
      median: new Float64Array(slots),
      p10: new Float64Array(slots),
      p25: new Float64Array(slots),
      p75: new Float64Array(slots),
      p90: new Float64Array(slots),
      stayMedian: new Float64Array(slots),
      lostFraction: new Float64Array(slots)
    };
    const distances = new Float64Array(count);
    const stays = new Float64Array(count);
    for (let day = 0; day < slots; day++) {
      let valid = 0;
      let withReal = 0;
      let lost = 0;
      for (let track = 0; track < count; track++) {
        if (!selection(track)) continue;
        const base = (track * slots + day) * 2;
        if (!Number.isFinite(drifters.daily[base])) continue;
        withReal++;
        if (day >= run.alive[track]) {
          lost++;
          continue;
        }
        const separation = run.distances[track * slots + day];
        if (!Number.isFinite(separation)) continue;
        distances[valid] = separation;
        stays[valid] = getGreatCircleMeters(
          drifters.daily[track * slots * 2],
          drifters.daily[track * slots * 2 + 1],
          drifters.daily[base],
          drifters.daily[base + 1]
        );
        valid++;
      }
      const sortedDistances = distances.slice(0, valid).sort();
      const sortedStays = stays.slice(0, valid).sort();
      result.count[day] = valid;
      result.median[day] = getSortedQuantile(sortedDistances, valid, 0.5);
      result.p10[day] = getSortedQuantile(sortedDistances, valid, 0.1);
      result.p25[day] = getSortedQuantile(sortedDistances, valid, 0.25);
      result.p75[day] = getSortedQuantile(sortedDistances, valid, 0.75);
      result.p90[day] = getSortedQuantile(sortedDistances, valid, 0.9);
      result.stayMedian[day] = getSortedQuantile(sortedStays, valid, 0.5);
      result.lostFraction[day] = withReal > 0 ? lost / withReal : 0;
    }
    return result;
  };

  const sumRegionLength = (lengths: Float32Array, regionId: string): number => {
    const region = regionOf(regionId);
    let total = 0;
    for (let row = 0; row < FIELD.height; row++) {
      const latitude = FIELD.south + (row + 0.5) * FIELD.cellDegrees;
      for (let column = 0; column < FIELD.width; column++) {
        const longitude = FIELD.west + (column + 0.5) * FIELD.cellDegrees;
        if (region.contains(longitude, latitude)) total += lengths[row * FIELD.width + column];
      }
    }
    return total;
  };

  const formatDensityReadout = (run: ModelRun, regionId: string) => {
    const real = sumRegionLength(run.realLengths, regionId);
    const model = sumRegionLength(run.modelLengths, regionId);
    const share = real > 0 ? Math.round((100 * model) / real) : 0;
    return `real ${formatKilometers(real)} · model ${formatKilometers(model)} (${share}% of real)`;
  };

  const publishRun = (run: ModelRun) => {
    ctx.setReadout('gulfStreamLength', formatDensityReadout(run, 'gulfStream'));
    ctx.setReadout('kuroshioLength', formatDensityReadout(run, 'kuroshio'));
    ctx.setReadout('agulhasLength', formatDensityReadout(run, 'agulhas'));
  };

  /** Recomputes statistics for the current selection and refreshes readouts and charts. */
  const refreshStatistics = () => {
    if (!lastRun) return;
    const run = lastRun;
    statistics = computeStatistics(run, track => mask[track] > 0.5);
    ctx.setReadout(
      'selected',
      `${selectedCount.toLocaleString('en-US')} of ${count.toLocaleString('en-US')} drifters`
    );
    refreshCharts(true);
  };

  const refreshCharts = (force: boolean) => {
    if (!statistics || !lastRun) return;
    const day = Math.min(LEAD_DAYS, Math.round(clock.time));
    const stats = statistics;
    const days = Array.from({length: slots}, (_, index) => index);
    const toKm = (values: Float64Array) => Array.from(values, value => value / 1000);
    if (force) {
      ctx.setChart('separationChart', {
        kind: 'line',
        height: 170,
        xLabel: 'days since release',
        yLabel: 'separation (km)',
        series: [
          {label: 'model median', x: days, y: toKm(stats.median), color: 1},
          {label: '90th percentile', x: days, y: toKm(stats.p90), color: 3, dashed: true},
          {
            label: 'staying put (median)',
            x: days,
            y: toKm(stats.stayMedian),
            color: 5,
            dashed: true
          }
        ],
        band: {
          x: days,
          low: toKm(stats.p25),
          high: toKm(stats.p75),
          label: 'middle half (25th to 75th percentile)'
        },
        markers: [{x: clock.time, label: `day ${clock.time.toFixed(1)}`}],
        formatY: value => `${Math.round(value)}`,
        description:
          'Median distance in kilometers between each virtual drifter and the real drifter it was released with, against days since release, with the middle half of drifters shaded.'
      });
    }
    // Histogram of the separation at the lead day and the median by region.
    const separations: number[] = [];
    for (let track = 0; track < count; track++) {
      if (mask[track] < 0.5 || day >= lastRun.alive[track]) continue;
      const value = lastRun.distances[track * slots + day];
      if (Number.isFinite(value) && Number.isFinite(drifters.daily[(track * slots + day) * 2])) {
        separations.push(value / 1000);
      }
    }
    const upper = Math.max(500, ctx.options.sepMax);
    const bins = new Float64Array(24);
    for (const value of separations) {
      bins[Math.min(bins.length - 1, Math.floor((value / upper) * bins.length))]++;
    }
    ctx.setChart('separationHistogram', {
      kind: 'histogram',
      height: 120,
      values: bins,
      xDomain: [0, upper],
      xLabel: `separation on day ${day} (km)`,
      yLabel: 'drifters',
      markers: Number.isFinite(stats.median[day])
        ? [{x: Math.min(upper, stats.median[day] / 1000), label: 'median'}]
        : []
    });
    const regionLabels: string[] = [];
    const regionValues: number[] = [];
    for (const region of OCEAN_REGIONS.slice(1)) {
      const regional = computeRegionMedian(lastRun, day, region.id);
      regionLabels.push(region.label);
      regionValues.push(Number.isFinite(regional) ? regional / 1000 : 0);
    }
    ctx.setChart('regionChart', {
      kind: 'bars',
      height: 130,
      values: regionValues,
      labels: regionLabels,
      highlight:
        ctx.options.region === 'all'
          ? []
          : [Math.max(0, OCEAN_REGIONS.findIndex(r => r.id === ctx.options.region) - 1)],
      yLabel: `median separation, day ${day} (km)`,
      formatY: value => `${Math.round(value)}`
    });
    lastChartDay = day;
    // Numbers the story cites.
    const median = stats.median[day];
    const stay = stats.stayMedian[day];
    ctx.setReadout('lead', `day ${clock.time.toFixed(1)} after release`);
    ctx.setReadout('comparedCount', stats.count[day]);
    ctx.setReadout('median', formatKilometers(median));
    ctx.setReadout(
      'middleHalf',
      `${formatKilometers(stats.p25[day])} to ${formatKilometers(stats.p75[day])}`
    );
    ctx.setReadout('stayPut', formatKilometers(stay));
    ctx.setReadout(
      'improvement',
      Number.isFinite(median) && stay > 0 ? `${Math.round((1 - median / stay) * 100)}%` : 'n/a'
    );
    ctx.setReadout('lost', `${(stats.lostFraction[day] * 100).toFixed(1)}%`);
  };

  const computeRegionMedian = (run: ModelRun, day: number, regionId: string): number => {
    const region = regionOf(regionId);
    const options = ctx.options;
    const values: number[] = [];
    for (let track = 0; track < count; track++) {
      if (!releaseValid[track] || day >= run.alive[track]) continue;
      const setOk =
        options.releaseSet === 'all' ||
        (options.releaseSet === 'deployed'
          ? drifters.deployed[track] === 1
          : drifters.deployed[track] === 0);
      if (!setOk || !region.contains(drifters.release[track * 2], drifters.release[track * 2 + 1]))
        continue;
      const value = run.distances[track * slots + day];
      if (Number.isFinite(value) && Number.isFinite(drifters.daily[(track * slots + day) * 2]))
        values.push(value);
    }
    values.sort((a, b) => a - b);
    return getSortedQuantile(values, values.length, 0.5);
  };

  const readInto = async (
    ring: GPUReadbackRing,
    sources: readonly {buffer: Buffer; size: number}[]
  ): Promise<ArrayBuffer> => {
    const ticket = await ring.acquire();
    const commandEncoder = device.createCommandEncoder({id: 'drifters-readback'});
    let offset = 0;
    for (const {buffer, size} of sources) {
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: buffer,
        destinationBuffer: ticket.buffer,
        destinationOffset: offset,
        size
      });
      offset += size;
    }
    ticket.markEncoded({byteOffset: 0, byteLength: offset});
    device.submit(commandEncoder.finish());
    const bytes = await ticket.read();
    return bytes.slice().buffer;
  };

  /**
   * Releases a virtual particle at every drifter and advects it for 30 daily steps, then measures
   * the separations. Runs outside `encode`; `commit` false keeps the display untouched (sweeps).
   */
  const runModel = async (scale: number, commit: boolean): Promise<ModelRun | null> => {
    const options = ctx.options;
    const started = performance.now();
    const runId = ++runCounter;
    // Initial state: particles at the real release points, trail slot 0 holds the release.
    velocityBuffer.write(options.coast === 'stick' ? field.filled : field.holes);
    particlePositions.write(drifters.release);
    particleAges.write(new Uint32Array(count));
    particleGenerations.write(new Uint32Array(count));
    const initialSnapshots = new Float32Array(trackRows * 2);
    for (let track = 0; track < count; track++) {
      for (let slot = 0; slot < slots; slot++) {
        initialSnapshots[(track * slots + slot) * 2] = drifters.release[track * 2];
        initialSnapshots[(track * slots + slot) * 2 + 1] = drifters.release[track * 2 + 1];
      }
    }
    snapshotBuffer.write(initialSnapshots);
    advectionParameters.write(
      getGPUParticleAdvectionParameterValues(
        {
          fieldExtent: FIELD.extent,
          timeStep: 1,
          speedScale: scale,
          // Particles that leave the data stay parked outside the field instead of respawning.
          spawnBounds: [PARKING[0], PARKING[1], PARKING[0], PARKING[1]]
        },
        [FIELD.width, FIELD.height]
      )
    );
    for (let day = 1; day <= LEAD_DAYS; day++) {
      advectionWords.write(
        getGPUParticleAdvectionWordParameterValues({
          seed: 1,
          frame: day,
          maximumAge: 0,
          reset: false
        })
      );
      const commandEncoder = device.createCommandEncoder({id: `drifters-advect-${day}`});
      advectionCompiled.encode(commandEncoder, {parameters: undefined});
      device.submit(commandEncoder.finish());
    }
    const snapshots = new Float32Array(
      await readInto(snapshotRing, [{buffer: snapshotBuffer, size: trackRows * 8}])
    );
    if (destroyed || runId !== runCounter) return null;

    // A lost particle is parked outside the field: hold its last position and count its live days.
    const alive = new Uint32Array(count);
    const sanitized = new Float32Array(trackRows * 2);
    for (let track = 0; track < count; track++) {
      let liveDays = 0;
      let lastLongitude = snapshots[track * slots * 2];
      let lastLatitude = snapshots[track * slots * 2 + 1];
      let lostSeen = false;
      for (let slot = 0; slot < slots; slot++) {
        const index = (track * slots + slot) * 2;
        const longitude = snapshots[index];
        const latitude = snapshots[index + 1];
        const parked = !Number.isFinite(longitude) || !Number.isFinite(latitude) || latitude > 89;
        if (parked) lostSeen = true;
        if (!lostSeen) {
          liveDays++;
          lastLongitude = longitude;
          lastLatitude = latitude;
        }
        sanitized[index] = lastLongitude;
        sanitized[index + 1] = lastLatitude;
      }
      alive[track] = Math.max(1, liveDays);
    }
    sanitizedTrack = sanitized;
    virtualTrackBuffer.write(sanitized);
    aliveBuffer.write(alive);

    const commandEncoder = device.createCommandEncoder({id: 'drifters-analysis'});
    pairsCompiled[options.distanceModel].encode(commandEncoder, {parameters: undefined});
    densityCompiled.encode(commandEncoder, {parameters: undefined});
    device.submit(commandEncoder.finish());
    const analysis = await readInto(analysisRing, [
      {buffer: distanceBuffer, size: trackRows * 4},
      {buffer: realLengths, size: FIELD_CELLS * 4},
      {buffer: modelLengths, size: FIELD_CELLS * 4}
    ]);
    if (destroyed || runId !== runCounter) return null;
    const run: ModelRun = {
      distances: new Float32Array(analysis, 0, trackRows),
      alive,
      realLengths: new Float32Array(analysis, trackRows * 4, FIELD_CELLS),
      modelLengths: new Float32Array(analysis, trackRows * 4 + FIELD_CELLS * 4, FIELD_CELLS),
      scale
    };
    if (commit) {
      lastRun = run;
      differenceDirty = true;
      writeVirtualSegments(run);
      publishRun(run);
      refreshStatistics();
      ctx.setReadout(
        'runTime',
        `${Math.round(performance.now() - started)} ms (30 steps, 2 readbacks)`
      );
      ctx.setStatus('');
      ctx.requestLayers();
    }
    return run;
  };

  /** Runs the model with the current options; coalesces requests made while a run is in flight. */
  const requestRun = async () => {
    if (running || sweeping) {
      rerunRequested = true;
      return;
    }
    running = true;
    try {
      do {
        rerunRequested = false;
        await runModel(ctx.options.speedScale, true);
      } while (rerunRequested && !destroyed);
    } catch (error) {
      // A run cut short by switching scenes is not an error.
      if (!destroyed) {
        // biome-ignore lint/suspicious/noConsole: a failed run must be visible
        console.error('[ocean-drifters-vs-model] model run failed', error);
      }
    } finally {
      running = false;
    }
  };

  /** Scores a run by the mean median separation over lead days 1 to 30 (lower is better). */
  const scoreRun = (run: ModelRun): number => {
    const scored = computeStatistics(run, track => mask[track] > 0.5);
    let total = 0;
    for (let day = 1; day < slots; day++) total += scored.median[day];
    return total / (slots - 1);
  };

  const runSweep = async () => {
    if (sweeping || destroyed) return;
    sweeping = true;
    ctx.setReadout('sweep', 'sweeping 9 speed scales...');
    try {
      while (running) await new Promise(resolve => setTimeout(resolve, 30));
      let best = {scale: SWEEP_SCALES[0], score: Number.POSITIVE_INFINITY};
      let baseline = Number.NaN;
      for (const scale of SWEEP_SCALES) {
        const run = await runModel(scale, false);
        if (!run) return;
        const score = scoreRun(run);
        if (scale === 1) baseline = score;
        if (score < best.score) best = {scale, score};
      }
      ctx.setReadout(
        'sweep',
        `best ${best.scale}x: mean median separation ${formatKilometers(best.score)} against ${formatKilometers(baseline)} at 1x`
      );
      ctx.setOptions({speedScale: best.scale});
      sweeping = false;
      await requestRun();
    } finally {
      sweeping = false;
    }
  };

  // ---- Initial state ------------------------------------------------------------------------------
  differenceParameters.write(Float32Array.of(ctx.options.densityMinKm * 1000, 0, 0, 0));
  updateSelection();
  ctx.setStatus('Releasing 2,811 virtual drifters');
  ctx.setReadout(
    'field',
    `${FIELD.width} x ${FIELD.height} cells (0.5 degree), ${field.validCells.toLocaleString('en-US')} with data; peak mean speed ${field.peakSpeed.toFixed(2)} m/s`
  );
  ctx.setReadout('samples', `${field.samples.toLocaleString('en-US')} ECCO 3.5-day displacements`);
  ctx.setReadout(
    'releases',
    `${count.toLocaleString('en-US')} released, ${validReleases.toLocaleString('en-US')} inside the model's data`
  );
  ctx.setReadout('sweep', 'press the button');
  void requestRun();

  // ---- Instance ------------------------------------------------------------------------------------
  const nearestRelease = (
    longitude: number,
    latitude: number,
    toleranceDegrees: number
  ): number => {
    let best = -1;
    let bestDistance = toleranceDegrees * toleranceDegrees;
    const cosine = Math.cos((latitude * Math.PI) / 180);
    for (let track = 0; track < count; track++) {
      if (mask[track] < 0.5) continue;
      const dx = (drifters.release[track * 2] - longitude) * cosine;
      const dy = drifters.release[track * 2 + 1] - latitude;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = track;
      }
    }
    return best;
  };

  return {
    getCompiledGraphs: () => [
      advectionCompiled,
      pairsCompiled[ctx.options.distanceModel],
      densityCompiled,
      differenceCompiled,
      nowCompiled
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'speedScale':
        case 'coast':
        case 'distanceModel':
          void requestRun();
          break;
        case 'releaseSet':
          updateSelection();
          refreshStatistics();
          break;
        case 'region': {
          updateSelection();
          refreshStatistics();
          const view = regionOf(state.region).view;
          ctx.flyTo(view, {transitionMs: 1400});
          break;
        }
        case 'densityMinKm':
          differenceParameters.write(Float32Array.of(state.densityMinKm * 1000, 0, 0, 0));
          differenceDirty = true;
          break;
        case 'sepMax':
          refreshCharts(false);
          break;
        default:
          break;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'sweep') void runSweep();
    },

    encode(commandEncoder, frame) {
      const lead = clock.advance(frame);
      leadParameters.write(Float32Array.of(lead, 0, 0, 0));
      nowCompiled.encode(commandEncoder, {parameters: undefined});
      trailCount.write(Uint32Array.of(Math.floor(lead) * count));
      commandEncoder.copyBufferToBuffer({
        sourceBuffer: trailCount,
        sourceOffset: 0,
        destinationBuffer: trailDraw.buffer,
        destinationOffset: 4,
        size: 4
      });
      if (differenceDirty && lastRun) {
        differenceCompiled.encode(commandEncoder, {parameters: undefined});
        differenceDirty = false;
      }
      if (
        lastRun &&
        frame.timeSeconds - lastChartSeconds > CHART_INTERVAL_SECONDS &&
        Math.round(lead) !== lastChartDay
      ) {
        lastChartSeconds = frame.timeSeconds;
        refreshCharts(false);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const realColor = dark ? REAL_COLORS.dark : REAL_COLORS.light;
      const modelColor = dark ? MODEL_COLORS.dark : MODEL_COLORS.light;
      const layers: Layer[] = [];
      const raster = {
        ...drawProps,
        gridSize: [FIELD.width, FIELD.height] as const,
        bounds: [FIELD.west, FIELD.south, FIELD.east, FIELD.north] as const,
        tessellation: 64,
        valueFormat: 'float32' as const,
        opacity: options.backgroundOpacity
      };
      switch (options.background) {
        case 'speed':
          layers.push(
            new SpatialAnalysisRasterLayer({
              id: 'drifters-speed',
              ...raster,
              values: speedBuffer,
              colormap: 'viridis',
              valueRange: [0, options.speedMax]
            })
          );
          break;
        case 'realDensity':
        case 'modelDensity':
          layers.push(
            new SpatialAnalysisRasterLayer({
              id: `drifters-${options.background}`,
              ...raster,
              values: options.background === 'realDensity' ? realDensities : modelDensities,
              colormap: 'inferno',
              valueScale: 1e6,
              valueRange: [0, options.densityMax],
              sqrtScale: true,
              discardAtOrBelow: 0
            })
          );
          break;
        case 'densityDiff':
          layers.push(
            new SpatialAnalysisRasterLayer({
              id: 'drifters-density-difference',
              ...raster,
              values: differenceBuffer,
              colormap: 'diverging',
              valueRange: [-1, 1]
            })
          );
          break;
        default:
          break;
      }
      if (options.showTrails && options.showReal) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'drifters-real-trails',
            ...drawProps,
            segments: realSegments,
            weights: realSegmentWeights,
            drawCommands: trailDraw,
            color: realColor,
            widthPixels: 1.4,
            opacity: 0.8
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'drifters-real-heads',
            ...drawProps,
            segments: headReal,
            instanceCount: count,
            color: realColor,
            widthPixels: 1.4,
            opacity: 0.8
          })
        );
      }
      if (options.showTrails && options.showVirtual) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'drifters-virtual-trails',
            ...drawProps,
            segments: virtualSegments,
            weights: virtualSegmentWeights,
            drawCommands: trailDraw,
            color: modelColor,
            widthPixels: 1.4,
            opacity: 0.8
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'drifters-virtual-heads',
            ...drawProps,
            segments: headVirtual,
            instanceCount: count,
            color: modelColor,
            widthPixels: 1.4,
            opacity: 0.8
          })
        );
      }
      if (options.showLinks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'drifters-links',
            ...drawProps,
            segments: linkSegments,
            instanceCount: count,
            values: separationNow,
            valueFormat: 'float32',
            valueScale: 0.001,
            valueRange: [0, options.sepMax],
            colormap: options.ramp,
            widthPixels: 1.6,
            opacity: 0.85
          })
        );
      }
      if (options.showReleases) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'drifters-release-halo',
            ...drawProps,
            positions: releaseBuffer,
            instanceCount: count,
            values: pairFlag,
            valueFormat: 'float32',
            colormap: 'mask',
            color: dark ? [255, 255, 255, 120] : [20, 24, 36, 120],
            radiusPixels: options.pointSize + 2
          }),
          new SpatialAnalysisPointLayer({
            id: 'drifters-releases',
            ...drawProps,
            positions: releaseBuffer,
            instanceCount: count,
            values: separationNow,
            valueFormat: 'float32',
            valueScale: 0.001,
            valueRange: [0, options.sepMax],
            colormap: options.ramp,
            radiusPixels: options.pointSize
          })
        );
      }
      if (options.showReal) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'drifters-real',
            ...drawProps,
            positions: realNow,
            instanceCount: count,
            color: realColor,
            radiusPixels: options.pointSize * 0.8
          })
        );
      }
      if (options.showVirtual) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'drifters-virtual',
            ...drawProps,
            positions: virtualNow,
            instanceCount: count,
            color: modelColor,
            radiusPixels: options.pointSize * 0.8
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate || !lastRun) return null;
      const viewport = ctx.getViewport();
      const zoom = viewport?.zoom ?? 2;
      const tolerance = (10 * 360) / (512 * 2 ** zoom);
      const track = nearestRelease(event.coordinate[0], event.coordinate[1], tolerance);
      if (track < 0) return null;
      const day = Math.min(LEAD_DAYS, Math.round(clock.time));
      const separation =
        day < lastRun.alive[track] ? lastRun.distances[track * slots + day] : Number.NaN;
      const real = [
        drifters.daily[(track * slots + day) * 2],
        drifters.daily[(track * slots + day) * 2 + 1]
      ];
      const stay = Number.isFinite(real[0])
        ? getGreatCircleMeters(
            drifters.release[track * 2],
            drifters.release[track * 2 + 1],
            real[0],
            real[1]
          )
        : Number.NaN;
      return [
        `Drifter released ${formatDriftDate(drifters.timeOriginMs, drifters.releaseDays[track])} at ${formatPosition(drifters.release[track * 2], drifters.release[track * 2 + 1])}`,
        `Day ${day}: model ${Number.isFinite(separation) ? formatKilometers(separation) : day >= lastRun.alive[track] ? 'lost (reached the model coast or edge)' : 'no real fix'} from the drifter`,
        `Staying put would be ${formatKilometers(stay)} away`,
        drifters.deployed[track] ? 'New deployment or record' : 'Already at sea on 1 January'
      ].join('\n');
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      destroyed = true;
      resources.destroy();
    }
  };
}
