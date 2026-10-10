// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Peaks: terrain features found on the GPU from one elevation raster.
 *
 * The explorer's DEM is not RGB-encoded, so the mode first synthesizes a Terrarium RGBA8 texture
 * from the decoded heights (CPU encode), optionally injects +/-256 m "high byte" spikes, and runs
 * `GPUTerrainRGBDecode` and `GPUTerrainSpikeRepair` to get heights back. Those heights feed
 *
 * - `GPUTerrainSummits`: disc maxima with a minimum drop, drawn as points sized by drop,
 * - `GPUTerrainPeakSnap`: clicked candidates snapped to the nearest summit,
 * - `GPUTerrainCriticalPoints`: peak, pit and saddle pixels from the 8- or 6-neighbour ring,
 * - `GPURasterProfile` + `GPUProfilePeaks`: an elevation profile along a drawn line with its
 *   prominent peaks.
 *
 * Every graph is encoded only when one of its inputs changed (outputs persist in GPU buffers).
 * Small results (summit lists, snap results, profile) are read back through `SummaryReader`.
 */

import type {Layer} from '@deck.gl/core';
import {Buffer, Texture, type CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainPeakSnapParameterValues,
  getGPUTerrainSummitsParameterValues,
  getGPUProfilePeaksParameterValues,
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
  GPU_TERRAIN_CRITICAL_POINT,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_STATUS,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH,
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
  GPUProfilePeaks,
  GPUTerrainCriticalPoints,
  GPUTerrainDerivatives,
  GPUTerrainPeakSnap,
  GPUTerrainRGBDecode,
  GPUTerrainSpikeRepair,
  GPUTerrainSummits
} from '@luma.gl/experimental/gpu-terrain';
import {
  getGPURasterProfileParameterValues,
  GPU_RASTER_PROFILE_PARAMETER_LENGTH,
  GPURasterProfile
} from '@luma.gl/experimental/gpu-raster';
import {importGraphBuffer} from '../graph-buffers';
import {LocalMetricProjection} from '../spatial-analysis-data';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {MiniChart} from './point-pattern-chart';
import {SummaryReader} from './summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from './vector-timing';

/** Capacity of the compact summit list. */
const SUMMIT_CAPACITY = 2048;
/** Compile-time bound of the summit and snap discs, in pixels. */
const MAXIMUM_RADIUS_PIXELS = 16;
/** Peak snap candidate slots; unused slots are parked outside the raster. */
const CANDIDATE_CAPACITY = 24;
/** Profile sample capacity at one sample per pixel. */
const PROFILE_CAPACITY = 1024;
/** Summit size classes: each is one point layer pair. */
const SUMMIT_BUCKET_COUNT = 4;
const SUMMIT_BUCKET_RADIUS = [3, 5, 7.5, 10.5];
const SUMMIT_BUCKET_COLOR: readonly (readonly [number, number, number, number])[] = [
  [255, 226, 110, 235],
  [255, 176, 70, 240],
  [255, 118, 60, 245],
  [235, 50, 70, 250]
];
/** Number of Terrarium-error blocks injected when spikes are on. */
const SPIKE_BLOCK_COUNT = 14;
const PROFILE_PEAK_CAPACITY = 64;

const CRITICAL_PALETTE = [
  [0, 0, 0, 0], // regular
  [235, 60, 70, 235], // peak
  [70, 140, 255, 235], // pit
  [255, 214, 70, 235], // saddle
  [0, 0, 0, 0], // boundary
  [0, 0, 0, 0], // noData
  [0, 0, 0, 0],
  [0, 0, 0, 0]
] as const;

const SNAP_STATUS_PALETTE = [
  [190, 200, 215, 255], // unchanged
  [70, 235, 130, 255], // snapped
  [255, 160, 60, 255], // onRing
  [235, 60, 70, 255], // rejectedMove
  [220, 90, 230, 255], // rejectedHeight
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0]
] as const;

const SNAP_STATUS_NAMES = [
  'unchanged',
  'snapped',
  'on ring',
  'move too far',
  'height change',
  'no data',
  'outside'
];

type BaseDisplay = 'hillshade' | 'elevation';
type ClickTool = 'snap' | 'profile';
type Ring = '8' | '6';
type ProfileWindow = '8' | '16' | '32' | '64';

type SpikeInjection = {blockCount: number; pixelCount: number; mask: Uint8Array};

export const terrainFeaturesMode: SpatialAnalysisModeDefinition = {
  id: 'terrain-features',
  title: 'Peaks',
  contributors: [
    'GPUTerrainRGBDecode',
    'GPUTerrainSpikeRepair',
    'GPUTerrainSummits',
    'GPUTerrainPeakSnap',
    'GPUTerrainCriticalPoints',
    'GPURasterProfile',
    'GPUProfilePeaks',
    'GPUTerrainDerivatives'
  ],
  description:
    'Summits, peak snapping, critical points and profile peaks from one DEM. The DEM is first ' +
    're-encoded as a Terrarium RGBA texture, optionally corrupted with +/-256 m high-byte ' +
    'spikes, then decoded and repaired on the GPU. Click the map with the chosen tool to add ' +
    'snap candidates or to draw the profile line (two clicks).',
  initialViewState: {longitude: -122.44, latitude: 37.76, zoom: 11.6},

  async create(context) {
    const terrain = await context.data.getSanFranciscoTerrain();
    context.signal.throwIfAborted();
    const {device} = context;
    const {width, height, bounds, cellSize} = terrain;
    const pixelCount = width * height;
    const cell = Math.min(cellSize[0], cellSize[1]);
    const projection = new LocalMetricProjection(terrain.origin);
    const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
    const resources = new SpatialAnalysisResources(device, 'terrain-features');

    // Raster pixel-centre index space (column, row) to planar meters, for GPU-side positions.
    const pixelScale: [number, number] = [cellSize[0], -cellSize[1]];
    const pixelOffset: [number, number] = [
      bounds[0] + 0.5 * cellSize[0],
      bounds[3] - 0.5 * cellSize[1]
    ];
    const pixelToMeters = (column: number, row: number): [number, number] => [
      pixelOffset[0] + column * pixelScale[0],
      pixelOffset[1] + row * pixelScale[1]
    ];

    // --- Source data and the synthetic Terrarium texture ----------------------------------------
    const sourceValid = new Uint8Array(pixelCount);
    for (let index = 0; index < pixelCount; index++) {
      sourceValid[index] = terrain.elevation[index] > 0.5 ? 1 : 0;
    }
    const cleanTerrarium = encodeTerrarium(terrain.elevation, sourceValid);
    let terrarium = cleanTerrarium;
    let injection: SpikeInjection = {
      blockCount: 0,
      pixelCount: 0,
      mask: new Uint8Array(pixelCount)
    };

    const texture = resources.track(
      device.createTexture({
        id: 'terrain-features-terrarium',
        format: 'rgba8unorm',
        width,
        height,
        usage: Texture.SAMPLE | Texture.COPY_DST
      })
    );
    texture.writeData(cleanTerrarium);

    // --- Buffers --------------------------------------------------------------------------------
    const decodedValues = resources.createBuffer('decoded-values', pixelCount * 4);
    const decodedValidity = resources.createBuffer('decoded-validity', pixelCount * 4);
    const repairedValues = resources.createBuffer('repaired-values', pixelCount * 4);
    const repairedValidity = resources.createBuffer('repaired-validity', pixelCount * 4);
    const repairStatistics = resources.createBuffer(
      'repair-statistics',
      GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH * 4
    );
    // The elevation every feature graph reads: decoded or repaired heights.
    const workValues = resources.createBuffer('work-values', pixelCount * 4);
    const workValidity = resources.createBuffer('work-validity', pixelCount * 4);
    const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
    const derivativesSettings = resources.createParameterBuffer(
      'derivatives-settings',
      'float32',
      GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
    );

    const summitIds = resources.createBuffer('summit-ids', SUMMIT_CAPACITY * 4);
    const summitDrops = resources.createBuffer('summit-drops', SUMMIT_CAPACITY * 4);
    const summitCount = resources.createBuffer('summit-count', 4);
    const summitTotal = resources.createBuffer('summit-total', 4);
    const summitOverflow = resources.createBuffer('summit-overflow', 4);
    const summitClamped = resources.createBuffer('summit-clamped', 4);
    const summitSettings = resources.createParameterBuffer(
      'summit-settings',
      'float32',
      GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH
    );
    const summitBuckets = Array.from({length: SUMMIT_BUCKET_COUNT}, (_, bucket) =>
      resources.createBuffer(`summit-bucket-${bucket}`, SUMMIT_CAPACITY * 8)
    );

    const criticalClasses = resources.createBuffer('critical-classes', pixelCount * 4);
    const criticalCounts = resources.createBuffer(
      'critical-counts',
      GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT * 4
    );

    const candidateBuffer = resources.createBuffer('candidates', CANDIDATE_CAPACITY * 8);
    const snapPositions = resources.createBuffer('snap-positions', CANDIDATE_CAPACITY * 8);
    const snapHeights = resources.createBuffer('snap-heights', CANDIDATE_CAPACITY * 4);
    const snapStatus = resources.createBuffer('snap-status', CANDIDATE_CAPACITY * 4);
    const snapDistance = resources.createBuffer('snap-distance', CANDIDATE_CAPACITY * 4);
    const snapOverflow = resources.createBuffer('snap-overflow', 4);
    const snapSegments = resources.createBuffer('snap-segments', CANDIDATE_CAPACITY * 16);
    const snapSettings = resources.createParameterBuffer(
      'snap-settings',
      'float32',
      GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH
    );

    const pathPositions = resources.createBuffer('path-positions', 16);
    const pathOffsets = resources.createBuffer('path-offsets', Uint32Array.of(0, 2));
    const profileSettings = resources.createParameterBuffer(
      'profile-settings',
      'float32',
      GPU_RASTER_PROFILE_PARAMETER_LENGTH
    );
    const profileCount = resources.createBuffer('profile-count', 4);
    const profileOverflow = resources.createBuffer('profile-overflow', 4);
    const profileValues = resources.createBuffer('profile-values', PROFILE_CAPACITY * 4);
    const profileOffsets = resources.createBuffer('profile-sample-offsets', 8);
    const peakSettings = resources.createParameterBuffer(
      'peak-settings',
      'float32',
      GPU_PROFILE_PEAKS_PARAMETER_LENGTH
    );
    const peakProminence = resources.createBuffer('peak-prominence', PROFILE_CAPACITY * 4);
    const peakIndex = resources.createBuffer('peak-index', PROFILE_CAPACITY * 4);
    const peakValue = resources.createBuffer('peak-value', PROFILE_CAPACITY * 4);
    const peakMask = resources.createBuffer('peak-mask', PROFILE_CAPACITY * 4);
    const peakConverged = resources.createBuffer('peak-converged', 4);
    const profileLineBuffer = resources.createBuffer('profile-line', 16);
    const profileEndpoints = resources.createBuffer('profile-endpoints', 16);
    const profilePeakMarkers = resources.createBuffer(
      'profile-peak-markers',
      PROFILE_PEAK_CAPACITY * 8
    );

    // --- Decode and repair graph (rerun when the texture changes) -------------------------------
    const prepGraph = new GPUCommandGraph<void>(device, {id: 'terrain-features-prep'});
    const textureHandle = prepGraph.importTexture(
      {id: 'terrarium', format: 'rgba8unorm', width, height, usage: texture.props.usage},
      texture
    );
    const decodedValuesView = importGraphBuffer(
      prepGraph,
      'decoded-values',
      decodedValues,
      'float32',
      pixelCount
    );
    const decodedValidityView = importGraphBuffer(
      prepGraph,
      'decoded-validity',
      decodedValidity,
      'uint32',
      pixelCount
    );
    prepGraph.add(
      new GPUTerrainRGBDecode({
        id: 'decode',
        width,
        height,
        encoding: 'terrarium',
        input: {
          texture: prepGraph.createTextureView(textureHandle, {
            mipLevelCount: 1
          }) as GraphTextureView<'rgba8unorm'>
        },
        values: decodedValuesView,
        validity: decodedValidityView
      })
    );
    const spikeRepair = resources.track(
      new GPUTerrainSpikeRepair({
        id: 'repair',
        width,
        height,
        elevation: {
          id: 'decoded',
          format: 'float32',
          storage: {kind: 'buffer', values: decodedValuesView},
          validity: decodedValidityView
        },
        values: importGraphBuffer(
          prepGraph,
          'repaired-values',
          repairedValues,
          'float32',
          pixelCount
        ),
        validity: importGraphBuffer(
          prepGraph,
          'repaired-validity',
          repairedValidity,
          'uint32',
          pixelCount
        ),
        statistics: importGraphBuffer(
          prepGraph,
          'repair-statistics',
          repairStatistics,
          'uint32',
          GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH
        )
      })
    );
    prepGraph.add(spikeRepair);
    const compiledPrep = resources.track(prepGraph.compile());

    // --- Helpers to build graphs over the shared work elevation ---------------------------------
    const getWorkElevation = (graph: GPUCommandGraph<void>, prefix: string) => ({
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(graph, `${prefix}-values`, workValues, 'float32', pixelCount)
      },
      validity: importGraphBuffer(graph, `${prefix}-validity`, workValidity, 'uint32', pixelCount)
    });

    // Context graph: hillshade of the working heights.
    const baseGraph = new GPUCommandGraph<void>(device, {id: 'terrain-features-base'});
    baseGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: getWorkElevation(baseGraph, 'base'),
        settings: derivativesSettings.importToGraph(baseGraph),
        hillshade: importGraphBuffer(
          baseGraph,
          'hillshade',
          hillshadeBuffer,
          'float32',
          pixelCount
        ),
        cellSizeMode: 'uniform',
        rowDirection: 'south'
      })
    );
    const compiledBase = resources.track(baseGraph.compile());

    // Summits.
    const summitGraph = new GPUCommandGraph<void>(device, {id: 'terrain-features-summits'});
    summitGraph.add(
      new GPUTerrainSummits({
        id: 'summits',
        width,
        height,
        elevation: getWorkElevation(summitGraph, 'summits'),
        cellSizeMode: 'uniform',
        maximumRadiusPixels: MAXIMUM_RADIUS_PIXELS,
        incompleteNeighborhood: 'ignore',
        settings: summitSettings.importToGraph(summitGraph),
        output: {
          ids: importGraphBuffer(summitGraph, 'ids', summitIds, 'uint32', SUMMIT_CAPACITY),
          count: importGraphBuffer(summitGraph, 'count', summitCount, 'uint32', 1),
          overflow: importGraphBuffer(summitGraph, 'overflow', summitOverflow, 'uint32', 1),
          requiredCount: importGraphBuffer(summitGraph, 'total', summitTotal, 'uint32', 1)
        },
        outputDrop: importGraphBuffer(
          summitGraph,
          'drops',
          summitDrops,
          'float32',
          SUMMIT_CAPACITY
        ),
        overflow: importGraphBuffer(summitGraph, 'clamped', summitClamped, 'uint32', 1)
      })
    );
    const compiledSummits = resources.track(summitGraph.compile());

    // Critical points (rebuilt when the ring connectivity changes).
    const buildCriticalGraph = (ring: Ring) => {
      const graph = new GPUCommandGraph<void>(device, {id: `terrain-features-critical-${ring}`});
      graph.add(
        new GPUTerrainCriticalPoints({
          id: `critical-${ring}`,
          width,
          height,
          elevation: getWorkElevation(graph, 'critical'),
          connectivity: ring === '8' ? 8 : 6,
          classes: importGraphBuffer(graph, 'classes', criticalClasses, 'uint32', pixelCount),
          counts: importGraphBuffer(
            graph,
            'counts',
            criticalCounts,
            'uint32',
            GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT
          )
        })
      );
      return resources.track(graph.compile());
    };

    // Peak snap.
    const snapGraph = new GPUCommandGraph<void>(device, {id: 'terrain-features-snap'});
    snapGraph.add(
      new GPUTerrainPeakSnap({
        id: 'snap',
        width,
        height,
        elevation: getWorkElevation(snapGraph, 'snap'),
        cellSizeMode: 'uniform',
        maximumRadiusPixels: MAXIMUM_RADIUS_PIXELS,
        candidates: importGraphBuffer(
          snapGraph,
          'candidates',
          candidateBuffer,
          'float32x2',
          CANDIDATE_CAPACITY
        ),
        settings: snapSettings.importToGraph(snapGraph),
        positions: importGraphBuffer(
          snapGraph,
          'positions',
          snapPositions,
          'float32x2',
          CANDIDATE_CAPACITY
        ),
        heights: importGraphBuffer(
          snapGraph,
          'heights',
          snapHeights,
          'float32',
          CANDIDATE_CAPACITY
        ),
        status: importGraphBuffer(snapGraph, 'status', snapStatus, 'uint32', CANDIDATE_CAPACITY),
        snapDistance: importGraphBuffer(
          snapGraph,
          'distance',
          snapDistance,
          'float32',
          CANDIDATE_CAPACITY
        ),
        overflow: importGraphBuffer(snapGraph, 'overflow', snapOverflow, 'uint32', 1)
      })
    );
    const compiledSnap = resources.track(snapGraph.compile());

    // Profile and its peaks (rebuilt when the prominence window changes).
    const buildProfileGraph = (window: number) => {
      const graph = new GPUCommandGraph<void>(device, {id: `terrain-features-profile-${window}`});
      const sampleValues = importGraphBuffer(
        graph,
        'values',
        profileValues,
        'float32',
        PROFILE_CAPACITY
      );
      const sampleOffsets = importGraphBuffer(graph, 'sample-offsets', profileOffsets, 'uint32', 2);
      graph.add(
        new GPURasterProfile({
          id: 'profile',
          width,
          height,
          values: importGraphBuffer(graph, 'dem', workValues, 'float32', pixelCount),
          validity: importGraphBuffer(graph, 'dem-validity', workValidity, 'uint32', pixelCount),
          pathPositions: importGraphBuffer(graph, 'path-positions', pathPositions, 'float32x2', 2),
          pathOffsets: importGraphBuffer(graph, 'path-offsets', pathOffsets, 'uint32', 2),
          sampleCapacity: PROFILE_CAPACITY,
          parameters: profileSettings.importToGraph(graph),
          output: {
            count: importGraphBuffer(graph, 'count', profileCount, 'uint32', 1),
            overflow: importGraphBuffer(graph, 'overflow', profileOverflow, 'uint32', 1),
            sampleValues,
            pathSampleOffsets: sampleOffsets
          }
        })
      );
      graph.add(
        new GPUProfilePeaks({
          id: 'peaks',
          values: sampleValues,
          offsets: sampleOffsets,
          settings: peakSettings.importToGraph(graph),
          window,
          prominence: importGraphBuffer(
            graph,
            'prominence',
            peakProminence,
            'float32',
            PROFILE_CAPACITY
          ),
          refinedIndex: importGraphBuffer(
            graph,
            'refined-index',
            peakIndex,
            'float32',
            PROFILE_CAPACITY
          ),
          refinedValue: importGraphBuffer(
            graph,
            'refined-value',
            peakValue,
            'float32',
            PROFILE_CAPACITY
          ),
          peakMask: importGraphBuffer(graph, 'mask', peakMask, 'uint32', PROFILE_CAPACITY),
          converged: importGraphBuffer(graph, 'converged', peakConverged, 'uint32', 1)
        })
      );
      return resources.track(graph.compile());
    };

    // --- State ----------------------------------------------------------------------------------
    let baseDisplay: BaseDisplay = 'hillshade';
    let showSummits = true;
    let showCritical = false;
    let showSnap = true;
    let showProfile = true;
    let clickTool: ClickTool = 'snap';
    let ring: Ring = '8';
    let profileWindow: ProfileWindow = '16';
    let summitRadius = 6 * cell;
    let summitMinimumDrop = 6;
    let snapRadius = 9 * cell;
    let snapMaximumMove = 8 * cell;
    let snapMaximumHeightChange = 80;
    let snapInterior = true;
    let minimumProminence = 8;
    let injectSpikes = true;
    let repairSpikes = true;
    let spikeSeed = 1;
    let destroyed = false;
    let preparing = false;
    let compiledCritical = buildCriticalGraph(ring);
    let compiledProfile = buildProfileGraph(Number(profileWindow));
    const dirty = {base: true, summits: true, critical: true, snap: true, profile: true};
    const markAllDirty = () => {
      dirty.base = dirty.summits = dirty.critical = dirty.snap = dirty.profile = true;
    };

    let workHeights = new Float32Array(pixelCount);
    const candidates: [number, number][] = [];
    let profileStart: [number, number] = [0, 0];
    let profileEnd: [number, number] = [0, 0];
    let pendingProfileStart: [number, number] | null = null;
    let summitBucketCounts = [0, 0, 0, 0];
    let snapRows: {status: number; distance: number; height: number}[] = [];
    let profileSamples: {
      values: Float32Array;
      peaks: {index: number; value: number; prominence: number}[];
    } = {
      values: new Float32Array(0),
      peaks: []
    };

    // --- Controls (readouts first so asynchronous results can update them) ----------------------
    const readouts = {} as Record<
      | 'decode'
      | 'injected'
      | 'repaired'
      | 'mismatch'
      | 'summits'
      | 'critical'
      | 'criticalOther'
      | 'euler'
      | 'snap'
      | 'profile',
      ReturnType<typeof context.controls.addReadout>
    >;

    function writeDerivatives(): void {
      derivativesSettings.write(
        getGPUTerrainDerivativesParameterValues({
          cellSize,
          azimuthDegrees: 315,
          altitudeDegrees: 40
        })
      );
    }
    function writeSummitSettings(): void {
      summitSettings.write(
        getGPUTerrainSummitsParameterValues({
          radius: summitRadius,
          minimumDrop: summitMinimumDrop,
          cellSize
        })
      );
      dirty.summits = true;
    }
    function writeSnapSettings(): void {
      snapSettings.write(
        getGPUTerrainPeakSnapParameterValues({
          radius: snapRadius,
          maximumMove: snapMaximumMove,
          maximumHeightChange: snapMaximumHeightChange,
          cellSize,
          interior: snapInterior
        })
      );
      dirty.snap = true;
    }
    function writeCandidates(): void {
      const values = new Float32Array(CANDIDATE_CAPACITY * 2).fill(-1);
      candidates.forEach(([column, row], index) => {
        values[index * 2] = column;
        values[index * 2 + 1] = row;
      });
      candidateBuffer.write(values);
      dirty.snap = true;
    }
    function writeProfileLine(): void {
      pathPositions.write(
        Float32Array.of(
          profileStart[0] + 0.5,
          profileStart[1] + 0.5,
          profileEnd[0] + 0.5,
          profileEnd[1] + 0.5
        )
      );
      profileLineBuffer.write(Float32Array.of(...profileStart, ...profileEnd));
      writeProfileEndpoints();
      dirty.profile = true;
    }
    function writeProfileEndpoints(): void {
      profileEndpoints.write(
        Float32Array.of(
          ...(pendingProfileStart ?? profileStart),
          ...(pendingProfileStart ? pendingProfileStart : profileEnd)
        )
      );
    }
    function writeProfileSettings(): void {
      profileSettings.write(
        getGPURasterProfileParameterValues({
          width,
          height,
          extent: [0, 0, width, height],
          method: 'bilinear',
          spacing: 1
        })
      );
      peakSettings.write(getGPUProfilePeaksParameterValues({minProminence: minimumProminence}));
      dirty.profile = true;
    }

    // --- Prep: encode, decode, repair, copy and verify ------------------------------------------
    async function runPrep(): Promise<void> {
      if (destroyed) return;
      preparing = true;
      try {
        texture.writeData(terrarium);
        const commandEncoder = device.createCommandEncoder({id: 'terrain-features-prep-encoder'});
        compiledPrep.encode(commandEncoder, {parameters: undefined});
        const sourceValues = repairSpikes ? repairedValues : decodedValues;
        const sourceValidity = repairSpikes ? repairedValidity : decodedValidity;
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: sourceValues,
          destinationBuffer: workValues,
          size: pixelCount * 4
        });
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: sourceValidity,
          destinationBuffer: workValidity,
          size: pixelCount * 4
        });
        device.submit(commandEncoder.finish());
        const [decodedBytes, workBytes, statisticsBytes] = await Promise.all([
          decodedValues.readAsync(),
          workValues.readAsync(),
          repairStatistics.readAsync()
        ]);
        if (destroyed) return;
        const decoded = new Float32Array(decodedBytes.buffer, decodedBytes.byteOffset, pixelCount);
        workHeights = Float32Array.from(
          new Float32Array(workBytes.buffer, workBytes.byteOffset, pixelCount)
        );
        const statistics = new Uint32Array(
          statisticsBytes.buffer,
          statisticsBytes.byteOffset,
          GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH
        );
        updateDecodeReadouts(decoded, workHeights, statistics);
        markAllDirty();
        context.updateLayers();
      } catch (error) {
        if (!destroyed) readouts.decode.setValue(`failed: ${(error as Error).message}`);
      } finally {
        preparing = false;
      }
    }

    function updateDecodeReadouts(
      decoded: Float32Array,
      repaired: Float32Array,
      statistics: Uint32Array
    ): void {
      let decodeError = 0;
      let invalidCount = 0;
      let remaining = 0;
      let remainingError = 0;
      for (let index = 0; index < pixelCount; index++) {
        if (!sourceValid[index]) continue;
        const source = terrain.elevation[index];
        if (!injection.mask[index]) {
          if (Number.isNaN(decoded[index])) invalidCount++;
          else decodeError = Math.max(decodeError, Math.abs(decoded[index] - source));
        }
        const difference = Math.abs(repaired[index] - source);
        if (Number.isNaN(repaired[index]) || difference > 0.001) {
          remaining++;
          if (!Number.isNaN(difference)) remainingError = Math.max(remainingError, difference);
        }
      }
      readouts.decode.setValue(
        `max error ${decodeError.toFixed(3)} m over ${formatCount(pixelCount - injection.pixelCount)} ` +
          `clean pixels` +
          (invalidCount ? ` (${formatCount(invalidCount)} undecoded)` : '')
      );
      readouts.injected.setValue(
        injection.blockCount
          ? `${injection.blockCount} blocks, ${formatCount(injection.pixelCount)} px at +/-256 m`
          : 'none'
      );
      const repairedPixels = statistics[GPU_TERRAIN_SPIKE_REPAIR_STATISTICS.repairedPixelCount];
      readouts.repaired.setValue(
        repairSpikes
          ? `${formatCount(repairedPixels)} px in ` +
              `${statistics[GPU_TERRAIN_SPIKE_REPAIR_STATISTICS.shiftedComponentCount]} components, ` +
              `jumps ${statistics[GPU_TERRAIN_SPIKE_REPAIR_STATISTICS.jumpCount]} to ` +
              `${statistics[GPU_TERRAIN_SPIKE_REPAIR_STATISTICS.remainingJumpCount]}` +
              (statistics[GPU_TERRAIN_SPIKE_REPAIR_STATISTICS.converged] ? '' : ' (not converged)')
          : 'off'
      );
      readouts.mismatch.setValue(
        `${formatCount(remaining)} px differ from the source` +
          (remaining ? `, max ${remainingError.toFixed(1)} m` : '')
      );
    }

    function rebuildSpikes(): void {
      if (injectSpikes) {
        terrarium = cleanTerrarium.slice();
        injection = injectTerrariumSpikes(terrarium, sourceValid, width, height, spikeSeed);
      } else {
        terrarium = cleanTerrarium;
        injection = {blockCount: 0, pixelCount: 0, mask: new Uint8Array(pixelCount)};
      }
    }

    // --- Readers --------------------------------------------------------------------------------
    const summitReader = new SummaryReader(
      resources,
      'summits',
      [
        {buffer: summitCount, size: 4},
        {buffer: summitTotal, size: 4},
        {buffer: summitOverflow, size: 4},
        {buffer: summitClamped, size: 4},
        {buffer: summitIds, size: SUMMIT_CAPACITY * 4},
        {buffer: summitDrops, size: SUMMIT_CAPACITY * 4}
      ],
      bytes => {
        const header = new Uint32Array(bytes, 0, 4);
        const count = Math.min(header[0], SUMMIT_CAPACITY);
        const ids = new Uint32Array(bytes, 16, SUMMIT_CAPACITY);
        const drops = new Float32Array(bytes, 16 + SUMMIT_CAPACITY * 4, SUMMIT_CAPACITY);
        updateSummits(count, header[1], header[2] !== 0, header[3] !== 0, ids, drops);
      }
    );
    const criticalReader = new SummaryReader(
      resources,
      'critical',
      [{buffer: criticalCounts, size: GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT * 4}],
      bytes => {
        const counts = new Uint32Array(bytes, 0, GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT);
        const peaks = counts[GPU_TERRAIN_CRITICAL_POINT.peak];
        const pits = counts[GPU_TERRAIN_CRITICAL_POINT.pit];
        const saddles = counts[GPU_TERRAIN_CRITICAL_POINT.saddle];
        readouts.critical.setValue(
          `${formatCount(peaks)} peaks, ${formatCount(pits)} pits, ${formatCount(saddles)} saddles`
        );
        readouts.euler.setValue(
          `${formatCount(peaks - saddles + pits)} (peaks - saddles + pits, ${ring}-ring)`
        );
        readouts.criticalOther.setValue(
          `${formatCount(counts[GPU_TERRAIN_CRITICAL_POINT.boundary])} boundary, ` +
            `${formatCount(counts[GPU_TERRAIN_CRITICAL_POINT.noData])} no data`
        );
      }
    );
    const snapReader = new SummaryReader(
      resources,
      'snap',
      [
        {buffer: snapPositions, size: CANDIDATE_CAPACITY * 8},
        {buffer: snapStatus, size: CANDIDATE_CAPACITY * 4},
        {buffer: snapDistance, size: CANDIDATE_CAPACITY * 4},
        {buffer: snapHeights, size: CANDIDATE_CAPACITY * 4}
      ],
      bytes => {
        const positions = new Float32Array(bytes, 0, CANDIDATE_CAPACITY * 2);
        const status = new Uint32Array(bytes, CANDIDATE_CAPACITY * 8, CANDIDATE_CAPACITY);
        const distance = new Float32Array(bytes, CANDIDATE_CAPACITY * 12, CANDIDATE_CAPACITY);
        const snapped = new Float32Array(bytes, CANDIDATE_CAPACITY * 16, CANDIDATE_CAPACITY);
        updateSnap(positions, status, distance, snapped);
      }
    );
    const profileReader = new SummaryReader(
      resources,
      'profile',
      [
        {buffer: profileCount, size: 4},
        {buffer: peakConverged, size: 4},
        {buffer: profileValues, size: PROFILE_CAPACITY * 4},
        {buffer: peakProminence, size: PROFILE_CAPACITY * 4},
        {buffer: peakIndex, size: PROFILE_CAPACITY * 4},
        {buffer: peakValue, size: PROFILE_CAPACITY * 4},
        {buffer: peakMask, size: PROFILE_CAPACITY * 4}
      ],
      bytes => {
        const header = new Uint32Array(bytes, 0, 2);
        const column = (index: number) => 8 + index * PROFILE_CAPACITY * 4;
        updateProfile(
          Math.min(header[0], PROFILE_CAPACITY),
          header[1] !== 0,
          new Float32Array(bytes, column(0), PROFILE_CAPACITY),
          new Float32Array(bytes, column(1), PROFILE_CAPACITY),
          new Float32Array(bytes, column(2), PROFILE_CAPACITY),
          new Float32Array(bytes, column(3), PROFILE_CAPACITY),
          new Uint32Array(bytes, column(4), PROFILE_CAPACITY)
        );
      }
    );

    function updateSummits(
      count: number,
      total: number,
      overflow: boolean,
      clamped: boolean,
      ids: Uint32Array,
      drops: Float32Array
    ): void {
      let maximumDrop = 0;
      for (let row = 0; row < count; row++) {
        if (Number.isFinite(drops[row])) maximumDrop = Math.max(maximumDrop, drops[row]);
      }
      // Four size classes of sqrt(drop / maximum drop), so mid-sized summits stay visible.
      const bucketPositions = Array.from({length: SUMMIT_BUCKET_COUNT}, () => [] as number[]);
      let heightest = -Infinity;
      for (let row = 0; row < count; row++) {
        const id = ids[row];
        const drop = Number.isFinite(drops[row]) ? drops[row] : maximumDrop;
        const t = maximumDrop > 0 ? Math.sqrt(Math.min(drop / maximumDrop, 1)) : 0;
        const bucket = Math.min(SUMMIT_BUCKET_COUNT - 1, Math.floor(t * SUMMIT_BUCKET_COUNT));
        bucketPositions[bucket].push(...pixelToMeters(id % width, Math.floor(id / width)));
        heightest = Math.max(heightest, workHeights[id]);
      }
      summitBucketCounts = bucketPositions.map(positions => positions.length / 2);
      bucketPositions.forEach((positions, bucket) => {
        if (positions.length) summitBuckets[bucket].write(Float32Array.from(positions));
      });
      readouts.summits.setValue(
        `${formatCount(total)}${overflow ? ` (list capped at ${formatCount(SUMMIT_CAPACITY)})` : ''}` +
          (clamped ? ', radius clamped' : '') +
          (count ? `, highest ${heightest.toFixed(0)} m, max drop ${maximumDrop.toFixed(0)} m` : '')
      );
      context.updateLayers();
    }

    function updateSnap(
      positions: Float32Array,
      status: Uint32Array,
      distance: Float32Array,
      snappedHeights: Float32Array
    ): void {
      const count = candidates.length;
      snapRows = [];
      const segments: number[] = [];
      let snapped = 0;
      let totalMove = 0;
      let rejected = 0;
      for (let index = 0; index < count; index++) {
        snapRows.push({
          status: status[index],
          distance: distance[index],
          height: snappedHeights[index]
        });
        if (status[index] === GPU_TERRAIN_PEAK_SNAP_STATUS.snapped) {
          snapped++;
          totalMove += distance[index];
          segments.push(
            candidates[index][0],
            candidates[index][1],
            positions[index * 2],
            positions[index * 2 + 1]
          );
        } else if (status[index] !== GPU_TERRAIN_PEAK_SNAP_STATUS.unchanged) {
          rejected++;
        }
      }
      snapSegmentCount = segments.length / 4;
      if (segments.length) snapSegments.write(Float32Array.from(segments));
      readouts.snap.setValue(
        count
          ? `${count} points: ${snapped} snapped` +
              (snapped ? ` (mean ${(totalMove / snapped).toFixed(0)} m)` : '') +
              `, ${count - snapped - rejected} unchanged, ${rejected} kept (${describeStatuses(status, count)})`
          : 'click the map with the Snap tool'
      );
      context.updateLayers();
    }
    let snapSegmentCount = 0;

    function describeStatuses(status: Uint32Array, count: number): string {
      const tally = new Map<number, number>();
      for (let index = 0; index < count; index++) {
        if (
          status[index] === GPU_TERRAIN_PEAK_SNAP_STATUS.snapped ||
          status[index] === GPU_TERRAIN_PEAK_SNAP_STATUS.unchanged
        ) {
          continue;
        }
        tally.set(status[index], (tally.get(status[index]) ?? 0) + 1);
      }
      return (
        [...tally].map(([code, n]) => `${n} ${SNAP_STATUS_NAMES[code] ?? code}`).join(', ') ||
        'none'
      );
    }

    const profileChart = new MiniChart('Elevation profile (m)', 'm');
    const readoutAnchor = document.querySelector('[data-mode-readouts]');
    profileChart.insertBefore(readoutAnchor);

    function updateProfile(
      count: number,
      converged: boolean,
      values: Float32Array,
      prominence: Float32Array,
      refinedIndex: Float32Array,
      refinedValue: Float32Array,
      mask: Uint32Array
    ): void {
      const peaks: {index: number; value: number; prominence: number}[] = [];
      for (let sample = 0; sample < count; sample++) {
        if (mask[sample] && peaks.length < PROFILE_PEAK_CAPACITY) {
          peaks.push({
            index: refinedIndex[sample],
            value: refinedValue[sample],
            prominence: prominence[sample]
          });
        }
      }
      profileSamples = {values: values.slice(0, count), peaks};
      const pixelLength = Math.hypot(
        profileEnd[0] - profileStart[0],
        profileEnd[1] - profileStart[1]
      );
      const spacingMeters = (pixelLength > 0 && count > 1 ? pixelLength / (count - 1) : 1) * cell;
      const xs = Array.from({length: count}, (_, sample) => sample * spacingMeters);
      let maximumProminence = 1;
      for (const peak of peaks) maximumProminence = Math.max(maximumProminence, peak.prominence);
      profileChart.update({
        series: [
          {kind: 'line', x: xs, y: profileSamples.values, color: '#9fb6e0'},
          {
            kind: 'points',
            x: peaks.map(peak => peak.index * spacingMeters),
            y: peaks.map(peak => peak.value),
            color: '#ff6d5a',
            radius: peaks.map(peak => 2.5 + 3.5 * Math.sqrt(peak.prominence / maximumProminence))
          }
        ],
        caption: `${peaks.length} peaks`
      });
      const markers: number[] = [];
      const length = Math.max(pixelLength, 1e-6);
      const directionX = (profileEnd[0] - profileStart[0]) / length;
      const directionY = (profileEnd[1] - profileStart[1]) / length;
      for (const peak of peaks) {
        markers.push(
          profileStart[0] + directionX * peak.index,
          profileStart[1] + directionY * peak.index
        );
      }
      profilePeakCount = peaks.length;
      if (markers.length) profilePeakMarkers.write(Float32Array.from(markers));
      readouts.profile.setValue(
        `${count} samples over ${(pixelLength * cell).toFixed(0)} m, ${peaks.length} peaks` +
          (converged ? '' : ' (suppression not converged)')
      );
      context.updateLayers();
    }
    let profilePeakCount = 0;

    // --- Controls -------------------------------------------------------------------------------
    const {controls} = context;
    controls.addSelect<BaseDisplay>({
      label: 'Base',
      options: [
        {value: 'hillshade', label: 'Hillshade'},
        {value: 'elevation', label: 'Elevation'}
      ],
      value: baseDisplay,
      onChange: value => {
        baseDisplay = value;
        context.updateLayers();
      }
    });
    controls.addToggle({
      label: 'Summits',
      value: showSummits,
      onChange: value => {
        showSummits = value;
        context.updateLayers();
      }
    });
    controls.addToggle({
      label: 'Critical points',
      value: showCritical,
      onChange: value => {
        showCritical = value;
        context.updateLayers();
      }
    });
    controls.addToggle({
      label: 'Snapped points',
      value: showSnap,
      onChange: value => {
        showSnap = value;
        context.updateLayers();
      }
    });
    controls.addToggle({
      label: 'Profile line',
      value: showProfile,
      onChange: value => {
        showProfile = value;
        context.updateLayers();
      }
    });
    controls.addSelect<ClickTool>({
      label: 'Map click',
      options: [
        {value: 'snap', label: 'Add snap candidate'},
        {value: 'profile', label: 'Draw profile (2 clicks)'}
      ],
      value: clickTool,
      onChange: value => {
        clickTool = value;
        pendingProfileStart = null;
        writeProfileEndpoints();
      }
    });
    controls.addSlider({
      label: 'Summit radius (per-frame)',
      min: 2 * cell,
      max: 15 * cell,
      step: cell / 2,
      value: summitRadius,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        summitRadius = value;
        writeSummitSettings();
      }
    });
    controls.addSlider({
      label: 'Summit minimum drop (per-frame)',
      min: 0,
      max: 80,
      step: 1,
      value: summitMinimumDrop,
      format: value => `${value} m`,
      onChange: value => {
        summitMinimumDrop = value;
        writeSummitSettings();
      }
    });
    controls.addSelect<Ring>({
      label: 'Critical point ring (rebuilds graph)',
      options: [
        {value: '8', label: '8 neighbours (Peucker-Douglas)'},
        {value: '6', label: '6 neighbours (Freudenthal)'}
      ],
      value: ring,
      onChange: value => {
        ring = value;
        resources.release(compiledCritical);
        compiledCritical = buildCriticalGraph(ring);
        dirty.critical = true;
      }
    });
    controls.addSlider({
      label: 'Snap radius (per-frame)',
      min: 2 * cell,
      max: 15 * cell,
      step: cell / 2,
      value: snapRadius,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        snapRadius = value;
        writeSnapSettings();
      }
    });
    controls.addSlider({
      label: 'Snap maximum move (per-frame)',
      min: cell,
      max: 15 * cell,
      step: cell / 2,
      value: snapMaximumMove,
      format: value => `${value.toFixed(0)} m`,
      onChange: value => {
        snapMaximumMove = value;
        writeSnapSettings();
      }
    });
    controls.addSlider({
      label: 'Snap maximum height change (per-frame)',
      min: 0,
      max: 200,
      step: 5,
      value: snapMaximumHeightChange,
      format: value => `${value} m`,
      onChange: value => {
        snapMaximumHeightChange = value;
        writeSnapSettings();
      }
    });
    controls.addToggle({
      label: 'Snap keeps flank points (interior rule)',
      value: snapInterior,
      onChange: value => {
        snapInterior = value;
        writeSnapSettings();
      }
    });
    controls.addButton({
      label: 'Clear snap candidates',
      onClick: () => {
        candidates.length = 0;
        writeCandidates();
        snapSegmentCount = 0;
        readouts.snap.setValue('click the map with the Snap tool');
        context.updateLayers();
      }
    });
    controls.addSelect<ProfileWindow>({
      label: 'Profile prominence window (rebuilds graph)',
      options: [
        {value: '8', label: '8 samples'},
        {value: '16', label: '16 samples'},
        {value: '32', label: '32 samples'},
        {value: '64', label: '64 samples'}
      ],
      value: profileWindow,
      onChange: value => {
        profileWindow = value;
        resources.release(compiledProfile);
        compiledProfile = buildProfileGraph(Number(profileWindow));
        dirty.profile = true;
      }
    });
    controls.addSlider({
      label: 'Profile minimum prominence (per-frame)',
      min: 0,
      max: 60,
      step: 1,
      value: minimumProminence,
      format: value => `${value} m`,
      onChange: value => {
        minimumProminence = value;
        writeProfileSettings();
      }
    });
    controls.addToggle({
      label: 'Inject +/-256 m spikes into the Terrarium texture',
      value: injectSpikes,
      onChange: value => {
        injectSpikes = value;
        rebuildSpikes();
        void runPrep();
      }
    });
    controls.addToggle({
      label: 'Repair spikes (GPUTerrainSpikeRepair)',
      value: repairSpikes,
      onChange: value => {
        repairSpikes = value;
        void runPrep();
      }
    });
    controls.addButton({
      label: 'Re-roll spikes',
      onClick: () => {
        spikeSeed++;
        if (injectSpikes) {
          rebuildSpikes();
          void runPrep();
        }
      }
    });
    controls.addLegend({
      title: 'Summit drop (size)',
      entries: SUMMIT_BUCKET_COLOR.map((color, bucket) => ({
        color,
        label: ['small', 'moderate', 'large', 'largest'][bucket]
      }))
    });
    controls.addLegend({
      title: 'Critical points',
      entries: [
        {color: CRITICAL_PALETTE[GPU_TERRAIN_CRITICAL_POINT.peak], label: 'Peak'},
        {color: CRITICAL_PALETTE[GPU_TERRAIN_CRITICAL_POINT.pit], label: 'Pit'},
        {color: CRITICAL_PALETTE[GPU_TERRAIN_CRITICAL_POINT.saddle], label: 'Saddle'}
      ]
    });
    controls.addLegend({
      title: 'Snap result',
      entries: [
        {color: SNAP_STATUS_PALETTE[GPU_TERRAIN_PEAK_SNAP_STATUS.snapped], label: 'Snapped'},
        {color: SNAP_STATUS_PALETTE[GPU_TERRAIN_PEAK_SNAP_STATUS.unchanged], label: 'Unchanged'},
        {color: SNAP_STATUS_PALETTE[GPU_TERRAIN_PEAK_SNAP_STATUS.onRing], label: 'On ring (flank)'},
        {
          color: SNAP_STATUS_PALETTE[GPU_TERRAIN_PEAK_SNAP_STATUS.rejectedMove],
          label: 'Move too far'
        },
        {
          color: SNAP_STATUS_PALETTE[GPU_TERRAIN_PEAK_SNAP_STATUS.rejectedHeight],
          label: 'Height change too large'
        }
      ]
    });
    controls.addNote(
      'White rings are the original candidates, filled discs the snapped result with a segment ' +
        'between them. The cyan line is the profile; its peaks are red.'
    );
    controls.addReadout('Raster', `${width} x ${height} cells, ${cell.toFixed(1)} m`);
    readouts.decode = controls.addReadout('Terrarium decode', '...');
    readouts.injected = controls.addReadout('Spikes injected', '...');
    readouts.repaired = controls.addReadout('Spikes repaired', '...');
    readouts.mismatch = controls.addReadout('After repair', '...');
    readouts.summits = controls.addReadout('Summits', '...');
    readouts.critical = controls.addReadout('Critical points', '...');
    readouts.criticalOther = controls.addReadout('Unclassified', '...');
    readouts.euler = controls.addReadout('Euler count', '...');
    readouts.snap = controls.addReadout('Peak snap', '...');
    readouts.profile = controls.addReadout('Profile peaks', '...');
    const timingReadouts = {
      prep: controls.addReadout('Decode + repair (GPU)', 'press Measure'),
      summits: controls.addReadout('Summits (GPU)', ''),
      critical: controls.addReadout('Critical points (GPU)', ''),
      snap: controls.addReadout('Peak snap (GPU)', ''),
      profile: controls.addReadout('Profile + peaks (GPU)', '')
    };
    controls.addReadout('Data', terrain.attribution);

    const measureAll = async () => {
      try {
        const run = async (
          compiled: CompiledGPUCommandGraph<void>,
          completionBuffer: Buffer,
          readout: (typeof timingReadouts)['prep']
        ) => {
          readout.setValue('measuring...');
          const timing = await measureCompiledGraph(device, compiled, {
            parameters: undefined,
            completionBuffer,
            signal: context.signal,
            runs: 5
          });
          if (!destroyed) {
            readout.setValue(
              `${compiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(timing)}`
            );
          }
        };
        await run(compiledPrep, repairStatistics, timingReadouts.prep);
        await run(compiledSummits, summitCount, timingReadouts.summits);
        await run(compiledCritical, criticalCounts, timingReadouts.critical);
        await run(compiledSnap, snapStatus, timingReadouts.snap);
        await run(compiledProfile, profileCount, timingReadouts.profile);
      } catch (error) {
        if (!destroyed) timingReadouts.prep.setValue(`failed: ${(error as Error).message}`);
      }
    };
    controls.addButton({label: 'Measure GPU cost', onClick: () => void measureAll()});

    // --- Initial state --------------------------------------------------------------------------
    writeDerivatives();
    writeSummitSettings();
    writeSnapSettings();
    writeProfileSettings();
    rebuildSpikes();
    await runPrep();
    context.signal.throwIfAborted();
    seedFromHeights();
    writeCandidates();
    writeProfileLine();

    /** Seeds four candidates near the highest separated summits and a profile across the top. */
    function seedFromHeights(): void {
      // Search near the initial camera so the seeds are on screen.
      const [centerX, centerY] = projection.project(
        terrainFeaturesMode.initialViewState.longitude,
        terrainFeaturesMode.initialViewState.latitude
      );
      const centerColumn = (centerX - bounds[0]) / cellSize[0] - 0.5;
      const centerRow = (bounds[3] - centerY) / cellSize[1] - 0.5;
      const picked: [number, number][] = [];
      for (let round = 0; round < 4; round++) {
        let best = -1;
        let bestHeight = -Infinity;
        for (let index = 0; index < pixelCount; index++) {
          const value = workHeights[index];
          if (!(value > bestHeight)) continue;
          const column = index % width;
          const row = Math.floor(index / width);
          if (Math.hypot(column - centerColumn, row - centerRow) > 150) continue;
          if (picked.some(([c, r]) => Math.hypot(c - column, r - row) < 36)) continue;
          best = index;
          bestHeight = value;
        }
        if (best < 0) break;
        picked.push([best % width, Math.floor(best / width)]);
      }
      const clamp = (value: number, size: number) => Math.min(Math.max(value, 1), size - 2);
      picked.forEach(([column, row], index) => {
        candidates.push([
          clamp(column + 3 - (index % 2) * 5, width),
          clamp(row + 2 + (index % 3), height)
        ]);
      });
      const [column, row] = picked[0] ?? [width / 2, height / 2];
      profileStart = [clamp(column - 90, width), clamp(row - 28, height)];
      profileEnd = [clamp(column + 90, width), clamp(row + 28, height)];
    }

    const getPixel = (event: SpatialAnalysisPointerEvent): [number, number] | null => {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const column = (x - bounds[0]) / cellSize[0] - 0.5;
      const row = (bounds[3] - y) / cellSize[1] - 0.5;
      if (column < 0 || row < 0 || column > width - 1 || row > height - 1) return null;
      return [column, row];
    };

    // --- Instance -------------------------------------------------------------------------------
    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [
        compiledPrep,
        compiledBase,
        compiledSummits,
        compiledCritical,
        compiledSnap,
        compiledProfile
      ],
      encode(commandEncoder: CommandEncoder) {
        if (preparing) return;
        const parameters = {parameters: undefined};
        if (dirty.base) {
          compiledBase.encode(commandEncoder, parameters);
          dirty.base = false;
        }
        if (dirty.summits) {
          compiledSummits.encode(commandEncoder, parameters);
          summitReader.request(commandEncoder);
          dirty.summits = false;
        }
        if (dirty.critical) {
          compiledCritical.encode(commandEncoder, parameters);
          criticalReader.request(commandEncoder);
          dirty.critical = false;
        }
        if (dirty.snap) {
          compiledSnap.encode(commandEncoder, parameters);
          snapReader.request(commandEncoder);
          dirty.snap = false;
        }
        if (dirty.profile) {
          compiledProfile.encode(commandEncoder, parameters);
          profileReader.request(commandEncoder);
          dirty.profile = false;
        }
        for (const reader of [summitReader, criticalReader, snapReader, profileReader]) {
          reader.flush(commandEncoder);
        }
      },
      getLayers() {
        const layers: Layer[] = [];
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [width, height] as const,
          bounds,
          rowOrigin: 'north' as const,
          valueFormat: 'float32' as const
        };
        if (baseDisplay === 'hillshade') {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'features-hillshade',
              values: hillshadeBuffer,
              colormap: 'grayscale',
              valueRange: [0, 1],
              color: [255, 255, 255, 190]
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'features-elevation',
              values: workValues,
              colormap: 'viridis',
              valueRange: [terrain.elevationRange[0], terrain.elevationRange[1]],
              color: [255, 255, 255, 200]
            })
          );
        }
        if (showCritical) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: 'features-critical',
              values: criticalClasses,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: CRITICAL_PALETTE
            })
          );
        }
        if (showProfile) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'features-profile-halo',
              coordinateOrigin: origin,
              segments: profileLineBuffer,
              instanceCount: 1,
              positionScale: pixelScale,
              positionOffset: pixelOffset,
              widthPixels: 5,
              color: [8, 16, 30, 200]
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'features-profile',
              coordinateOrigin: origin,
              segments: profileLineBuffer,
              instanceCount: 1,
              positionScale: pixelScale,
              positionOffset: pixelOffset,
              widthPixels: 2.2,
              color: [70, 225, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'features-profile-ends',
              coordinateOrigin: origin,
              positions: profileEndpoints,
              instanceCount: pendingProfileStart ? 1 : 2,
              positionScale: pixelScale,
              positionOffset: pixelOffset,
              radiusPixels: 6,
              color: [70, 225, 255, 255]
            })
          );
          if (profilePeakCount > 0) {
            layers.push(
              new SpatialAnalysisPointLayer({
                id: 'features-profile-peaks-halo',
                coordinateOrigin: origin,
                positions: profilePeakMarkers,
                instanceCount: profilePeakCount,
                positionScale: pixelScale,
                positionOffset: pixelOffset,
                radiusPixels: 7.5,
                color: [255, 255, 255, 240]
              }),
              new SpatialAnalysisPointLayer({
                id: 'features-profile-peaks',
                coordinateOrigin: origin,
                positions: profilePeakMarkers,
                instanceCount: profilePeakCount,
                positionScale: pixelScale,
                positionOffset: pixelOffset,
                radiusPixels: 5,
                color: [255, 90, 70, 255]
              })
            );
          }
        }
        if (showSummits) {
          summitBuckets.forEach((buffer, bucket) => {
            if (summitBucketCounts[bucket] === 0) return;
            layers.push(
              new SpatialAnalysisPointLayer({
                id: `features-summit-halo-${bucket}`,
                coordinateOrigin: origin,
                positions: buffer,
                instanceCount: summitBucketCounts[bucket],
                radiusPixels: SUMMIT_BUCKET_RADIUS[bucket] + 1.5,
                color: [10, 14, 24, 200]
              }),
              new SpatialAnalysisPointLayer({
                id: `features-summit-${bucket}`,
                coordinateOrigin: origin,
                positions: buffer,
                instanceCount: summitBucketCounts[bucket],
                radiusPixels: SUMMIT_BUCKET_RADIUS[bucket],
                color: SUMMIT_BUCKET_COLOR[bucket]
              })
            );
          });
        }
        if (showSnap && candidates.length > 0) {
          if (snapSegmentCount > 0) {
            layers.push(
              new SpatialAnalysisSegmentLayer({
                id: 'features-snap-segments',
                coordinateOrigin: origin,
                segments: snapSegments,
                instanceCount: snapSegmentCount,
                positionScale: pixelScale,
                positionOffset: pixelOffset,
                widthPixels: 2,
                color: [255, 255, 255, 230]
              })
            );
          }
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'features-snap-original-halo',
              coordinateOrigin: origin,
              positions: candidateBuffer,
              instanceCount: candidates.length,
              positionScale: pixelScale,
              positionOffset: pixelOffset,
              radiusPixels: 8,
              color: [255, 255, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'features-snap-original',
              coordinateOrigin: origin,
              positions: candidateBuffer,
              instanceCount: candidates.length,
              positionScale: pixelScale,
              positionOffset: pixelOffset,
              radiusPixels: 5,
              color: [30, 40, 60, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: 'features-snap-result',
              coordinateOrigin: origin,
              positions: snapPositions,
              instanceCount: candidates.length,
              positionScale: pixelScale,
              positionOffset: pixelOffset,
              radiusPixels: 5.5,
              values: snapStatus,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: SNAP_STATUS_PALETTE
            })
          );
        }
        return layers;
      },
      onClick(event) {
        const pixel = getPixel(event);
        if (!pixel) return false;
        if (clickTool === 'snap') {
          if (candidates.length >= CANDIDATE_CAPACITY) candidates.shift();
          candidates.push(pixel);
          writeCandidates();
        } else if (!pendingProfileStart) {
          pendingProfileStart = pixel;
          writeProfileEndpoints();
        } else {
          profileStart = pendingProfileStart;
          profileEnd = pixel;
          pendingProfileStart = null;
          writeProfileLine();
        }
        context.updateLayers();
        return true;
      },
      getTooltip(event) {
        const pixel = getPixel(event);
        if (!pixel) return null;
        const index = Math.round(pixel[1]) * width + Math.round(pixel[0]);
        const value = workHeights[index];
        return Number.isFinite(value) ? `${value.toFixed(0)} m` : null;
      },
      destroy() {
        destroyed = true;
        profileChart.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};

/** Encodes heights as Terrarium RGBA8: `h = R * 256 + G + B / 256 - 32768`, alpha 0 for nodata. */
function encodeTerrarium(elevation: Float32Array, valid: Uint8Array): Uint8Array {
  const bytes = new Uint8Array(elevation.length * 4);
  for (let index = 0; index < elevation.length; index++) {
    if (!valid[index]) continue;
    const shifted = elevation[index] + 32768;
    const whole = Math.floor(shifted);
    bytes[index * 4] = whole >> 8;
    bytes[index * 4 + 1] = whole & 255;
    bytes[index * 4 + 2] = Math.min(255, Math.round((shifted - whole) * 256));
    bytes[index * 4 + 3] = 255;
  }
  return bytes;
}

/**
 * Adds +/-1 to the Terrarium R byte (a +/-256 m error) of a few square blocks, like canvas noise
 * or a failed decode would. Blocks avoid nodata and each other, so repair is well-posed.
 */
function injectTerrariumSpikes(
  bytes: Uint8Array,
  valid: Uint8Array,
  width: number,
  height: number,
  seed: number
): SpikeInjection {
  const mask = new Uint8Array(width * height);
  let state = (seed * 2654435761) >>> 0 || 1;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  let blockCount = 0;
  let pixelCount = 0;
  for (let attempt = 0; attempt < 600 && blockCount < SPIKE_BLOCK_COUNT; attempt++) {
    const size = 4 + Math.floor(random() * 6);
    const left = 3 + Math.floor(random() * (width - size - 6));
    const top = 3 + Math.floor(random() * (height - size - 6));
    let usable = true;
    for (let y = top - 3; y < top + size + 3 && usable; y++) {
      for (let x = left - 3; x < left + size + 3; x++) {
        if (!valid[y * width + x] || mask[y * width + x]) {
          usable = false;
          break;
        }
      }
    }
    if (!usable) continue;
    const sign = blockCount % 2 === 0 ? 1 : -1;
    for (let y = top; y < top + size; y++) {
      for (let x = left; x < left + size; x++) {
        const index = y * width + x;
        const red = bytes[index * 4] + sign;
        if (red < 0 || red > 255) continue;
        bytes[index * 4] = red;
        mask[index] = 1;
        pixelCount++;
      }
    }
    blockCount++;
  }
  return {blockCount, pixelCount, mask};
}
