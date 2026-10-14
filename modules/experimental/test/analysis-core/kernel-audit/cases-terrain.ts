// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {DrawCommandBuffer} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPURasterExtremaPyramidLayout,
  makeGPURasterBand,
  makeGPURasterMetadata,
  type GPURasterExtremaPyramidOutput
} from '../../../src/gpu-raster';
import {
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPUGeomorphons,
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  GPUTerrainFlow,
  GPUTerrainHeightAboveDrainage,
  GPUTerrainHydrologicIndices,
  GPUTerrainStreamOrder,
  GPUTerrainWatersheds,
  GPUSimpleLocalRelief,
  GPUMultiScaleRelief,
  GPULocalDominance,
  GPUReliefBlend,
  GPU_LOCAL_DOMINANCE_MAXIMUM_SHIFT_COUNT,
  GPU_RELIEF_BLEND_MAXIMUM_LAYER_COUNT,
  GPU_RELIEF_BLEND_PARAMETER_LENGTH,
  GPUTerrainDerivatives,
  GPUTerrainContours,
  GPUTerrainViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainCumulativeViewshed,
  GPUTerrainCurvature,
  GPUTerrainRGBDecode,
  GPUTerrainSpikeRepair,
  GPUTerrainSummits,
  GPUTerrainPeakProminence,
  GPUTerrainPeakSnap,
  GPUTerrainCriticalPoints,
  GPUProfilePeaks,
  GPUTerrainFlowField,
  GPUSolarPosition,
  GPUTerrainHorizon,
  GPUTerrainCastShadow,
  GPUSolarIrradiance,
  GPUSolarShadowMask,
  GPUReliefShading,
  GPUTextureShading,
  GPUTerrainRuggedness,
  GPUTerrainVectorRuggedness,
  GPUTerrainTopographicPosition,
  GPUTerrainWeissLandforms,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT,
  getGPUTerrainPeakSnapMaximumCandidateCount,
  getGPUTerrainSpikeRepairMaximumCellCount,
  getGPUTerrainSummedAreaMaximumCellCount,
  GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES,
  GPU_TEXTURE_SHADING_MAXIMUM_LEVEL_COUNT,
  type GPUTerrainHorizonFormat,
  GPU_PROFILE_PEAKS_MAXIMUM_RADIUS,
  GPU_PROFILE_PEAKS_SUPPRESSION_ITERATION_LIMIT,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT,
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureMethod,
  type GPUGeomorphonComparison,
  type GPUTerrainFlowRouting
} from '../../../src/gpu-terrain';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';
import {getGPURasterRelaxationContinuationStateLength} from '../../../src/gpu-raster/cost-distance/raster-relaxation';

/** Small grid for the "every option" cases. */
const SMALL = makeGPURasterMetadata({width: 20, height: 15});
/** Largest grid whose float32 and uint32 per-cell buffers (64 MiB) fit one binding. */
const LARGE = makeGPURasterMetadata({width: 4096, height: 4096});

/**
 * Largest grid whose textures the fake core device accepts (its `maxTextureDimension2D` is 2048, below
 * the core 8192); texture outputs are only wired on grids up to this size.
 */
const MIDDLE = makeGPURasterMetadata({width: 2048, height: 2048});

/** Grid whose 64-direction float32 horizon map (256 bytes per pixel) fills one 128 MiB binding. */
const HORIZON_LIMIT = makeGPURasterMetadata({width: 512, height: 1024});

/** Grid at exactly the summed-area table maximum, `floor(2^27 / 12)` cells (3315 x 3374). */
const SUMMED_AREA_LIMIT = makeGPURasterMetadata({width: 3315, height: 3374});

type Grid = typeof SMALL;

const cellCount = (grid: Grid) => grid.width * grid.height;

/** Elevation band over a transient float32 buffer view. */
function elevationBand({view}: KernelAuditContext, grid: Grid, name = 'elevation') {
  return makeGPURasterBand(view('float32', cellCount(grid), name), grid, {id: name});
}

/** Both grid sizes as `[label, grid]`. */
const GRIDS: [string, Grid][] = [
  ['every option', SMALL],
  ['maximum 4096 x 4096 grid', LARGE]
];

type CellSizeMode = 'uniform' | 'web-mercator' | 'geographic';
const CELL_SIZE_MODES: CellSizeMode[] = ['uniform', 'web-mercator', 'geographic'];

/** Extrema pyramid a consumer can share, as a `GPURasterExtremaPyramid` would output. */
function sharedPyramid({view}: KernelAuditContext, grid: Grid): GPURasterExtremaPyramidOutput {
  const layout = getGPURasterExtremaPyramidLayout(grid, {firstBlockSize: 4, footprint: 'bilinear'});
  return {layout, combined: view('float32', 2 * layout.length, 'pyramid')};
}

/** Compact id list with count, required count and overflow words. */
function compactOutput({view}: KernelAuditContext, capacity: number, name: string) {
  return {
    ids: view('uint32', capacity, `${name}-ids`),
    status: {
      count: view('uint32', 1, `${name}-count`),
      requiredCount: view('uint32', 1, `${name}-required`),
      overflow: view('uint32', 1, `${name}-overflow`)
    }
  };
}

const CURVATURE_KINDS: GPUTerrainCurvatureKind[] = [
  'profile',
  'plan',
  'tangential',
  'mean',
  'gaussian',
  'minimal',
  'maximal',
  'unsphericity',
  'difference',
  'horizontal-excess',
  'vertical-excess',
  'accumulation',
  'ring',
  'rotor',
  'laplacian'
];

/** Sight-line traversals, with whether the pyramid is shared (it only exists for `'pyramid'`). */
const SIGHT_MODES: ['march' | 'pyramid', boolean][] = [
  ['march', false],
  ['pyramid', false],
  ['pyramid', true]
];

/** Grids for contributors with texture inputs or outputs: textures are skipped on the large grid. */
const TEXTURE_GRIDS: [string, Grid][] = [...GRIDS, ['2048 x 2048 grid with textures', MIDDLE]];

/** Horizon map storage over `pixels` pixels and `directions` sectors (two codes per word if unorm16). */
function horizonView(
  {view}: KernelAuditContext,
  format: GPUTerrainHorizonFormat,
  pixels: number,
  directions: number,
  name = 'horizon'
) {
  return format === 'unorm16'
    ? view('uint32', Math.ceil((pixels * directions) / 2), name)
    : view('float32', pixels * directions, name);
}

const HORIZON_FORMATS: GPUTerrainHorizonFormat[] = ['float32', 'unorm16'];

const ROUTINGS: GPUTerrainFlowRouting[] = ['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'];
const COMPARISONS: GPUGeomorphonComparison[] = ['anglev1', 'anglev2', 'anglev2-distance'];

/** Kernel-audit cases for terrain. */
export const TERRAIN_CASES: KernelAuditCase[] = [
  {
    exportName: 'GPUTerrainFlow',
    name: 'continued flats with GPU stage dependencies',
    build: context => {
      const {graph, view} = context;
      const stage = () => ({
        state: view('uint32', getGPURasterRelaxationContinuationStateLength(SMALL)),
        values: view('float32', cellCount(SMALL))
      });
      graph.add(
        new GPUTerrainFlow({
          grid: SMALL,
          elevation: elevationBand(context, SMALL),
          parameters: view('float32', 8),
          resolveFlats: true,
          flatContinuation: {lower: stage(), higher: stage(), maximum: stage()},
          output: {
            flowDirections: view('uint32', cellCount(SMALL)),
            status: {flatsConverged: view('uint32', 1), flatsIterationCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainPeakProminence',
    name: 'spill graph, dense outputs',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUTerrainPeakProminence({
          grid: SMALL,
          elevation: elevationBand(context, SMALL),
          output: {
            prominence: view('float32', cellCount(SMALL)),
            keySaddle: view('float32', cellCount(SMALL)),
            parentPeak: view('uint32', cellCount(SMALL))
          }
        })
      );
    }
  },
  // ---- geomorphons
  ...COMPARISONS.flatMap(comparison =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUGeomorphons',
        name: `${comparison}, ${label}`,
        build: context => {
          const {graph, view} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUGeomorphons({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 8),
              searchRadius: grid === LARGE ? 1000 : 64,
              skipRadius: 3,
              comparison,
              cellSizeMode: 'geographic',
              rowDirection: 'north',
              output: {
                forms: view('uint32', count),
                ternary: view('uint32', count),
                pattern: view('uint32', count),
                outputValidity: view('uint32', count)
              }
            })
          );
        }
      })
    )
  ),

  // ---- hydrology: flow
  ...ROUTINGS.flatMap(flowRouting =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainFlow',
        name: `${flowRouting}, ${label}`,
        build: context => {
          const {graph, view} = context;
          const count = cellCount(grid);
          const iterations = grid === LARGE ? 1024 : 8;
          graph.add(
            new GPUTerrainFlow({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 8),
              cellSizeMode: flowRouting === 'd8' ? 'uniform' : 'web-mercator',
              fillDepressions: true,
              maximumFillIterations: iterations,
              resolveFlats: true,
              maximumFlatIterations: iterations,
              flowRouting,
              maximumAccumulationIterations: iterations,
              runoff: view('float32', count),
              accumulationUnits: 'area',
              output: {
                filledElevation: view('float32', count),
                flowDirections: view('uint32', count),
                cellClasses: view('uint32', count),
                accumulation: view('float32', count),
                streams: view('uint32', count),
                status: {
                  fillConverged: view('uint32', 1),
                  accumulationConverged: view('uint32', 1),
                  flatsConverged: view('uint32', 1),
                  fillIterationCount: view('uint32', 1),
                  flatsIterationCount: view('uint32', 1),
                  accumulationIterationCount: view('uint32', 1)
                }
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUTerrainFlow',
    name: 'geographic cells, directions only',
    build: context => {
      const {graph, view} = context;
      graph.add(
        new GPUTerrainFlow({
          grid: SMALL,
          elevation: elevationBand(context, SMALL),
          parameters: view('float32', 8),
          cellSizeMode: 'geographic',
          output: {flowDirections: view('uint32', cellCount(SMALL))}
        })
      );
    }
  },

  // ---- hydrology: tracing
  ...GRIDS.map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPUTerrainHeightAboveDrainage',
      name: label,
      build: context => {
        const {graph, view} = context;
        const count = cellCount(grid);
        graph.add(
          new GPUTerrainHeightAboveDrainage({
            grid,
            elevation: elevationBand(context, grid),
            flowDirections: view('uint32', count),
            streams: view('uint32', count),
            maximumIterations: 1024,
            output: {
              heightAboveDrainage: view('float32', count),
              drainageCells: view('uint32', count),
              status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...GRIDS.map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPUTerrainWatersheds',
      name: label,
      build: context => {
        const {graph, view} = context;
        const count = cellCount(grid);
        graph.add(
          new GPUTerrainWatersheds({
            grid,
            flowDirections: view('uint32', count),
            pourPoints: view('uint32', count),
            maximumIterations: 1024,
            output: {
              labels: view('uint32', count),
              status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...GRIDS.map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPUTerrainStreamOrder',
      name: label,
      build: context => {
        const {graph, view} = context;
        const count = cellCount(grid);
        graph.add(
          new GPUTerrainStreamOrder({
            grid,
            flowDirections: view('uint32', count),
            streams: view('uint32', count),
            maximumIterations: 1024,
            output: {
              streamOrder: view('uint32', count),
              status: {converged: view('uint32', 1), iterationCount: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...(['uniform', 'web-mercator', 'geographic'] as const).flatMap(cellSizeMode =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainHydrologicIndices',
        name: `${cellSizeMode}, ${label}`,
        build: context => {
          const {graph, view} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUTerrainHydrologicIndices({
              grid,
              elevation: elevationBand(context, grid),
              accumulation: view('float32', count),
              parameters: view('float32', 8),
              cellSizeMode,
              output: {
                slope: view('float32', count),
                specificCatchmentArea: view('float32', count),
                wetnessIndex: view('float32', count),
                streamPowerIndex: view('float32', count)
              }
            })
          );
        }
      })
    )
  ),

  // ---- point horizon
  ...(['planar', 'web-mercator'] as const).flatMap(projection =>
    (['pyramid', 'march'] as const).flatMap(traversal =>
      [true, false].map(
        (withSamples): KernelAuditCase => ({
          exportName: 'GPUPointHorizonProfile',
          name: `${projection}, ${traversal}, ${withSamples ? 'every output' : 'without samples'}`,
          build: context => {
            const {graph, view} = context;
            const observers = 40;
            const azimuthCount = 720;
            const rays = observers * (azimuthCount - 10);
            graph.add(
              new GPUPointHorizonProfile({
                grid: SMALL,
                elevation: elevationBand(context, SMALL),
                observers: view('float32x4', observers),
                parameters: view('float32', 16),
                projection,
                rowDirection: 'north',
                heightReference: 'absolute',
                traversal,
                azimuthCount,
                firstAzimuth: 10,
                azimuthSpan: azimuthCount - 10,
                minimumDistance: 5,
                maximumDistance: 20000,
                cellSize: 30,
                stepFactor: 1e-3,
                nearFactor: 0.02,
                cellSteps: 0.25,
                minimumStep: 0.5,
                maximumLatitude: 85,
                segmentTolerance: 1e-5,
                raysPerDispatch: 1000,
                output: {
                  tangent: view('float32', rays),
                  skylineAngle: view('float32', rays),
                  distance: view('float32', rays),
                  ...(withSamples ? {samples: view('uint32', rays)} : {})
                }
              })
            );
          }
        })
      )
    )
  ),
  ...[true, false].map(
    (withSamples): KernelAuditCase => ({
      exportName: 'GPUPointHorizonProfile',
      name: `maximum azimuth count and rays at the binding limit${withSamples ? '' : ', without samples'}`,
      build: context => {
        const {graph, view, maximumRows} = context;
        const azimuthCount = 1 << 24;
        const rays = Math.min(azimuthCount, maximumRows(4));
        graph.add(
          new GPUPointHorizonProfile({
            grid: LARGE,
            elevation: elevationBand(context, LARGE),
            observers: view('float32x4', 1),
            parameters: view('float32', 16),
            azimuthCount,
            azimuthSpan: rays,
            maximumDistance: 100000,
            cellSize: 30,
            output: {
              tangent: view('float32', rays),
              skylineAngle: view('float32', rays),
              distance: view('float32', rays),
              ...(withSamples ? {samples: view('uint32', rays)} : {})
            }
          })
        );
      }
    })
  ),
  ...(['planar', 'web-mercator'] as const).flatMap(projection =>
    (['pyramid', 'march'] as const).map(
      (traversal): KernelAuditCase => ({
        exportName: 'GPUPointHorizonVisibility',
        name: `${projection}, ${traversal}, every output`,
        build: context => {
          const {graph, view} = context;
          const targets = 500;
          graph.add(
            new GPUPointHorizonVisibility({
              grid: SMALL,
              elevation: elevationBand(context, SMALL),
              observers: view('float32x4', 4),
              targets: view('float32x4', targets),
              parameters: view('float32', 16),
              projection,
              rowDirection: 'north',
              heightReference: 'absolute',
              traversal,
              minimumDistance: 5,
              maximumDistance: 20000,
              cellSize: 30,
              stepFactor: 1e-3,
              nearFactor: 0.02,
              cellSteps: 0.25,
              minimumStep: 0.5,
              maximumLatitude: 85,
              segmentTolerance: 1e-5,
              targetsPerDispatch: 100,
              output: {
                visibility: view('uint32', targets),
                details: view('float32x4', targets)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUPointHorizonVisibility',
    name: 'targets at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const targets = maximumRows(16);
      graph.add(
        new GPUPointHorizonVisibility({
          grid: LARGE,
          elevation: elevationBand(context, LARGE),
          observers: view('float32x4', 1000),
          targets: view('float32x4', targets),
          parameters: view('float32', 16),
          maximumDistance: 100000,
          cellSize: 30,
          output: {
            visibility: view('uint32', targets),
            details: view('float32x4', targets)
          }
        })
      );
    }
  },

  ...(['planar', 'web-mercator'] as const).flatMap(projection =>
    [false, true].map(
      (dense): KernelAuditCase => ({
        exportName: 'GPUPointHorizonProfile',
        name: `${projection}, shared pyramid${dense ? ', dense lattice' : ''}`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUPointHorizonProfile({
              grid: SMALL,
              elevation: elevationBand(context, SMALL),
              observers: view('float32x4', 3),
              parameters: view('float32', 16),
              projection,
              traversal: 'pyramid',
              pyramid: sharedPyramid(context, SMALL),
              maximumDistance: dense ? 1000000 : 20000,
              minimumDistance: dense ? 1 : undefined,
              cellSize: dense ? 1 : 30,
              stepFactor: dense ? 1e-5 : undefined,
              segmentTolerance: dense ? 1e-9 : undefined,
              output: {tangent: view('float32', 3 * 720), distance: view('float32', 3 * 720)}
            })
          );
        }
      })
    )
  ),
  ...(['planar', 'web-mercator'] as const).flatMap(projection =>
    [false, true].map(
      (dense): KernelAuditCase => ({
        exportName: 'GPUPointHorizonVisibility',
        name: `${projection}, shared pyramid${dense ? ', dense lattice' : ''}`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUPointHorizonVisibility({
              grid: SMALL,
              elevation: elevationBand(context, SMALL),
              observers: view('float32x4', 3),
              targets: view('float32x4', 100),
              parameters: view('float32', 16),
              projection,
              traversal: 'pyramid',
              pyramid: sharedPyramid(context, SMALL),
              maximumDistance: dense ? 1000000 : 20000,
              minimumDistance: dense ? 1 : undefined,
              cellSize: dense ? 1 : 30,
              stepFactor: dense ? 1e-5 : undefined,
              segmentTolerance: dense ? 1e-9 : undefined,
              output: {visibility: view('uint32', 100)}
            })
          );
        }
      })
    )
  ),

  // ---- relief visualization
  ...GRIDS.flatMap(([label, grid]) =>
    (['both outputs', 'relief only', 'validity only'] as const).map(
      (outputs): KernelAuditCase => ({
        exportName: 'GPUSimpleLocalRelief',
        name: `${outputs}, ${label}`,
        build: context => {
          const {graph, view} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUSimpleLocalRelief({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 4),
              maximumRadius: grid === LARGE ? 1000 : 20,
              output: {
                ...(outputs !== 'validity only' ? {relief: view('float32', count)} : {}),
                ...(outputs !== 'relief only' ? {outputValidity: view('uint32', count)} : {})
              }
            })
          );
        }
      })
    )
  ),
  ...GRIDS.map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPUMultiScaleRelief',
      name: label,
      build: context => {
        const {graph, view} = context;
        const count = cellCount(grid);
        graph.add(
          new GPUMultiScaleRelief({
            grid,
            elevation: elevationBand(context, grid),
            parameters: view('float32', 4),
            resolution: 1,
            featureMinimum: 1,
            featureMaximum: grid === LARGE ? 2001 : 41,
            scalingFactor: grid === LARGE ? 2 : 1,
            output: {relief: view('float32', count), outputValidity: view('uint32', count)}
          })
        );
      }
    })
  ),
  ...GRIDS.map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPULocalDominance',
      name: label,
      build: context => {
        const {graph, view} = context;
        const count = cellCount(grid);
        // Maximum sample offsets: 22 distances x 360 angles = 7920 of at most 8192.
        const maximal = grid === LARGE;
        graph.add(
          new GPULocalDominance({
            grid,
            elevation: elevationBand(context, grid),
            parameters: view('float32', 4),
            minimumRadius: 1,
            maximumRadius: maximal ? Math.floor(GPU_LOCAL_DOMINANCE_MAXIMUM_SHIFT_COUNT / 360) : 20,
            radiusIncrement: 1,
            angularResolution: maximal ? 1 : 15,
            output: {dominance: view('float32', count), outputValidity: view('uint32', count)}
          })
        );
      }
    })
  ),
  ...[1, GPU_RELIEF_BLEND_MAXIMUM_LAYER_COUNT].flatMap(layerCount =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUReliefBlend',
        name: `${layerCount} layers, ${label}`,
        build: context => {
          const {graph, view} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUReliefBlend({
              grid,
              layers: Array.from({length: layerCount}, (_, index) =>
                view('float32', count, `layer-${index}`)
              ),
              parameters: view('float32', GPU_RELIEF_BLEND_PARAMETER_LENGTH),
              output: {
                blend: view('float32', count),
                color: view('uint32', count),
                outputValidity: view('uint32', count)
              }
            })
          );
        }
      })
    )
  ),

  // ---- terrain analysis: derivatives
  ...(['clamp', 'reflect', 'constant', 'nodata'] as const).flatMap((borderMode, modeIndex) =>
    TEXTURE_GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainDerivatives',
        name: `${borderMode}, ${label}`,
        build: context => {
          const {graph, view, texture} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUTerrainDerivatives({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 8),
              cellSizeMode: CELL_SIZE_MODES[modeIndex % 3],
              slopeUnits: modeIndex % 2 === 0 ? 'degrees' : 'percent',
              rowDirection: modeIndex % 2 === 0 ? 'south' : 'north',
              borderMode,
              output: {
                slope: view('float32', count),
                aspect: view('float32', count),
                hillshade: view('float32', count),
                outputValidity: view('uint32', count),
                ...(grid === LARGE
                  ? {}
                  : {
                      hillshadeTexture: graph.createTextureView(
                        texture('rgba32float', grid.width, grid.height),
                        {mipLevelCount: 1}
                      )
                    })
              }
            })
          );
        }
      })
    )
  ),

  // ---- terrain analysis: contours
  ...GRIDS.map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPUTerrainContours',
      name: label,
      build: context => {
        const {graph, view, device, maximumRows} = context;
        // `capacity` counts segments, two vertices each.
        const capacity = grid === LARGE ? Math.floor(maximumRows(8) / 2) : 1000;
        const draw = new DrawCommandBuffer(device, {
          id: 'draw',
          type: 'draw',
          capacity: 16
        }).importToGraph(graph);
        const levelCount = grid === LARGE ? 16 : 6;
        graph.add(
          new GPUTerrainContours({
            grid,
            elevation: elevationBand(context, grid),
            levels: Array.from({length: levelCount}, (_, index) => ({
              level: index % 2 === 0 ? 100 * index : view('float32', 1, `level-${index}`),
              vertices: view('float32x2', 2 * capacity, `vertices-${index}`),
              capacity,
              status: {
                count: view('uint32', 1, `count-${index}`),
                requiredCount: view('uint32', 1, `required-${index}`),
                overflow: view('uint32', 1, `level-overflow-${index}`)
              },
              draw,
              drawCommandIndex: index,
              ...(index % 3 === 0
                ? {drawLayout: 'line-list' as const}
                : {verticesPerInstance: 2 + index})
            })),
            output: {status: {overflow: view('uint32', 1)}}
          })
        );
      }
    })
  ),

  // ---- terrain analysis: viewshed
  ...SIGHT_MODES.flatMap(([traversal, shared]) =>
    TEXTURE_GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainViewshed',
        name: `${traversal}${shared ? ' shared pyramid' : ''}, ${label}`,
        build: context => {
          const {graph, view, texture} = context;
          // A shared pyramid only exists for the pyramid traversal.
          if (shared && traversal !== 'pyramid') {
            return;
          }
          graph.add(
            new GPUTerrainViewshed({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH),
              toleranceParameters: view('float32', 4),
              traversal,
              ...(shared ? {pyramid: sharedPyramid(context, grid)} : {}),
              output: {
                visibility: view('uint32', cellCount(grid)),
                ...(grid === LARGE
                  ? {}
                  : {
                      visibilityTexture: graph.createTextureView(
                        texture('rgba32uint', grid.width, grid.height),
                        {mipLevelCount: 1}
                      )
                    })
              }
            })
          );
        }
      })
    )
  ),

  // ---- terrain analysis: line of sight
  ...SIGHT_MODES.flatMap(([traversal, shared]) =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainLineOfSight',
        name: `${traversal}${shared ? ' shared pyramid' : ''}, ${label}`,
        build: context => {
          const {graph, view, maximumRows} = context;
          if (shared && traversal !== 'pyramid') {
            return;
          }
          const pairs = grid === LARGE ? maximumRows(16) : 500;
          graph.add(
            new GPUTerrainLineOfSight({
              grid,
              elevation: elevationBand(context, grid),
              pairs: view('float32x4', pairs),
              pairHeights: view('float32x2', pairs),
              parameters: view('float32', 12),
              traversal,
              ...(shared ? {pyramid: sharedPyramid(context, grid)} : {}),
              output: {visibility: view('uint32', pairs), clearance: view('float32', pairs)}
            })
          );
        }
      })
    )
  ),

  // ---- terrain analysis: cumulative viewshed
  ...SIGHT_MODES.flatMap(([traversal, shared]) =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainCumulativeViewshed',
        name: `${traversal}${shared ? ' shared pyramid' : ''}, ${label}`,
        build: context => {
          const {graph, view} = context;
          if (shared && traversal !== 'pyramid') {
            return;
          }
          const count = cellCount(grid);
          const observers = grid === LARGE ? 1000 : 10;
          graph.add(
            new GPUTerrainCumulativeViewshed({
              grid,
              elevation: elevationBand(context, grid),
              observers: view('float32x2', observers),
              observerHeights: view('float32', observers),
              parameters: view('float32', 12),
              traversal,
              // The largest batch whose invocations plus one pixel pass fit 32 bits.
              observersPerDispatch: grid === LARGE ? Math.floor(0xffffffff / count) - 1 : 4,
              ...(shared ? {pyramid: sharedPyramid(context, grid)} : {}),
              output: {visibleCount: view('uint32', count), marginalCount: view('uint32', count)}
            })
          );
        }
      })
    )
  ),

  // ---- terrain curvature
  ...(['evans-young', 'zevenbergen-thorne', 'florinsky'] as GPUTerrainCurvatureMethod[]).flatMap(
    (method, methodIndex) =>
      GRIDS.map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTerrainCurvature',
          name: `${method}, ${label}`,
          build: context => {
            const {graph, view} = context;
            const count = cellCount(grid);
            const curvatures = Object.fromEntries(
              CURVATURE_KINDS.map(kind => [kind, view('float32', count, kind)])
            ) as Record<GPUTerrainCurvatureKind, ReturnType<typeof view<'float32'>>>;
            graph.add(
              new GPUTerrainCurvature({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', 12),
                method,
                cellSizeMode: CELL_SIZE_MODES[methodIndex],
                rowDirection: methodIndex === 1 ? 'north' : 'south',
                borderMode: methodIndex === 2 ? 'nodata' : 'clamp',
                ringRadii: grid === LARGE ? [16, 64, 256, 1024] : [1, 2, 3, 4],
                ringSquash: methodIndex === 0 ? 'none' : 'pade-tanh',
                output: {
                  curvatures,
                  ringCurvature: view('float32', count),
                  outputValidity: view('uint32', count)
                }
              })
            );
          }
        })
      )
  ),

  // ---- terrain decode: RGB
  ...(['terrarium', 'mapbox'] as const).flatMap(encoding =>
    ['texture', 'buffer'].flatMap(inputKind =>
      TEXTURE_GRIDS.filter(([, grid]) => inputKind === 'buffer' || grid !== LARGE).map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTerrainRGBDecode',
          name: `${encoding}, ${inputKind} input, ${label}`,
          build: context => {
            const {graph, view, texture} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUTerrainRGBDecode({
                grid,
                encoding,
                input:
                  inputKind === 'texture'
                    ? {
                        texture: graph.createTextureView(
                          texture('rgba8unorm', grid.width, grid.height),
                          {mipLevelCount: 1}
                        )
                      }
                    : {buffer: view('uint32', count)},
                ...(inputKind === 'buffer' ? {validity: view('uint32', count)} : {}),
                alphaNoData: true,
                noDataRGB: [0, 0, 0],
                validRange: [-11000, 9000],
                clampBathymetry: true,
                output: {values: view('float32', count), outputValidity: view('uint32', count)}
              })
            );
          }
        })
      )
    )
  ),

  // ---- terrain decode: spike repair
  ...(
    [
      ['every option', SMALL],
      ['grid at the maximum cell count', MIDDLE]
    ] as [string, Grid][]
  ).map(
    ([label, grid]): KernelAuditCase => ({
      exportName: 'GPUTerrainSpikeRepair',
      name: label,
      build: context => {
        const {graph, view} = context;
        const count = cellCount(grid);
        graph.add(
          new GPUTerrainSpikeRepair({
            grid,
            elevation: elevationBand(context, grid),
            step: 256,
            jump: 200,
            tolerance: 40,
            agreement: 0.99,
            maximumComponentFraction: 1,
            maximumComponentIterations: 1024,
            output: {
              values: view('float32', count),
              outputValidity: view('uint32', count),
              labels: view('uint32', count),
              statistics: view('uint32', GPU_TERRAIN_SPIKE_REPAIR_STATISTICS_LENGTH)
            }
          })
        );
      }
    })
  ),

  // ---- terrain features: summits
  ...(['reject', 'ignore'] as const).flatMap(incompleteNeighborhood =>
    CELL_SIZE_MODES.flatMap(cellSizeMode =>
      GRIDS.map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTerrainSummits',
          name: `${incompleteNeighborhood}, ${cellSizeMode}, ${label}`,
          build: context => {
            const {graph, view} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUTerrainSummits({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', 8),
                cellSizeMode,
                maximumRadiusPixels: 64,
                incompleteNeighborhood,
                output: {
                  summitMask: view('uint32', count),
                  drop: view('float32', count),
                  summits: compactOutput(context, count, 'summits'),
                  summitDrop: view('float32', count),
                  status: {radiusClamped: view('uint32', 1)}
                }
              })
            );
          }
        })
      )
    )
  ),

  // ---- terrain features: peak snap
  ...CELL_SIZE_MODES.flatMap(cellSizeMode =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainPeakSnap',
        name: `${cellSizeMode}, ${label}`,
        build: context => {
          const {graph, view} = context;
          const candidates =
            grid === LARGE ? getGPUTerrainPeakSnapMaximumCandidateCount(context.device) : 500;
          graph.add(
            new GPUTerrainPeakSnap({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 8),
              cellSizeMode,
              maximumRadiusPixels: 64,
              candidates: view('float32x2', candidates),
              candidateHeights: view('float32', candidates),
              candidateRadii: view('float32', candidates),
              output: {
                positions: view('float32x2', candidates),
                heights: view('float32', candidates),
                snapStatus: view('uint32', candidates),
                snapDistance: view('float32', candidates),
                status: {radiusClamped: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),

  {
    exportName: 'GPUTerrainPeakSnap',
    name: 'candidates at the maximum',
    build: context => {
      const {graph, view} = context;
      // The kernel keeps five uint32 result words per candidate in one binding.
      const candidates = getGPUTerrainPeakSnapMaximumCandidateCount(context.device);
      graph.add(
        new GPUTerrainPeakSnap({
          grid: LARGE,
          elevation: elevationBand(context, LARGE),
          parameters: view('float32', 8),
          maximumRadiusPixels: 64,
          candidates: view('float32x2', candidates),
          candidateHeights: view('float32', candidates),
          candidateRadii: view('float32', candidates),
          output: {
            positions: view('float32x2', candidates),
            heights: view('float32', candidates),
            snapStatus: view('uint32', candidates),
            snapDistance: view('float32', candidates),
            status: {radiusClamped: view('uint32', 1)}
          }
        })
      );
    }
  },

  // ---- terrain features: critical points
  ...([8, 6] as const).flatMap(connectivity =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainCriticalPoints',
        name: `connectivity ${connectivity}, ${label}`,
        build: context => {
          const {graph, view} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUTerrainCriticalPoints({
              grid,
              elevation: elevationBand(context, grid),
              connectivity,
              output: {
                classes: view('uint32', count),
                signChanges: view('uint32', count),
                counts: view('uint32', GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT)
              }
            })
          );
        }
      })
    )
  ),

  // ---- terrain features: profile peaks
  ...[false, true].flatMap(wrap =>
    [false, true].map(
      (maximal): KernelAuditCase => ({
        exportName: 'GPUProfilePeaks',
        name: `${wrap ? 'wrapped' : 'open'}, ${maximal ? 'maximum window and samples' : 'every option'}`,
        build: context => {
          const {graph, view, maximumRows} = context;
          const samples = maximal ? maximumRows(4) : 600;
          graph.add(
            new GPUProfilePeaks({
              values: view('float32', samples),
              validity: view('uint32', samples),
              offsets: view('uint32', maximal ? 100001 : 4),
              parameters: view('float32', 1),
              window: maximal ? GPU_PROFILE_PEAKS_MAXIMUM_RADIUS : 12,
              minimumSide: maximal ? 1000000 : 4,
              suppressionRadius: maximal ? GPU_PROFILE_PEAKS_MAXIMUM_RADIUS : 5,
              wrap,
              maximumSuppressionIterations: maximal
                ? GPU_PROFILE_PEAKS_SUPPRESSION_ITERATION_LIMIT
                : 16,
              output: {
                prominence: view('float32', samples),
                refinedIndex: view('float32', samples),
                refinedValue: view('float32', samples),
                peakMask: view('uint32', samples),
                peaks: compactOutput(context, samples, 'peaks'),
                status: {converged: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),

  // ---- terrain flow field
  ...CELL_SIZE_MODES.flatMap(cellSizeMode =>
    GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainFlowField',
        name: `${cellSizeMode}, ${label}`,
        build: context => {
          const {graph, view} = context;
          graph.add(
            new GPUTerrainFlowField({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 8),
              cellSizeMode,
              output: {velocities: view('float32x2', cellCount(grid))}
            })
          );
        }
      })
    )
  ),

  // ---- illumination: solar position
  ...[false, true].flatMap(refraction =>
    [false, true].map(
      (maximal): KernelAuditCase => ({
        exportName: 'GPUSolarPosition',
        name: `refraction ${refraction}, ${maximal ? 'rows at the binding limit' : 'every output'}`,
        build: context => {
          const {graph, view, maximumRows} = context;
          const rows = maximal ? maximumRows(8) : 300;
          graph.add(
            new GPUSolarPosition({
              positions: view('float32x2', rows),
              parameters: view('float32', 4),
              refraction,
              output: {
                azimuth: view('float32', rows),
                altitude: view('float32', rows),
                daylight: view('uint32', rows)
              }
            })
          );
        }
      })
    )
  ),

  // ---- illumination: horizon
  ...(['march', 'sweep'] as const).flatMap(algorithm =>
    HORIZON_FORMATS.flatMap(horizonFormat =>
      [false, true].map(
        (maximal): KernelAuditCase => ({
          exportName: 'GPUTerrainHorizon',
          name: `${algorithm}, ${horizonFormat}, ${maximal ? 'maximum 64 directions' : 'every option'}`,
          build: context => {
            const {graph, view, texture} = context;
            const grid = maximal
              ? horizonFormat === 'unorm16'
                ? makeGPURasterMetadata({width: 1024, height: 1024})
                : HORIZON_LIMIT
              : SMALL;
            const count = cellCount(grid);
            const directionCount = maximal ? GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT : 16;
            graph.add(
              new GPUTerrainHorizon({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH),
                directionCount,
                maximumRadius: maximal ? 2000 : 12,
                stepGrowth: algorithm === 'march' ? 1.1 : 1,
                algorithm,
                maximumStepsPerDispatch: 64,
                cellSizeMode: 'geographic',
                rowDirection: 'north',
                horizonFormat,
                output: {
                  horizon: horizonView(context, horizonFormat, count, directionCount),
                  skyViewFactor: view('float32', count),
                  positiveOpenness: view('float32', count),
                  negativeOpenness: view('float32', count),
                  anisotropicSkyViewFactor: view('float32', count),
                  outputValidity: view('uint32', count),
                  skyViewFactorTexture: graph.createTextureView(
                    texture('r32float', grid.width, grid.height),
                    {mipLevelCount: 1}
                  )
                }
              })
            );
          }
        })
      )
    )
  ),
  ...(['march', 'sweep'] as const).map(
    (algorithm): KernelAuditCase => ({
      exportName: 'GPUTerrainHorizon',
      name: `${algorithm}, large grid, sky view factor only`,
      build: context => {
        const {graph, view} = context;
        graph.add(
          new GPUTerrainHorizon({
            grid: LARGE,
            elevation: elevationBand(context, LARGE),
            parameters: view('float32', 8),
            directionCount: 64,
            maximumRadius: 4096,
            algorithm,
            output: {skyViewFactor: view('float32', cellCount(LARGE))}
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUTerrainHorizon',
    name: 'sweep, maximum extent 32767',
    build: context => {
      const {graph, view} = context;
      const grid = makeGPURasterMetadata({width: 32767, height: 1024});
      graph.add(
        new GPUTerrainHorizon({
          grid,
          elevation: elevationBand(context, grid),
          parameters: view('float32', 8),
          maximumRadius: 32767,
          algorithm: 'sweep',
          output: {skyViewFactor: view('float32', cellCount(grid))}
        })
      );
    }
  },

  // ---- illumination: cast shadow
  ...CELL_SIZE_MODES.flatMap(cellSizeMode =>
    TEXTURE_GRIDS.map(
      ([label, grid]): KernelAuditCase => ({
        exportName: 'GPUTerrainCastShadow',
        name: `${cellSizeMode}, ${label}`,
        build: context => {
          const {graph, view, texture} = context;
          const count = cellCount(grid);
          graph.add(
            new GPUTerrainCastShadow({
              grid,
              elevation: elevationBand(context, grid),
              parameters: view('float32', 16),
              maximumRadius: grid === SMALL ? 40 : 8192,
              cellSizeMode,
              rowDirection: cellSizeMode === 'uniform' ? 'south' : 'north',
              output: {
                sunVisibility: view('float32', count),
                horizonAngle: view('float32', count),
                outputValidity: view('uint32', count),
                ...(grid === LARGE
                  ? {}
                  : {
                      sunVisibilityTexture: graph.createTextureView(
                        texture('r32float', grid.width, grid.height),
                        {mipLevelCount: 1}
                      )
                    })
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUTerrainCastShadow',
    name: 'maximum extent 32767',
    build: context => {
      const {graph, view} = context;
      const grid = makeGPURasterMetadata({width: 32767, height: 1024});
      graph.add(
        new GPUTerrainCastShadow({
          grid,
          elevation: elevationBand(context, grid),
          parameters: view('float32', 16),
          output: {
            sunVisibility: view('float32', cellCount(grid)),
            horizonAngle: view('float32', cellCount(grid))
          }
        })
      );
    }
  },

  // ---- illumination: solar irradiance
  ...HORIZON_FORMATS.flatMap(horizonFormat =>
    [false, true].map(
      (maximal): KernelAuditCase => ({
        exportName: 'GPUSolarIrradiance',
        name: `${horizonFormat}, ${maximal ? 'maximum 64 directions and sun table' : 'every option'}`,
        build: context => {
          const {graph, view, texture, maximumRows} = context;
          const grid = maximal
            ? horizonFormat === 'unorm16'
              ? makeGPURasterMetadata({width: 1024, height: 1024})
              : HORIZON_LIMIT
            : SMALL;
          const count = cellCount(grid);
          const directionCount = maximal ? GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT : 16;
          const sampleCapacity = maximal
            ? maximumRows(4 * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE)
            : 96;
          graph.add(
            new GPUSolarIrradiance({
              grid,
              directionCount,
              horizonFormat,
              horizon: horizonView(context, horizonFormat, count, directionCount),
              sunTable: view('float32', sampleCapacity * GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE),
              sampleCapacity,
              samplesPerNode: maximal ? undefined : 7,
              parameters: view('float32', 8),
              slope: view('float32', count),
              aspect: view('float32', count),
              skyViewFactor: view('float32', count),
              output: {
                sunHours: view('float32', count),
                insolation: view('float32', count),
                outputValidity: view('uint32', count),
                ...(maximal
                  ? {}
                  : {
                      sunHoursTexture: graph.createTextureView(
                        texture('r32float', grid.width, grid.height),
                        {mipLevelCount: 1}
                      ),
                      insolationTexture: graph.createTextureView(
                        texture('rgba32float', grid.width, grid.height),
                        {mipLevelCount: 1}
                      )
                    })
              }
            })
          );
        }
      })
    )
  ),

  // ---- illumination: solar shadow mask
  ...HORIZON_FORMATS.flatMap(horizonFormat =>
    [false, true].map(
      (maximal): KernelAuditCase => ({
        exportName: 'GPUSolarShadowMask',
        name: `${horizonFormat}, ${maximal ? 'maximum 64 directions' : 'every option'}`,
        build: context => {
          const {graph, view, texture} = context;
          const grid = maximal
            ? horizonFormat === 'unorm16'
              ? makeGPURasterMetadata({width: 1024, height: 1024})
              : HORIZON_LIMIT
            : SMALL;
          const count = cellCount(grid);
          const directionCount = maximal ? GPU_TERRAIN_HORIZON_MAXIMUM_DIRECTION_COUNT : 16;
          graph.add(
            new GPUSolarShadowMask({
              grid,
              directionCount,
              horizonFormat,
              horizon: horizonView(context, horizonFormat, count, directionCount),
              parameters: view('float32', 8),
              slope: view('float32', count),
              aspect: view('float32', count),
              skyViewFactor: view('float32', count),
              output: {
                sunVisibility: view('float32', count),
                illumination: view('float32', count),
                ...(maximal
                  ? {}
                  : {
                      sunVisibilityTexture: graph.createTextureView(
                        texture('r32float', grid.width, grid.height),
                        {mipLevelCount: 1}
                      ),
                      illuminationTexture: graph.createTextureView(
                        texture('rgba32float', grid.width, grid.height),
                        {mipLevelCount: 1}
                      )
                    })
              }
            })
          );
        }
      })
    )
  ),

  // ---- illumination: relief shading
  ...[false, true].flatMap(imhofSwing =>
    (['clamp', 'reflect', 'constant', 'nodata'] as const).flatMap((borderMode, modeIndex) =>
      TEXTURE_GRIDS.map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUReliefShading',
          name: `imhof swing ${imhofSwing}, ${borderMode}, ${label}`,
          build: context => {
            const {graph, view, texture} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUReliefShading({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', GPU_RELIEF_SHADING_PARAMETER_LENGTH),
                skyViewFactor: view('float32', count),
                textureShade: view('float32', count),
                curvature: view('float32', count),
                cellSizeMode: CELL_SIZE_MODES[modeIndex % 3],
                rowDirection: modeIndex % 2 === 0 ? 'south' : 'north',
                borderMode,
                imhofSwing,
                output: {
                  hillshade: view('float32', count),
                  relief: view('float32', count),
                  color: view('uint32', count),
                  outputValidity: view('uint32', count),
                  ...(grid === LARGE
                    ? {}
                    : {
                        hillshadeTexture: graph.createTextureView(
                          texture('r32float', grid.width, grid.height),
                          {mipLevelCount: 1}
                        ),
                        reliefTexture: graph.createTextureView(
                          texture('rgba32float', grid.width, grid.height),
                          {mipLevelCount: 1}
                        )
                      })
                }
              })
            );
          }
        })
      )
    )
  ),

  // ---- illumination: texture shading
  ...[1, 16].flatMap(baseSigma =>
    [false, true].flatMap(downsampleLevels =>
      TEXTURE_GRIDS.map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTextureShading',
          name: `base sigma ${baseSigma}, downsample ${downsampleLevels}, ${label}`,
          build: context => {
            const {graph, view, texture} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUTextureShading({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', 12),
                levelCount: GPU_TEXTURE_SHADING_MAXIMUM_LEVEL_COUNT,
                baseSigma,
                hasNodata: true,
                downsampleLevels,
                output: {
                  textureShade: view('float32', count),
                  outputValidity: view('uint32', count),
                  ...(grid === LARGE
                    ? {}
                    : {
                        textureShadeTexture: graph.createTextureView(
                          texture('r32float', grid.width, grid.height),
                          {mipLevelCount: 1}
                        )
                      })
                }
              })
            );
          }
        })
      )
    )
  ),

  // ---- topographic position: ruggedness
  ...(['riley', 'wilson'] as const).flatMap(terrainRuggednessAlgorithm =>
    (['nodata', 'extrapolate'] as const).flatMap(edgeMode =>
      GRIDS.map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTerrainRuggedness',
          name: `${terrainRuggednessAlgorithm}, ${edgeMode}, ${label}`,
          build: context => {
            const {graph, view} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUTerrainRuggedness({
                grid,
                elevation: elevationBand(context, grid),
                terrainRuggednessAlgorithm,
                edgeMode,
                output: {
                  topographicPositionIndex: view('float32', count),
                  terrainRuggednessIndex: view('float32', count),
                  roughness: view('float32', count),
                  outputValidity: view('uint32', count)
                }
              })
            );
          }
        })
      )
    )
  ),

  // ---- topographic position: vector ruggedness
  ...(['clamp', 'nodata'] as const).flatMap(borderMode =>
    CELL_SIZE_MODES.flatMap(cellSizeMode =>
      GRIDS.map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTerrainVectorRuggedness',
          name: `${borderMode}, ${cellSizeMode}, ${label}`,
          build: context => {
            const {graph, view} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUTerrainVectorRuggedness({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', 8),
                maximumRadius: grid === LARGE ? 1000 : 10,
                cellSizeMode,
                rowDirection: cellSizeMode === 'uniform' ? 'south' : 'north',
                borderMode,
                output: {
                  vectorRuggedness: view('float32', count),
                  outputValidity: view('uint32', count)
                }
              })
            );
          }
        })
      )
    )
  ),

  // ---- topographic position: topographic position index
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'every output, 3 scales',
    build: context => {
      const {graph, view} = context;
      const count = cellCount(SMALL);
      graph.add(
        new GPUTerrainTopographicPosition({
          grid: SMALL,
          elevation: elevationBand(context, SMALL),
          scales: [{radius: 2}, {radius: 5, innerRadius: 2}, {radius: 9, innerRadius: 1}],
          quantum: 0.25,
          output: {
            topographicPositionIndex: view('float32', count * 3),
            deviationFromMean: view('float32', count * 3),
            maximumDeviation: view('float32', count),
            maximumDeviationRadius: view('uint32', count),
            outputValidity: view('uint32', count)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'maximum 64 scales on the largest grid with per-scale outputs',
    build: context => {
      const {graph, view} = context;
      const scaleCount = GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES;
      const count = cellCount(HORIZON_LIMIT);
      graph.add(
        new GPUTerrainTopographicPosition({
          grid: HORIZON_LIMIT,
          elevation: elevationBand(context, HORIZON_LIMIT),
          scales: Array.from({length: scaleCount}, (_, index) => ({
            radius: 8 * (index + 1),
            innerRadius: index % 3
          })),
          output: {
            topographicPositionIndex: view('float32', count * scaleCount),
            deviationFromMean: view('float32', count * scaleCount),
            maximumDeviation: view('float32', count),
            maximumDeviationRadius: view('uint32', count),
            outputValidity: view('uint32', count)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'grid at the maximum cell count with 2 scales',
    build: context => {
      const {graph, view} = context;
      // The exact integer summed-area table keeps three words (12 bytes) per pixel.
      const grid = SUMMED_AREA_LIMIT;
      const count = cellCount(grid);
      graph.add(
        new GPUTerrainTopographicPosition({
          grid,
          elevation: elevationBand(context, grid),
          scales: [{radius: 50}, {radius: 500, innerRadius: 100}],
          output: {
            topographicPositionIndex: view('float32', count * 2),
            deviationFromMean: view('float32', count * 2),
            maximumDeviation: view('float32', count),
            maximumDeviationRadius: view('uint32', count),
            outputValidity: view('uint32', count)
          }
        })
      );
    }
  },

  // ---- topographic position: Weiss landforms
  ...(['global', 'local'] as const).flatMap(standardization =>
    CELL_SIZE_MODES.flatMap(cellSizeMode =>
      (
        [
          ['every option', SMALL],
          ['grid at the maximum cell count', SUMMED_AREA_LIMIT]
        ] as [string, Grid][]
      ).map(
        ([label, grid]): KernelAuditCase => ({
          exportName: 'GPUTerrainWeissLandforms',
          name: `${standardization}, ${cellSizeMode}, ${label}`,
          build: context => {
            const {graph, view} = context;
            const count = cellCount(grid);
            graph.add(
              new GPUTerrainWeissLandforms({
                grid,
                elevation: elevationBand(context, grid),
                parameters: view('float32', 8),
                smallScale: {radius: grid === SMALL ? 3 : 100, innerRadius: 1},
                largeScale: {radius: grid === SMALL ? 9 : 1000, innerRadius: 2},
                standardization,
                quantum: 0.5,
                cellSizeMode,
                output: {
                  landforms: view('uint32', count),
                  standardizedPosition: view('float32', 2 * count),
                  outputValidity: view('uint32', count)
                }
              })
            );
          }
        })
      )
    )
  ),

  // ---- one step above the documented maxima (refused, see rejections-terrain.ts)
  {
    exportName: 'GPUTerrainSpikeRepair',
    name: 'grid above the maximum cell count',
    build: context => {
      const {graph, view} = context;
      const grid = makeGPURasterMetadata({
        width: getGPUTerrainSpikeRepairMaximumCellCount(context.device) + 1,
        height: 1
      });
      const count = cellCount(grid);
      graph.add(
        new GPUTerrainSpikeRepair({
          grid,
          elevation: elevationBand(context, grid),
          output: {
            values: view('float32', count),
            outputValidity: view('uint32', count)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainPeakSnap',
    name: 'candidates above the maximum',
    build: context => {
      const {graph, view} = context;
      const candidates = getGPUTerrainPeakSnapMaximumCandidateCount(context.device) + 1;
      graph.add(
        new GPUTerrainPeakSnap({
          grid: SMALL,
          elevation: elevationBand(context, SMALL),
          parameters: view('float32', 8),
          candidates: view('float32x2', candidates),
          output: {
            positions: view('float32x2', candidates),
            heights: view('float32', candidates),
            snapStatus: view('uint32', candidates)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'grid above the maximum cell count',
    build: context => {
      const {graph, view} = context;
      const grid = makeGPURasterMetadata({
        width: getGPUTerrainSummedAreaMaximumCellCount(context.device) + 1,
        height: 1
      });
      graph.add(
        new GPUTerrainTopographicPosition({
          grid,
          elevation: elevationBand(context, grid),
          scales: [{radius: 2}],
          output: {outputValidity: view('uint32', cellCount(grid))}
        })
      );
    }
  },
  {
    exportName: 'GPUTerrainTopographicPosition',
    name: 'grid x scales above the maximum plane count',
    build: context => {
      const {graph, view} = context;
      const scaleCount = GPU_TERRAIN_TOPOGRAPHIC_POSITION_MAXIMUM_SCALES;
      // One more row than the 512 x 1024 limit grid of the 64-scale case above.
      const grid = makeGPURasterMetadata({width: 512, height: 1025});
      graph.add(
        new GPUTerrainTopographicPosition({
          grid,
          elevation: elevationBand(context, grid),
          scales: Array.from({length: scaleCount}, (_, index) => ({radius: index + 1})),
          output: {topographicPositionIndex: view('float32', cellCount(grid) * scaleCount)}
        })
      );
    }
  },
  ...(['global', 'local'] as const).map(
    (standardization): KernelAuditCase => ({
      exportName: 'GPUTerrainWeissLandforms',
      name: `${standardization}, grid above the maximum cell count`,
      build: context => {
        const {graph, view} = context;
        const grid = makeGPURasterMetadata({
          width: getGPUTerrainSummedAreaMaximumCellCount(context.device) + 1,
          height: 1
        });
        graph.add(
          new GPUTerrainWeissLandforms({
            grid,
            elevation: elevationBand(context, grid),
            parameters: view('float32', 8),
            standardization,
            output: {landforms: view('uint32', cellCount(grid))}
          })
        );
      }
    })
  )
];
