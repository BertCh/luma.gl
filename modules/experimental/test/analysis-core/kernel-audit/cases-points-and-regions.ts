// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS,
  GPUArealInterpolation,
  GPUCatchmentAccessibility,
  GPUDotDensity,
  GPUGeographicDistribution,
  GPUHuffTradeAreas,
  GPUKernelDensity,
  GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT,
  GPUPointDensity,
  GPUFocalStatistics,
  GPUInverseDistanceWeighting,
  GPUHierarchicalDensityClustering,
  GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT,
  GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_SAMPLES,
  GPUKMeans,
  GPUKriging,
  GPU_FOCAL_STATISTICS_PARAMETER_LENGTH,
  GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH,
  GPU_KRIGING_PARAMETER_LENGTH,
  GPUAZPRegions,
  GPUMaxPRegions,
  GPURegionPartitionEvaluation,
  GPUSegregation,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree,
  GPUWardRegions,
  GPU_AZP_MAXIMUM_COLUMNS,
  GPU_AZP_MAXIMUM_ITERATIONS,
  getGPUAZPMaximumTableWordCount,
  GPU_AZP_REGIONS_PARAMETER_LENGTH,
  GPU_AZP_STATUS,
  GPU_MAX_P_MAXIMUM_COLUMNS,
  GPU_MAX_P_MAXIMUM_ROUNDS,
  getGPUMaxPMaximumSlotCount,
  GPU_MAX_P_MAXIMUM_START_COUNT,
  GPU_MAX_P_REGIONS_PARAMETER_LENGTH,
  GPU_MAX_P_STATUS,
  GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS,
  GPU_REGION_PARTITION_EVALUATION_LAYOUT,
  GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS,
  GPU_SKATER_MAXIMUM_COLUMNS,
  getGPUSkaterMaximumStatisticCount,
  getGPURegionalizationMaximumSlotCount,
  getGPURegionPartitionEvaluationMaximumRowCount,
  GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT,
  GPU_SKATER_REGIONS_PARAMETER_LENGTH,
  GPU_WARD_MAXIMUM_COLUMNS,
  GPU_WARD_REGIONS_PARAMETER_LENGTH,
  GPU_WARD_STATUS,
  GPUSpatialClustering,
  GPU_KMEANS_MAXIMUM_CLUSTERS,
  GPU_KMEANS_MAXIMUM_ITERATIONS,
  GPU_KMEANS_MAXIMUM_RESTARTS,
  GPU_KMEANS_PARAMETER_LENGTH,
  GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH,
  GPUSimilarLocations,
  GPU_SEGREGATION_MAXIMUM_GROUPS,
  getGPUSegregationLayout,
  getGPUSimilarLocationsParameterLength,
  GPU_KERNEL_DENSITY_PARAMETER_LENGTH,
  GPULocalOutlierFactor,
  GPULocationAllocation,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH,
  getGPULocationAllocationMaximumFacilityCount,
  getGPULocationAllocationMaximumDemandCount,
  getGPULocationAllocationMaximumSparseEntryCount,
  GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS,
  GPU_LOCATION_ALLOCATION_STATUS,
  GPURoseStatistic,
  GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH,
  GPUPycnophylactic,
  GPURandomPointsInPolygon,
  getGPUPolygonTriangulationMaximumNodeCount,
  GPURandomPointsOnLine,
  getGPUDotDensityMaximumCapacity,
  GPU_DOT_DENSITY_PARAMETER_LENGTH,
  GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH,
  GPU_ROSE_STATISTIC_SUMMARY_STRIDE
} from '../../../src/gpu-spatial-analysis';
import type {
  GPUKernelDensityBandwidthRule,
  GPUKernelDensityKernel
} from '../../../src/gpu-spatial-analysis';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const ROWS = 300;

/** Longitude-latitude raster context with its extent. */
const GEOGRAPHIC = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

/** Polygon geometry fields: `featureCount` features of one polygon with one ring of `ringSize` vertices. */
function polygons(context: KernelAuditContext, featureCount: number, ringSize = 8) {
  return {
    positions: context.view('float32x2', featureCount * ringSize, 'polygon-positions'),
    featureOffsets: context.view('uint32', featureCount + 1, 'feature-offsets'),
    polygonOffsets: context.view('uint32', featureCount + 1, 'polygon-offsets'),
    ringOffsets: context.view('uint32', featureCount + 1, 'ring-offsets')
  };
}

/** Compact output of `capacity` rows. */
function compact(context: KernelAuditContext, capacity: number) {
  return {
    ids: context.view('uint32', capacity, 'compact-ids'),
    status: {
      count: context.view('uint32', 1, 'compact-count'),
      overflow: context.view('uint32', 1, 'compact-overflow'),
      requiredCount: context.view('uint32', 1, 'compact-required')
    }
  };
}

/** Kernel-audit cases for points-and-regions. */
export const POINTS_AND_REGIONS_CASES: KernelAuditCase[] = [
  // Areal interpolation
  ...(['extensive', 'intensive'] as const).flatMap(mode =>
    (['zone', 'overlap'] as const).map(
      (denominator): KernelAuditCase => ({
        exportName: 'GPUArealInterpolation',
        name: `planar every option ${mode} ${denominator}`,
        build: context => {
          const {graph, view} = context;
          const sources = 12;
          const targets = 20;
          const capacity = ROWS;
          graph.add(
            new GPUArealInterpolation({
              grid: makeGPURasterMetadata({width: ROWS, height: 1}),
              sourceZones: view('uint32', ROWS),
              targetZones: view('uint32', ROWS),
              sourceZoneCount: sources,
              targetZoneCount: targets,
              cellWeights: view('float32', ROWS),
              denominator,
              sourceAnchorCells: view('uint32', sources),
              mode,
              categories: {sourceCategories: view('uint32', sources), categoryCount: 5},
              output: {
                weights: {
                  offsets: view('uint32', targets + 1),
                  neighbors: view('uint32', capacity),
                  weights: view('float32', capacity),
                  distances: view('float32', capacity)
                },
                alternateWeights: view('float32', capacity),
                areas: view('float32', capacity),
                sourceAreas: view('float32', sources),
                sourceTransferredFractions: view('float32', sources),
                categoryShares: view('float32', targets * 5),
                status: {
                  overflow: view('uint32', 1),
                  requiredPairCount: view('uint32', 1),
                  invalidCount: view('uint32', 1)
                }
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUArealInterpolation',
    name: 'geographic raster both denominators',
    build: context => {
      const {graph, view} = context;
      const width = 20;
      const cells = width * 15;
      graph.add(
        new GPUArealInterpolation({
          grid: makeGPURasterMetadata({width, height: 15}),
          sourceZones: view('uint32', cells),
          targetZones: view('uint32', cells),
          sourceZoneCount: 10,
          targetZoneCount: 12,
          cellWeights: view('float32', cells),
          denominator: {extensive: 'overlap', intensive: 'zone'},
          spatialContext: GEOGRAPHIC,
          parameters: view('float32', 4),
          output: {
            weights: {
              offsets: view('uint32', 13),
              neighbors: view('uint32', cells),
              weights: view('float32', cells)
            },
            status: {overflow: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUArealInterpolation',
    name: 'unweighted fast path',
    build: ({graph, view}) =>
      graph.add(
        new GPUArealInterpolation({
          grid: makeGPURasterMetadata({width: ROWS, height: 1}),
          sourceZones: view('uint32', ROWS),
          targetZones: view('uint32', ROWS),
          sourceZoneCount: 12,
          targetZoneCount: 20,
          denominator: 'overlap',
          unweightedFastPath: true,
          output: {
            weights: {
              offsets: view('uint32', 21),
              neighbors: view('uint32', ROWS),
              weights: view('float32', ROWS)
            },
            status: {overflow: view('uint32', 1)}
          }
        })
      )
  },
  ...[
    ['cells above the maximum, pairs at the maximum', 255, 0],
    ['cells at the maximum, pairs above the maximum', 0, 1],
    ['cells and pairs at the maximum', 255, 1]
  ].map(
    ([name, trim, below]): KernelAuditCase => ({
      exportName: 'GPUArealInterpolation',
      name: String(name),
      build: ({graph, view, maximumRows}) => {
        const cells = maximumRows(4) - Number(below);
        const pairs = trim === 0 ? maximumRows(4) - 1 : (1 << 24) - 1;
        const targets = 1 << 20;
        graph.add(
          new GPUArealInterpolation({
            grid: makeGPURasterMetadata({width: cells, height: 1}),
            sourceZones: view('uint32', cells),
            targetZones: view('uint32', cells),
            sourceZoneCount: 1 << 20,
            targetZoneCount: targets,
            cellWeights: view('float32', cells),
            denominator: 'overlap',
            output: {
              weights: {
                offsets: view('uint32', targets + 1),
                neighbors: view('uint32', pairs),
                weights: view('float32', pairs),
                distances: view('float32', pairs)
              },
              alternateWeights: view('float32', pairs),
              areas: view('float32', pairs),
              sourceTransferredFractions: view('float32', 1 << 20),
              status: {overflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  // Pycnophylactic interpolation
  ...(['rook', 'box'] as const).flatMap(kernel =>
    [false, true].map(
      (geographic): KernelAuditCase => ({
        exportName: 'GPUPycnophylactic',
        name: `every option ${kernel} ${geographic ? 'geographic' : 'planar'}`,
        build: ({graph, view}) => {
          const width = 20;
          const height = 15;
          const cells = width * height;
          const zoneCount = 9;
          graph.add(
            new GPUPycnophylactic({
              grid: makeGPURasterMetadata({width, height}),
              zones: view('uint32', cells),
              zoneCount,
              totals: view('float32', zoneCount),
              maximumIterations: GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS,
              tolerance: view('float32', 1),
              zoneAnchorCells: view('uint32', zoneCount),
              kernel,
              ...(geographic ? {spatialContext: GEOGRAPHIC, parameters: view('float32', 4)} : {}),
              output: {
                mass: view('float32', cells),
                density: view('float32', cells),
                status: {
                  converged: view('uint32', 1),
                  iterationCount: view('uint32', 1),
                  invalidCount: view('uint32', 1)
                }
              }
            })
          );
        }
      })
    )
  ),
  ...(['rook', 'box'] as const).map(
    (kernel): KernelAuditCase => ({
      exportName: 'GPUPycnophylactic',
      name: `raster at the binding limit ${kernel}`,
      build: ({graph, view, maximumRows}) => {
        const width = 4096;
        const height = Math.floor(maximumRows(4) / width);
        const cells = width * height;
        const zoneCount = 1 << 20;
        graph.add(
          new GPUPycnophylactic({
            grid: makeGPURasterMetadata({width, height}),
            zones: view('uint32', cells),
            zoneCount,
            totals: view('float32', zoneCount),
            maximumIterations: GPU_PYCNOPHYLACTIC_MAXIMUM_ITERATIONS,
            tolerance: view('float32', 1),
            zoneAnchorCells: view('uint32', zoneCount),
            kernel,
            output: {
              mass: view('float32', cells),
              density: view('float32', cells),
              status: {
                converged: view('uint32', 1),
                iterationCount: view('uint32', 1),
                invalidCount: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),
  // Catchment accessibility
  ...(['2sfca', '3sfca'] as const).map(
    (method): KernelAuditCase => ({
      exportName: 'GPUCatchmentAccessibility',
      name: `every output ${method}`,
      build: ({graph, view, weights}) => {
        const facilities = ROWS;
        const demand = 500;
        graph.add(
          new GPUCatchmentAccessibility({
            method,
            supply: view('float32', facilities),
            demand: view('float32', demand),
            facilityWeights: {
              ...weights(facilities, 8),
              distances: view('float32', facilities * 8)
            },
            demandWeights: {...weights(demand, 8), distances: view('float32', demand * 8)},
            output: {
              accessibility: view('float32', demand),
              ratios: view('float32', facilities),
              reachableFacilities: view('uint32', demand)
            }
          })
        );
      }
    })
  ),
  ...(['2sfca', '3sfca'] as const).map(
    (method): KernelAuditCase => ({
      exportName: 'GPUCatchmentAccessibility',
      name: `rows and slots at the binding limit ${method}`,
      build: ({graph, view, maximumRows}) => {
        const rows = maximumRows(4) - 1;
        const slots = maximumRows(4);
        const weightsOf = () => ({
          offsets: view('uint32', rows + 1),
          neighbors: view('uint32', slots),
          weights: view('float32', slots),
          distances: view('float32', slots)
        });
        graph.add(
          new GPUCatchmentAccessibility({
            method,
            supply: view('float32', rows),
            demand: view('float32', rows),
            facilityWeights: weightsOf(),
            demandWeights: weightsOf(),
            output: {
              accessibility: view('float32', rows),
              ratios: view('float32', rows),
              reachableFacilities: view('uint32', rows)
            }
          })
        );
      }
    })
  ),
  // Huff trade areas
  {
    exportName: 'GPUHuffTradeAreas',
    name: 'every output',
    build: ({graph, view, weights}) => {
      const facilities = ROWS;
      const demand = 500;
      graph.add(
        new GPUHuffTradeAreas({
          attractiveness: view('float32', facilities),
          demandWeights: {...weights(demand, 8), distances: view('float32', demand * 8)},
          facilityWeights: weights(facilities, 8),
          demand: view('float32', demand),
          parameters: view('float32', GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH),
          output: {
            tradeArea: view('uint32', demand),
            probabilities: view('float32', demand * 8),
            tradeAreaProbability: view('float32', demand),
            expectedDemand: view('float32', facilities)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUHuffTradeAreas',
    name: 'rows and slots at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4) - 1;
      const slots = maximumRows(4);
      graph.add(
        new GPUHuffTradeAreas({
          attractiveness: view('float32', rows),
          demandWeights: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', slots),
            weights: view('float32', slots),
            distances: view('float32', slots)
          },
          facilityWeights: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', slots),
            weights: view('float32', slots)
          },
          demand: view('float32', rows),
          parameters: view('float32', GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH),
          output: {
            tradeArea: view('uint32', rows),
            probabilities: view('float32', slots),
            tradeAreaProbability: view('float32', rows),
            expectedDemand: view('float32', rows)
          }
        })
      );
    }
  },
  // Dot density
  {
    exportName: 'GPUDotDensity',
    name: 'every option',
    build: context => {
      const {graph, view} = context;
      const features = 40;
      const categories = 6;
      const capacity = ROWS;
      graph.add(
        new GPUDotDensity({
          ...polygons(context, features),
          values: view('float32', features * categories),
          categoryCount: categories,
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          maximumAttempts: 1024,
          mask: {weights: view('float32', 64 * 64), width: 64, height: 64},
          spatialContext: {coordinateSpace: 'planar', metric: 'native'},
          output: {
            positions: view('float32x2', capacity),
            dots: compact(context, capacity),
            categories: view('uint32', capacity),
            slotCounts: view('uint32', features * categories),
            slotOffsets: view('uint32', features * categories),
            status: {failedCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUDotDensity',
    name: 'capacity above the maximum',
    build: context => {
      const {graph, view} = context;
      const features = 1 << 16;
      const categories = 8;
      const capacity = getGPUDotDensityMaximumCapacity(graph.device) + 1;
      graph.add(
        new GPUDotDensity({
          ...polygons(context, features),
          values: view('float32', features * categories),
          categoryCount: categories,
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          maximumAttempts: 1 << 20,
          mask: {weights: view('float32', 4096 * 4096), width: 4096, height: 4096},
          output: {
            positions: view('float32x2', capacity),
            dots: compact(context, capacity),
            categories: view('uint32', capacity),
            slotCounts: view('uint32', features * categories),
            slotOffsets: view('uint32', features * categories),
            status: {failedCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUDotDensity',
    name: 'capacity at the maximum',
    build: context => {
      const {graph, view} = context;
      const features = 1 << 16;
      const categories = 8;
      const capacity = getGPUDotDensityMaximumCapacity(graph.device);
      graph.add(
        new GPUDotDensity({
          ...polygons(context, features),
          values: view('float32', features * categories),
          categoryCount: categories,
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', capacity),
            dots: compact(context, capacity),
            categories: view('uint32', capacity)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURandomPointsInPolygon',
    name: 'every option',
    build: context => {
      const {graph, view} = context;
      const features = 40;
      const capacity = ROWS;
      graph.add(
        new GPURandomPointsInPolygon({
          ...polygons(context, features),
          counts: view('uint32', features),
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          maximumAttempts: 1024,
          mask: {weights: view('float32', 64 * 64), width: 64, height: 64},
          spatialContext: {coordinateSpace: 'planar', metric: 'native'},
          output: {
            positions: view('float32x2', capacity),
            dots: compact(context, capacity),
            slotCounts: view('uint32', features),
            slotOffsets: view('uint32', features),
            status: {failedCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPURandomPointsInPolygon',
    name: 'triangulation scratch at the maximum',
    build: context => {
      const {graph, view} = context;
      // Four vertices per feature: the triangulation scratch takes 16 nodes per feature.
      const features = Math.floor(getGPUPolygonTriangulationMaximumNodeCount(context.device) / 16);
      const capacity = 4096;
      graph.add(
        new GPURandomPointsInPolygon({
          ...polygons(context, features, 4),
          counts: view('uint32', features),
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          maximumAttempts: 1 << 20,
          output: {
            positions: view('float32x2', capacity),
            dots: compact(context, capacity),
            slotCounts: view('uint32', features),
            slotOffsets: view('uint32', features),
            status: {failedCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPURandomPointsInPolygon',
    name: 'triangulation scratch above the maximum',
    build: context => {
      const {graph, view} = context;
      // Four vertices per feature: the triangulation scratch takes 16 nodes per feature.
      const features =
        Math.floor(getGPUPolygonTriangulationMaximumNodeCount(context.device) / 16) + 1;
      const capacity = 4096;
      graph.add(
        new GPURandomPointsInPolygon({
          ...polygons(context, features, 4),
          counts: view('uint32', features),
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          maximumAttempts: 1 << 20,
          output: {
            positions: view('float32x2', capacity),
            dots: compact(context, capacity),
            slotCounts: view('uint32', features),
            slotOffsets: view('uint32', features),
            status: {failedCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPURandomPointsOnLine',
    name: 'every option',
    build: context => {
      const {graph, view} = context;
      const lines = 40;
      const capacity = ROWS;
      graph.add(
        new GPURandomPointsOnLine({
          positions: view('float32x2', lines * 8),
          lineOffsets: view('uint32', lines + 1),
          counts: view('uint32', lines),
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          spatialContext: {coordinateSpace: 'planar', metric: 'native'},
          output: {
            positions: view('float32x2', capacity),
            points: compact(context, capacity),
            fractions: view('float32', capacity)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURandomPointsOnLine',
    name: 'capacity at the binding limit',
    build: context => {
      const {graph, view, maximumRows} = context;
      const lines = maximumRows(4) - 1;
      const capacity = maximumRows(8);
      graph.add(
        new GPURandomPointsOnLine({
          positions: view('float32x2', maximumRows(8)),
          lineOffsets: view('uint32', lines + 1),
          counts: view('uint32', lines),
          parameters: view('uint32', GPU_DOT_DENSITY_PARAMETER_LENGTH),
          output: {
            positions: view('float32x2', capacity),
            points: compact(context, capacity),
            fractions: view('float32', capacity)
          }
        })
      );
    }
  },
  // Geographic distribution
  ...(['auto', 'chunked', 'sorted'] as const).flatMap(reduction =>
    [1, 200, 256].map(
      (groups): KernelAuditCase => ({
        exportName: 'GPUGeographicDistribution',
        name: `every output ${reduction} ${groups} groups`,
        build: ({graph, view}) => {
          const rows = 2000;
          const vertices = 64;
          graph.add(
            new GPUGeographicDistribution({
              positions: view('float32x2', rows),
              weights: view('float32', rows),
              ...(groups > 1 ? {groupIds: view('uint32', rows), groupCount: groups} : {}),
              mask: view('uint32', rows),
              lineEnds: view('float32x2', rows),
              parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
              maximumMedianIterations: 256,
              polygonVertexCount: vertices,
              reduction,
              output: {
                counts: view('uint32', groups),
                weightSums: view('float32', groups),
                meanCenters: view('float32x2', groups),
                medianCenters: view('float32x2', groups),
                medianConverged: view('uint32', groups),
                standardDistances: view('float32', groups),
                ellipses: view('float32', groups * 3),
                directionalMeans: view('float32', groups * 3),
                ellipseVertices: view('float32x2', groups * vertices),
                circleVertices: view('float32x2', groups * vertices)
              }
            })
          );
        }
      })
    )
  ),
  ...(['chunked', 'sorted'] as const).map(
    (reduction): KernelAuditCase => ({
      exportName: 'GPUGeographicDistribution',
      name: `rows at the documented maximum ${reduction}`,
      build: ({graph, view}) => {
        const rows = 2 ** 24 - 1;
        const groups = reduction === 'chunked' ? 256 : 4096;
        graph.add(
          new GPUGeographicDistribution({
            positions: view('float32x2', rows),
            weights: view('float32', rows),
            groupIds: view('uint32', rows),
            groupCount: groups,
            mask: view('uint32', rows),
            lineEnds: view('float32x2', rows),
            parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
            maximumMedianIterations: 256,
            reduction,
            output: {
              counts: view('uint32', groups),
              weightSums: view('float32', groups),
              meanCenters: view('float32x2', groups),
              medianCenters: view('float32x2', groups),
              medianConverged: view('uint32', groups),
              standardDistances: view('float32', groups),
              ellipses: view('float32', groups * 3),
              directionalMeans: view('float32', groups * 3)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUGeographicDistribution',
    name: 'groups at the binding limit sorted',
    build: ({graph, view, maximumRows}) => {
      const rows = 2 ** 24 - 1;
      const groups = maximumRows(24);
      graph.add(
        new GPUGeographicDistribution({
          positions: view('float32x2', rows),
          weights: view('float32', rows),
          groupIds: view('uint32', rows),
          groupCount: groups,
          mask: view('uint32', rows),
          lineEnds: view('float32x2', rows),
          parameters: view('float32', GPU_GEOGRAPHIC_DISTRIBUTION_PARAMETER_LENGTH),
          maximumMedianIterations: 256,
          polygonVertexCount: 3,
          reduction: 'sorted',
          output: {
            counts: view('uint32', groups),
            weightSums: view('float32', groups),
            meanCenters: view('float32x2', groups),
            medianCenters: view('float32x2', groups),
            medianConverged: view('uint32', groups),
            standardDistances: view('float32', groups),
            ellipses: view('float32', groups * 3),
            directionalMeans: view('float32', groups * 3),
            ellipseVertices: view('float32x2', groups * 3),
            circleVertices: view('float32x2', groups * 3)
          }
        })
      );
    }
  },
  // Rose statistic
  ...(['directional', 'axial'] as const).flatMap(mode =>
    [1, 50].map(
      (groups): KernelAuditCase => ({
        exportName: 'GPURoseStatistic',
        name: `every option ${mode} ${groups} groups at 360 bins`,
        build: ({graph, view}) => {
          const rows = ROWS;
          graph.add(
            new GPURoseStatistic({
              starts: view('float32x2', rows),
              ends: view('float32x2', rows),
              weights: view('float32', rows),
              ...(groups > 1 ? {groupIds: view('uint32', rows), groupCount: groups} : {}),
              mask: view('uint32', rows),
              circular: {
                mode,
                angleUnit: 'radians',
                origin: 'positive-x-counter-clockwise',
                binCount: 360
              },
              output: {
                summary: view('float32', groups * GPU_ROSE_STATISTIC_SUMMARY_STRIDE),
                bins: view('float32', groups * 360)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPURoseStatistic',
    name: 'rows and groups at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(8);
      const groups = maximumRows(360 * 4);
      graph.add(
        new GPURoseStatistic({
          starts: view('float32x2', rows),
          ends: view('float32x2', rows),
          weights: view('float32', rows),
          groupIds: view('uint32', rows),
          groupCount: groups,
          mask: view('uint32', rows),
          circular: {
            mode: 'axial',
            angleUnit: 'radians',
            origin: 'positive-x-counter-clockwise',
            binCount: 360
          },
          output: {
            summary: view('float32', groups * GPU_ROSE_STATISTIC_SUMMARY_STRIDE),
            bins: view('float32', groups * 360)
          }
        })
      );
    }
  },
  // Location allocation
  ...(['p-median', 'maximum-coverage', 'set-covering'] as const).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPULocationAllocation',
      name: `every option ${operation}`,
      build: ({graph, view}) => {
        const facilities = 40;
        const demand = 120;
        graph.add(
          new GPULocationAllocation({
            costs: view('float32', facilities * demand),
            facilityCount: facilities,
            demandWeights: view('float32', demand),
            operation,
            maximumFacilityCount: 10,
            ...(operation === 'set-covering'
              ? {}
              : {maximumSwapIterations: GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS}),
            coverageDistance: view('float32', 1),
            output: {
              selectedFacilities: view('uint32', facilities),
              assignments: view('uint32', demand),
              assignedCosts: view('float32', demand),
              objective: view('float32', 1),
              status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
            }
          })
        );
      }
    })
  ),
  ...(['p-median', 'maximum-coverage', 'set-covering'] as const).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPULocationAllocation',
      name: `facilities and demand at the documented maximum ${operation}`,
      build: ({graph, view, maximumRows}) => {
        const swapping = operation !== 'set-covering';
        // The swap phase allows maximumFacilityCount * facilityCount <= 2^24.
        const facilities = swapping
          ? 4096
          : getGPULocationAllocationMaximumFacilityCount(graph.device);
        // Every greedy or swap step is an unrolled set of command nodes, so
        // `maximumFacilityCount` = `facilityCount` (33.5M) would declare tens of millions of
        // nodes and exhaust memory; 4096 is the largest step count the swap phase can reach.
        const maximumFacilities = 4096;
        const demand = Math.floor(maximumRows(4) / facilities);
        graph.add(
          new GPULocationAllocation({
            costs: view('float32', facilities * demand),
            facilityCount: facilities,
            demandWeights: view('float32', demand),
            operation,
            maximumFacilityCount: maximumFacilities,
            ...(swapping
              ? {maximumSwapIterations: GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS}
              : {}),
            coverageDistance: view('float32', 1),
            output: {
              selectedFacilities: view('uint32', facilities),
              assignments: view('uint32', demand),
              assignedCosts: view('float32', demand),
              objective: view('float32', 1),
              status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULocationAllocation',
    name: 'facilities above the maximum set-covering',
    build: ({graph, view}) => {
      const facilities = getGPULocationAllocationMaximumFacilityCount(graph.device) + 1;
      graph.add(
        new GPULocationAllocation({
          costs: view('float32', facilities),
          facilityCount: facilities,
          demandWeights: view('float32', 1),
          operation: 'set-covering',
          maximumFacilityCount: 4096,
          coverageDistance: view('float32', 1),
          output: {
            selectedFacilities: view('uint32', facilities),
            assignments: view('uint32', 1),
            assignedCosts: view('float32', 1),
            objective: view('float32', 1),
            status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
          }
        })
      );
    }
  },
  // Location allocation, sparse (CSR) costs
  ...(['p-median', 'maximum-coverage', 'set-covering'] as const).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPULocationAllocation',
      name: `every option sparse ${operation}`,
      build: ({graph, view}) => {
        const facilities = 40;
        const demand = 120;
        graph.add(
          new GPULocationAllocation({
            sparseCosts: {
              offsets: view('uint32', demand + 1),
              facilities: view('uint32', 600),
              costs: view('float32', 600)
            },
            facilityCount: facilities,
            demandWeights: view('float32', demand),
            operation,
            maximumFacilityCount: 10,
            ...(operation === 'set-covering'
              ? {}
              : {maximumSwapIterations: GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS}),
            coverageDistance: view('float32', 1),
            output: {
              selectedFacilities: view('uint32', facilities),
              assignments: view('uint32', demand),
              assignedCosts: view('float32', demand),
              objective: view('float32', 1),
              status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
            }
          })
        );
      }
    })
  ),
  ...(['p-median', 'maximum-coverage', 'set-covering'] as const).map(
    (operation): KernelAuditCase => ({
      exportName: 'GPULocationAllocation',
      name: `sparse entries and demand at the documented maximum ${operation}`,
      build: ({graph, view}) => {
        const swapping = operation !== 'set-covering';
        // The swap phase allows maximumFacilityCount * facilityCount <= 2^24; every step is an
        // unrolled set of command nodes, so the step count stays at 4096 (see the dense case).
        const facilities = swapping ? 4096 : 65536;
        const demand = getGPULocationAllocationMaximumDemandCount(graph.device);
        const entries = getGPULocationAllocationMaximumSparseEntryCount(graph.device);
        graph.add(
          new GPULocationAllocation({
            sparseCosts: {
              offsets: view('uint32', demand + 1),
              facilities: view('uint32', entries),
              costs: view('float32', entries)
            },
            facilityCount: facilities,
            demandWeights: view('float32', demand),
            operation,
            maximumFacilityCount: 4096,
            ...(swapping
              ? {maximumSwapIterations: GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS}
              : {}),
            coverageDistance: view('float32', 1),
            output: {
              selectedFacilities: view('uint32', facilities),
              assignments: view('uint32', demand),
              assignedCosts: view('float32', demand),
              objective: view('float32', 1),
              status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPULocationAllocation',
    name: 'sparse entries above the maximum',
    build: ({graph, view}) => {
      const facilities = 4096;
      const demand = 1024;
      const entries = getGPULocationAllocationMaximumSparseEntryCount(graph.device) + 1;
      graph.add(
        new GPULocationAllocation({
          sparseCosts: {
            offsets: view('uint32', demand + 1),
            facilities: view('uint32', entries),
            costs: view('float32', entries)
          },
          facilityCount: facilities,
          operation: 'p-median',
          maximumFacilityCount: 8,
          output: {
            selectedFacilities: view('uint32', facilities),
            assignments: view('uint32', demand),
            objective: view('float32', 1),
            status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
          }
        })
      );
    }
  },
  // Local outlier factor
  {
    exportName: 'GPULocalOutlierFactor',
    name: 'every output',
    build: ({graph, view, weights}) =>
      graph.add(
        new GPULocalOutlierFactor({
          neighbors: {...weights(ROWS, 32), distances: view('float32', ROWS * 32)},
          parameters: view('float32', GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH),
          output: {
            kDistance: view('float32', ROWS),
            localReachabilityDensity: view('float32', ROWS),
            lof: view('float32', ROWS),
            outlier: view('uint32', ROWS),
            outlierCount: view('uint32', 1)
          }
        })
      )
  },
  {
    exportName: 'GPULocalOutlierFactor',
    name: 'rows and slots at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4) - 1;
      const slots = maximumRows(4);
      graph.add(
        new GPULocalOutlierFactor({
          neighbors: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', slots),
            weights: view('float32', slots),
            distances: view('float32', slots)
          },
          parameters: view('float32', GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH),
          output: {
            kDistance: view('float32', rows),
            localReachabilityDensity: view('float32', rows),
            lof: view('float32', rows),
            outlier: view('uint32', rows),
            outlierCount: view('uint32', 1)
          }
        })
      );
    }
  },
  // Point density
  ...(
    [
      ['count', 'auto'],
      ['sum', 'workgroup'],
      ['sum', 'atomic'],
      ['mean', 'workgroup'],
      ['mean', 'atomic'],
      ['mean', 'auto']
    ] as const
  ).map(
    ([statistic, sumAccumulation]): KernelAuditCase => ({
      exportName: 'GPUPointDensity',
      name: `grid every output ${statistic} ${sumAccumulation}`,
      build: ({graph, view, texture}) => {
        const [columns, rows] = [64, 48];
        const cells = columns * rows;
        graph.add(
          new GPUPointDensity({
            positions: view('float32x2', 5000),
            weights: view('float32', 5000),
            mask: view('uint32', 5000),
            bounds: view('float32x4', 1),
            gridSize: [columns, rows],
            binning: 'grid',
            statistic,
            sumAccumulation,
            output: {
              values: view('float32', cells),
              counts: view('uint32', cells),
              sums: view('float32', cells),
              means: view('float32', cells),
              extent: view('float32', 2),
              histogram: view('uint32', 256),
              texture: graph.createTextureView(texture('r32float', columns, rows), {
                mipLevelCount: 1
              }),
              status: {overflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...(['count', 'sum', 'mean'] as const).flatMap((statistic, statisticIndex) =>
    [statisticIndex === 1].map(
      (radiusView): KernelAuditCase => ({
        exportName: 'GPUPointDensity',
        name: `hexagon every output ${statistic} ${radiusView ? 'radius view' : 'radius number'}`,
        build: ({graph, view}) => {
          const [columns, rows] = [64, 48];
          const cells = columns * rows;
          graph.add(
            new GPUPointDensity({
              positions: view('float32x2', 5000),
              weights: view('float32', 5000),
              mask: view('uint32', 5000),
              bounds: [0, 0, 100, 100],
              gridSize: [columns, rows],
              binning: 'hexagon',
              hexagonRadius: radiusView ? view('float32', 1) : 2.5,
              statistic,
              output: {
                values: view('float32', cells),
                counts: view('uint32', cells),
                sums: view('float32', cells),
                means: view('float32', cells),
                extent: view('float32', 2),
                histogram: view('uint32', 256),
                status: {overflow: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),
  ...(['auto', 'direct', 'fft'] as const).flatMap(strategy =>
    [false, true].map(
      (separable): KernelAuditCase => ({
        exportName: 'GPUPointDensity',
        name: `grid smoothing ${strategy} ${separable ? 'separable' : 'full'} kernel radius 32`,
        build: ({graph, view, texture}) => {
          const [columns, rows] = [256, 192];
          const cells = columns * rows;
          const size = 2 * 32 + 1;
          graph.add(
            new GPUPointDensity({
              positions: view('float32x2', 5000),
              weights: view('float32', 5000),
              bounds: view('float32', 4),
              gridSize: [columns, rows],
              statistic: 'mean',
              smoothing: {
                ...(separable
                  ? {
                      separableKernel: {
                        horizontal: view('float32', size),
                        vertical: view('float32', size)
                      }
                    }
                  : {kernel: view('float32', size * size)}),
                kernelWidth: size,
                kernelHeight: size,
                strategy
              },
              output: {
                values: view('float32', cells),
                counts: view('uint32', cells),
                sums: view('float32', cells),
                means: view('float32', cells),
                extent: view('float32', 2),
                histogram: view('uint32', 256),
                texture: graph.createTextureView(texture('r32float', columns, rows), {
                  mipLevelCount: 1
                }),
                status: {overflow: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),
  ...(['workgroup', 'atomic'] as const).map(
    (sumAccumulation): KernelAuditCase => ({
      exportName: 'GPUPointDensity',
      name: `points and grid at the binding limit ${sumAccumulation}`,
      build: ({graph, view, texture, maximumRows}) => {
        // Deterministic sums hold a 12-byte fixed-point table per cell, a lower documented maximum.
        const maximumCells =
          sumAccumulation === 'workgroup'
            ? GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT
            : maximumRows(4);
        const [columns, rows] = [8192, Math.floor(maximumCells / 8192)];
        const cells = columns * rows;
        const points = maximumRows(8);
        graph.add(
          new GPUPointDensity({
            positions: view('float32x2', points),
            weights: view('float32', points),
            mask: view('uint32', points),
            bounds: view('float32x4', 1),
            gridSize: [columns, rows],
            statistic: 'mean',
            sumAccumulation,
            output: {
              values: view('float32', cells),
              counts: view('uint32', cells),
              sums: view('float32', cells),
              means: view('float32', cells),
              extent: view('float32', 2),
              histogram: view('uint32', 256),
              texture: graph.createTextureView(texture('r32float', columns, rows), {
                mipLevelCount: 1
              }),
              status: {overflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  // Kernel density
  ...(['gaussian', 'quartic', 'epanechnikov'] as const).flatMap((kernel, kernelIndex) =>
    (['none', 'uniform', 'diggle'] as const).map((edgeCorrection, edgeIndex): KernelAuditCase => {
      const bandwidthRule: GPUKernelDensityBandwidthRule | undefined = [
        'scott',
        'silverman',
        undefined
      ][(kernelIndex + edgeIndex) % 3] as GPUKernelDensityBandwidthRule | undefined;
      const normalization = (kernelIndex + edgeIndex) % 2 === 0 ? 'intensity' : 'probability';
      return {
        exportName: 'GPUKernelDensity',
        name: `every output ${kernel} ${edgeCorrection} ${normalization} ${bandwidthRule ?? 'fixed'}`,
        build: ({graph, view}) =>
          graph.add(
            new GPUKernelDensity({
              positions: view('float32x2', 5000),
              weights: view('float32', 5000),
              parameters: view('float32', GPU_KERNEL_DENSITY_PARAMETER_LENGTH),
              grid: makeGPURasterMetadata({width: 128, height: 96}),
              indexGridSize: [64, 64],
              indexBounds: [0, 0, 100, 100],
              kernel: kernel as GPUKernelDensityKernel,
              edgeCorrection,
              normalization,
              ...(bandwidthRule ? {bandwidthRule} : {}),
              output: {
                values: view('float32', 128 * 96),
                weightSum: view('float32', 1),
                ...(bandwidthRule ? {bandwidth: view('float32', 1)} : {}),
                status: {invalidCount: view('uint32', 1)}
              }
            })
          )
      };
    })
  ),
  // 2^25 - 1 = 18631 x 1801 index cells is the documented maximum; one row more is refused.
  ...(['gaussian', 'epanechnikov'] as const).flatMap(kernel =>
    (
      [
        [`points, raster and index grid at the maximum ${kernel}`, [18631, 1801]],
        [`index grid above the maximum ${kernel}`, [8192, 4096]]
      ] as const
    ).map(
      ([name, indexGridSize]): KernelAuditCase => ({
        exportName: 'GPUKernelDensity',
        name,
        build: ({graph, view, maximumRows}) => {
          const width = 8192;
          const height = Math.floor(maximumRows(4) / width);
          graph.add(
            new GPUKernelDensity({
              positions: view('float32x2', maximumRows(8)),
              weights: view('float32', maximumRows(8)),
              parameters: view('float32', GPU_KERNEL_DENSITY_PARAMETER_LENGTH),
              grid: makeGPURasterMetadata({width, height}),
              indexGridSize,
              indexBounds: [0, 0, 100, 100],
              kernel,
              edgeCorrection: 'diggle',
              bandwidthRule: 'scott',
              output: {
                values: view('float32', width * height),
                weightSum: view('float32', 1),
                bandwidth: view('float32', 1),
                status: {invalidCount: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),
  // Segregation
  ...(['environment', 'smoothed-population'] as const).flatMap(spatialForm =>
    [2, 7, GPU_SEGREGATION_MAXIMUM_GROUPS].map(
      (groupCount): KernelAuditCase => ({
        exportName: 'GPUSegregation',
        name: `every output ${spatialForm} ${groupCount} groups`,
        build: ({graph, view, weights}) => {
          const units = 200;
          const scaleCount = 3;
          graph.add(
            new GPUSegregation({
              groupCount,
              groupCounts: view('float32', units * groupCount),
              scales: [null, weights(units, 8), weights(units, 24)],
              selfWeight: 0.5,
              spatialForm,
              atkinsonInequalityAversion: 0.9,
              output: {
                indices: view('float32', scaleCount * getGPUSegregationLayout(groupCount).stride),
                localEnvironment: view('float32', scaleCount * units * groupCount),
                localEntropy: view('float32', scaleCount * units),
                localDissimilarity: view('float32', scaleCount * units * groupCount),
                localTheil: view('float32', scaleCount * units)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUSegregation',
    name: 'units and slots at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const groupCount = GPU_SEGREGATION_MAXIMUM_GROUPS;
      const scaleCount = 3;
      const units = Math.floor(maximumRows(4) / (groupCount * scaleCount));
      const slots = maximumRows(4);
      const weightsOf = () => ({
        offsets: view('uint32', units + 1),
        neighbors: view('uint32', slots),
        weights: view('float32', slots)
      });
      graph.add(
        new GPUSegregation({
          groupCount,
          groupCounts: view('float32', units * groupCount),
          scales: [null, weightsOf(), weightsOf()],
          spatialForm: 'smoothed-population',
          output: {
            indices: view('float32', scaleCount * getGPUSegregationLayout(groupCount).stride),
            localEnvironment: view('float32', scaleCount * units * groupCount),
            localEntropy: view('float32', scaleCount * units),
            localDissimilarity: view('float32', scaleCount * units * groupCount),
            localTheil: view('float32', scaleCount * units)
          }
        })
      );
    }
  },
  // Similar locations
  ...(['zscore', 'rank'] as const).map(
    (standardization): KernelAuditCase => ({
      exportName: 'GPUSimilarLocations',
      name: `every option ${standardization}`,
      build: ({graph, view}) => {
        const rows = 500;
        const columns = 12;
        graph.add(
          new GPUSimilarLocations({
            values: view('float32', rows * columns),
            columnCount: columns,
            mask: view('uint32', rows),
            referenceMask: view('uint32', rows),
            parameters: view('float32', getGPUSimilarLocationsParameterLength(columns)),
            standardization,
            resultCapacity: rows,
            output: {
              ranks: view('uint32', rows),
              distances: view('float32', rows),
              topIds: view('uint32', rows),
              status: {count: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...(['zscore', 'rank'] as const).map(
    (standardization): KernelAuditCase => ({
      exportName: 'GPUSimilarLocations',
      name: `rows and results at the binding limit ${standardization}`,
      build: ({graph, view, maximumRows}) => {
        const rows = maximumRows(4);
        graph.add(
          new GPUSimilarLocations({
            values: view('float32', rows),
            columnCount: 1,
            mask: view('uint32', rows),
            referenceMask: view('uint32', rows),
            parameters: view('float32', getGPUSimilarLocationsParameterLength(1)),
            standardization,
            resultCapacity: rows,
            output: {
              ranks: view('uint32', rows),
              distances: view('float32', rows),
              topIds: view('uint32', rows),
              status: {count: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...(
    [
      ['columns above the maximum', GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT + 1],
      ['columns at the maximum', GPU_SIMILAR_LOCATIONS_MAXIMUM_COLUMN_COUNT],
      ['columns below the reduction dispatch limit', 1 << 23]
    ] as const
  ).map(
    ([name, columns]): KernelAuditCase => ({
      exportName: 'GPUSimilarLocations',
      name,
      build: ({graph, view}) => {
        graph.add(
          new GPUSimilarLocations({
            values: view('float32', columns),
            columnCount: columns,
            referenceMask: view('uint32', 1),
            parameters: view('float32', getGPUSimilarLocationsParameterLength(columns)),
            standardization: 'rank',
            resultCapacity: 1,
            output: {
              ranks: view('uint32', 1),
              distances: view('float32', 1),
              topIds: view('uint32', 1),
              status: {count: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  // Spatial clustering
  ...(['atomic', 'sorted'] as const).flatMap(sumOrder =>
    [
      {geographic: false, denseBoxShortcut: false},
      {geographic: false, denseBoxShortcut: true},
      {geographic: true, denseBoxShortcut: false}
    ].map(
      ({geographic, denseBoxShortcut}): KernelAuditCase => ({
        exportName: 'GPUSpatialClustering',
        name: `every output ${geographic ? 'geographic' : 'planar'} ${sumOrder} dense ${denseBoxShortcut}`,
        build: context => {
          const {graph, view} = context;
          const rows = 5000;
          const capacity = 512;
          graph.add(
            new GPUSpatialClustering({
              positions: view('float32x2', rows),
              parameters: view('float32', GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH),
              gridSize: [128, 128],
              sourceIds: view('uint32', rows),
              sumOrder,
              denseBoxShortcut,
              ...(geographic ? {spatialContext: GEOGRAPHIC} : {}),
              output: {
                labels: view('uint32', rows),
                rootRows: view('uint32', rows),
                coreFlags: view('uint32', rows),
                clusterCount: view('uint32', 1),
                clusters: compact(context, capacity),
                drawInstanceCount: view('uint32', 1),
                clusterSizes: view('uint32', capacity),
                clusterCentroids: view('float32x2', capacity)
              }
            })
          );
        }
      })
    )
  ),
  ...(
    [
      'atomic',
      'sorted, clusters above the sorted maximum',
      'sorted below the segment-sum dispatch limit'
    ] as const
  ).flatMap(variant =>
    (
      [
        ['index grid at the maximum', [18631, 1801]],
        ['index grid above the maximum', [8192, 4096]]
      ] as const
    ).map(
      ([gridName, gridSize]): KernelAuditCase => ({
        exportName: 'GPUSpatialClustering',
        name: `rows and clusters, ${gridName} ${variant}`,
        build: context => {
          const {graph, view, maximumRows} = context;
          const sumOrder = variant === 'atomic' ? 'atomic' : 'sorted';
          const rows = variant.includes('below') ? 2 ** 24 - 1 : maximumRows(8);
          const capacity = variant.includes('below') ? 2 ** 24 - 1 : maximumRows(8);
          graph.add(
            new GPUSpatialClustering({
              positions: view('float32x2', rows),
              parameters: view('float32', GPU_SPATIAL_CLUSTERING_PARAMETER_LENGTH),
              gridSize,
              sourceIds: view('uint32', rows),
              sumOrder,
              denseBoxShortcut: true,
              output: {
                labels: view('uint32', rows),
                rootRows: view('uint32', rows),
                coreFlags: view('uint32', rows),
                clusterCount: view('uint32', 1),
                clusters: compact(context, capacity),
                drawInstanceCount: view('uint32', 1),
                clusterSizes: view('uint32', capacity),
                clusterCentroids: view('float32x2', capacity)
              }
            })
          );
        }
      })
    )
  ),
  // K-means
  // ---------------------------------------------------------------- GPUHierarchicalDensityClustering
  ...[
    {name: 'small', rows: 5000, minimumSamples: 5},
    {
      name: 'rows at the maximum',
      rows: GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_ROW_COUNT,
      minimumSamples: GPU_HIERARCHICAL_DENSITY_CLUSTERING_MAXIMUM_SAMPLES
    }
  ].map(
    ({name, rows, minimumSamples}): KernelAuditCase => ({
      exportName: 'GPUHierarchicalDensityClustering',
      name: `${name} every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUHierarchicalDensityClustering({
            positions: view('float32x2', rows),
            minimumClusterSize: 2,
            minimumSamples,
            output: {
              labels: view('uint32', rows),
              coreDistances: view('float32', rows),
              status: {count: view('uint32', 1)}
            }
          })
        )
    })
  ),
  ...(
    [
      ['first-valid', 1, GPU_KMEANS_MAXIMUM_CLUSTERS],
      ['first-valid', 1, 1],
      ['kmeans++', 1, GPU_KMEANS_MAXIMUM_CLUSTERS],
      ['kmeans++', GPU_KMEANS_MAXIMUM_RESTARTS, GPU_KMEANS_MAXIMUM_CLUSTERS],
      ['kmeans++', GPU_KMEANS_MAXIMUM_RESTARTS, 7]
    ] as const
  ).map(
    ([initialization, restartCount, k]): KernelAuditCase => ({
      exportName: 'GPUKMeans',
      name: `every output ${initialization} ${restartCount} restarts k ${k}`,
      build: ({graph, view}) => {
        const rows = 5000;
        graph.add(
          new GPUKMeans({
            positions: view('float32x2', rows),
            k,
            maximumIterations: GPU_KMEANS_MAXIMUM_ITERATIONS,
            parameters: view('uint32', GPU_KMEANS_PARAMETER_LENGTH),
            initialization,
            restartCount,
            output: {
              labels: view('uint32', rows),
              centers: view('float32x2', k),
              sizes: view('uint32', k),
              squaredDistances: view('float32', rows),
              inertia: view('float32', 1),
              status: {iterationCount: view('uint32', 1), converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...(['first-valid', 'kmeans++'] as const).map(
    (initialization): KernelAuditCase => ({
      exportName: 'GPUKMeans',
      name: `rows at the binding limit ${initialization}`,
      build: ({graph, view, maximumRows}) => {
        const rows = maximumRows(8);
        const k = GPU_KMEANS_MAXIMUM_CLUSTERS;
        graph.add(
          new GPUKMeans({
            positions: view('float32x2', rows),
            k,
            maximumIterations: GPU_KMEANS_MAXIMUM_ITERATIONS,
            parameters: view('uint32', GPU_KMEANS_PARAMETER_LENGTH),
            initialization,
            restartCount: initialization === 'kmeans++' ? GPU_KMEANS_MAXIMUM_RESTARTS : 1,
            output: {
              labels: view('uint32', rows),
              centers: view('float32x2', k),
              sizes: view('uint32', k),
              squaredDistances: view('float32', rows),
              inertia: view('float32', 1),
              status: {iterationCount: view('uint32', 1), converged: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  // Inverse distance weighting
  ...[false, true].flatMap(geographic =>
    [8, 64].map(
      (neighbors): KernelAuditCase => ({
        exportName: 'GPUInverseDistanceWeighting',
        name: `every output ${geographic ? 'geographic' : 'planar'} ${neighbors} neighbors`,
        build: ({graph, view}) =>
          graph.add(
            new GPUInverseDistanceWeighting({
              positions: view('float32x2', 5000),
              values: view('float32', 5000),
              mask: view('uint32', 5000),
              parameters: view('float32', GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH),
              grid: makeGPURasterMetadata({width: 128, height: 96}),
              indexGridSize: [64, 64],
              indexBounds: [0, 0, 100, 100],
              maximumNeighborCount: neighbors,
              ...(geographic ? {spatialContext: GEOGRAPHIC} : {}),
              output: {values: view('float32', 128 * 96), counts: view('uint32', 128 * 96)}
            })
          )
      })
    )
  ),
  ...(
    [
      ['samples, raster and index grid at the maximum', [18631, 1801]],
      ['index grid above the maximum', [8192, 4096]]
    ] as const
  ).map(
    ([name, indexGridSize]): KernelAuditCase => ({
      exportName: 'GPUInverseDistanceWeighting',
      name,
      build: ({graph, view, maximumRows}) => {
        const width = 8192;
        const height = Math.floor(maximumRows(4) / width);
        graph.add(
          new GPUInverseDistanceWeighting({
            positions: view('float32x2', maximumRows(8)),
            values: view('float32', maximumRows(8)),
            mask: view('uint32', maximumRows(8)),
            parameters: view('float32', GPU_INVERSE_DISTANCE_WEIGHTING_PARAMETER_LENGTH),
            grid: makeGPURasterMetadata({width, height}),
            indexGridSize,
            indexBounds: [0, 0, 100, 100],
            maximumNeighborCount: 64,
            output: {
              values: view('float32', width * height),
              counts: view('uint32', width * height)
            }
          })
        );
      }
    })
  ),
  // Kriging
  ...[false, true].flatMap(geographic =>
    [false, true].map(
      (crossValidation): KernelAuditCase => ({
        exportName: 'GPUKriging',
        name: `every output ${geographic ? 'geographic' : 'planar'} ${crossValidation ? 'cross validation' : 'grid only'} 16 neighbors`,
        build: ({graph, view}) => {
          const samples = 5000;
          graph.add(
            new GPUKriging({
              positions: view('float32x2', samples),
              values: view('float32', samples),
              mask: view('uint32', samples),
              parameters: view('float32', GPU_KRIGING_PARAMETER_LENGTH),
              grid: makeGPURasterMetadata({width: 128, height: 96}),
              indexGridSize: [64, 64],
              indexBounds: [0, 0, 100, 100],
              maximumNeighborCount: 16,
              ...(geographic ? {spatialContext: GEOGRAPHIC} : {}),
              output: {
                values: view('float32', 128 * 96),
                variance: view('float32', 128 * 96),
                ...(crossValidation
                  ? {
                      crossValidationPrediction: view('float32', samples),
                      crossValidationVariance: view('float32', samples)
                    }
                  : {}),
                status: {
                  invalidCount: view('uint32', 1),
                  uncertainCount: view('uint32', 1)
                }
              }
            })
          );
        }
      })
    )
  ),
  ...(
    [
      ['samples, raster and index grid at the maximum', [18631, 1801]],
      ['index grid above the maximum', [8192, 4096]]
    ] as const
  ).map(
    ([name, indexGridSize]): KernelAuditCase => ({
      exportName: 'GPUKriging',
      name,
      build: ({graph, view, maximumRows}) => {
        const width = 8192;
        const height = Math.floor(maximumRows(4) / width);
        const samples = maximumRows(8);
        graph.add(
          new GPUKriging({
            positions: view('float32x2', samples),
            values: view('float32', samples),
            mask: view('uint32', samples),
            parameters: view('float32', GPU_KRIGING_PARAMETER_LENGTH),
            grid: makeGPURasterMetadata({width, height}),
            indexGridSize,
            indexBounds: [0, 0, 100, 100],
            maximumNeighborCount: 16,
            output: {
              values: view('float32', width * height),
              variance: view('float32', width * height),
              crossValidationPrediction: view('float32', samples),
              crossValidationVariance: view('float32', samples),
              status: {invalidCount: view('uint32', 1), uncertainCount: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  // Focal statistics
  ...[8, 64].map(
    (maximumRadius): KernelAuditCase => ({
      exportName: 'GPUFocalStatistics',
      name: `every output radius ${maximumRadius}`,
      build: ({graph, view}) => {
        const [width, height] = [256, 192];
        const cells = width * height;
        graph.add(
          new GPUFocalStatistics({
            values: view('float32', cells),
            validity: view('uint32', cells),
            noDataValue: -9999,
            grid: makeGPURasterMetadata({width, height}),
            maximumRadius,
            parameters: view('float32', GPU_FOCAL_STATISTICS_PARAMETER_LENGTH),
            output: {
              mean: view('float32', cells),
              sum: view('float32', cells),
              min: view('float32', cells),
              max: view('float32', cells),
              range: view('float32', cells),
              standardDeviation: view('float32', cells),
              counts: view('uint32', cells)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUFocalStatistics',
    name: 'raster at the binding limit radius 64',
    build: ({graph, view, maximumRows}) => {
      const width = 8192;
      const height = Math.floor(maximumRows(4) / width);
      const cells = width * height;
      graph.add(
        new GPUFocalStatistics({
          values: view('float32', cells),
          validity: view('uint32', cells),
          noDataValue: -9999,
          grid: makeGPURasterMetadata({width, height}),
          maximumRadius: 64,
          parameters: view('float32', GPU_FOCAL_STATISTICS_PARAMETER_LENGTH),
          output: {
            mean: view('float32', cells),
            sum: view('float32', cells),
            min: view('float32', cells),
            max: view('float32', cells),
            range: view('float32', cells),
            standardDeviation: view('float32', cells),
            counts: view('uint32', cells)
          }
        })
      );
    }
  },
  // Minimum spanning tree of spatial weights
  ...[1, GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS].flatMap(columnCount =>
    [false, true].map(
      (standardize): KernelAuditCase => ({
        exportName: 'GPUSpatialWeightsMinimumSpanningTree',
        name: `every output ${columnCount} columns standardize ${standardize}`,
        build: context => {
          const {graph, view, weights} = context;
          const rows = 1000;
          const slots = rows * 8;
          graph.add(
            new GPUSpatialWeightsMinimumSpanningTree({
              weights: weights(rows, 8),
              values: view('float32', rows * columnCount),
              columnCount,
              standardize,
              output: {
                treeEdgeFlags: view('uint32', slots),
                componentLabels: view('uint32', rows),
                ...(standardize ? {standardizedValues: view('float32', rows * columnCount)} : {}),
                edges: compact(context, rows),
                edgeEndpoints: view('uint32', 2 * rows),
                edgeDissimilarities: view('float32', rows)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUSpatialWeightsMinimumSpanningTree',
    name: 'rows and slots at the binding limit 32 columns',
    build: context => {
      const {graph, view, maximumRows} = context;
      const columnCount = GPU_MINIMUM_SPANNING_TREE_MAXIMUM_COLUMNS;
      const rows = maximumRows(4 * columnCount);
      const slots = maximumRows(4);
      graph.add(
        new GPUSpatialWeightsMinimumSpanningTree({
          weights: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', slots),
            weights: view('float32', slots)
          },
          values: view('float32', rows * columnCount),
          columnCount,
          output: {
            treeEdgeFlags: view('uint32', slots),
            componentLabels: view('uint32', rows),
            standardizedValues: view('float32', rows * columnCount),
            edges: compact(context, rows),
            edgeEndpoints: view('uint32', 2 * rows),
            edgeDissimilarities: view('float32', rows)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUSpatialWeightsMinimumSpanningTree',
    name: 'rows and slots at the binding limit 1 column',
    build: context => {
      const {graph, view, maximumRows} = context;
      const rows = maximumRows(8) - 1;
      const slots = maximumRows(4);
      graph.add(
        new GPUSpatialWeightsMinimumSpanningTree({
          weights: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', slots),
            weights: view('float32', slots)
          },
          values: view('float32', rows),
          output: {
            treeEdgeFlags: view('uint32', slots),
            componentLabels: view('uint32', rows),
            standardizedValues: view('float32', rows),
            edges: compact(context, rows),
            edgeEndpoints: view('uint32', 2 * rows),
            edgeDissimilarities: view('float32', rows)
          }
        })
      );
    }
  },
  // SKATER regions
  ...[1, GPU_SKATER_MAXIMUM_COLUMNS].map(
    (columnCount): KernelAuditCase => ({
      exportName: 'GPUSkaterRegions',
      name: `every output ${columnCount} columns 64 regions`,
      build: ({graph, view, weights}) => {
        const rows = 1000;
        const regions = 64;
        graph.add(
          new GPUSkaterRegions({
            weights: weights(rows, 8),
            treeEdgeFlags: view('uint32', rows * 8),
            componentLabels: view('uint32', rows),
            values: view('float32', rows * columnCount),
            columnCount,
            maximumRegionCount: regions,
            parameters: view('uint32', GPU_SKATER_REGIONS_PARAMETER_LENGTH),
            output: {
              labels: view('uint32', rows),
              regionCount: view('uint32', 1),
              cutEdges: view('uint32', regions - 1),
              cutGains: view('float32', regions - 1)
            }
          })
        );
      }
    })
  ),
  ...(
    [
      ['16 columns at the maximum', GPU_SKATER_MAXIMUM_COLUMNS, 0],
      ['16 columns above the maximum', GPU_SKATER_MAXIMUM_COLUMNS, 1],
      ['1 columns at the maximum', 1, 0]
    ] as const
  ).map(
    ([name, columnCount, excess]): KernelAuditCase => ({
      exportName: 'GPUSkaterRegions',
      name: `slots ${name} 1024 regions`,
      build: ({graph, view}) => {
        const maximumStatisticCount = getGPUSkaterMaximumStatisticCount(graph.device);
        // Arc statistics hold `2 + columnCount` floats per slot: slots x (2 + columnCount) is capped.
        const slots = Math.floor(maximumStatisticCount / (2 + columnCount)) + excess;
        const rows = Math.min(slots, Math.floor(maximumStatisticCount / (2 + columnCount)));
        // Each region is one unrolled cut step of command nodes, so the region count is kept at 1024.
        const regions = 1024;
        graph.add(
          new GPUSkaterRegions({
            weights: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', slots),
              weights: view('float32', slots)
            },
            treeEdgeFlags: view('uint32', slots),
            componentLabels: view('uint32', rows),
            values: view('float32', rows * columnCount),
            columnCount,
            maximumRegionCount: regions,
            parameters: view('uint32', GPU_SKATER_REGIONS_PARAMETER_LENGTH),
            output: {
              labels: view('uint32', rows),
              regionCount: view('uint32', 1),
              cutEdges: view('uint32', regions - 1),
              cutGains: view('float32', regions - 1)
            }
          })
        );
      }
    })
  ),
  // Region partition evaluation
  ...[false, true].map(
    (withWeights): KernelAuditCase => ({
      exportName: 'GPURegionPartitionEvaluation',
      name: `every output 32 columns ${withWeights ? 'with' : 'without'} weights`,
      build: ({graph, view, weights}) => {
        const rows = 1000;
        const columnCount = GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS;
        graph.add(
          new GPURegionPartitionEvaluation({
            values: view('float32', rows * columnCount),
            columnCount,
            labels: view('uint32', rows),
            labelCapacity: 128,
            ...(withWeights ? {weights: weights(rows, 8)} : {}),
            output: {
              summary: view('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length),
              regionSizes: view('uint32', 128),
              regionWithinSumOfSquares: view('float32', 128)
            }
          })
        );
      }
    })
  ),
  ...(
    [
      ['rows at the maximum 1 columns', 1, 0],
      ['rows above the maximum 1 columns', 1, 1],
      ['values at the binding limit 32 columns', GPU_REGION_PARTITION_EVALUATION_MAXIMUM_COLUMNS, 0]
    ] as const
  ).map(
    ([name, columnCount, excess]): KernelAuditCase => ({
      exportName: 'GPURegionPartitionEvaluation',
      name,
      build: ({graph, view, maximumRows}) => {
        const rows =
          Math.min(
            getGPURegionPartitionEvaluationMaximumRowCount(graph.device),
            maximumRows(4 * columnCount)
          ) + excess;
        const labelCapacity = Math.min(
          maximumRows(4) - 1,
          Math.floor(maximumRows(4) / columnCount)
        );
        graph.add(
          new GPURegionPartitionEvaluation({
            values: view('float32', rows * columnCount),
            columnCount,
            labels: view('uint32', rows),
            labelCapacity,
            weights: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', maximumRows(4)),
              weights: view('float32', maximumRows(4))
            },
            output: {
              summary: view('float32', GPU_REGION_PARTITION_EVALUATION_LAYOUT.length),
              regionSizes: view('uint32', labelCapacity),
              regionWithinSumOfSquares: view('float32', labelCapacity)
            }
          })
        );
      }
    })
  ),
  // AZP regions
  ...[false, true].flatMap(extensive =>
    [1, GPU_AZP_MAXIMUM_COLUMNS].map(
      (columnCount): KernelAuditCase => ({
        exportName: 'GPUAZPRegions',
        name: `every option ${columnCount} columns ${extensive ? 'with' : 'without'} threshold`,
        build: ({graph, view, weights}) => {
          const rows = 1000;
          graph.add(
            new GPUAZPRegions({
              weights: weights(rows, 8),
              values: view('float32', rows * columnCount),
              columnCount,
              standardize: true,
              ...(extensive
                ? {extensiveValues: view('float32', rows), extensiveThreshold: view('float32', 1)}
                : {}),
              initialLabels: view('uint32', rows),
              regionCapacity: rows,
              minimumRegionSize: 5,
              maximumIterations: GPU_AZP_MAXIMUM_ITERATIONS,
              parameters: view('uint32', GPU_AZP_REGIONS_PARAMETER_LENGTH),
              output: {
                labels: view('uint32', rows),
                objective: view('float32', 1),
                status: view('uint32', GPU_AZP_STATUS.length)
              }
            })
          );
        }
      })
    )
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUAZPRegions',
      name: excess ? 'slots above the maximum' : 'rows and slots at the maximum',
      build: ({graph, view}) => {
        const columnCount = GPU_AZP_MAXIMUM_COLUMNS;
        // The widest table: one region per row at the region record (columns + extensive + count).
        const rows = Math.floor(getGPUAZPMaximumTableWordCount(graph.device) / (columnCount + 2));
        const slots = getGPURegionalizationMaximumSlotCount(graph.device) + excess;
        graph.add(
          new GPUAZPRegions({
            weights: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', slots),
              weights: view('float32', slots)
            },
            values: view('float32', rows * columnCount),
            columnCount,
            standardize: true,
            extensiveValues: view('float32', rows),
            extensiveThreshold: view('float32', 1),
            initialLabels: view('uint32', rows),
            regionCapacity: rows,
            minimumRegionSize: rows,
            maximumIterations: GPU_AZP_MAXIMUM_ITERATIONS,
            parameters: view('uint32', GPU_AZP_REGIONS_PARAMETER_LENGTH),
            output: {
              labels: view('uint32', rows),
              objective: view('float32', 1),
              status: view('uint32', GPU_AZP_STATUS.length)
            }
          })
        );
      }
    })
  ),
  // Ward regions
  ...[1, GPU_WARD_MAXIMUM_COLUMNS].flatMap(columnCount =>
    [false, true].map(
      (standardize): KernelAuditCase => ({
        exportName: 'GPUWardRegions',
        name: `every option ${columnCount} columns standardize ${standardize}`,
        build: ({graph, view, weights}) => {
          const rows = 1000;
          graph.add(
            new GPUWardRegions({
              weights: weights(rows, 8),
              values: view('float32', rows * columnCount),
              columnCount,
              standardize,
              targetRegionCount: 10,
              maximumIterations: rows - 10,
              parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
              output: {
                labels: view('uint32', rows),
                objective: view('float32', 1),
                status: view('uint32', GPU_WARD_STATUS.length)
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUWardRegions',
    name: 'every option mutual-nearest schedule 8 columns round-count status word',
    build: ({graph, view, weights}) => {
      const rows = 1000;
      const columnCount = GPU_WARD_MAXIMUM_COLUMNS;
      graph.add(
        new GPUWardRegions({
          weights: weights(rows, 8),
          values: view('float32', rows * columnCount),
          columnCount,
          standardize: true,
          targetRegionCount: 10,
          maximumIterations: rows - 10,
          mergeSchedule: 'mutual-nearest',
          parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
          output: {
            labels: view('uint32', rows),
            objective: view('float32', 1),
            status: view('uint32', GPU_WARD_STATUS.length + 1)
          }
        })
      );
    }
  },
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUWardRegions',
      name: excess ? 'slots above the maximum' : 'rows and slots at the maximum',
      build: ({graph, view, maximumRows}) => {
        const columnCount = GPU_WARD_MAXIMUM_COLUMNS;
        const rows = maximumRows(4 * (columnCount + 1));
        const slots = getGPURegionalizationMaximumSlotCount(graph.device) + excess;
        // Every merge iteration is unrolled into command nodes; 2048 is kept to stay buildable.
        graph.add(
          new GPUWardRegions({
            weights: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', slots),
              weights: view('float32', slots)
            },
            values: view('float32', rows * columnCount),
            columnCount,
            standardize: true,
            targetRegionCount: 1,
            maximumIterations: 2048,
            parameters: view('uint32', GPU_WARD_REGIONS_PARAMETER_LENGTH),
            output: {
              labels: view('uint32', rows),
              objective: view('float32', 1),
              status: view('uint32', GPU_WARD_STATUS.length)
            }
          })
        );
      }
    })
  ),
  // Max-p regions
  ...[0, GPU_AZP_MAXIMUM_ITERATIONS].flatMap(maximumIterations =>
    [1, GPU_MAX_P_MAXIMUM_START_COUNT].map(
      (startCount): KernelAuditCase => ({
        exportName: 'GPUMaxPRegions',
        name: `every option ${startCount} starts ${maximumIterations} iterations`,
        build: ({graph, view, weights}) => {
          const rows = 1000;
          const columnCount = GPU_MAX_P_MAXIMUM_COLUMNS;
          graph.add(
            new GPUMaxPRegions({
              weights: weights(rows, 8),
              values: view('float32', rows * columnCount),
              columnCount,
              standardize: true,
              extensiveValues: view('float32', rows),
              threshold: view('float32', 1),
              startCount,
              // The largest round count only on the largest start count: 4096 rounds are about
              // 37,000 command nodes.
              maximumGrowthSteps:
                startCount === GPU_MAX_P_MAXIMUM_START_COUNT ? GPU_MAX_P_MAXIMUM_ROUNDS : undefined,
              maximumIterations,
              parameters: view('uint32', GPU_MAX_P_REGIONS_PARAMETER_LENGTH),
              output: {
                labels: view('uint32', rows),
                objective: view('float32', 1),
                status: view('uint32', GPU_MAX_P_STATUS.length)
              }
            })
          );
        }
      })
    )
  ),
  ...[0, 1].map(
    (excess): KernelAuditCase => ({
      exportName: 'GPUMaxPRegions',
      name: excess ? 'slots above the maximum' : 'rows and slots at the maximum',
      build: ({graph, view}) => {
        const columnCount = GPU_MAX_P_MAXIMUM_COLUMNS;
        // No row cap: the region table of every start (columnCount + 3 words per row) fills one
        // 128 MiB storage binding.
        const rows = Math.floor(
          graph.device.limits.maxStorageBufferBindingSize /
            (GPU_MAX_P_MAXIMUM_START_COUNT * (columnCount + 3) * 4)
        );
        const slots = getGPUMaxPMaximumSlotCount(graph.device) + excess;
        graph.add(
          new GPUMaxPRegions({
            weights: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', slots),
              weights: view('float32', slots)
            },
            values: view('float32', rows * columnCount),
            columnCount,
            standardize: true,
            extensiveValues: view('float32', rows),
            threshold: view('float32', 1),
            startCount: GPU_MAX_P_MAXIMUM_START_COUNT,
            parameters: view('uint32', GPU_MAX_P_REGIONS_PARAMETER_LENGTH),
            output: {
              labels: view('uint32', rows),
              objective: view('float32', 1),
              status: view('uint32', GPU_MAX_P_STATUS.length)
            }
          })
        );
      }
    })
  )
];
