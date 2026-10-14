// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUBoundsFilter,
  GPUBufferSelection,
  GPUCellAggregation,
  GPUCellCompaction,
  GPUCellCover,
  GPUCellGeometry,
  GPUCellGridPath,
  GPUCellLevelSelection,
  GPUCellMeasures,
  GPUCellPyramid,
  GPUCellRollup,
  GPUCellSetOutline,
  GPUCellTopology,
  GPUHilbertKeys,
  GPUNearestFeatureJoin,
  GPUNearestFeatureWeights,
  GPUNeighborSearch,
  GPUPairGather,
  GPUPointInPolygonJoin,
  GPUSpatialJoinCandidates,
  GPUSpatialJoinPrepared,
  GPUSpatialPredicateJoin,
  GPUTrackSimilarity,
  GPUTrajectoryClosestApproach,
  GPUTrajectoryEncounters,
  GPUTrajectoryMetrics,
  GPUTrajectoryPlayhead,
  GPUTrajectoryResample,
  GPUZoneEvents,
  GPUPointToCell,
  GPU_BOUNDS_FILTER_PARAMETER_LENGTH,
  getGPUCellCoverMaximumCandidateCapacity,
  getGPUGridIndexMaximumCellCount,
  getGPUSpatialJoinCandidatesMaximumPairCapacity,
  getGPUTrajectoryPlayheadMaximumTrackCount,
  GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT,
  GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH,
  GPU_CELL_INDEX_FAMILIES,
  GPU_CELL_INDEX_RESOLUTION_RANGES,
  GPU_CELL_MAXIMUM_RESOLUTION,
  GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT,
  GPU_CELL_SET_OUTLINE_MAXIMUM_EDGES,
  GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE,
  GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS,
  GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH,
  GPU_HILBERT_MAXIMUM_ORDER,
  GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT,
  GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH,
  GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  type GPUNearestFeatureJoinProps,
  type GPUSpatialJoinGeometry,
  type GPUSpatialPredicate,
  type GPUCellTable
} from '../../../src/gpu-spatial-analysis';
import type {KernelAuditContext, KernelAuditCase} from './kernel-audit-context';

const ROWS = 300;

/** Most BVH leaves whose node arrays (8 B per node, 2 nodes per leaf) fit one 128 MiB binding. */
const MAXIMUM_BVH_FEATURES = 1 << 23;

/** Cell table with every optional column at `capacity` rows. */
function cellTable(
  {view}: KernelAuditContext,
  capacity: number,
  columns: boolean = true
): GPUCellTable {
  return {
    cells: view('uint32x2', capacity),
    counts: view('uint32', capacity),
    ...(columns
      ? {
          sums: view('uint32x2', capacity),
          sumValues: view('float32', capacity),
          minimums: view('float32', capacity),
          maximums: view('float32', capacity)
        }
      : {}),
    status: {
      count: view('uint32', 1),
      overflow: view('uint32', 1),
      requiredCount: view('uint32', 1)
    }
  };
}

/** Polygon layout views with `vertices` positions and `features` features. */
function polygons({view}: KernelAuditContext, features: number, vertices: number) {
  return {
    positions: view('float32x2', vertices),
    featureOffsets: view('uint32', features + 1),
    polygonOffsets: view('uint32', features + 1),
    ringOffsets: view('uint32', features + 1)
  };
}

type TimestampFormat = 'float32' | 'uint32x2';

/** Packed trajectory timestamps of the given format. */
function timestamps({view}: KernelAuditContext, format: TimestampFormat, rows: number) {
  return format === 'float32' ? view('float32', rows) : view('uint32x2', rows);
}

/** A great-circle context, as used by the trajectory contributors. */
const GREAT_CIRCLE = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;

type GPUTrackSimilarityLines = ConstructorParameters<typeof GPUTrackSimilarity>[0]['linesA'];

type GeometryKind = 'points' | 'lines' | 'polygons';

/** Packed point, line or polygon features: `features` rows over `vertices` positions. */
function geometry(
  {view}: KernelAuditContext,
  kind: GeometryKind,
  features: number,
  vertices: number
): GPUSpatialJoinGeometry {
  if (kind === 'points') {
    return {kind, positions: view('float32x2', features), sourceIds: view('uint32', features)};
  }
  if (kind === 'lines') {
    return {
      kind,
      positions: view('float32x2', vertices),
      lineOffsets: view('uint32', features + 1),
      sourceIds: view('uint32', features)
    };
  }
  return {
    kind,
    positions: view('float32x2', vertices),
    featureOffsets: view('uint32', features + 1),
    polygonOffsets: view('uint32', features + 1),
    ringOffsets: view('uint32', features + 1),
    sourceIds: view('uint32', features)
  };
}

/** Status scalars of a compact output. */
function compactStatus({view}: KernelAuditContext) {
  return {
    count: view('uint32', 1),
    requiredCount: view('uint32', 1),
    overflow: view('uint32', 1)
  };
}

/** Kernel-audit cases for cells-and-joins. */
export const CELLS_AND_JOINS_CASES: KernelAuditCase[] = [
  // ---------------------------------------------------------------- GPUBoundsFilter
  ...(['intersects', 'within', 'contains'] as const).map(
    (mode): KernelAuditCase => ({
      exportName: 'GPUBoundsFilter',
      name: `${mode} every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUBoundsFilter({
            bounds: view('float32x4', ROWS),
            parameters: view('float32', GPU_BOUNDS_FILTER_PARAMETER_LENGTH),
            mode,
            output: {
              ids: view('uint32', ROWS),
              outputMask: view('uint32', ROWS),
              status: {
                count: view('uint32', 1),
                overflow: view('uint32', 1),
                requiredCount: view('uint32', 1)
              }
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUBoundsFilter',
    name: 'rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(16);
      graph.add(
        new GPUBoundsFilter({
          bounds: view('float32x4', rows),
          parameters: view('float32', GPU_BOUNDS_FILTER_PARAMETER_LENGTH),
          output: {
            ids: view('uint32', rows),
            outputMask: view('uint32', rows),
            status: {count: view('uint32', 1), overflow: view('uint32', 1)}
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUPointToCell
  ...GPU_CELL_INDEX_FAMILIES.flatMap((family): KernelAuditCase[] => {
    const {maximum} = GPU_CELL_INDEX_RESOLUTION_RANGES[family];
    return [
      {
        exportName: 'GPUPointToCell',
        name: `${family} resolution ${maximum} every output`,
        build: ({graph, view}) =>
          graph.add(
            new GPUPointToCell({
              family,
              resolution: maximum,
              positions: view('float32x2', ROWS),
              mask: view('uint32', ROWS),
              output: {cells: view('uint32x2', ROWS), validity: view('uint32', ROWS)}
            })
          )
      },
      {
        exportName: 'GPUPointToCell',
        name: `${family} rows at the binding limit`,
        build: ({graph, view, maximumRows}) => {
          const rows = maximumRows(8);
          graph.add(
            new GPUPointToCell({
              family,
              resolution: maximum,
              positions: view('float32x2', rows),
              mask: view('uint32', rows),
              output: {cells: view('uint32x2', rows), validity: view('uint32', rows)}
            })
          );
        }
      }
    ];
  }),

  // ---------------------------------------------------------------- GPUCellAggregation
  {
    exportName: 'GPUCellAggregation',
    name: 'quadbin positions resolution 26 every output',
    build: context =>
      context.graph.add(
        new GPUCellAggregation({
          family: 'quadbin',
          resolution: GPU_CELL_MAXIMUM_RESOLUTION.quadbin,
          positions: context.view('float32x2', ROWS),
          values: context.view('float32', ROWS),
          mask: context.view('uint32', ROWS),
          sumScale: 1 << 20,
          output: cellTable(context, ROWS)
        })
      )
  },
  ...(['little-endian', 'high-low'] as const).flatMap((wordOrder): KernelAuditCase[] =>
    (['quadbin', 'h3'] as const).map(
      (family): KernelAuditCase => ({
        exportName: 'GPUCellAggregation',
        name: `${family} cells ${wordOrder} resolution max every output`,
        build: context =>
          context.graph.add(
            new GPUCellAggregation({
              family,
              resolution: GPU_CELL_MAXIMUM_RESOLUTION[family],
              cells: context.view('uint32x2', ROWS),
              wordOrder,
              values: context.view('float32', ROWS),
              mask: context.view('uint32', ROWS),
              output: cellTable(context, ROWS)
            })
          )
      })
    )
  ),
  ...(['quadbin', 'h3'] as const).map(
    (family): KernelAuditCase => ({
      exportName: 'GPUCellAggregation',
      name: `${family} cells single-sort resolution`,
      build: context =>
        context.graph.add(
          new GPUCellAggregation({
            family,
            resolution: family === 'quadbin' ? 15 : 8,
            cells: context.view('uint32x2', ROWS),
            values: context.view('float32', ROWS),
            mask: context.view('uint32', ROWS),
            output: cellTable(context, ROWS)
          })
        )
    })
  ),
  {
    exportName: 'GPUCellAggregation',
    name: 'counts only',
    build: context =>
      context.graph.add(
        new GPUCellAggregation({
          family: 'quadbin',
          resolution: 12,
          positions: context.view('float32x2', ROWS),
          output: cellTable(context, ROWS, false)
        })
      )
  },
  ...(['quadbin', 'h3'] as const).map(
    (family): KernelAuditCase => ({
      exportName: 'GPUCellAggregation',
      name: `${family} cells rows at the binding limit`,
      build: context => {
        const rows = context.maximumRows(8);
        context.graph.add(
          new GPUCellAggregation({
            family,
            resolution: GPU_CELL_MAXIMUM_RESOLUTION[family],
            cells: context.view('uint32x2', rows),
            values: context.view('float32', rows),
            mask: context.view('uint32', rows),
            output: cellTable(context, rows)
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUCellRollup
  ...(['quadbin', 'h3'] as const).map(
    (family): KernelAuditCase => ({
      exportName: 'GPUCellRollup',
      name: `${family} every column`,
      build: context =>
        context.graph.add(
          new GPUCellRollup({
            family,
            sourceResolution: GPU_CELL_MAXIMUM_RESOLUTION[family],
            resolution: 0,
            sumScale: 1 << 20,
            source: cellTable(context, ROWS),
            output: cellTable(context, ROWS)
          })
        )
    })
  ),
  {
    exportName: 'GPUCellRollup',
    name: 'counts only',
    build: context =>
      context.graph.add(
        new GPUCellRollup({
          family: 'quadbin',
          sourceResolution: 20,
          resolution: 10,
          source: cellTable(context, ROWS, false),
          output: cellTable(context, ROWS, false)
        })
      )
  },
  {
    exportName: 'GPUCellRollup',
    name: 'capacity at the binding limit',
    build: context => {
      const capacity = context.maximumRows(8);
      context.graph.add(
        new GPUCellRollup({
          family: 'h3',
          sourceResolution: 15,
          resolution: 5,
          source: cellTable(context, capacity),
          output: cellTable(context, capacity)
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUCellPyramid
  {
    exportName: 'GPUCellPyramid',
    name: 'quadbin positions 27 levels every output',
    build: context => {
      const resolutions = Array.from({length: 27}, (_, level) => 26 - level);
      context.graph.add(
        new GPUCellPyramid({
          family: 'quadbin',
          positions: context.view('float32x2', ROWS),
          values: context.view('float32', ROWS),
          mask: context.view('uint32', ROWS),
          resolutions,
          output: {
            levels: resolutions.map(() => cellTable(context, ROWS)),
            levelCounts: context.view('uint32', resolutions.length)
          }
        })
      );
    }
  },
  ...(['quadbin', 'h3'] as const).map(
    (family): KernelAuditCase => ({
      exportName: 'GPUCellPyramid',
      name: `${family} cells every output`,
      build: context => {
        const resolutions = [GPU_CELL_MAXIMUM_RESOLUTION[family], 8, 4, 0];
        context.graph.add(
          new GPUCellPyramid({
            family,
            cells: context.view('uint32x2', ROWS),
            wordOrder: 'high-low',
            values: context.view('float32', ROWS),
            mask: context.view('uint32', ROWS),
            sumScale: 1 << 12,
            resolutions,
            output: {
              levels: resolutions.map(() => cellTable(context, ROWS)),
              levelCounts: context.view('uint32', resolutions.length)
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUCellPyramid',
    name: 'rows and capacities at the binding limit',
    build: context => {
      const rows = context.maximumRows(8);
      const resolutions = [26, 20, 10];
      context.graph.add(
        new GPUCellPyramid({
          family: 'quadbin',
          cells: context.view('uint32x2', rows),
          values: context.view('float32', rows),
          mask: context.view('uint32', rows),
          resolutions,
          output: {
            levels: resolutions.map(() => cellTable(context, rows)),
            levelCounts: context.view('uint32', resolutions.length)
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUCellLevelSelection
  {
    exportName: 'GPUCellLevelSelection',
    name: '27 levels every output',
    build: ({graph, view}) =>
      graph.add(
        new GPUCellLevelSelection({
          levelCounts: view('uint32', 27),
          activeLevel: view('uint32', 1),
          levelFirstRows: Array.from({length: 27}, (_, level) => level * 1000),
          output: {
            status: {count: view('uint32', 1), requiredCount: view('uint32', 1)},
            firstRow: view('uint32', 1),
            drawArguments: view('uint32', 4)
          }
        })
      )
  },

  // ---------------------------------------------------------------- GPUCellCover
  ...(['center', 'full', 'intersects'] as const).flatMap((containment): KernelAuditCase[] =>
    [true, false].map(
      (edgeSlabs): KernelAuditCase => ({
        exportName: 'GPUCellCover',
        name: `quadbin ${containment} resolution 26 edgeSlabs ${edgeSlabs} every output`,
        build: context => {
          const {view} = context;
          context.graph.add(
            new GPUCellCover({
              family: 'quadbin',
              resolution: GPU_CELL_MAXIMUM_RESOLUTION.quadbin,
              containment,
              ...polygons(context, 20, ROWS),
              featureIds: view('uint32', 20),
              candidateCapacity: 4096,
              edgeSlabs,
              edgeSlabEntryCapacity: 4096,
              output: {
                featureIds: view('uint32', 1024),
                cells: view('uint32x2', 1024),
                core: view('uint32', 1024),
                status: {
                  count: view('uint32', 1),
                  requiredCount: view('uint32', 1),
                  overflow: view('uint32', 1),
                  candidateOverflow: view('uint32', 1),
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
  {
    exportName: 'GPUCellCover',
    name: 'quadbin full resolution 26 edgeSlabs true without uncertainCount',
    build: context => {
      const {view} = context;
      context.graph.add(
        new GPUCellCover({
          family: 'quadbin',
          resolution: GPU_CELL_MAXIMUM_RESOLUTION.quadbin,
          containment: 'full',
          ...polygons(context, 20, ROWS),
          featureIds: view('uint32', 20),
          candidateCapacity: 4096,
          edgeSlabs: true,
          edgeSlabEntryCapacity: 4096,
          output: {
            featureIds: view('uint32', 1024),
            cells: view('uint32x2', 1024),
            core: view('uint32', 1024),
            status: {
              count: view('uint32', 1),
              requiredCount: view('uint32', 1),
              overflow: view('uint32', 1),
              candidateOverflow: view('uint32', 1),
              invalidCount: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  ...[true, false].map(
    (edgeSlabs): KernelAuditCase => ({
      exportName: 'GPUCellCover',
      name: `h3 center resolution 15 edgeSlabs ${edgeSlabs} every output`,
      build: context => {
        const {view} = context;
        context.graph.add(
          new GPUCellCover({
            family: 'h3',
            resolution: GPU_CELL_MAXIMUM_RESOLUTION.h3,
            containment: 'center',
            ...polygons(context, 20, ROWS),
            featureIds: view('uint32', 20),
            candidateCapacity: 4096,
            edgeSlabs,
            edgeSlabEntryCapacity: 4096,
            output: {
              featureIds: view('uint32', 1024),
              cells: view('uint32x2', 1024),
              core: view('uint32', 1024),
              status: {
                count: view('uint32', 1),
                requiredCount: view('uint32', 1),
                overflow: view('uint32', 1),
                candidateOverflow: view('uint32', 1),
                invalidCount: view('uint32', 1),
                uncertainCount: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),
  ...(['quadbin', 'h3'] as const).flatMap((family): KernelAuditCase[] =>
    [true, false].map(
      (diagnostics): KernelAuditCase => ({
        exportName: 'GPUCellCover',
        name: `${family} capacities at the maximum${diagnostics ? '' : ' without uncertainCount'}`,
        build: context => {
          const {view, maximumRows} = context;
          // Vertices plus slab entries fill the 2^24-entry slab table exactly.
          const vertices = maximumRows(16);
          const capacity = maximumRows(8);
          context.graph.add(
            new GPUCellCover({
              family,
              resolution: GPU_CELL_MAXIMUM_RESOLUTION[family],
              ...polygons(context, 100, vertices),
              featureIds: view('uint32', 100),
              candidateCapacity: maximumRows(8),
              edgeSlabs: true,
              edgeSlabEntryCapacity: maximumRows(16),
              output: {
                featureIds: view('uint32', capacity),
                cells: view('uint32x2', capacity),
                core: view('uint32', capacity),
                status: {
                  count: view('uint32', 1),
                  requiredCount: view('uint32', 1),
                  overflow: view('uint32', 1),
                  candidateOverflow: view('uint32', 1),
                  invalidCount: view('uint32', 1),
                  ...(diagnostics ? {uncertainCount: view('uint32', 1)} : {})
                }
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUCellCover',
    name: 'quadbin edge slab entries above the maximum',
    build: context => {
      const {view, maximumRows} = context;
      context.graph.add(
        new GPUCellCover({
          family: 'quadbin',
          resolution: 10,
          ...polygons(context, 100, maximumRows(16)),
          candidateCapacity: 1024,
          edgeSlabs: true,
          edgeSlabEntryCapacity: maximumRows(16) + 1,
          output: {
            featureIds: view('uint32', 1024),
            cells: view('uint32x2', 1024),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              candidateOverflow: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  ...[
    {name: 'at the maximum', extra: 0},
    {name: 'above the maximum', extra: 1}
  ].map(
    ({name, extra}): KernelAuditCase => ({
      exportName: 'GPUCellCover',
      name: `quadbin candidateCapacity ${name}`,
      build: context => {
        const {view} = context;
        const candidateCapacity = getGPUCellCoverMaximumCandidateCapacity(context.device) + extra;
        context.graph.add(
          new GPUCellCover({
            family: 'quadbin',
            resolution: 10,
            ...polygons(context, 1, ROWS),
            candidateCapacity,
            output: {
              featureIds: view('uint32', 1024),
              cells: view('uint32x2', 1024),
              status: {
                count: view('uint32', 1),
                overflow: view('uint32', 1),
                candidateOverflow: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUCellGeometry
  ...(['quadbin', 'h3', 'quadkey', 'geohash', 's2', 'a5'] as const).flatMap(
    (family): KernelAuditCase[] => {
      const maximumVertexCount = family === 'h3' ? GPU_CELL_GEOMETRY_H3_MAXIMUM_VERTEX_COUNT : 16;
      return [
        {
          exportName: 'GPUCellGeometry',
          name: `${family} every output`,
          build: ({graph, view}) =>
            graph.add(
              new GPUCellGeometry({
                family,
                cells: view('uint32x2', ROWS),
                wordOrder: 'high-low',
                maximumVertexCount,
                output: {
                  centers: view('float32x2', ROWS),
                  boundaries: view('float32x2', ROWS * maximumVertexCount),
                  vertexCounts: view('uint32', ROWS)
                }
              })
            )
        },
        {
          exportName: 'GPUCellGeometry',
          name: `${family} rows at the binding limit`,
          build: ({graph, view, maximumRows}) => {
            const rows = Math.min(maximumRows(8), maximumRows(8 * maximumVertexCount));
            graph.add(
              new GPUCellGeometry({
                family,
                cells: view('uint32x2', rows),
                maximumVertexCount,
                output: {
                  centers: view('float32x2', rows),
                  boundaries: view('float32x2', rows * maximumVertexCount),
                  vertexCounts: view('uint32', rows)
                }
              })
            );
          }
        }
      ];
    }
  ),
  {
    exportName: 'GPUCellGeometry',
    name: 'h3 centers only',
    build: ({graph, view}) =>
      graph.add(
        new GPUCellGeometry({
          family: 'h3',
          cells: view('uint32x2', ROWS),
          output: {centers: view('float32x2', ROWS)}
        })
      )
  },

  // ---------------------------------------------------------------- GPUCellMeasures
  ...(['meters', 'kilometers'] as const).map(
    (units): KernelAuditCase => ({
      exportName: 'GPUCellMeasures',
      name: `h3 ${units} every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUCellMeasures({
            family: 'h3',
            cells: view('uint32x2', ROWS),
            wordOrder: 'high-low',
            maximumEdgeCount: GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT,
            units,
            output: {
              areas: view('float32', ROWS),
              perimeters: view('float32', ROWS),
              edgeLengths: view('float32', ROWS * GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT),
              edgeCounts: view('uint32', ROWS),
              pentagons: view('uint32', ROWS)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUCellMeasures',
    name: 'h3 rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(4 * GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT);
      graph.add(
        new GPUCellMeasures({
          family: 'h3',
          cells: view('uint32x2', rows),
          maximumEdgeCount: GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT,
          output: {
            areas: view('float32', rows),
            perimeters: view('float32', rows),
            edgeLengths: view('float32', rows * GPU_CELL_MEASURES_H3_MAXIMUM_EDGE_COUNT),
            edgeCounts: view('uint32', rows),
            pentagons: view('uint32', rows)
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUCellSetOutline
  ...(['h3', 'quadbin'] as const).flatMap((family): KernelAuditCase[] => {
    const edges = GPU_CELL_SET_OUTLINE_MAXIMUM_EDGES[family];
    const outline = (
      {view}: KernelAuditContext,
      rows: number,
      capacity: number,
      rings: boolean
    ) => ({
      family,
      cells: view('uint32x2', rows),
      count: view('uint32', 1),
      groups: view('uint32', rows),
      output: {
        rows: view('uint32', capacity),
        cells: view('uint32x2', capacity),
        edgeIndices: view('uint32', capacity),
        endpoints: view('float32x4', capacity),
        groups: view('uint32', capacity),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        }
      },
      ...(rings
        ? {
            rings: {
              vertexTolerance: 1e-4,
              normalizeWinding: true,
              output: {
                ringOffsets: view('uint32', capacity + 1),
                positions: view('float32x2', capacity + capacity),
                ringAreas: view('float32', capacity),
                ringIsHole: view('uint32', capacity),
                ringShells: view('uint32', capacity),
                ringGroups: view('uint32', capacity),
                segmentRings: view('uint32', capacity),
                segmentVertices: view('uint32', capacity),
                segmentFlags: view('uint32', capacity),
                polygons: {
                  kind: 'polygons' as const,
                  positions: view('float32x2', capacity + capacity),
                  ringOffsets: view('uint32', capacity + 1),
                  polygonOffsets: view('uint32', capacity + 1),
                  featureOffsets: view('uint32', capacity + 1),
                  sourceIds: view('uint32', capacity)
                },
                status: {
                  count: view('uint32', 1),
                  overflow: view('uint32', 1),
                  requiredCount: view('uint32', 1),
                  openSegmentCount: view('uint32', 1),
                  touchingSegmentCount: view('uint32', 1)
                }
              }
            }
          }
        : {})
    });
    return [
      {
        exportName: 'GPUCellSetOutline',
        name: `${family} every output without rings`,
        build: context =>
          context.graph.add(new GPUCellSetOutline(outline(context, ROWS, ROWS * edges, false)))
      },
      {
        exportName: 'GPUCellSetOutline',
        name: `${family} every output with rings`,
        build: context =>
          context.graph.add(new GPUCellSetOutline(outline(context, ROWS, ROWS * edges, true)))
      },
      {
        exportName: 'GPUCellSetOutline',
        name: `${family} rows and capacity at the binding limit`,
        build: context => {
          const rows = context.maximumRows(16 * edges);
          context.graph.add(
            new GPUCellSetOutline(outline(context, rows, context.maximumRows(16), false))
          );
        }
      }
    ];
  }),

  // ---------------------------------------------------------------- GPUCellTopology
  ...(['quadbin', 'h3'] as const).flatMap((family): KernelAuditCase[] => {
    const depth = family === 'h3' ? 4 : 6;
    const operations = [
      {type: 'disk', k: GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS} as const,
      {type: 'ring', k: GPU_CELL_TOPOLOGY_MAXIMUM_RADIUS} as const,
      {type: 'parent', resolution: 0} as const,
      {type: 'children', inputResolution: 0, resolution: depth} as const
    ];
    const strides = {
      disk: family === 'h3' ? 3 * 8 * 9 + 1 : 17 ** 2,
      ring: (family === 'h3' ? 6 : 8) * 8,
      parent: 1,
      children: (family === 'h3' ? 7 : 4) ** depth
    };
    return operations.flatMap((operation): KernelAuditCase[] => {
      const stride = strides[operation.type];
      const props = (context: KernelAuditContext, rows: number) => ({
        family,
        operation,
        cells: context.view('uint32x2', rows),
        wordOrder: 'high-low' as const,
        mask: context.view('uint32', rows),
        output: {
          cells: context.view('uint32x2', rows * stride),
          ...(operation.type === 'disk' ? {distances: context.view('uint32', rows * stride)} : {}),
          counts: context.view('uint32', rows)
        }
      });
      return [
        {
          exportName: 'GPUCellTopology',
          name: `${family} ${operation.type} every output`,
          build: context => context.graph.add(new GPUCellTopology(props(context, ROWS)))
        },
        {
          exportName: 'GPUCellTopology',
          name: `${family} ${operation.type} rows at the binding limit`,
          build: context =>
            context.graph.add(new GPUCellTopology(props(context, context.maximumRows(8 * stride))))
        }
      ];
    });
  }),
  ...(['quadbin', 'h3'] as const).map(
    (family): KernelAuditCase => ({
      exportName: 'GPUCellTopology',
      name: `${family} children at the maximum stride`,
      build: context => {
        // Largest depth whose stride stays within the documented maximum.
        const base = family === 'h3' ? 7 : 4;
        let depth = 0;
        while (base ** (depth + 1) <= GPU_CELL_TOPOLOGY_MAXIMUM_CHILDREN_STRIDE) depth++;
        const stride = base ** depth;
        context.graph.add(
          new GPUCellTopology({
            family,
            operation: {type: 'children', inputResolution: 0, resolution: depth},
            cells: context.view('uint32x2', ROWS),
            mask: context.view('uint32', ROWS),
            output: {
              cells: context.view('uint32x2', ROWS * stride),
              counts: context.view('uint32', ROWS)
            }
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUCellCompaction
  ...(['quadbin', 'h3'] as const).flatMap((family): KernelAuditCase[] => {
    const maximumResolution = GPU_CELL_MAXIMUM_RESOLUTION[family];
    const status = ({view}: KernelAuditContext) => ({
      count: view('uint32', 1),
      requiredCount: view('uint32', 1),
      overflow: view('uint32', 1),
      droppedCount: view('uint32', 1)
    });
    return [
      ...[false, true].map(
        (sorted): KernelAuditCase => ({
          exportName: 'GPUCellCompaction',
          name: `${family} compact sorted ${sorted} every option`,
          build: context =>
            context.graph.add(
              new GPUCellCompaction({
                family,
                operation: {
                  type: 'compact',
                  resolution: maximumResolution,
                  minimumResolution: 0,
                  sorted
                },
                cells: context.view('uint32x2', ROWS),
                wordOrder: 'high-low',
                mask: context.view('uint32', ROWS),
                count: context.view('uint32', 1),
                output: {cells: context.view('uint32x2', ROWS), status: status(context)}
              })
            )
        })
      ),
      {
        exportName: 'GPUCellCompaction',
        name: `${family} compact rows at the binding limit`,
        build: context => {
          const rows = context.maximumRows(8);
          context.graph.add(
            new GPUCellCompaction({
              family,
              operation: {type: 'compact', resolution: maximumResolution},
              cells: context.view('uint32x2', rows),
              mask: context.view('uint32', rows),
              count: context.view('uint32', 1),
              output: {cells: context.view('uint32x2', rows), status: status(context)}
            })
          );
        }
      },
      ...[GPU_CELL_UNCOMPACT_DEFAULT_MAXIMUM_DEPTH, family === 'h3' ? 11 : 15].map(
        (maximumDepth): KernelAuditCase => ({
          exportName: 'GPUCellCompaction',
          name: `${family} uncompact depth ${maximumDepth} every option`,
          build: context =>
            context.graph.add(
              new GPUCellCompaction({
                family,
                operation: {type: 'uncompact', resolution: maximumResolution, maximumDepth},
                cells: context.view('uint32x2', ROWS),
                wordOrder: 'high-low',
                mask: context.view('uint32', ROWS),
                count: context.view('uint32', 1),
                output: {cells: context.view('uint32x2', 4096), status: status(context)}
              })
            )
        })
      ),
      {
        exportName: 'GPUCellCompaction',
        name: `${family} uncompact capacity at the binding limit`,
        build: context => {
          const rows = context.maximumRows(8);
          context.graph.add(
            new GPUCellCompaction({
              family,
              operation: {type: 'uncompact', resolution: maximumResolution, maximumDepth: 11},
              cells: context.view('uint32x2', rows),
              mask: context.view('uint32', rows),
              count: context.view('uint32', 1),
              output: {cells: context.view('uint32x2', rows), status: status(context)}
            })
          );
        }
      }
    ];
  }),

  // ---------------------------------------------------------------- GPUCellGridPath
  {
    exportName: 'GPUCellGridPath',
    name: 'h3 every output at the maximum path length',
    build: ({graph, view}) =>
      graph.add(
        new GPUCellGridPath({
          family: 'h3',
          origins: view('uint32x2', ROWS),
          destinations: view('uint32x2', ROWS),
          wordOrder: 'high-low',
          mask: view('uint32', ROWS),
          maximumPathLength: GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH,
          output: {
            distances: view('uint32', ROWS),
            cells: view('uint32x2', ROWS * GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH),
            counts: view('uint32', ROWS)
          }
        })
      )
  },
  {
    exportName: 'GPUCellGridPath',
    name: 'h3 distances only',
    build: ({graph, view}) =>
      graph.add(
        new GPUCellGridPath({
          family: 'h3',
          origins: view('uint32x2', ROWS),
          destinations: view('uint32x2', ROWS),
          output: {distances: view('uint32', ROWS)}
        })
      )
  },
  {
    exportName: 'GPUCellGridPath',
    name: 'h3 rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(8 * GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH);
      graph.add(
        new GPUCellGridPath({
          family: 'h3',
          origins: view('uint32x2', rows),
          destinations: view('uint32x2', rows),
          mask: view('uint32', rows),
          maximumPathLength: GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH,
          output: {
            distances: view('uint32', rows),
            cells: view('uint32x2', rows * GPU_CELL_GRID_PATH_MAXIMUM_PATH_LENGTH),
            counts: view('uint32', rows)
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUHilbertKeys
  ...[1, 8, GPU_HILBERT_MAXIMUM_ORDER].map(
    (order): KernelAuditCase => ({
      exportName: 'GPUHilbertKeys',
      name: `positions order ${order} every output`,
      build: ({graph, view}) =>
        graph.add(
          new GPUHilbertKeys({
            order,
            positions: view('float32x2', ROWS),
            bounds: view('float32', 4),
            output: {
              keys: view('uint32', ROWS),
              sortedRows: view('uint32', ROWS),
              sortedKeys: view('uint32', ROWS)
            }
          })
        )
    })
  ),
  {
    exportName: 'GPUHilbertKeys',
    name: 'boxes with reduced bounds every output',
    build: ({graph, view}) =>
      graph.add(
        new GPUHilbertKeys({
          order: GPU_HILBERT_MAXIMUM_ORDER,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'},
          minima: view('float32x2', ROWS),
          maxima: view('float32x2', ROWS),
          output: {
            keys: view('uint32', ROWS),
            sortedRows: view('uint32', ROWS),
            sortedKeys: view('uint32', ROWS)
          }
        })
      )
  },
  {
    exportName: 'GPUHilbertKeys',
    name: 'keys only',
    build: ({graph, view}) =>
      graph.add(
        new GPUHilbertKeys({
          positions: view('float32x2', ROWS),
          output: {keys: view('uint32', ROWS)}
        })
      )
  },
  {
    exportName: 'GPUHilbertKeys',
    name: 'rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(8);
      graph.add(
        new GPUHilbertKeys({
          order: GPU_HILBERT_MAXIMUM_ORDER,
          positions: view('float32x2', rows),
          output: {
            keys: view('uint32', rows),
            sortedRows: view('uint32', rows),
            sortedKeys: view('uint32', rows)
          }
        })
      );
    }
  },
  {
    exportName: 'GPUHilbertKeys',
    name: 'box rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const rows = maximumRows(16);
      graph.add(
        new GPUHilbertKeys({
          order: GPU_HILBERT_MAXIMUM_ORDER,
          minima: view('float32x2', rows),
          maxima: view('float32x2', rows),
          bounds: view('float32', 4),
          output: {
            keys: view('uint32', rows),
            sortedRows: view('uint32', rows),
            sortedKeys: view('uint32', rows)
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUNeighborSearch
  ...(['knn', 'radius'] as const).flatMap((mode): KernelAuditCase[] =>
    [
      {name: 'planar', spatialContext: undefined},
      {
        name: 'great-circle',
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'kilometers'
        } as const
      }
    ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
      [
        {rowStandardize: undefined, weightTransform: 'raw'},
        {rowStandardize: true, weightTransform: 'row'},
        {rowStandardize: false, weightTransform: 'kernel'}
      ].flatMap(({rowStandardize, weightTransform}): KernelAuditCase[] => {
        const props = (context: KernelAuditContext, cross: boolean) => {
          const {view} = context;
          const queryRows = cross ? 200 : ROWS;
          const slots = mode === 'knn' ? queryRows * GPU_NEIGHBOR_SEARCH_MAXIMUM_K : 4096;
          return {
            mode,
            k: mode === 'knn' ? GPU_NEIGHBOR_SEARCH_MAXIMUM_K : undefined,
            spatialContext,
            positions: view('float32x2', ROWS),
            ...(cross ? {queryPositions: view('float32x2', queryRows)} : {}),
            mask: view('uint32', ROWS),
            queryMask: view('uint32', queryRows),
            parameters: view('float32', GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH),
            gridSize: [64, 64] as [number, number],
            rowStandardize,
            weightTransform: weightTransform as 'raw' | 'row' | 'kernel',
            output: {
              offsets: view('uint32', queryRows + 1),
              neighbors: view('uint32', slots),
              weights: view('float32', slots),
              distances: view('float32', slots),
              neighborCounts: view('uint32', queryRows),
              status: {
                overflow: view('uint32', 1),
                requiredCount: view('uint32', 1),
                invalidCount: view('uint32', 1)
              }
            }
          };
        };
        return [false, true].map(
          (cross): KernelAuditCase => ({
            exportName: 'GPUNeighborSearch',
            name: `${mode} ${name} ${cross ? 'cross join' : 'self join'} rowStandardize ${rowStandardize} every output`,
            build: context => context.graph.add(new GPUNeighborSearch(props(context, cross)))
          })
        );
      })
    )
  ),
  ...(['knn', 'radius'] as const).map(
    (mode): KernelAuditCase => ({
      exportName: 'GPUNeighborSearch',
      name: `${mode} rows and slots at the binding limit`,
      build: ({graph, view, maximumRows}) => {
        const rows =
          mode === 'knn' ? maximumRows(4 * GPU_NEIGHBOR_SEARCH_MAXIMUM_K) - 1 : maximumRows(12) - 1;
        const slots = maximumRows(4);
        graph.add(
          new GPUNeighborSearch({
            mode,
            k: mode === 'knn' ? GPU_NEIGHBOR_SEARCH_MAXIMUM_K : undefined,
            positions: view('float32x2', rows),
            mask: view('uint32', rows),
            queryMask: view('uint32', rows),
            parameters: view('float32', GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH),
            gridSize: [1024, Math.floor((maximumRows(4) - 1) / 1024)],
            output: {
              offsets: view('uint32', rows + 1),
              neighbors: view('uint32', slots),
              weights: view('float32', slots),
              distances: view('float32', slots),
              neighborCounts: view('uint32', rows),
              status: {
                overflow: view('uint32', 1),
                requiredCount: view('uint32', 1),
                invalidCount: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),
  ...[
    {name: 'at the maximum', extra: 0},
    {name: 'above the maximum', extra: 1}
  ].map(
    ({name, extra}): KernelAuditCase => ({
      exportName: 'GPUNeighborSearch',
      name: `radius gridSize ${name}`,
      build: ({graph, view, device}) =>
        graph.add(
          new GPUNeighborSearch({
            mode: 'radius',
            positions: view('float32x2', ROWS),
            parameters: view('float32', GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH),
            gridSize: [getGPUGridIndexMaximumCellCount(device) + extra, 1],
            output: {
              offsets: view('uint32', ROWS + 1),
              neighbors: view('uint32', 4096),
              weights: view('float32', 4096),
              status: {overflow: view('uint32', 1)}
            }
          })
        )
    })
  ),

  // ---------------------------------------------------------------- GPUPairGather
  ...(['inner', 'left'] as const).flatMap((how): KernelAuditCase[] => {
    const columns = ({view}: KernelAuditContext, rows: number, outputRows: number) => [
      {source: view('float32', rows), output: view('float32', outputRows)},
      {source: view('uint32', rows), output: view('uint32', outputRows)}
    ];
    const unmatched = (context: KernelAuditContext, rows: number) =>
      how === 'left'
        ? {unmatchedLeft: {ids: context.view('uint32', rows), status: compactStatus(context)}}
        : {};
    return [
      {
        exportName: 'GPUPairGather',
        name: `${how} pairs every output`,
        build: context => {
          const {view} = context;
          context.graph.add(
            new GPUPairGather({
              how,
              pairs: {
                leftIds: view('uint32', ROWS),
                rightIds: view('uint32', ROWS),
                status: {
                  count: view('uint32', 1),
                  requiredCount: view('uint32', 1),
                  overflow: view('uint32', 1),
                  candidateOverflow: view('uint32', 1)
                }
              },
              ...unmatched(context, ROWS),
              leftColumns: columns(context, ROWS, ROWS),
              rightColumns: columns(context, ROWS, ROWS),
              slotColumns: columns(context, ROWS, ROWS),
              overflowSources: [view('uint32', 1), view('uint32', 1)],
              output: {
                leftRows: view('uint32', ROWS),
                rightRows: view('uint32', ROWS),
                rowCount: view('uint32', 1),
                status: {overflow: view('uint32', 1), matchedCount: view('uint32', 1)}
              }
            })
          );
        }
      },
      {
        exportName: 'GPUPairGather',
        name: `${how} neighbors every output`,
        build: context => {
          const {view} = context;
          const capacity = 8;
          const slots = ROWS * capacity;
          context.graph.add(
            new GPUPairGather({
              how,
              neighbors: {ids: view('uint32', slots), counts: view('uint32', ROWS), capacity},
              ...unmatched(context, ROWS),
              leftColumns: columns(context, ROWS, slots),
              rightColumns: columns(context, ROWS, slots),
              slotColumns: columns(context, slots, slots),
              overflowSources: [view('uint32', 1)],
              output: {
                leftRows: view('uint32', slots),
                rightRows: view('uint32', slots),
                rowCount: view('uint32', 1),
                status: {overflow: view('uint32', 1), matchedCount: view('uint32', 1)}
              }
            })
          );
        }
      },
      {
        exportName: 'GPUPairGather',
        name: `${how} pairs at the binding limit`,
        build: context => {
          const {view, maximumRows} = context;
          const rows = maximumRows(4);
          context.graph.add(
            new GPUPairGather({
              how,
              pairs: {
                leftIds: view('uint32', rows),
                rightIds: view('uint32', rows),
                status: {
                  count: view('uint32', 1),
                  overflow: view('uint32', 1),
                  candidateOverflow: view('uint32', 1)
                }
              },
              ...unmatched(context, rows),
              leftColumns: columns(context, rows, rows),
              rightColumns: columns(context, rows, rows),
              slotColumns: columns(context, rows, rows),
              output: {
                leftRows: view('uint32', rows),
                rightRows: view('uint32', rows),
                rowCount: view('uint32', 1),
                status: {overflow: view('uint32', 1)}
              }
            })
          );
        }
      }
    ];
  }),

  // ---------------------------------------------------------------- GPUPointInPolygonJoin
  ...(['half-open', 'closed', 'open'] as const).flatMap((boundaryRule): KernelAuditCase[] =>
    [
      {spatialSort: true, spatialSortCurve: 'hilbert' as const, planar: true},
      {spatialSort: true, spatialSortCurve: 'morton' as const, planar: false},
      {spatialSort: false, spatialSortCurve: undefined, planar: true}
    ].map(
      ({spatialSort, spatialSortCurve, planar}): KernelAuditCase => ({
        exportName: 'GPUPointInPolygonJoin',
        name: `${boundaryRule} sort ${spatialSort} ${spatialSortCurve ?? 'default'} ${planar ? 'planar' : 'degrees'} every output`,
        build: context => {
          const {view} = context;
          context.graph.add(
            new GPUPointInPolygonJoin({
              points: {
                kind: 'points',
                positions: view('float32x2', ROWS),
                sourceIds: view('uint32', ROWS)
              },
              polygons: geometry(context, 'polygons', 50, 2000) as never,
              candidateCapacity: 4096,
              leafCapacity: 64,
              boundaryRule,
              spatialSort,
              spatialSortCurve,
              spatialContext: planar
                ? undefined
                : {coordinateSpace: 'longitude-latitude', metric: 'none'},
              output: {
                pointFeatureIds: view('uint32', ROWS),
                featureCounts: view('uint32', 50),
                matches: {ids: view('uint32', ROWS), status: compactStatus(context)},
                status: {
                  overflow: view('uint32', 1),
                  candidateCount: view('uint32', 1),
                  uncertainCount: view('uint32', 1)
                }
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUPointInPolygonJoin',
    name: 'prepared polygons every output',
    build: context => {
      const {view, graph} = context;
      const polygons = geometry(context, 'polygons', 50, 2000) as never;
      const prepared = new GPUSpatialJoinPrepared({geometry: polygons, leafCapacity: 64});
      graph.add(prepared);
      graph.add(
        new GPUPointInPolygonJoin({
          points: {kind: 'points', positions: view('float32x2', ROWS)},
          polygons,
          prepared,
          candidateCapacity: 4096,
          output: {
            pointFeatureIds: view('uint32', ROWS),
            featureCounts: view('uint32', 50),
            matches: {ids: view('uint32', ROWS), status: compactStatus(context)},
            status: {
              overflow: view('uint32', 1),
              candidateCount: view('uint32', 1),
              uncertainCount: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  {
    exportName: 'GPUPointInPolygonJoin',
    name: 'rows and capacities at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      const points = maximumRows(8);
      const features = MAXIMUM_BVH_FEATURES;
      context.graph.add(
        new GPUPointInPolygonJoin({
          points: {
            kind: 'points',
            positions: view('float32x2', points),
            sourceIds: view('uint32', points)
          },
          polygons: geometry(context, 'polygons', features, maximumRows(8)) as never,
          candidateCapacity: maximumRows(8),
          spatialSort: true,
          output: {
            pointFeatureIds: view('uint32', points),
            featureCounts: view('uint32', features),
            matches: {
              ids: view('uint32', points),
              status: compactStatus(context)
            },
            status: {
              overflow: view('uint32', 1),
              candidateCount: view('uint32', 1),
              uncertainCount: view('uint32', 1)
            }
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUNearestFeatureJoin
  ...(['points', 'segments'] as const).flatMap((featureKind): KernelAuditCase[] =>
    [
      {name: 'planar', spatialContext: undefined},
      {
        name: 'great-circle',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const
      }
    ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
      [true, false].map(
        (exclusive): KernelAuditCase => ({
          exportName: 'GPUNearestFeatureJoin',
          name: `nearest-feature ${featureKind} ${name} exclusive ${exclusive} every output`,
          build: context => {
            const {view} = context;
            const features =
              featureKind === 'points'
                ? {
                    kind: 'points' as const,
                    positions: view('float32x2', 200),
                    sourceIds: view('uint32', 200)
                  }
                : {
                    kind: 'segments' as const,
                    starts: view('float32x2', 200),
                    ends: view('float32x2', 200),
                    sourceIds: view('uint32', 200)
                  };
            context.graph.add(
              new GPUNearestFeatureJoin({
                spatialContext,
                queries: {
                  kind: 'points',
                  positions: view('float32x2', ROWS),
                  sourceIds: view('uint32', ROWS)
                },
                features,
                radius: view('float32', 1),
                ...(exclusive ? {exclusive, queryIds: view('uint32', ROWS)} : {}),
                onAttribute: {left: view('uint32', ROWS), right: view('uint32', 200)},
                candidateCapacity: 4096,
                leafCapacity: 256,
                spatialSort: true,
                spatialSortCurve: 'morton',
                output: {
                  nearestFeatureIds: view('uint32', ROWS),
                  nearestDistances: view('float32', ROWS),
                  featureCounts: view('uint32', 200),
                  matches: {ids: view('uint32', ROWS), status: compactStatus(context)},
                  status: {overflow: view('uint32', 1), candidateCount: view('uint32', 1)}
                }
              })
            );
          }
        })
      )
    )
  ),
  ...(['points', 'lines', 'polygons'] as const).flatMap((queryKind): KernelAuditCase[] =>
    (['points', 'segments', 'lines', 'polygons'] as const).flatMap(
      (featureKind): KernelAuditCase[] =>
        [
          {name: 'planar', spatialContext: undefined},
          {
            name: 'great-circle',
            spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const
          }
        ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
          (['lowest-id', 'all'] as const).map(
            (ties): KernelAuditCase => ({
              exportName: 'GPUNearestFeatureJoin',
              name: `neighbors ${queryKind} to ${featureKind} ${name} ties ${ties} every output`,
              build: context => {
                const {view} = context;
                const capacity = ties === 'all' ? 64 : GPU_NEIGHBOR_SEARCH_MAXIMUM_K;
                const slots = ROWS * capacity;
                const features =
                  featureKind === 'segments'
                    ? {
                        kind: 'segments' as const,
                        starts: view('float32x2', 200),
                        ends: view('float32x2', 200),
                        sourceIds: view('uint32', 200)
                      }
                    : (geometry(context, featureKind, 200, 2000) as never);
                const queries =
                  queryKind === 'points'
                    ? {
                        kind: 'points' as const,
                        positions: view('float32x2', ROWS),
                        sourceIds: view('uint32', ROWS)
                      }
                    : (geometry(context, queryKind, ROWS, 3000) as never);
                const props: GPUNearestFeatureJoinProps = {
                  spatialContext,
                  queries,
                  features,
                  radius: view('float32', 1),
                  exclusive: true,
                  queryIds: view('uint32', ROWS),
                  onAttribute: {left: view('uint32', ROWS), right: view('uint32', 200)},
                  k: GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
                  neighborCapacity: capacity,
                  ties,
                  leafCapacity: 256,
                  spatialSort: true,
                  output: {
                    neighborIds: view('uint32', slots),
                    neighborCounts: view('uint32', ROWS),
                    neighborDistances: view('float32', slots),
                    neighborFootPoints: view('float32x2', slots),
                    neighborSegmentIndices: view('uint32', slots),
                    neighborQueryPoints: view('float32x2', slots),
                    status: {overflow: view('uint32', 1), candidateCount: view('uint32', 1)}
                  }
                };
                context.graph.add(new GPUNearestFeatureJoin(props));
              }
            })
          )
        )
    )
  ),
  {
    exportName: 'GPUNearestFeatureJoin',
    name: 'prepared polygons neighbors',
    build: context => {
      const {view, graph} = context;
      const features = geometry(context, 'polygons', 200, 2000) as never;
      const prepared = new GPUSpatialJoinPrepared({geometry: features, leafCapacity: 256});
      graph.add(prepared);
      graph.add(
        new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: view('float32x2', ROWS)},
          features,
          prepared,
          k: 4,
          output: {
            neighborIds: view('uint32', ROWS * 4),
            neighborCounts: view('uint32', ROWS),
            neighborDistances: view('float32', ROWS * 4),
            status: {overflow: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUNearestFeatureJoin',
    name: 'nearest-feature at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      const rows = maximumRows(8);
      const features = MAXIMUM_BVH_FEATURES;
      context.graph.add(
        new GPUNearestFeatureJoin({
          queries: {
            kind: 'points',
            positions: view('float32x2', rows),
            sourceIds: view('uint32', rows)
          },
          features: {
            kind: 'segments',
            starts: view('float32x2', features),
            ends: view('float32x2', features),
            sourceIds: view('uint32', features)
          },
          radius: view('float32', 1),
          candidateCapacity: maximumRows(8),
          spatialSort: true,
          output: {
            nearestFeatureIds: view('uint32', rows),
            nearestDistances: view('float32', rows),
            featureCounts: view('uint32', features),
            matches: {ids: view('uint32', rows), status: compactStatus(context)},
            status: {overflow: view('uint32', 1), candidateCount: view('uint32', 1)}
          }
        })
      );
    }
  },
  ...[
    // 5 * (2^22 - 1) node words and two per query fill the 2^25-word node table exactly.
    {name: 'at the maximum', features: 1 << 21, queries: 6291458},
    {name: 'above the maximum', features: 1 << 22, queries: 1}
  ].map(
    ({name, features, queries}): KernelAuditCase => ({
      exportName: 'GPUNearestFeatureJoin',
      name: `neighbors node table ${name}`,
      build: context => {
        const {view, maximumRows} = context;
        context.graph.add(
          new GPUNearestFeatureJoin({
            queries: {kind: 'points', positions: view('float32x2', queries)},
            features: geometry(context, 'polygons', features, maximumRows(8)) as never,
            k: 1,
            output: {
              neighborIds: view('uint32', queries),
              neighborCounts: view('uint32', queries),
              neighborDistances: view('float32', queries),
              status: {overflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUNearestFeatureJoin',
    name: 'neighbors at the node-data binding limit',
    build: context => {
      const {view, maximumRows} = context;
      // Packed node data takes 5 words per BVH node (2 nodes per leaf).
      const capacity = 64;
      const queries = maximumRows(8 * capacity);
      const slots = queries * capacity;
      context.graph.add(
        new GPUNearestFeatureJoin({
          queries: {kind: 'points', positions: view('float32x2', queries)},
          features: geometry(context, 'polygons', 1 << 21, maximumRows(8)) as never,
          k: GPU_NEIGHBOR_SEARCH_MAXIMUM_K,
          neighborCapacity: capacity,
          ties: 'all',
          spatialSort: true,
          output: {
            neighborIds: view('uint32', slots),
            neighborCounts: view('uint32', queries),
            neighborDistances: view('float32', slots),
            neighborFootPoints: view('float32x2', slots),
            neighborSegmentIndices: view('uint32', slots),
            neighborQueryPoints: view('float32x2', slots),
            status: {overflow: view('uint32', 1), candidateCount: view('uint32', 1)}
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUBufferSelection
  ...(['points', 'segments'] as const).flatMap((featureKind): KernelAuditCase[] =>
    [
      {name: 'planar', spatialContext: undefined},
      {
        name: 'great-circle',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const
      }
    ].map(
      ({name, spatialContext}): KernelAuditCase => ({
        exportName: 'GPUBufferSelection',
        name: `${featureKind} ${name} every output`,
        build: context => {
          const {view} = context;
          context.graph.add(
            new GPUBufferSelection({
              spatialContext,
              points: {
                kind: 'points',
                positions: view('float32x2', ROWS),
                sourceIds: view('uint32', ROWS)
              },
              features:
                featureKind === 'points'
                  ? {kind: 'points', positions: view('float32x2', 200)}
                  : {
                      kind: 'segments',
                      starts: view('float32x2', 200),
                      ends: view('float32x2', 200)
                    },
              distance: view('float32', 1),
              candidateCapacity: 4096,
              leafCapacity: 256,
              spatialSort: true,
              spatialSortCurve: 'hilbert',
              output: {
                outputMask: view('uint32', ROWS),
                matches: {ids: view('uint32', ROWS), status: compactStatus(context)},
                drawInstanceCount: view('uint32', 1),
                distances: view('float32', ROWS),
                nearestFeatureIds: view('uint32', ROWS),
                status: {overflow: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUBufferSelection',
    name: 'rows and capacities at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      const rows = maximumRows(8);
      context.graph.add(
        new GPUBufferSelection({
          points: {
            kind: 'points',
            positions: view('float32x2', rows),
            sourceIds: view('uint32', rows)
          },
          features: {
            kind: 'segments',
            starts: view('float32x2', MAXIMUM_BVH_FEATURES),
            ends: view('float32x2', MAXIMUM_BVH_FEATURES)
          },
          distance: view('float32', 1),
          candidateCapacity: maximumRows(8),
          output: {
            outputMask: view('uint32', rows),
            matches: {ids: view('uint32', rows), status: compactStatus(context)},
            drawInstanceCount: view('uint32', 1),
            distances: view('float32', rows),
            nearestFeatureIds: view('uint32', rows),
            status: {overflow: view('uint32', 1)}
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUSpatialPredicateJoin
  ...(
    [
      ...(['intersects', 'contains', 'within', 'covers', 'coveredBy'] as const),
      ...(['touches', 'crosses', 'overlaps', 'equals', 'containsProperly'] as const),
      'relate',
      'dwithin'
    ] as GPUSpatialPredicate[]
  ).flatMap((predicate): KernelAuditCase[] =>
    (['points', 'lines', 'polygons'] as const).flatMap(leftKind =>
      (['points', 'lines', 'polygons'] as const).flatMap((rightKind): KernelAuditCase[] => {
        const isDistance = predicate === 'dwithin';
        const isLegacy = ['intersects', 'contains', 'within', 'dwithin'].includes(predicate);
        const build = (
          context: KernelAuditContext,
          engine: 'auto' | 'fast' | 'relate',
          fullStatus: boolean
        ) => {
          const {view} = context;
          const pairCapacity = 2048;
          context.graph.add(
            new GPUSpatialPredicateJoin({
              left: geometry(context, leftKind, ROWS, 3000),
              right: geometry(context, rightKind, 200, 2000),
              predicate,
              engine,
              ...(isDistance ? {distance: engine === 'fast' ? 3 : view('float32', 1)} : {}),
              ...(predicate === 'relate' ? {pattern: ['T********', '*T*******']} : {}),
              excludeSameRow: true,
              onAttribute: {left: view('uint32', ROWS), right: view('uint32', 200)},
              candidateCapacity: 8192,
              leafCapacity: 256,
              output: {
                pairs: {
                  leftIds: view('uint32', pairCapacity),
                  rightIds: view('uint32', pairCapacity),
                  status: {
                    count: view('uint32', 1),
                    ...(fullStatus ? {requiredCount: view('uint32', 1)} : {}),
                    overflow: view('uint32', 1),
                    candidateOverflow: view('uint32', 1)
                  }
                },
                ...(isDistance ? {} : {relate: view('uint32', pairCapacity)}),
                weights: {
                  offsets: view('uint32', ROWS + 1),
                  neighbors: view('uint32', pairCapacity),
                  weights: view('float32', pairCapacity)
                },
                status: {
                  overflow: view('uint32', 1),
                  ...(fullStatus ? {candidateCount: view('uint32', 1)} : {}),
                  uncertainCount: view('uint32', 1)
                }
              }
            })
          );
        };
        return [
          // Every status scalar wired: the finalize kernel then binds nine storage buffers.
          ...(predicate === 'intersects' || (leftKind === 'polygons' && rightKind === 'polygons')
            ? [
                {
                  exportName: 'GPUSpatialPredicateJoin',
                  name: `${predicate} ${leftKind} to ${rightKind} auto every status scalar`,
                  build: (context: KernelAuditContext) => build(context, 'auto', true)
                }
              ]
            : []),
          {
            exportName: 'GPUSpatialPredicateJoin',
            name: `${predicate} ${leftKind} to ${rightKind} auto every output within eight bindings`,
            build: context => build(context, 'auto', false)
          },
          ...(isLegacy
            ? (['fast', 'relate'] as const).map(
                (engine): KernelAuditCase => ({
                  exportName: 'GPUSpatialPredicateJoin',
                  name: `${predicate} ${leftKind} to ${rightKind} ${engine} every output within eight bindings`,
                  build: context => build(context, engine, false)
                })
              )
            : [])
        ];
      })
    )
  ),
  ...(['intersects', 'dwithin', 'touches', 'relate'] as const).flatMap(predicate =>
    (['points', 'lines', 'polygons'] as const).map(
      (rightKind): KernelAuditCase => ({
        exportName: 'GPUSpatialPredicateJoin',
        name: `${predicate} anti join polygons to ${rightKind} every output`,
        build: context => {
          const {view} = context;
          context.graph.add(
            new GPUSpatialPredicateJoin({
              left: geometry(context, 'polygons', ROWS, 3000),
              right: geometry(context, rightKind, 200, 2000),
              predicate,
              how: 'anti',
              ...(predicate === 'dwithin' ? {distance: view('float32', 1)} : {}),
              ...(predicate === 'relate' ? {pattern: view('uint32', 8)} : {}),
              onAttribute: {left: view('uint32', ROWS), right: view('uint32', 200)},
              candidateCapacity: 8192,
              output: {
                unmatched: {ids: view('uint32', ROWS), status: compactStatus(context)},
                status: {
                  overflow: view('uint32', 1),
                  candidateCount: view('uint32', 1),
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
      {
        name: 'great-circle number distance',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const,
        predicate: 'dwithin' as const,
        view: false
      },
      {
        name: 'great-circle view distance',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const,
        predicate: 'dwithin' as const,
        view: true
      },
      {
        name: 'degrees touches',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'} as const,
        predicate: 'touches' as const,
        view: false
      }
    ] as const
  ).flatMap(variant =>
    (['points', 'lines', 'polygons'] as const).flatMap(leftKind =>
      (['points', 'lines', 'polygons'] as const).map(
        (rightKind): KernelAuditCase => ({
          exportName: 'GPUSpatialPredicateJoin',
          name: `${variant.name} ${leftKind} to ${rightKind} every output`,
          build: context => {
            const {view} = context;
            context.graph.add(
              new GPUSpatialPredicateJoin({
                left: geometry(context, leftKind, ROWS, 3000),
                right: geometry(context, rightKind, 200, 2000),
                predicate: variant.predicate,
                spatialContext: variant.spatialContext,
                ...(variant.predicate === 'dwithin'
                  ? {distance: variant.view ? view('float32', 1) : 7_900_000}
                  : {}),
                candidateCapacity: 8192,
                output: {
                  pairs: {
                    leftIds: view('uint32', 2048),
                    rightIds: view('uint32', 2048),
                    status: {
                      count: view('uint32', 1),
                      overflow: view('uint32', 1),
                      candidateOverflow: view('uint32', 1)
                    }
                  },
                  status: {
                    overflow: view('uint32', 1),
                    uncertainCount: view('uint32', 1)
                  }
                }
              })
            );
          }
        })
      )
    )
  ),
  ...[
    {name: 'string', pattern: 'T********'},
    {name: 'any-of list', pattern: ['T********', '*T*******', '***T*****']}
  ].map(
    ({name, pattern}): KernelAuditCase => ({
      exportName: 'GPUSpatialPredicateJoin',
      name: `relate ${name} pattern prepared polygons`,
      build: context => {
        const {view, graph} = context;
        const right = geometry(context, 'polygons', 200, 2000);
        const prepared = new GPUSpatialJoinPrepared({geometry: right, leafCapacity: 256});
        graph.add(prepared);
        graph.add(
          new GPUSpatialPredicateJoin({
            left: geometry(context, 'polygons', ROWS, 3000),
            right,
            predicate: 'relate',
            pattern,
            prepared,
            candidateCapacity: 8192,
            output: {
              pairs: {
                leftIds: view('uint32', 2048),
                rightIds: view('uint32', 2048),
                status: {
                  count: view('uint32', 1),
                  overflow: view('uint32', 1),
                  candidateOverflow: view('uint32', 1)
                }
              },
              relate: view('uint32', 2048),
              status: {overflow: view('uint32', 1), uncertainCount: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  ...(['intersects', 'relate', 'dwithin'] as const).map(
    (predicate): KernelAuditCase => ({
      exportName: 'GPUSpatialPredicateJoin',
      name: `${predicate} polygons at the binding limit`,
      build: context => {
        const {view, maximumRows} = context;
        const pairs = maximumRows(4);
        const left = maximumRows(8);
        // GPUSpatialPredicateJoin rejects candidateCapacity above 2^24 - 1.
        const candidateCapacity = maximumRows(8) - 1;
        context.graph.add(
          new GPUSpatialPredicateJoin({
            left: geometry(context, 'polygons', left, maximumRows(8)),
            right: geometry(context, 'polygons', MAXIMUM_BVH_FEATURES, maximumRows(8)),
            predicate,
            ...(predicate === 'dwithin' ? {distance: view('float32', 1)} : {}),
            ...(predicate === 'relate' ? {pattern: 'T********'} : {}),
            candidateCapacity,
            output: {
              pairs: {
                leftIds: view('uint32', pairs),
                rightIds: view('uint32', pairs),
                status: {
                  count: view('uint32', 1),
                  overflow: view('uint32', 1),
                  candidateOverflow: view('uint32', 1)
                }
              },
              ...(predicate === 'dwithin' ? {} : {relate: view('uint32', pairs)}),
              weights: {
                offsets: view('uint32', left + 1),
                neighbors: view('uint32', pairs),
                weights: view('float32', pairs)
              },
              status: {overflow: view('uint32', 1), uncertainCount: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUSpatialJoinPrepared
  ...(['points', 'lines', 'polygons'] as const).flatMap((kind): KernelAuditCase[] =>
    [true, false].map(
      (spatialSort): KernelAuditCase => ({
        exportName: 'GPUSpatialJoinPrepared',
        name: `${kind} spatialSort ${spatialSort}`,
        build: context => {
          context.graph.add(
            new GPUSpatialJoinPrepared({
              geometry: geometry(context, kind, 500, 5000),
              leafCapacity: 512,
              spatialSort,
              rebuildWhen: () => false
            })
          );
        }
      })
    )
  ),
  ...(['points', 'lines', 'polygons'] as const).map(
    (kind): KernelAuditCase => ({
      exportName: 'GPUSpatialJoinPrepared',
      name: `${kind} features at the binding limit`,
      build: context => {
        const {maximumRows} = context;
        context.graph.add(
          new GPUSpatialJoinPrepared({
            geometry: geometry(context, kind, MAXIMUM_BVH_FEATURES, maximumRows(8)),
            spatialSort: true
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUSpatialJoinCandidates
  ...(['points', 'lines', 'polygons'] as const).flatMap(leftKind =>
    (['points', 'lines', 'polygons'] as const).flatMap((rightKind): KernelAuditCase[] =>
      [true, false].map(
        (viewDistance): KernelAuditCase => ({
          exportName: 'GPUSpatialJoinCandidates',
          name: `${leftKind} to ${rightKind} ${viewDistance ? 'view' : 'number'} distance every output`,
          build: context => {
            const {view} = context;
            context.graph.add(
              new GPUSpatialJoinCandidates({
                left: geometry(context, leftKind, ROWS, 3000),
                right: geometry(context, rightKind, 200, 2000),
                distance: viewDistance ? view('float32', 1) : 5,
                leafCapacity: 256,
                output: {
                  leftIds: view('uint32', 4096),
                  rightIds: view('uint32', 4096),
                  status: {
                    count: view('uint32', 1),
                    requiredCount: view('uint32', 1),
                    overflow: view('uint32', 1),
                    candidateOverflow: view('uint32', 1)
                  }
                }
              })
            );
          }
        })
      )
    )
  ),
  {
    exportName: 'GPUSpatialJoinCandidates',
    name: 'prepared polygons',
    build: context => {
      const {view, graph} = context;
      const right = geometry(context, 'polygons', 200, 2000);
      const prepared = new GPUSpatialJoinPrepared({
        geometry: right,
        leafCapacity: 256,
        spatialSort: false
      });
      graph.add(prepared);
      graph.add(
        new GPUSpatialJoinCandidates({
          left: geometry(context, 'points', ROWS, ROWS),
          right,
          prepared,
          output: {
            leftIds: view('uint32', 4096),
            rightIds: view('uint32', 4096),
            status: {
              count: view('uint32', 1),
              overflow: view('uint32', 1),
              candidateOverflow: view('uint32', 1)
            }
          }
        })
      );
    }
  },
  ...[
    {name: 'at the maximum', extra: 0},
    {name: 'above the maximum', extra: 1}
  ].map(
    ({name, extra}): KernelAuditCase => ({
      exportName: 'GPUSpatialJoinCandidates',
      name: `polygons pair capacity ${name}`,
      build: context => {
        const {view, maximumRows} = context;
        const pairs = getGPUSpatialJoinCandidatesMaximumPairCapacity(context.device) + extra;
        context.graph.add(
          new GPUSpatialJoinCandidates({
            left: geometry(context, 'polygons', maximumRows(8), maximumRows(8)),
            right: geometry(context, 'polygons', MAXIMUM_BVH_FEATURES, maximumRows(8)),
            distance: view('float32', 1),
            output: {
              leftIds: view('uint32', pairs),
              rightIds: view('uint32', pairs),
              status: {
                count: view('uint32', 1),
                requiredCount: view('uint32', 1),
                overflow: view('uint32', 1),
                candidateOverflow: view('uint32', 1)
              }
            }
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUNearestFeatureWeights
  ...(['binary', 'inverse-distance'] as const).flatMap((weightType): KernelAuditCase[] =>
    [true, false].map(
      (excludeSelf): KernelAuditCase => ({
        exportName: 'GPUNearestFeatureWeights',
        name: `${weightType} excludeSelf ${excludeSelf} every output`,
        build: ({graph, view}) => {
          const slotCapacity = 64;
          graph.add(
            new GPUNearestFeatureWeights({
              neighborIds: view('uint32', ROWS * slotCapacity),
              neighborCounts: view('uint32', ROWS),
              neighborDistances: view('float32', ROWS * slotCapacity),
              slotCapacity,
              k: slotCapacity - 1,
              excludeSelf,
              weightType,
              power: 2,
              distanceFloor: 1e-3,
              output: {
                offsets: view('uint32', ROWS + 1),
                neighbors: view('uint32', ROWS * slotCapacity),
                weights: view('float32', ROWS * slotCapacity),
                distances: view('float32', ROWS * slotCapacity),
                status: {overflow: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUNearestFeatureWeights',
    name: 'rows at the binding limit',
    build: ({graph, view, maximumRows}) => {
      const slotCapacity = 64;
      const rows = maximumRows(4 * slotCapacity) - 1;
      graph.add(
        new GPUNearestFeatureWeights({
          neighborIds: view('uint32', rows * slotCapacity),
          neighborCounts: view('uint32', rows),
          neighborDistances: view('float32', rows * slotCapacity),
          slotCapacity,
          excludeSelf: true,
          weightType: 'inverse-distance',
          output: {
            offsets: view('uint32', rows + 1),
            neighbors: view('uint32', rows * slotCapacity),
            weights: view('float32', rows * slotCapacity),
            distances: view('float32', rows * slotCapacity),
            status: {overflow: view('uint32', 1)}
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUTrackSimilarity
  ...[
    {name: 'planar', spatialContext: undefined},
    {name: 'great-circle', spatialContext: GREAT_CIRCLE}
  ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
    [
      {densify: 1 / 4096, maximumFrechetVertexCount: 2048},
      {densify: 1, maximumFrechetVertexCount: 1},
      {densify: undefined, maximumFrechetVertexCount: undefined}
    ].flatMap(({densify, maximumFrechetVertexCount}): KernelAuditCase[] =>
      [true, false].map(
        (twoSets): KernelAuditCase => ({
          exportName: 'GPUTrackSimilarity',
          name: `${name} ${twoSets ? 'two sets' : 'one set'} densify ${densify} frechet ${maximumFrechetVertexCount} every output`,
          build: context => {
            const {view} = context;
            const lines = (tracks: number) =>
              geometry(context, 'lines', tracks, 4000) as GPUTrackSimilarityLines;
            context.graph.add(
              new GPUTrackSimilarity({
                spatialContext,
                linesA: lines(100),
                ...(twoSets ? {linesB: lines(120)} : {}),
                pairA: view('uint32', ROWS),
                pairB: view('uint32', ROWS),
                activePairCount: view('uint32', 1),
                densify,
                maximumFrechetVertexCount,
                frechetEpsilon: view('float32', 1),
                dynamicTimeWarpingBandRadius: densify === undefined ? undefined : 16,
                output: {
                  hausdorff: view('float32', ROWS),
                  frechet: view('float32', ROWS),
                  frechetWithin: view('uint32', ROWS),
                  dynamicTimeWarping: view('float32', ROWS),
                  maximumDistance: view('float32', ROWS),
                  pairStatus: view('uint32', ROWS)
                }
              })
            );
          }
        })
      )
    )
  ),
  ...[
    {name: 'hausdorff', output: 'hausdorff'},
    {name: 'frechet', output: 'frechet'},
    {name: 'maximumDistance', output: 'maximumDistance'}
  ].map(
    ({name, output}): KernelAuditCase => ({
      exportName: 'GPUTrackSimilarity',
      name: `single measure ${name}`,
      build: ({graph, view}) =>
        graph.add(
          new GPUTrackSimilarity({
            linesA: {
              kind: 'lines',
              positions: view('float32x2', 1000),
              lineOffsets: view('uint32', 11)
            },
            pairA: view('uint32', ROWS),
            pairB: view('uint32', ROWS),
            output: {[output]: view('float32', ROWS)}
          })
        )
    })
  ),
  {
    exportName: 'GPUTrackSimilarity',
    name: 'pairs above the maximum',
    build: context => {
      const {view, maximumRows} = context;
      const pairs = GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT + 1;
      context.graph.add(
        new GPUTrackSimilarity({
          spatialContext: GREAT_CIRCLE,
          linesA: {
            kind: 'lines',
            positions: view('float32x2', maximumRows(8)),
            lineOffsets: view('uint32', maximumRows(4))
          },
          linesB: {
            kind: 'lines',
            positions: view('float32x2', maximumRows(8)),
            lineOffsets: view('uint32', maximumRows(4))
          },
          pairA: view('uint32', pairs),
          pairB: view('uint32', pairs),
          activePairCount: view('uint32', 1),
          densify: 1 / 4096,
          maximumFrechetVertexCount: 2048,
          output: {
            hausdorff: view('float32', pairs),
            frechet: view('float32', pairs),
            maximumDistance: view('float32', pairs),
            pairStatus: view('uint32', pairs)
          }
        })
      );
    }
  },

  {
    exportName: 'GPUTrackSimilarity',
    name: 'pairs at the maximum',
    build: context => {
      const {view, maximumRows} = context;
      const pairs = GPU_TRACK_SIMILARITY_MAXIMUM_PAIR_COUNT;
      context.graph.add(
        new GPUTrackSimilarity({
          spatialContext: GREAT_CIRCLE,
          linesA: {
            kind: 'lines',
            positions: view('float32x2', maximumRows(8)),
            lineOffsets: view('uint32', maximumRows(4))
          },
          pairA: view('uint32', pairs),
          pairB: view('uint32', pairs),
          activePairCount: view('uint32', 1),
          densify: 1 / 4096,
          maximumFrechetVertexCount: 2048,
          output: {
            hausdorff: view('float32', pairs),
            frechet: view('float32', pairs),
            maximumDistance: view('float32', pairs),
            pairStatus: view('uint32', pairs)
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUTrajectoryMetrics
  ...[
    {name: 'planar', spatialContext: undefined},
    {name: 'great-circle', spatialContext: GREAT_CIRCLE}
  ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
    (
      [
        {format: 'float32', low: false},
        {format: 'float32', low: true},
        {format: 'uint32x2', low: false}
      ] as const
    ).map(
      ({format, low}): KernelAuditCase => ({
        exportName: 'GPUTrajectoryMetrics',
        name: `${name} ${format} timestamps low ${low} every output`,
        build: context => {
          const {view} = context;
          const tracks = 40;
          context.graph.add(
            new GPUTrajectoryMetrics({
              spatialContext,
              positions: view('float32x2', ROWS),
              trackOffsets: view('uint32', tracks + 1),
              timestamps: timestamps(context, format, ROWS),
              ...(low ? {timestampsLow: view('float32', ROWS)} : {}),
              parameters: view('float32', GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH),
              output: {
                trackLengths: view('float32', tracks),
                trackDurations: view('float32', tracks),
                averageSpeeds: view('float32', tracks),
                maximumSpeeds: view('float32', tracks),
                stepSpeeds: view('float32', ROWS),
                stepHeadings: view('float32', ROWS),
                stepAccelerations: view('float32', ROWS),
                trackStopCounts: view('uint32', tracks),
                stops: {
                  ids: view('uint32', 128),
                  status: compactStatus(context),
                  drawInstanceCount: view('uint32', 1),
                  startRows: view('uint32', 128),
                  endRows: view('uint32', 128),
                  centroids: view('float32x2', 128),
                  durations: view('float32', 128)
                }
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUTrajectoryMetrics',
    name: 'rows and tracks at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      const rows = maximumRows(8);
      const tracks = maximumRows(4) - 1;
      const stops = maximumRows(8);
      context.graph.add(
        new GPUTrajectoryMetrics({
          spatialContext: GREAT_CIRCLE,
          positions: view('float32x2', rows),
          trackOffsets: view('uint32', tracks + 1),
          timestamps: view('float32', rows),
          timestampsLow: view('float32', rows),
          parameters: view('float32', GPU_TRAJECTORY_METRICS_PARAMETER_LENGTH),
          output: {
            trackLengths: view('float32', tracks),
            trackDurations: view('float32', tracks),
            averageSpeeds: view('float32', tracks),
            maximumSpeeds: view('float32', tracks),
            stepSpeeds: view('float32', rows),
            stepHeadings: view('float32', rows),
            stepAccelerations: view('float32', rows),
            trackStopCounts: view('uint32', tracks),
            stops: {
              ids: view('uint32', stops),
              status: compactStatus(context),
              drawInstanceCount: view('uint32', 1),
              startRows: view('uint32', stops),
              endRows: view('uint32', stops),
              centroids: view('float32x2', stops),
              durations: view('float32', stops)
            }
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUTrajectoryClosestApproach
  ...(['float32', 'uint32x2'] as const).flatMap((timeFormat): KernelAuditCase[] =>
    [true, false].map(
      (trackValid): KernelAuditCase => ({
        exportName: 'GPUTrajectoryClosestApproach',
        name: `${timeFormat} trackValid ${trackValid} every output`,
        build: context => {
          const {view} = context;
          const trackCount = 200;
          const rows = trackCount * 50;
          const capacity = 4096;
          context.graph.add(
            new GPUTrajectoryClosestApproach({
              positions: view('float32x2', rows),
              timestamps: view(timeFormat, rows),
              trackOffsets: view('uint32', trackCount + 1),
              ...(trackValid ? {trackValid: view('uint32', trackCount)} : {}),
              distance: view('float32', 1),
              cellSize: 25,
              timeCellSize: 60,
              entryCapacity: 1 << 20,
              hitCapacity: 1 << 18,
              output: {
                pairs: {
                  ids: view('uint32', capacity),
                  status: compactStatus(context),
                  partners: view('uint32', capacity),
                  minimumDistances: view('float32', capacity),
                  closestTimes: view('float32', capacity),
                  firstTimes: view('float32', capacity),
                  durations: view('float32', capacity)
                },
                status: {entryOverflow: view('uint32', 1), hitOverflow: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),

  // ---------------------------------------------------------------- GPUTrajectoryEncounters
  ...[
    {
      name: 'planar',
      spatialContext: undefined,
      bounds: [0, 0, 1000, 1000] as const,
      cellSize: 10
    },
    {
      name: 'great-circle',
      spatialContext: GREAT_CIRCLE,
      bounds: [-180, -60, 180, 60] as const,
      cellSize: 1
    },
    {
      name: 'great-circle regional',
      spatialContext: GREAT_CIRCLE,
      bounds: [-10, 30, 30, 60] as const,
      cellSize: 0.5
    }
  ].flatMap(({name, spatialContext, bounds, cellSize}): KernelAuditCase[] =>
    [true, false].map(
      (trackValid): KernelAuditCase => ({
        exportName: 'GPUTrajectoryEncounters',
        name: `${name} trackValid ${trackValid} every output`,
        build: context => {
          const {view} = context;
          const trackCount = 50;
          const bucketCount = 20;
          const capacity = 512;
          context.graph.add(
            new GPUTrajectoryEncounters({
              spatialContext,
              samples: view('float32x2', trackCount * bucketCount),
              trackCount,
              bucketCount,
              ...(trackValid ? {trackValid: view('uint32', trackCount)} : {}),
              distance: view('float32', 1),
              cellSize,
              bounds: [...bounds],
              hitCapacity: 4096,
              bucketTimes: view('float32', bucketCount),
              output: {
                pairs: {
                  ids: view('uint32', capacity),
                  status: compactStatus(context),
                  partners: view('uint32', capacity),
                  firstBuckets: view('uint32', capacity),
                  minimumDistances: view('float32', capacity),
                  bucketCounts: view('uint32', capacity),
                  firstTimes: view('float32', capacity)
                },
                status: {candidateOverflow: view('uint32', 1)}
              }
            })
          );
        }
      })
    )
  ),
  ...[
    // 1801 x 18631 cells is exactly getGPUGridIndexMaximumCellCount (2^25 - 1) with one bucket.
    {name: 'at the maximum', rows: 18631},
    {name: 'above the maximum', rows: 18632}
  ].map(
    ({name, rows}): KernelAuditCase => ({
      exportName: 'GPUTrajectoryEncounters',
      name: `lattice ${name}`,
      build: context => {
        const {view, maximumRows} = context;
        const trackCount = 4096;
        const bucketCount = 1;
        const capacity = maximumRows(8);
        context.graph.add(
          new GPUTrajectoryEncounters({
            samples: view('float32x2', trackCount * bucketCount),
            trackCount,
            bucketCount,
            trackValid: view('uint32', trackCount),
            distance: view('float32', 1),
            cellSize: 1,
            bounds: [0, 0, 1801, rows],
            hitCapacity: maximumRows(12),
            bucketTimes: view('float32', bucketCount),
            output: {
              pairs: {
                ids: view('uint32', capacity),
                status: compactStatus(context),
                partners: view('uint32', capacity),
                firstBuckets: view('uint32', capacity),
                minimumDistances: view('float32', capacity),
                bucketCounts: view('uint32', capacity),
                firstTimes: view('float32', capacity)
              },
              status: {candidateOverflow: view('uint32', 1)}
            }
          })
        );
      }
    })
  ),
  {
    exportName: 'GPUTrajectoryEncounters',
    name: 'lattice at the binding limit and capacities at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      // 2047 x 2048 cells x 8 buckets: the cell offsets just fit one 128 MiB binding.
      const trackCount = 4096;
      const bucketCount = 8;
      const capacity = maximumRows(8);
      context.graph.add(
        new GPUTrajectoryEncounters({
          samples: view('float32x2', trackCount * bucketCount),
          trackCount,
          bucketCount,
          trackValid: view('uint32', trackCount),
          distance: view('float32', 1),
          cellSize: 1,
          bounds: [0, 0, 2047, 2048],
          hitCapacity: maximumRows(12),
          bucketTimes: view('float32', bucketCount),
          output: {
            pairs: {
              ids: view('uint32', capacity),
              status: compactStatus(context),
              partners: view('uint32', capacity),
              firstBuckets: view('uint32', capacity),
              minimumDistances: view('float32', capacity),
              bucketCounts: view('uint32', capacity),
              firstTimes: view('float32', capacity)
            },
            status: {candidateOverflow: view('uint32', 1)}
          }
        })
      );
    }
  },
  {
    exportName: 'GPUTrajectoryEncounters',
    name: 'samples at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      // The grid index keeps float32x3 positions: 12 bytes per sample.
      const samples = maximumRows(12);
      const bucketCount = 64;
      const trackCount = Math.floor(samples / bucketCount);
      const capacity = maximumRows(8);
      context.graph.add(
        new GPUTrajectoryEncounters({
          samples: view('float32x2', trackCount * bucketCount),
          trackCount,
          bucketCount,
          distance: view('float32', 1),
          cellSize: 64,
          bounds: [0, 0, 1024, 1024],
          hitCapacity: maximumRows(12),
          output: {
            pairs: {ids: view('uint32', capacity), status: compactStatus(context)},
            status: {candidateOverflow: view('uint32', 1)}
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUTrajectoryPlayhead
  ...[
    {name: 'planar', spatialContext: undefined},
    {name: 'great-circle', spatialContext: GREAT_CIRCLE}
  ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
    (['float32', 'uint32x2'] as const).flatMap((format): KernelAuditCase[] =>
      [true, false].map(
        (elevations): KernelAuditCase => ({
          exportName: 'GPUTrajectoryPlayhead',
          name: `${name} ${format} timestamps elevations ${elevations} every output`,
          build: context => {
            const {view} = context;
            const tracks = 40;
            context.graph.add(
              new GPUTrajectoryPlayhead({
                spatialContext,
                positions: view('float32x2', ROWS),
                trackOffsets: view('uint32', tracks + 1),
                timestamps: timestamps(context, format, ROWS),
                ...(elevations ? {elevations: view('float32', ROWS)} : {}),
                parameters: view(
                  format === 'float32' ? 'float32' : 'uint32',
                  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
                ) as never,
                output: {
                  currentPositions: view('float32x2', tracks),
                  ...(elevations ? {currentElevations: view('float32', tracks)} : {}),
                  headings: view('float32', tracks),
                  speeds: view('float32', tracks),
                  trackStatus: view('uint32', tracks),
                  segmentRows: view('uint32', tracks),
                  segmentFractions: view('float32', tracks),
                  activeTracks: {ids: view('uint32', tracks), status: compactStatus(context)},
                  drawInstanceCount: view('uint32', 1)
                }
              })
            );
          }
        })
      )
    )
  ),
  {
    exportName: 'GPUTrajectoryPlayhead',
    name: 'tracks above the maximum',
    build: context => {
      const {view, maximumRows} = context;
      const rows = maximumRows(8);
      const tracks = getGPUTrajectoryPlayheadMaximumTrackCount(context.device) + 1;
      context.graph.add(
        new GPUTrajectoryPlayhead({
          spatialContext: GREAT_CIRCLE,
          positions: view('float32x2', rows),
          trackOffsets: view('uint32', tracks + 1),
          timestamps: view('float32', rows),
          elevations: view('float32', rows),
          parameters: view('float32', GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH),
          output: {
            currentPositions: view('float32x2', tracks),
            currentElevations: view('float32', tracks),
            headings: view('float32', tracks),
            speeds: view('float32', tracks),
            trackStatus: view('uint32', tracks),
            segmentRows: view('uint32', tracks),
            segmentFractions: view('float32', tracks),
            activeTracks: {ids: view('uint32', tracks), status: compactStatus(context)},
            drawInstanceCount: view('uint32', 1)
          }
        })
      );
    }
  },

  {
    exportName: 'GPUTrajectoryPlayhead',
    name: 'tracks at the maximum',
    build: context => {
      const {view, maximumRows} = context;
      const tracks = getGPUTrajectoryPlayheadMaximumTrackCount(context.device);
      context.graph.add(
        new GPUTrajectoryPlayhead({
          spatialContext: GREAT_CIRCLE,
          positions: view('float32x2', maximumRows(8)),
          trackOffsets: view('uint32', tracks + 1),
          timestamps: view('float32', maximumRows(8)),
          elevations: view('float32', maximumRows(8)),
          parameters: view('float32', GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH),
          output: {
            currentPositions: view('float32x2', tracks),
            currentElevations: view('float32', tracks),
            headings: view('float32', tracks),
            speeds: view('float32', tracks),
            trackStatus: view('uint32', tracks),
            segmentRows: view('uint32', tracks),
            segmentFractions: view('float32', tracks),
            activeTracks: {ids: view('uint32', tracks), status: compactStatus(context)},
            drawInstanceCount: view('uint32', 1)
          }
        })
      );
    }
  },

  // ---------------------------------------------------------------- GPUTrajectoryResample
  ...[
    {name: 'planar', spatialContext: undefined},
    {name: 'great-circle', spatialContext: GREAT_CIRCLE}
  ].flatMap(({name, spatialContext}): KernelAuditCase[] =>
    (['time', 'arc-length', 'clock'] as const).flatMap((spacing): KernelAuditCase[] =>
      (['float32', 'uint32x2'] as const).map(
        (format): KernelAuditCase => ({
          exportName: 'GPUTrajectoryResample',
          name: `${name} ${spacing} ${format} timestamps every output`,
          build: context => {
            const {view} = context;
            const tracks = 40;
            const sampleCount = 64;
            const samples = tracks * sampleCount;
            context.graph.add(
              new GPUTrajectoryResample({
                spatialContext,
                positions: view('float32x2', ROWS),
                trackOffsets: view('uint32', tracks + 1),
                timestamps: timestamps(context, format, ROWS),
                elevations: view('float32', ROWS),
                sampleCount,
                spacing,
                ...(spacing === 'clock'
                  ? {
                      clock: view(
                        format === 'float32' ? 'float32' : 'uint32',
                        GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH
                      ) as never
                    }
                  : {}),
                output: {
                  samples: view('float32x2', samples),
                  sampleElevations: view('float32', samples),
                  sampleTimes: view('float32', samples)
                }
              })
            );
          }
        })
      )
    )
  ),
  ...(['time', 'arc-length', 'clock'] as const).map(
    (spacing): KernelAuditCase => ({
      exportName: 'GPUTrajectoryResample',
      name: `${spacing} samples at the binding limit`,
      build: context => {
        const {view, maximumRows} = context;
        const rows = maximumRows(8);
        const sampleCount = 4096;
        const tracks = Math.floor(maximumRows(8) / sampleCount);
        const samples = tracks * sampleCount;
        context.graph.add(
          new GPUTrajectoryResample({
            spatialContext: GREAT_CIRCLE,
            positions: view('float32x2', rows),
            trackOffsets: view('uint32', tracks + 1),
            timestamps: view('float32', rows),
            elevations: view('float32', rows),
            sampleCount,
            spacing,
            ...(spacing === 'clock'
              ? {clock: view('float32', GPU_TRAJECTORY_CLOCK_PARAMETER_LENGTH)}
              : {}),
            output: {
              samples: view('float32x2', samples),
              sampleElevations: view('float32', samples),
              sampleTimes: view('float32', samples)
            }
          })
        );
      }
    })
  ),

  // ---------------------------------------------------------------- GPUZoneEvents
  ...(['float32', 'uint32x2'] as const).flatMap((format): KernelAuditCase[] =>
    [true, false].map(
      (spatialSort): KernelAuditCase => ({
        exportName: 'GPUZoneEvents',
        name: `${format} timestamps spatialSort ${spatialSort} every output`,
        build: context => {
          const {view} = context;
          const tracks = 40;
          const zones = 25;
          const eventCapacity = 256;
          context.graph.add(
            new GPUZoneEvents({
              tracks: {
                kind: 'trajectories',
                positions: view('float32x2', ROWS),
                trackOffsets: view('uint32', tracks + 1),
                timestamps: timestamps(context, format, ROWS) as never
              },
              zones: geometry(context, 'polygons', zones, 1000) as never,
              candidateCapacity: 8192,
              trackEventCapacity: 64,
              leafCapacity: 64,
              spatialSort,
              output: {
                events: {
                  ids: view('uint32', eventCapacity),
                  status: compactStatus(context),
                  eventZones: view('uint32', eventCapacity),
                  eventTypes: view('uint32', eventCapacity),
                  eventTimes: view('float32', eventCapacity),
                  eventRows: view('uint32', eventCapacity),
                  eventPositions: view('float32x2', eventCapacity)
                },
                dwellTimes: view('float32', tracks * zones),
                visitCounts: view('uint32', tracks * zones),
                visitTable: {
                  ids: view('uint32', eventCapacity),
                  status: compactStatus(context),
                  zones: view('uint32', eventCapacity),
                  visits: view('uint32', eventCapacity),
                  dwellTimes: view('float32', eventCapacity),
                  firstEnterTimes: view('float32', eventCapacity),
                  lastExitTimes: view('float32', eventCapacity)
                },
                trackEventCounts: view('uint32', tracks),
                status: {
                  candidateCount: view('uint32', 1),
                  candidateOverflow: view('uint32', 1),
                  trackOverflow: view('uint32', 1),
                  eventOverflow: view('uint32', 1)
                }
              }
            })
          );
        }
      })
    )
  ),
  {
    exportName: 'GPUZoneEvents',
    name: 'rows, matrices and capacities at the binding limit',
    build: context => {
      const {view, maximumRows} = context;
      const tracks = 4096;
      const zones = 4096;
      const eventCapacity = maximumRows(8);
      context.graph.add(
        new GPUZoneEvents({
          tracks: {
            kind: 'trajectories',
            positions: view('float32x2', maximumRows(8)),
            trackOffsets: view('uint32', tracks + 1),
            timestamps: view('float32', maximumRows(8))
          },
          // The edge BVH holds at most 2^23 leaves in one binding.
          zones: geometry(context, 'polygons', zones, MAXIMUM_BVH_FEATURES) as never,
          candidateCapacity: maximumRows(12),
          trackEventCapacity: 4096,
          spatialSort: true,
          output: {
            events: {
              ids: view('uint32', eventCapacity),
              status: compactStatus(context),
              eventZones: view('uint32', eventCapacity),
              eventTypes: view('uint32', eventCapacity),
              eventTimes: view('float32', eventCapacity),
              eventRows: view('uint32', eventCapacity),
              eventPositions: view('float32x2', eventCapacity)
            },
            dwellTimes: view('float32', tracks * zones),
            visitCounts: view('uint32', tracks * zones),
            visitTable: {
              ids: view('uint32', eventCapacity),
              status: compactStatus(context),
              zones: view('uint32', eventCapacity),
              visits: view('uint32', eventCapacity),
              dwellTimes: view('float32', eventCapacity),
              firstEnterTimes: view('float32', eventCapacity),
              lastExitTimes: view('float32', eventCapacity)
            },
            trackEventCounts: view('uint32', tracks),
            status: {
              candidateCount: view('uint32', 1),
              candidateOverflow: view('uint32', 1),
              trackOverflow: view('uint32', 1),
              eventOverflow: view('uint32', 1)
            }
          }
        })
      );
    }
  }
];
