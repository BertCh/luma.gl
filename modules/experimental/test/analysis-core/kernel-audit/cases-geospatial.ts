// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGridIndex,
  GPUHaversineDistance,
  GPUPairwisePointDistance,
  GPUPairwisePointInPolygon,
  GPUPairwisePointLinestringNearest,
  GPUPairwisePointSegmentDistance,
  GPUPointSpatialQuery,
  GPUSinusoidalProjection
} from '../../../src/geospatial';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const ROWS = 300;

/** Position encodings: local f32 XY or raw binary64 XY. */
const PRECISIONS = [
  {label: 'f32', format: 'float32x2', bytes: 8},
  {label: 'f64', format: 'uint32x4', bytes: 16}
] as const;

/** Row counts: a moderate "every option" size and the binding-limited maximum. */
function getSizes(
  context: KernelAuditContext,
  bytesPerRow: number
): {label: string; rows: number}[] {
  return [
    {label: 'moderate', rows: ROWS},
    {label: 'binding-limited maximum', rows: context.maximumRows(bytesPerRow)}
  ];
}

const SIZE_LABELS = ['moderate', 'binding-limited maximum'] as const;

/** Builds one case per precision and size for a pairwise contributor. */
function pairwiseCases(
  exportName: string,
  largestRowBytes: number,
  build: (context: KernelAuditContext, rows: number, precision: (typeof PRECISIONS)[number]) => void
): KernelAuditCase[] {
  const cases: KernelAuditCase[] = [];
  for (const precision of PRECISIONS) {
    SIZE_LABELS.forEach((sizeLabel, sizeIndex) => {
      cases.push({
        exportName,
        name: `${precision.label} ${sizeLabel}`,
        build: context =>
          build(context, getSizes(context, largestRowBytes)[sizeIndex].rows, precision)
      });
    });
  }
  return cases;
}

/** Kernel-audit cases for geospatial. */
export const GEOSPATIAL_CASES: KernelAuditCase[] = [
  ...pairwiseCases('GPUSinusoidalProjection', 16, ({graph, view}, rows, precision) => {
    for (const wrapLongitude of [false, true]) {
      graph.add(
        new GPUSinusoidalProjection({
          id: `projection-${wrapLongitude}`,
          positions: view(precision.format, rows) as never,
          output: view('float32x2', rows),
          origin: [180, 90],
          wrapLongitude
        })
      );
    }
  }),

  ...(['meters', 'kilometers'] as const).flatMap(units =>
    pairwiseCases(`GPUHaversineDistance`, 16, ({graph, view}, rows, precision) => {
      graph.add(
        new GPUHaversineDistance({
          id: `haversine-${units}`,
          left: view(precision.format, rows, 'left') as never,
          right: view(precision.format, rows, 'right') as never,
          output: view('float32', rows),
          units,
          sphereRadius: units === 'meters' ? 6_371_008.8 : 6371.0088
        } as never)
      );
    }).map(auditCase => ({...auditCase, name: `${units} ${auditCase.name}`}))
  ),

  ...pairwiseCases('GPUPairwisePointDistance', 16, ({graph, view}, rows, precision) => {
    const output =
      precision.format === 'uint32x4' ? view('float32x2', rows) : view('float32', rows);
    graph.add(
      new GPUPairwisePointDistance({
        left: view(precision.format, rows, 'left') as never,
        right: view(precision.format, rows, 'right') as never,
        output: output as never,
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'}
      } as never)
    );
  }),

  ...pairwiseCases('GPUPairwisePointSegmentDistance', 24, ({graph, view}, rows, precision) => {
    const output =
      precision.format === 'uint32x4' ? view('float32x2', rows) : view('float32', rows);
    graph.add(
      new GPUPairwisePointSegmentDistance({
        positions: view(precision.format, rows, 'points') as never,
        segmentStarts: view(precision.format, rows, 'starts') as never,
        segmentEnds: view(precision.format, rows, 'ends') as never,
        output: output as never,
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'}
      } as never)
    );
  }),

  ...pairwiseCases('GPUPairwisePointInPolygon', 16, ({view, graph}, rows, precision) => {
    const vertices = Math.min(rows, 1 << 20);
    graph.add(
      new GPUPairwisePointInPolygon({
        queryPositions: view(precision.format, rows, 'query') as never,
        positions: view(precision.format, rows, 'vertices') as never,
        featureOffsets: view('uint32', rows + 1, 'features'),
        polygonOffsets: view('uint32', vertices + 1, 'polygons'),
        ringOffsets: view('uint32', vertices + 1, 'rings'),
        output: view('uint32', rows)
      } as never)
    );
  }),

  ...[false, true].flatMap(withOptional =>
    pairwiseCases('GPUPairwisePointLinestringNearest', 16, ({view, graph}, rows, precision) => {
      const precise = precision.format === 'uint32x4';
      graph.add(
        new GPUPairwisePointLinestringNearest({
          queryPositions: view(precision.format, rows, 'query') as never,
          positions: view(precision.format, rows, 'vertices') as never,
          featureOffsets: view('uint32', rows + 1, 'features'),
          lineOffsets: view('uint32', rows + 1, 'lines'),
          spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
          output: {
            distances: precise ? view('float32x2', rows) : view('float32', rows),
            ...(withOptional
              ? {
                  nearestPositions: precise ? view('float32x4', rows) : view('float32x2', rows),
                  lineIndices: view('uint32', rows),
                  segmentIndices: view('uint32', rows)
                }
              : {})
          }
        } as never)
      );
    }).map(auditCase => ({
      ...auditCase,
      name: `${withOptional ? 'every output' : 'distances only'} ${auditCase.name}`
    }))
  ),

  ...gridIndexCases(),
  ...pointQueryCases()
];

/** Largest 2D/3D grids whose `cellCount + 1` offsets fit one binding. */
function getGridSizes(
  context: KernelAuditContext,
  dimension: 2 | 3
): {size: readonly number[]; cells: number} {
  const maximumCells = context.maximumRows(4) - 1;
  const side = Math.floor(maximumCells ** (1 / dimension));
  const size = dimension === 2 ? [side, side] : [side, side, side];
  return {size, cells: size.reduce((product, value) => product * value, 1)};
}

function gridIndexCases(): KernelAuditCase[] {
  const cases: KernelAuditCase[] = [];
  for (const dimension of [2, 3] as const) {
    const format = dimension === 2 ? 'float32x2' : 'float32x3';
    const bytes = dimension === 2 ? 8 : 12;
    const bounds = dimension === 2 ? [0, 0, 1000, 1000] : [0, 0, 0, 1000, 1000, 1000];
    for (const [label, withSourceIds, withBoundsBuffer, maximum] of [
      ['every option', true, true, false],
      ['literal bounds', false, false, false],
      ['binding-limited maximum', true, true, true]
    ] as const) {
      cases.push({
        exportName: 'GPUGridIndex',
        name: `${dimension}d ${label}`,
        build: context => {
          const {graph, view} = context;
          const {size, cells} = maximum
            ? getGridSizes(context, dimension)
            : {size: dimension === 2 ? [16, 16] : [8, 8, 8], cells: dimension === 2 ? 256 : 512};
          const rows = maximum
            ? Math.min(context.maximumRows(bytes), context.maximumRows(4))
            : ROWS;
          graph.add(
            new GPUGridIndex({
              positions: view(format, rows) as never,
              ...(withSourceIds ? {sourceIds: view('uint32', rows)} : {}),
              gridSize: size as never,
              bounds: bounds as never,
              ...(withBoundsBuffer ? {boundsBuffer: view('float32', dimension * 2)} : {}),
              cellOffsets: view('uint32', cells + 1),
              objectIds: view('uint32', rows),
              count: view('uint32', 1),
              overflow: view('uint32', 1)
            })
          );
        }
      });
    }
  }
  return cases;
}

function pointQueryCases(): KernelAuditCase[] {
  const cases: KernelAuditCase[] = [];
  const kinds = [
    {kind: 'bounds', dimension: 2},
    {kind: 'bounds', dimension: 3},
    {kind: 'radius', dimension: 2},
    {kind: 'radius', dimension: 3},
    {kind: 'polygon', dimension: 2}
  ] as const;
  for (const {kind, dimension} of kinds) {
    for (const indexed of [false, true]) {
      for (const maximum of [false, true]) {
        cases.push({
          exportName: 'GPUPointSpatialQuery',
          name: `${kind} ${dimension}d ${indexed ? 'indexed' : 'scan'} ${maximum ? 'binding-limited maximum' : 'every option'}`,
          build: context => {
            const {graph, view} = context;
            const format = dimension === 2 ? 'float32x2' : 'float32x3';
            const bytes = dimension === 2 ? 8 : 12;
            const rows = maximum ? context.maximumRows(bytes) : ROWS;
            const capacity = maximum ? context.maximumRows(4) : ROWS;
            const parameterCount = kind === 'radius' ? dimension + 1 : dimension * 2;
            const bounds = dimension === 2 ? [0, 0, 1000, 1000] : [0, 0, 0, 1000, 1000, 1000];
            const {size, cells} = maximum
              ? getGridSizes(context, dimension)
              : {size: dimension === 2 ? [16, 16] : [8, 8, 8], cells: dimension === 2 ? 256 : 512};
            graph.add(
              new GPUPointSpatialQuery({
                positions: view(format, rows) as never,
                sourceIds: view('uint32', rows),
                kind,
                ...(kind !== 'radius'
                  ? {
                      spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
                    }
                  : {}),
                parameters: view('float32', parameterCount),
                ...(kind === 'polygon'
                  ? {
                      polygon: {
                        positions: view(
                          'float32x2',
                          maximum ? context.maximumRows(8) : 64,
                          'polygon'
                        ),
                        ringOffsets: view('uint32', 8, 'rings')
                      }
                    }
                  : {}),
                ...(indexed
                  ? {
                      index: {
                        gridSize: size as never,
                        bounds: bounds as never,
                        cellOffsets: view('uint32', cells + 1),
                        rowIndices: view('uint32', capacity),
                        count: view('uint32', 1),
                        overflow: view('uint32', 1)
                      }
                    }
                  : {}),
                output: {
                  ids: view('uint32', capacity),
                  status: {
                    count: view('uint32', 1),
                    overflow: view('uint32', 1),
                    requiredCount: view('uint32', 1)
                  },
                  intersectedCellCount: view('uint32', 1),
                  candidateCount: view('uint32', 1)
                }
              } as never)
            );
          }
        });
      }
    }
  }
  return cases;
}
