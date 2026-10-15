// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUGridCellCount,
  getGPUGridVerticesPerCell,
  GPUGridGenerator,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Grid size, origin parameters and the area the intersection mask tests against (UTM 16N, m). */
export type GridGeneratorExampleInput = {
  columns: number;
  rows: number;
  /** `[minX, minY, cellWidth, cellHeight]` from `getGPUGridGeneratorParameterValues`, in metres. */
  parameters: GraphDataView<'float32'>;
  /** One community area; hexagons that touch it get `intersects = 1`. */
  area: GPUPolygonGeometryFields;
};

/** Pointy-top hexagon grid over Wicker Park, flagged where it touches a community area. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {columns, rows, parameters, area}: GridGeneratorExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  // #region example
  const cellCount = getGPUGridCellCount('hex', columns, rows);
  const output = {
    positions: createTransientView(
      graph,
      'cellCorners',
      'float32x2',
      cellCount * getGPUGridVerticesPerCell('hex'),
      usage
    ),
    centers: createTransientView(graph, 'centers', 'float32x2', cellCount, usage),
    intersects: createTransientView(graph, 'intersects', 'uint32', cellCount, usage)
  };
  graph.add(
    new GPUGridGenerator({
      gridType: 'hex',
      hexOrientation: 'pointy',
      columns,
      rows,
      extent: {kind: 'polygons', ...area},
      parameters,
      output
    })
  );
  // positions: 6 corners per hexagon, counter-clockwise; intersects: 1 when it touches the area
  const {positions, centers, intersects} = output;
  // #endregion example
  return {...output, positions, centers, intersects};
}
