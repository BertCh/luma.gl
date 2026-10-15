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
  GPUOffsetCurve,
  getGPUOffsetCurveRowsPerVertex,
  type GPULineGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Streets in UTM 16N metres and the packed offset distance (20 m) for GPUOffsetCurve. */
export type OffsetCurveExampleInput = {
  roads: GPULineGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** One-sided offset of every street, with round joins on the outside of turns. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {roads, parameters}: OffsetCurveExampleInput
) {
  // #region example
  // Parameters: 20 m to the left of each street's direction (UTM 16N metres, planar).
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const rowsPerVertex = getGPUOffsetCurveRowsPerVertex('round', 8);
  const output = {
    positions: createTransientView(
      graph,
      'offsetCurve',
      'float32x2',
      roads.positions.length * rowsPerVertex,
      usage
    ),
    counts: createTransientView(graph, 'offsetCounts', 'uint32', roads.positions.length, usage)
  };
  graph.add(
    new GPUOffsetCurve({...roads, joinStyle: 'round', quadSegments: 8, parameters, output})
  );
  // positions: 2 * 8 + 1 rows per vertex; counts: distinct points per vertex, the rest repeat
  const {positions: offsetCurve, counts} = output;
  // #endregion example
  return {offsetCurve, counts};
}
