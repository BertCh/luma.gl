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
  GPUGeometryCleanup,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks in UTM 16N metres and the packed grid size and tolerance for GPUGeometryCleanup. */
export type GeometryCleanupExampleInput = {
  parks: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Snaps park vertices to a grid and drops repeated vertices. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, parameters}: GeometryCleanupExampleInput
) {
  // #region example
  // Parameters: snap to a 10 m grid, then drop vertices within 20 m of the last kept one.
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const view = <Format extends 'float32x2' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const output = {
    positions: view('cleanedParks', 'float32x2', parks.positions.length),
    ringOffsets: view('cleanedRingOffsets', 'uint32', parks.ringOffsets.length),
    status: {
      count: view('cleanedCount', 'uint32', 1),
      overflow: view('cleanedOverflow', 'uint32', 1),
      requiredCount: view('cleanedRequiredCount', 'uint32', 1),
      collapsedCount: view('collapsedCount', 'uint32', 1)
    }
  };
  graph.add(new GPUGeometryCleanup({...parks, geometryType: 'polygons', parameters, output}));
  // output.ringOffsets delimit the compacted rings; collapsedCount counts emptied rings
  const {positions: cleanedParks, ringOffsets: cleanedRingOffsets} = output;
  // #endregion example
  return {...output, cleanedParks, cleanedRingOffsets};
}
