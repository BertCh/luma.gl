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
  GPUGeometryOrientation,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks in UTM 16N metres and the packed winding convention for GPUGeometryOrientation. */
export type GeometryOrientationExampleInput = {
  parks: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Orients every park exterior counter-clockwise and reports which rings were reversed. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, parameters}: GeometryOrientationExampleInput
) {
  // #region example
  // Parameters: exteriorClockwise false, the GeoJSON convention (counter-clockwise shells).
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = {
    positions: createTransientView(
      graph,
      'orientedParks',
      'float32x2',
      parks.positions.length,
      usage
    ),
    reversedRings: createTransientView(
      graph,
      'reversedRings',
      'uint32',
      parks.ringOffsets.length - 1,
      usage
    )
  };
  graph.add(new GPUGeometryOrientation({...parks, mode: 'orient-polygons', parameters, output}));
  // reversedRings: 1 for every ring whose vertex order was flipped
  const {positions: orientedParks, reversedRings} = output;
  // #endregion example
  return {orientedParks, reversedRings};
}
