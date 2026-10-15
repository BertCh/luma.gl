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
  GPUAffineTransform,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks in UTM 16N metres and the packed affine matrix for GPUAffineTransform. */
export type AffineTransformExampleInput = {
  parks: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Rotates and shrinks every park about its own centroid. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, parameters}: AffineTransformExampleInput
) {
  // #region example
  // Parameters: rotate 30 degrees, scale 0.6, about each park's centroid (UTM 16N metres).
  const output = {
    positions: createTransientView(
      graph,
      'transformedParks',
      'float32x2',
      parks.positions.length,
      Buffer.STORAGE | Buffer.COPY_SRC
    )
  };
  graph.add(
    new GPUAffineTransform({
      ...parks,
      geometryType: 'polygons',
      origins: ['centroid'],
      parameters,
      output
    })
  );
  // positions keep the park layout, so the input ring offsets still apply
  const {positions: transformedParks} = output;
  // #endregion example
  return {transformedParks};
}
