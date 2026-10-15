// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGeometryPredicates,
  type GPUGeometryPredicatesProps,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-parks.json`. */
export type GeometryPredicatesExampleInput = {
  /** Park polygons, projected to local-frame metres. */
  parks: Extract<GPUGeometryPredicatesProps['geometry'], {kind: 'polygons'}>;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Simplicity, vertex count and hole count of every park polygon. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, spatialContext}: GeometryPredicatesExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const parkCount = parks.featureOffsets.length - 1;
  const isSimple = createTransientView(graph, 'isSimple', 'uint32', parkCount, usage);
  const numCoordinates = createTransientView(graph, 'numCoordinates', 'uint32', parkCount, usage);
  const numInteriorRings = createTransientView(
    graph,
    'numInteriorRings',
    'uint32',
    parkCount,
    usage
  );
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1, usage);
  // #region example
  graph.add(
    new GPUGeometryPredicates({
      spatialContext,
      geometry: parks,
      intersectionCapacity: 4096,
      output: {isSimple, numCoordinates, numInteriorRings, status: {overflow}}
    })
  );
  // numCoordinates colours parks; isSimple is 1 where a ring never meets itself
  // #endregion example
  return {isSimple, numCoordinates, numInteriorRings, overflow};
}
