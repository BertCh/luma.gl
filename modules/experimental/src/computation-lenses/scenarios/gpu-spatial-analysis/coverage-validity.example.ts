// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUCoverageValidityParameterValues,
  GPUCoverageValidity,
  GPUParameterBuffer,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from `s2-chicago-north-tracts.json`. */
export type CoverageValidityExampleInput = {
  /** Census tracts as one polygon coverage, projected to local-frame metres. */
  tracts: GPUPolygonGeometryFields;
  /** Planar context of the local frame. */
  spatialContext: GPUSpatialContext;
};

/** Gap, overlap and unmatched-edge check of the tracts as a polygon coverage. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {tracts, spatialContext}: CoverageValidityExampleInput
) {
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const segmentFlags = createTransientView(
    graph,
    'segmentFlags',
    'uint32',
    tracts.positions.length,
    usage
  );
  const featureInvalidCounts = createTransientView(
    graph,
    'featureInvalidCounts',
    'uint32',
    tracts.featureOffsets.length - 1,
    usage
  );
  const isValid = createTransientView(graph, 'isValid', 'uint32', 1, usage);
  const invalidCount = createTransientView(graph, 'invalidCount', 'uint32', 1, usage);
  // #region example
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'coverage-parameters',
    format: 'float32',
    length: 4,
    values: getGPUCoverageValidityParameterValues({gapWidth: 1}) // metres
  });
  graph.add(
    new GPUCoverageValidity({
      spatialContext,
      ...tracts,
      parameters: parameters.importToGraph(graph),
      output: {segmentFlags, featureInvalidCounts, isValid, status: {invalidCount}}
    })
  );
  // featureInvalidCounts colours tracts; segmentFlags marks each offending edge
  // #endregion example
  return {segmentFlags, featureInvalidCounts, isValid, invalidCount, parameters};
}
