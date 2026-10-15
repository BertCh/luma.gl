// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUStudyWindowPreparation,
  type GPUPolygonGeometryFields,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';

export type StudyWindowPreparationExampleInput = {
  communityAreas: GPUPolygonGeometryFields;
  spatialContext: GPUSpatialContext;
};

export async function runExample(
  graph: GPUCommandGraph<undefined>,
  input: StudyWindowPreparationExampleInput
) {
  // #region example
  const {communityAreas, spatialContext} = input;
  const summary = createTransientView(
    graph,
    'summary',
    'float32',
    8,
    Buffer.STORAGE | Buffer.COPY_SRC
  );
  const studyWindow = new GPUStudyWindowPreparation({
    ...communityAreas,
    spatialContext,
    samplingGrid: [64, 64],
    output: {summary}
  });
  graph.add(studyWindow);
  // studyWindow: prepared community-area polygons; summary: bounds, area, perimeter, edges, flags
  // #endregion example
  return {studyWindow, summary};
}
