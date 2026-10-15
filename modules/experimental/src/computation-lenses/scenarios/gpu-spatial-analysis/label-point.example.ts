// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, type GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULabelPoint,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks uploaded by the lens scenario, in UTM 16N metres (planar, as the element requires). */
export type LabelPointExampleInput = {parks: GPUPolygonGeometryFields};

/** Label point (pole of inaccessibility) and inscribed radius of every park. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks}: LabelPointExampleInput
) {
  // #region example
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const parkCount = parks.featureOffsets.length - 1;
  const output = {
    points: createTransientView(graph, 'labelPoints', 'float32x2', parkCount, usage),
    distances: createTransientView(graph, 'inscribedRadii', 'float32', parkCount, usage),
    degenerate: createTransientView(graph, 'degenerate', 'uint32', parkCount, usage)
  };
  graph.add(
    new GPULabelPoint({
      ...parks,
      initialGridSize: 16,
      refinementGridSize: 8,
      refinementRounds: 6,
      refinementCandidates: 4,
      output
    })
  );
  // distances: radius in metres of the largest circle found around each label point
  const {points: labelPoints, distances} = output;
  // #endregion example
  return {...output, labelPoints, distances};
}
