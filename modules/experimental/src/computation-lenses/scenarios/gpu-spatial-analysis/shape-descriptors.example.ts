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
  GPUShapeDescriptors,
  type GPUPolygonGeometryFields
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Parks in UTM 16N metres and the packed sliver threshold for GPUShapeDescriptors. */
export type ShapeDescriptorsExampleInput = {
  parks: GPUPolygonGeometryFields;
  parameters: GraphDataView<'float32'>;
};

/** Compactness, elongation, orientation and convexity of every park. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {parks, parameters}: ShapeDescriptorsExampleInput
) {
  // #region example
  // Parameters: sliver threshold 0.05 (Polsby-Popper); parks are UTM 16N metres.
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const parkCount = parks.featureOffsets.length - 1;
  const column = <Format extends 'float32' | 'uint32'>(id: string, format: Format) =>
    createTransientView(graph, id, format, parkCount, usage);
  const output = {
    areas: column('areas', 'float32'),
    perimeters: column('perimeters', 'float32'),
    polsbyPopper: column('polsbyPopper', 'float32'),
    elongation: column('elongation', 'float32'),
    orientation: column('orientation', 'float32'),
    convexity: column('convexity', 'float32'),
    sliver: column('sliver', 'uint32')
  };
  graph.add(new GPUShapeDescriptors({...parks, parameters, output}));
  // polsbyPopper: 4 * pi * area / perimeter^2, 1 for a circle; orientation in radians
  const {polsbyPopper, elongation, convexity} = output;
  // #endregion example
  return {...output, polsbyPopper, elongation, convexity};
}
