// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {
  GPURandomPointsInPolygon,
  getGPUDotDensityParameterValues,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SpatialAnalysisResources} from '../../engine/resources';

/**
 * Compiles one deterministic jitter graph per subject. The public polygon sampler supplies
 * Philox offsets uniformly in a unit 256-gon (the inscribed-disc approximation); only the
 * radius parameter changes. Base coordinates remain on the GPU and never accumulate offsets.
 */
export async function createPatternJitter(
  resources: SpatialAnalysisResources,
  subject: {id: string; count: number; baseBuffer: Buffer; positionsBuffer: Buffer},
  spatialContext: GPUSpatialContext,
  seed: number
) {
  const graph = new GPUCommandGraph<void>(resources.device, {id: `${subject.id}-jitter`});
  const upload = (name: string, values: Float32Array | Uint32Array) =>
    resources.createBuffer(`${subject.id}-jitter-${name}`, values);
  const vertices = new Float32Array(256 * 2);
  for (let vertex = 0; vertex < 256; vertex++) {
    const angle = (vertex * Math.PI * 2) / 256;
    vertices[vertex * 2] = Math.cos(angle);
    vertices[vertex * 2 + 1] = Math.sin(angle);
  }
  const radius = resources.createParameterBuffer(`${subject.id}-jitter-radius`, 'float32', 1);
  const offsets = createTransientView(graph, 'offsets', 'float32x2', subject.count);
  graph.add(
    new GPURandomPointsInPolygon({
      id: 'jitter-offsets',
      spatialContext,
      positions: importGraphBuffer(graph, 'disc', upload('disc', vertices), 'float32x2'),
      featureOffsets: importGraphBuffer(
        graph,
        'features',
        upload('features', Uint32Array.of(0, 1)),
        'uint32'
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygons',
        upload('polygons', Uint32Array.of(0, 1)),
        'uint32'
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'rings',
        upload('rings', Uint32Array.of(0, 256)),
        'uint32'
      ),
      counts: importGraphBuffer(
        graph,
        'counts',
        upload('counts', Uint32Array.of(subject.count)),
        'uint32'
      ),
      parameters: importGraphBuffer(
        graph,
        'parameters',
        upload('parameters', getGPUDotDensityParameterValues({seed})),
        'uint32'
      ),
      output: {
        positions: offsets,
        dots: {
          ids: createTransientView(graph, 'ids', 'uint32', subject.count),
          status: {
            count: createTransientView(graph, 'count', 'uint32', 1),
            overflow: createTransientView(graph, 'overflow', 'uint32', 1)
          }
        }
      }
    })
  );
  addKernelPass(graph, {
    id: 'apply-jitter',
    invocationCount: subject.count * 2,
    bindings: [
      {
        name: 'base',
        view: importGraphBuffer(graph, 'base', subject.baseBuffer, 'float32x2'),
        type: 'f32',
        access: 'read'
      },
      {name: 'offsets', view: offsets, type: 'f32', access: 'read'},
      {name: 'radius', view: radius.importToGraph(graph), type: 'f32', access: 'read'},
      {
        name: 'positions',
        view: importGraphBuffer(graph, 'positions', subject.positionsBuffer, 'float32x2'),
        type: 'f32',
        access: 'read_write'
      }
    ],
    body: `let amount = radius[radiusOffset];
      var value = base[baseOffset + index];
      if (amount > 0.0) { value += amount * offsets[offsetsOffset + index]; }
      positions[positionsOffset + index] = value;`
  });
  return {compiled: resources.track(await graph.compileAsync()), radius};
}
