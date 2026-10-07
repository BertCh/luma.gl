// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {importGraphBuffer} from '../../engine/graph-buffers';

/**
 * Returns an importer that adds an application buffer to `graph` once and hands back the same
 * typed view on every later call with the same `name`. A graph rejects two imports with one id,
 * and a buffer is often both an input of one node and an output of another.
 */
export function createViewImporter(graph: GPUCommandGraph<void>, prefix: string) {
  const imported = new Map<string, GraphDataView<GPUVectorFormat>>();
  return <Format extends GPUVectorFormat>(
    name: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ): GraphDataView<Format> => {
    let view = imported.get(name);
    if (!view) {
      view = importGraphBuffer(graph, `${prefix}-${name}`, buffer, format, length);
      imported.set(name, view);
    }
    return view as GraphDataView<Format>;
  };
}
