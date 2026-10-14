// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

/** One compute pipeline request recorded by {@link getRasterPipelineKeys}. */
export type RecordedRasterPipeline = {
  /** Pipeline (kernel) ID. */
  id: string;
  /** Complete WGSL module. */
  source: string;
  /** `(source, sorted constants)` key; `PipelineFactory` keeps one pipeline per key. */
  key: string;
};

/**
 * Compiles command nodes on a CPU-only WebGPU device that records every compute pipeline request
 * (WGSL source and override constants) instead of creating it.
 *
 * Library rule 12: the number of distinct keys must not grow with a loop count (rounds, gate
 * slots, observers, sectors).
 */
export function getRasterPipelines(nodes: readonly unknown[]): RecordedRasterPipeline[] {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  const pipelines: RecordedRasterPipeline[] = [];
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string; shader: {source: string}; constants?: Record<string, number>}) => {
      const constants = Object.entries(props.constants ?? {}).sort(([left], [right]) =>
        left < right ? -1 : left > right ? 1 : 0
      );
      pipelines.push({
        id: props.id ?? '',
        source: props.shader.source,
        key: `${props.shader.source}\n${JSON.stringify(constants)}`
      });
      return {id: props.id ?? '', hash: '', destroy: () => {}};
    }
  });
  for (const node of nodes as {compile?: (context: {device: Device}) => unknown}[]) {
    node.compile?.({device});
  }
  device.destroy();
  return pipelines;
}

/** Returns the number of distinct `(source, constants)` pipelines that `nodes` compile. */
export function countRasterPipelines(nodes: readonly unknown[]): number {
  return new Set(getRasterPipelines(nodes).map(pipeline => pipeline.key)).size;
}

/** Returns the distinct pipeline count of the nodes whose ID matches `pattern`. */
export function countRasterPipelinesMatching(nodes: readonly unknown[], pattern: RegExp): number {
  return new Set(
    getRasterPipelines(nodes)
      .filter(pipeline => pattern.test(pipeline.id))
      .map(pipeline => pipeline.key)
  ).size;
}
