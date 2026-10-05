// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphTextureView
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterNeighborhood,
  GPURasterTextureToBuffer,
  type GPURasterBand,
  type GPURasterBufferBand,
  type GPURasterTextureBand
} from '../../gpu-raster/index';
import {captureGraphCommandNodes, validateGraphViewsBelongToGraph} from '../../utils/gpu-contributor-utils';

/** WGSL helper shared by terrain kernels. @internal */
export const TERRAIN_WGSL_HELPERS = /* wgsl */ `
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }`;

/**
 * Throws unless width and height are positive integers whose product fits in uint32.
 *
 * @returns The pixel count.
 * @internal
 */
export function validateTerrainGrid(id: string, width: number, height: number): number {
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    width * height > 0xffffffff
  ) {
    throw new Error(`${id} dimensions must be positive integers`);
  }
  return width * height;
}

/** Throws unless `settings` is packed float32 with at least `length` values. @internal */
export function validateTerrainSettings(id: string, settings: GraphDataView, length: number): void {
  validatePackedView(settings, ['float32'], `${id} settings`);
  if (settings.length < length) {
    throw new Error(`${id} settings must contain at least ${length} float32 values`);
  }
}

/** Throws when output buffers repeat or alias an input buffer. @internal */
export function validateTerrainBuffersDistinct(
  id: string,
  outputs: readonly (GraphDataView | undefined)[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  const outputBuffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
  const inputBuffers = inputs.filter(view => view !== undefined).map(view => view.buffer);
  if (
    new Set(outputBuffers).size !== outputBuffers.length ||
    outputBuffers.some(buffer => inputBuffers.includes(buffer))
  ) {
    throw new Error(`${id} outputs must not share buffers with each other or with inputs`);
  }
}

/** Returns the buffer views of a raster band (values for buffer bands, plus validity). @internal */
export function getTerrainBandViews(band: GPURasterBand): GraphDataView[] {
  return [
    ...(band.storage.kind === 'buffer' ? [band.storage.values] : []),
    ...(band.validity ? [band.validity] : [])
  ];
}

/** Throws when band storage or any texture belongs to a different graph. @internal */
export function validateTerrainBandBelongsToGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  band: GPURasterBand,
  textures: readonly (GraphTextureView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, getTerrainBandViews(band));
  const allTextures = [
    ...textures,
    band.storage.kind === 'texture' ? band.storage.view : undefined
  ];
  for (const texture of allTextures) {
    if (texture && texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }
  }
}

/** Throws unless a storage texture has an allowed format and the grid extent. @internal */
export function validateTerrainTexture(
  id: string,
  name: string,
  texture: GraphTextureView | undefined,
  formats: readonly string[],
  width: number,
  height: number
): void {
  if (
    texture &&
    (!formats.includes(texture.format) || texture.width !== width || texture.height !== height)
  ) {
    throw new Error(`${id} ${name} must be ${formats.join(' or ')} with the grid extent`);
  }
}

/** Buffer-backed elevation band and the nodes that produce it. @internal */
export type TerrainElevationSource<Parameters> = {
  band: GPURasterBufferBand;
  nodes: GPUCommandNode<Parameters>[];
};

/**
 * Produces a buffer-backed elevation band for downstream raster operators.
 *
 * Texture bands are gathered with `GPURasterTextureToBuffer`. Buffer bands pass through unless
 * `canonicalize` is set, in which case an identity `GPURasterNeighborhood` resolves scale, offset,
 * nodata, validity, and non-finite samples into canonical float32 values plus uint32 validity.
 *
 * @internal
 */
export function getTerrainElevationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  band: GPURasterBand,
  width: number,
  height: number,
  canonicalize: boolean
): TerrainElevationSource<Parameters> {
  if (band.storage.kind === 'buffer' && !canonicalize) {
    return {band: band as GPURasterBufferBand, nodes: []};
  }
  const pixelCount = width * height;
  const values = createTransientView(graph, `${id}-elevation-values`, 'float32', pixelCount);
  const validity = createTransientView(graph, `${id}-elevation-validity`, 'uint32', pixelCount);
  const nodes =
    band.storage.kind === 'texture'
      ? captureGraphCommandNodes(graph, () =>
          new GPURasterTextureToBuffer({
            id: `${id}-elevation`,
            input: band as GPURasterTextureBand,
            output: values,
            outputValidity: validity,
            applyCalibration: true
          }).addToGraph(graph)
        )
      : captureGraphCommandNodes(graph, () =>
          new GPURasterNeighborhood({
            id: `${id}-elevation`,
            width,
            height,
            input: band as GPURasterBufferBand,
            output: values,
            outputValidity: validity,
            radius: 0,
            kernel: [1],
            borderMode: 'clamp',
            noDataPolicy: 'propagate',
            normalize: false
          }).addToGraph(graph)
        );
  return {
    band: {
      id: `${id}-elevation`,
      format: 'float32',
      storage: {kind: 'buffer', values},
      validity
    },
    nodes
  };
}
