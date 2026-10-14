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
import {GPURasterNeighborhood} from './gpu-raster-neighborhood';
import {GPURasterTextureToBuffer} from './gpu-raster-texture-to-buffer';
import type {
  GPURasterBand,
  GPURasterBufferBand,
  GPURasterMetadata,
  GPURasterTextureBand
} from './types';
import {
  captureGraphCommandNodes,
  validateGraphViewsBelongToGraph
} from '../utils/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../utils/wgsl-common';
import type {GPURasterCellSizeMode} from '../analysis-core/raster-grid';

/** WGSL helper shared by raster and terrain kernels. @internal */
export const RASTER_WGSL_HELPERS = /* wgsl */ IS_FINITE_VALUE_WGSL;

/**
 * Throws unless the grid width and height are positive integers whose product fits in uint32 and
 * the affine is finite.
 *
 * @returns The pixel count.
 * @internal
 */
export function validateRasterGrid(id: string, grid: GPURasterMetadata): number {
  const {width, height, affine} = grid;
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    width * height > 0xffffffff
  ) {
    throw new Error(`${id} grid dimensions must be positive integers`);
  }
  if (affine.length !== 6 || !affine.every(Number.isFinite)) {
    throw new Error(`${id} grid affine must be six finite values`);
  }
  return width * height;
}

/**
 * Throws when a latitude-dependent cell size mode meets a rotated or sheared grid: the
 * `'web-mercator'` and `'geographic'` row scales assume rows of constant world y.
 *
 * @internal
 */
export function validateRasterGridAxisAligned(
  id: string,
  grid: GPURasterMetadata,
  cellSizeMode: GPURasterCellSizeMode | undefined
): void {
  if ((cellSizeMode ?? 'uniform') !== 'uniform' && (grid.affine[1] !== 0 || grid.affine[3] !== 0)) {
    throw new Error(`${id} ${cellSizeMode} cell sizes need an axis-aligned grid`);
  }
}

/** Throws unless `parameters` is packed float32 with at least `length` values. @internal */
export function validateRasterParameters(
  id: string,
  parameters: GraphDataView,
  length: number,
  name: string = 'parameters'
): void {
  validatePackedView(parameters, ['float32'], `${id} ${name}`);
  if (parameters.length < length) {
    throw new Error(`${id} ${name} must contain at least ${length} values`);
  }
}

/** Throws unless a packer target holds at least `length` values. @internal */
export function validateParameterTarget(
  name: string,
  target: ArrayLike<number>,
  length: number
): void {
  if (target.length < length) {
    throw new Error(`${name} target must hold ${length} values`);
  }
}

/** Throws when output buffers repeat or alias an input buffer. @internal */
export function validateRasterBuffersDistinct(
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
export function getRasterBandViews(band: GPURasterBand): GraphDataView[] {
  return [
    ...(band.storage.kind === 'buffer' ? [band.storage.values] : []),
    ...(band.validity ? [band.validity] : [])
  ];
}

/** Throws when band storage or any texture belongs to a different graph. @internal */
export function validateRasterBandBelongsToGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  band: GPURasterBand,
  textures: readonly (GraphTextureView | undefined)[]
): void {
  validateGraphViewsBelongToGraph(id, graph, getRasterBandViews(band));
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
export function validateRasterOutputTexture(
  id: string,
  name: string,
  texture: GraphTextureView | undefined,
  formats: readonly string[],
  grid: GPURasterMetadata
): void {
  if (
    texture &&
    (!formats.includes(texture.format) ||
      texture.width !== grid.width ||
      texture.height !== grid.height)
  ) {
    throw new Error(`${id} ${name} must be ${formats.join(' or ')} with the grid extent`);
  }
}

/** Buffer-backed band and the nodes that produce it. @internal */
export type RasterBufferBandSource<Parameters> = {
  band: GPURasterBufferBand;
  nodes: GPUCommandNode<Parameters>[];
};

/**
 * Produces a buffer-backed band (typically elevation) for downstream raster operators.
 *
 * Texture bands are gathered with `GPURasterTextureToBuffer`. Buffer bands pass through unless
 * `canonicalize` is set, in which case an identity `GPURasterNeighborhood` resolves scale, offset,
 * nodata, validity, and non-finite samples into canonical float32 values plus uint32 validity.
 *
 * `canonicalize: 'reuse-canonical'` behaves like `true`, except that a float32 buffer band that
 * already has validity and no scale, offset, or noData passes through without a copy. Its values
 * may still hold non-finite samples, so the consumer must treat them as invalid itself.
 *
 * @internal
 */
export function getRasterBufferBandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  band: GPURasterBand,
  grid: GPURasterMetadata,
  canonicalize: boolean | 'reuse-canonical'
): RasterBufferBandSource<Parameters> {
  const {width, height} = grid;
  if (band.storage.kind === 'buffer' && !canonicalize) {
    return {band: band as GPURasterBufferBand, nodes: []};
  }
  if (
    band.storage.kind === 'buffer' &&
    canonicalize === 'reuse-canonical' &&
    band.format === 'float32' &&
    band.validity &&
    band.scale === undefined &&
    band.offset === undefined &&
    band.noDataValue === undefined
  ) {
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
