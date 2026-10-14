// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../contracts/earth';
import type {GPUSpatialContext} from '../contracts/spatial-context';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';

/** Number of float32 rows of a raster extent: `[originLongitude, originLatitude, cellWidth, cellHeight]`. */
const RASTER_EXTENT_LENGTH = 4;
const DEGREES_TO_RADIANS = Math.PI / 180;

/**
 * Area in squared meters of one cell of a longitude/latitude raster on a sphere:
 * `R^2 * dLon * (sin(latTop) - sin(latBottom))`, evaluated as
 * `2 R^2 dLon cos(latMiddle) sin(dLat / 2)` so thin cells do not cancel.
 *
 * @param originLatitude Latitude in degrees of the lower edge of row 0 (the raster extent origin).
 * @param cellWidth Cell width in degrees of longitude.
 * @param cellHeight Cell height in degrees of latitude.
 * @param row Raster row; row 0 has the smallest latitude.
 * @param radius Sphere radius in meters. Defaults to the WGS84 authalic radius.
 */
export function getSphericalRasterCellArea(
  originLatitude: number,
  cellWidth: number,
  cellHeight: number,
  row: number,
  radius: number = GPU_EARTH_AUTHALIC_RADIUS_METERS
): number {
  const middle = (originLatitude + (row + 0.5) * cellHeight) * DEGREES_TO_RADIANS;
  return (
    2 *
    radius *
    radius *
    (cellWidth * DEGREES_TO_RADIANS) *
    Math.cos(middle) *
    Math.sin(0.5 * cellHeight * DEGREES_TO_RADIANS)
  );
}

/** Sphere radius in meters used for the cell areas of `context`. @internal */
export function getRasterSphereRadius(context: GPUSpatialContext): number {
  return context.sphereRadius ?? GPU_EARTH_AUTHALIC_RADIUS_METERS;
}

/** Validates a longitude/latitude raster extent view and the raster width. @internal */
export function validateGeographicRaster(
  id: string,
  extent: GraphDataView<'float32'> | undefined,
  width: number | undefined,
  cellCount: number
): void {
  if (!extent || width === undefined) {
    throw new Error(`${id} longitude-latitude rasters need extent and width`);
  }
  validatePackedView(extent, ['float32'], `${id} extent`);
  if (extent.length < RASTER_EXTENT_LENGTH) {
    throw new Error(`${id} extent must contain ${RASTER_EXTENT_LENGTH} float32 rows`);
  }
  if (!Number.isSafeInteger(width) || width < 1 || cellCount % width !== 0) {
    throw new Error(`${id} width must be a positive integer dividing the cell count`);
  }
}

/**
 * Declares `getCellArea(row)` in squared meters. Needs an `extent` f32 binding holding
 * `[originLongitude, originLatitude, cellWidth, cellHeight]` in degrees. A cell size that is not
 * positive gives a non-positive or NaN area, which callers treat as no area.
 *
 * @internal
 */
export function getSphericalCellAreaWGSL(radius: number): string {
  return /* wgsl */ `
const SPHERE_RADIUS: f32 = ${getWGSLFloatLiteral(radius)};
const DEGREES_TO_RADIANS: f32 = 0.017453292519943295;
fn getCellArea(row: u32) -> f32 {
  let cellWidth = extent[extentOffset + 2u];
  let cellHeight = extent[extentOffset + 3u];
  let middle = (extent[extentOffset + 1u] + (f32(row) + 0.5) * cellHeight) * DEGREES_TO_RADIANS;
  // R^2 dLon (sin(top) - sin(bottom)) = 2 R^2 dLon cos(middle) sin(dLat / 2).
  return 2.0 * SPHERE_RADIUS * SPHERE_RADIUS * (cellWidth * DEGREES_TO_RADIANS) * cos(middle) *
    sin(0.5 * cellHeight * DEGREES_TO_RADIANS);
}`;
}

/**
 * Writes one effective area weight per cell: `cellWeights[cell]` (or 1) times the spherical cell
 * area. Non-finite or non-positive results are written as 0 (no area).
 *
 * @internal
 */
export function createSphericalCellWeightsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    cellCount: number;
    width: number;
    radius: number;
    extent: GraphDataView<'float32'>;
    cellWeights?: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: props.operation,
    variant: `spherical-cell-weights${props.cellWeights ? '-masked' : ''}`,
    bindings: [
      {name: 'extent', view: props.extent, type: 'f32', access: 'read'},
      ...(props.cellWeights
        ? [
            {
              name: 'cellWeights',
              view: props.cellWeights,
              type: 'f32' as const,
              access: 'read' as const
            }
          ]
        : []),
      {name: 'output', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: `const WIDTH: u32 = ${props.width}u;
${getSphericalCellAreaWGSL(props.radius)}`,
    body: `var weight = ${props.cellWeights ? 'cellWeights[cellWeightsOffset + index]' : '1.0'};
  weight = weight * getCellArea(index / WIDTH);
  // Non-finite (NaN or infinity) and non-positive weights carry no area.
  if ((bitcast<u32>(weight) & 0x7f800000u) == 0x7f800000u || !(weight > 0.0)) {
    weight = 0.0;
  }
  output[outputOffset + index] = weight;`
  });
}
