// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {PolygonEdgeSlabs} from '../../utils/polygon-edge-slabs';
import {assertCompatibleGPUSpatialContexts, type GPUSpatialContext} from './spatial-context';

/** Per-frame rectangle from each operator's parameters (current behavior). */
export type GPURectangleStudyWindow = 'rectangle';

/** Summary row length written by the GPU preparation (float32). */
export const GPU_STUDY_WINDOW_SUMMARY_LENGTH = 8;

/** Summary row layout written by the GPU preparation (float32). */
export const GPU_STUDY_WINDOW_SUMMARY = Object.freeze({
  minX: 0,
  minY: 1,
  maxX: 2,
  maxY: 3,
  area: 4,
  perimeter: 5,
  edgeCount: 6,
  flags: 7
} as const);

/** Summary flags bit for uncertain edge orientation. */
export const GPU_STUDY_WINDOW_FLAG_ORIENTATION_UNCERTAIN = 1;

/** Summary flags bit for zero/negative area: the window is unusable. */
export const GPU_STUDY_WINDOW_FLAG_EMPTY = 2;

/** Edge-slab index views of a study window (owner = the single window feature). @internal */
export type GPUStudyWindowSlabViews = PolygonEdgeSlabs;

/** Views a consumer binds; valid only in the graph the handle was added to. */
export type GPUPolygonStudyWindowViews = {
  /** 4 floats per vertex: oriented edge [ax, ay, bx, by], interior on the LEFT; NaN rows unused. */
  edges: GraphDataView<'float32'>;
  /** GPU_STUDY_WINDOW_SUMMARY_LENGTH floats, laid out by GPU_STUDY_WINDOW_SUMMARY. */
  summary: GraphDataView<'float32'>;
  /** Edge-slab index (owner = the single window feature). @internal */
  slabs: GPUStudyWindowSlabViews;
  /**
   * Compacted sampler cells: row-major ids of inside|boundary cells; bit 31 set marks a boundary
   * cell (needs a point-in-window test).
   */
  sampleCells: GraphDataView<'uint32'>;
  /** Number of sampler cells (1 word). */
  sampleCellCount: GraphDataView<'uint32'>;
  /** Sampler grid dimensions. */
  sampleGrid: readonly [number, number];
  /**
   * One-binding copy of everything a consumer kernel reads: summary, oriented edges, slab index,
   * sampler cell count and cells, laid out back to back as 32-bit words (floats as bits). Layout is
   * computed from the other views' lengths by the WGSL helpers. @internal
   */
  packed: GraphDataView<'uint32'>;
};

/** A polygon (multi)polygon-with-holes study window in the operator's planar frame. */
export interface GPUPolygonStudyWindow {
  readonly kind: 'polygon';
  readonly id: string;
  /** Planar, resolved. */
  readonly spatialContext: GPUSpatialContext;
  /** Views for `graph`; throws `${id} study window needs graph.add before its consumers`. */
  getViews(graph: GPUCommandGraph<unknown>): GPUPolygonStudyWindowViews;
}

/** Rectangle or prepared polygon study window used by an operator. */
export type GPUStudyWindow = GPURectangleStudyWindow | GPUPolygonStudyWindow;

/** Validates a consumer's window against its resolved context. Returns 'rectangle' when omitted. */
export function resolveGPUStudyWindow(
  id: string,
  window: GPUStudyWindow | undefined,
  spatialContext: GPUSpatialContext
): GPUStudyWindow {
  if (window === undefined || window === 'rectangle') {
    return 'rectangle';
  }
  if (
    window !== null &&
    typeof window === 'object' &&
    window.kind === 'polygon' &&
    typeof window.id === 'string' &&
    window.spatialContext !== null &&
    typeof window.spatialContext === 'object' &&
    typeof window.getViews === 'function'
  ) {
    assertCompatibleGPUSpatialContexts(`${id} studyWindow`, window.spatialContext, spatialContext);
    return window;
  }
  throw new Error(`${id} studyWindow must be 'rectangle' or a GPUStudyWindowPreparation`);
}
