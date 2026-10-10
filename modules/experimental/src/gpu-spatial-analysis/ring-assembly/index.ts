// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  GPUSegmentRingAssembly,
  GPU_SEGMENT_RING_ASSEMBLY_NONE,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_TOUCHING,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_DANGLING,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT,
  GPU_SEGMENT_RING_ASSEMBLY_FLAG_CANCELLED
} from './gpu-segment-ring-assembly';
export type {
  GPUSegmentRingAssemblyInteriorSide,
  GPUSegmentRingAssemblyOutput,
  GPUSegmentRingPolygonOutput,
  GPUSegmentRingAssemblyProps
} from './gpu-segment-ring-assembly';
export {getSegmentPolygonizationDiagnostics} from './segment-polygonization-diagnostics';
export type {
  PolygonizationSegment,
  SegmentPolygonizationClassification,
  SegmentPolygonizationDiagnostics,
  SegmentPolygonizationDiagnosticsInput,
  SegmentPolygonizationRing
} from './segment-polygonization-diagnostics';
