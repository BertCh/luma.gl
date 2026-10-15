// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {
  GPUComputationAttributeValue,
  GPUComputationCapture,
  GPUComputationCaptureGeometry
} from '@luma.gl/gpgpu';
import {createSpatialChoice} from '../../../gpu-spatial-analysis/computation-pack/choices';
import type {LensScenarioResult} from '../../types';
import {runS2Lens, type S2CaptureSpec, type S2LensRunProps, type S2Uploads} from './s2-lens-run';
import type {S3ProjectedFrame} from './s3-lens-data';

/**
 * One captured view of an S3 lens run: the shared S2 capture contract plus the `context` role
 * (drawn faint under every result panel, never counted as an input) and the output `highlight`
 * (the single answer the card figure points at; chosen from data by the scenario, never by a view).
 */
export type S3CaptureSpec = Omit<S2CaptureSpec, 'role'> & {
  role?: NonNullable<GPUComputationCapture['role']>;
  highlight?: {key: number; label: string};
};
/** Upload helpers handed to an S3 scenario's `build` (the shared S2 contract). */
export type S3Uploads = S2Uploads;

/** Run metadata that guards read, as plain values. */
export type S3RunMetadata = Readonly<Record<string, GPUComputationAttributeValue>>;

/** Inputs of one recorded S3 lens run: the shared S2 run plus capture roles and guard metadata. */
export type S3LensRunProps = Omit<S2LensRunProps, 'build'> & {
  /** Adds uploads and the element (through the scenario's `runExample`); returns views to capture. */
  build(
    graph: Parameters<S2LensRunProps['build']>[0],
    uploads: S3Uploads
  ): Promise<Record<string, S3CaptureSpec>> | Record<string, S3CaptureSpec>;
  /**
   * Factual run metadata for the element's guards (coordinate model, CRS, conservation sums). A
   * function receives the captured snapshots, so sums are taken from GPU readbacks, never estimated.
   * Recorded as `presentation`-stage choices, the convention S1's spatial guards read.
   */
  metadata?: S3RunMetadata | ((snapshots: readonly GPUComputationCapture[]) => S3RunMetadata);
};

/**
 * Runs one S3 lens scenario through the shared spatial lens run (`runS2Lens`: recording graph,
 * `describe()`, one end-of-run capture), then applies the `context` roles and highlights and
 * appends the run metadata to the record's choices. The run id hashes structure, adapter, dataset
 * and seed, none of which these annotations change.
 */
export async function runS3Lens(
  device: Device,
  props: S3LensRunProps
): Promise<LensScenarioResult> {
  let specs: Record<string, S3CaptureSpec> = {};
  const result = await runS2Lens(device, {
    ...props,
    async build(graph, uploads) {
      specs = await props.build(graph, uploads);
      return Object.fromEntries(
        Object.entries(specs).map(([name, {role, highlight, ...spec}]) => [
          name,
          {...spec, ...(role && role !== 'context' && {role})}
        ])
      );
    }
  });
  const snapshots = (result.captures?.snapshots ?? []).map(capture => {
    const spec = specs[capture.name];
    return {
      ...capture,
      ...(spec?.role === 'context' && {role: 'context' as const}),
      ...(spec?.highlight && {highlight: spec.highlight})
    };
  });
  const metadata =
    typeof props.metadata === 'function' ? props.metadata(snapshots) : (props.metadata ?? {});
  const metadataChoices = Object.entries(metadata).map(([name, value]) =>
    createSpatialChoice({name, stage: 'presentation', sweepable: false}, value)
  );
  return {
    ...result,
    choices: [...result.choices, ...metadataChoices],
    ...(result.captures && {captures: {...result.captures, snapshots}})
  };
}

/** Geometry tag for coordinates in a dataset's projected frame (metres). */
export function projectedGeometry(
  kind: 'points' | 'lines' | 'polygons',
  options: {offsetsCapture?: string; valuesCapture?: string} = {}
): GPUComputationCaptureGeometry {
  return {kind, coordinateModel: 'projected', unit: 'm', ...options};
}

/** Geometry tag for lon/lat coordinates in degrees. */
export function geographicGeometry(
  kind: 'points' | 'lines' | 'polygons',
  options: {offsetsCapture?: string; valuesCapture?: string} = {}
): GPUComputationCaptureGeometry {
  return {kind, coordinateModel: 'geographic', ...options};
}

/** Coordinate-model guard metadata for a run in a dataset's projected frame. */
export function projectedMetadata(frame: S3ProjectedFrame): S3RunMetadata {
  return {
    coordinateModel: 'projected',
    crs: frame.crs,
    falseOrigin: frame.falseOrigin.join(', ')
  };
}

/** Coordinate-model guard metadata for a lon/lat run. */
export function geographicMetadata(metric?: 'geodesic'): S3RunMetadata {
  return {coordinateModel: 'geographic', crs: 'EPSG:4326', ...(metric && {metric})};
}

/**
 * Display offsets for fixed-size output shapes (`size` vertices each, e.g. 4 rectangle corners):
 * `[0, size, 2·size, …]`. Uploaded as a captured view, so the result figure draws the GPU output
 * vertices exactly as written; the offsets are an index, not data.
 */
export function createFixedOffsets(count: number, size: number): Uint32Array {
  return Uint32Array.from({length: count + 1}, (_, index) => index * size);
}

/** Sums one captured scalar column (finite values only), for conservation metadata. */
export function sumCapture(
  snapshots: readonly GPUComputationCapture[],
  name: string
): number | undefined {
  const capture = snapshots.find(candidate => candidate.name === name);
  if (!capture) return undefined;
  let sum = 0;
  for (let index = 0; index < capture.values.length; index++) {
    const value = Number(capture.values[index]);
    if (Number.isFinite(value)) sum += value;
  }
  return sum;
}
