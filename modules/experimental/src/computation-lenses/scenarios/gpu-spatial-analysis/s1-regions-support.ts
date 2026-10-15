// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUComputationRecordWithSidecars} from '@luma.gl/gpgpu';
import {loadChicagoTracts, type ChicagoTracts} from './spatial-lens-data';

/**
 * Chicago community areas 1 to 8 (Rogers Park to Near North Side): 136 contiguous tracts, each area
 * itself connected, with no missing attribute. The regionalization cards use them so the unrolled
 * algorithms stay docs-sized; the captions say so.
 */
export const REGION_COMMUNITY_AREAS: readonly number[] = [1, 2, 3, 4, 5, 6, 7, 8];

/** Keeps the tracts of the given community areas, in source order; all columns are kept aligned. */
export function selectCommunityAreaTracts(
  tracts: ChicagoTracts,
  areas: readonly number[] = REGION_COMMUNITY_AREAS
): ChicagoTracts {
  const selected: number[] = [];
  for (let tract = 0; tract < tracts.count; tract++) {
    if (areas.includes(tracts.columns['communityArea'][tract])) selected.push(tract);
  }
  const ringOffsets = new Uint32Array(selected.length + 1);
  const vertexChunks: Float32Array[] = [];
  for (const [index, tract] of selected.entries()) {
    // The compact extract has exactly one ring per feature.
    assert(tracts.polygonRingOffsets[tract + 1] - tracts.polygonRingOffsets[tract] === 1);
    const chunk = tracts.vertices.subarray(
      tracts.ringOffsets[tract] * 2,
      tracts.ringOffsets[tract + 1] * 2
    );
    vertexChunks.push(chunk);
    ringOffsets[index + 1] = ringOffsets[index] + chunk.length / 2;
  }
  const vertices = new Float32Array(ringOffsets[selected.length] * 2);
  let offset = 0;
  for (const chunk of vertexChunks) {
    vertices.set(chunk, offset);
    offset += chunk.length;
  }
  const sequence = Uint32Array.from({length: selected.length + 1}, (_, index) => index);
  return {
    count: selected.length,
    vertices,
    ringOffsets,
    polygonRingOffsets: sequence,
    partFeature: sequence.slice(0, selected.length),
    columns: Object.fromEntries(
      Object.entries(tracts.columns).map(([name, column]) => [
        name,
        Float32Array.from(selected, tract => column[tract])
      ])
    ),
    dataset: tracts.dataset
  };
}

/** Loads the docs-sized regionalization study area: community areas 1 to 8 of the Chicago tracts. */
export async function loadRegionTracts(): Promise<ChicagoTracts> {
  return selectCommunityAreaTracts(await loadChicagoTracts());
}

/** Community area of each tract as a dense label `0..areas.length-1` (the areas' order). */
export function getCommunityAreaLabels(
  tracts: ChicagoTracts,
  areas: readonly number[] = REGION_COMMUNITY_AREAS
): Uint32Array {
  return Uint32Array.from(tracts.columns['communityArea'], area => {
    const label = areas.indexOf(area);
    assert(label >= 0);
    return label;
  });
}

/**
 * Demand-by-facility straight-line cost matrix in kilometres between tract vertex-mean centres, in a
 * local equirectangular frame at the study area's mean latitude. It is prepared on the CPU as the
 * caller-supplied input of location allocation; no result figure draws it.
 */
export function getCentreCostMatrix(tracts: ChicagoTracts): Float32Array {
  const centres: [number, number][] = [];
  for (let tract = 0; tract < tracts.count; tract++) {
    let longitude = 0;
    let latitude = 0;
    const start = tracts.ringOffsets[tract];
    const end = tracts.ringOffsets[tract + 1];
    for (let vertex = start; vertex < end; vertex++) {
      longitude += tracts.vertices[vertex * 2];
      latitude += tracts.vertices[vertex * 2 + 1];
    }
    centres.push([longitude / (end - start), latitude / (end - start)]);
  }
  const meanLatitude = centres.reduce((sum, centre) => sum + centre[1], 0) / centres.length;
  const kilometresPerDegree = 111.32;
  const longitudeScale = Math.cos((meanLatitude * Math.PI) / 180) * kilometresPerDegree;
  const costs = new Float32Array(tracts.count * tracts.count);
  for (let demand = 0; demand < tracts.count; demand++) {
    for (let facility = 0; facility < tracts.count; facility++) {
      costs[demand * tracts.count + facility] = Math.hypot(
        (centres[demand][0] - centres[facility][0]) * longitudeScale,
        (centres[demand][1] - centres[facility][1]) * kilometresPerDegree
      );
    }
  }
  return costs;
}

/**
 * Fails when a regionalization capture says the run rejected its input, so a plausible-looking
 * figure is never drawn from an invalid partition. `invalidWord` is the status word that counts it.
 */
export function assertRegionalizationValid(
  record: GPUComputationRecordWithSidecars,
  statusName: string,
  invalidWord: number
): GPUComputationRecordWithSidecars {
  const status = record.captures?.snapshots.find(capture => capture.name === statusName);
  assert(status && status.values[invalidWord] === 0);
  return record;
}
