// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphBufferHandle, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainPeakSnap,
  type GPUTerrainPeakSnapProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-peak-snap';

function createPeakSnapProps(): GPUTerrainPeakSnapProps {
  const owner = {id: 'peak-snap-datum'};
  const makeView = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ): GraphDataView<Format> => {
    const byteStride = format === 'float32x2' ? 8 : 4;
    const buffer = new GraphBufferHandle(
      owner,
      {id, byteLength: length * byteStride, usage: 0},
      false
    );
    return new GraphDataView(buffer, {
      format,
      length,
      byteOffset: 0,
      byteStride,
      rowByteLength: byteStride
    });
  };
  const grid = makeGPURasterMetadata({width: 4, height: 4});
  return {
    id: 'datum-peak-snap',
    grid,
    elevation: makeGPURasterBand(makeView('elevation', 'float32', 16), grid, {
      verticalDatum: 'EPSG:5773'
    }),
    candidates: makeView('candidates', 'float32x2', 2),
    candidateHeights: makeView('candidate-heights', 'float32', 2),
    parameters: makeView('parameters', 'float32', 8),
    output: {
      positions: makeView('positions', 'float32x2', 2),
      heights: makeView('heights', 'float32', 2),
      snapStatus: makeView('snap-status', 'uint32', 2)
    }
  };
}

it('GPUTerrainPeakSnap requires candidate heights when their datum is declared', () => {
  const props = createPeakSnapProps();
  expect(
    () =>
      new GPUTerrainPeakSnap({
        ...props,
        candidateHeights: undefined,
        candidateVerticalDatum: 'EPSG:5773'
      })
  ).toThrow(new Error('datum-peak-snap candidateVerticalDatum needs candidateHeights'));
});

it('GPUTerrainPeakSnap rejects mismatched declared candidate and elevation datums', () => {
  expect(
    () => new GPUTerrainPeakSnap({...createPeakSnapProps(), candidateVerticalDatum: 'ellipsoidal'})
  ).toThrow(
    new Error(
      'datum-peak-snap candidate heights are ellipsoidal but elevation is EPSG:5773; convert to one vertical datum first'
    )
  );
});

it('GPUTerrainPeakSnap accepts matching or undeclared vertical datums', () => {
  const props = createPeakSnapProps();
  for (const [candidateVerticalDatum, verticalDatum] of [
    ['EPSG:5773', 'EPSG:5773'],
    ['epsg:5773', 'EPSG:5773'],
    [' EPSG:5773 ', 'EPSG:5773'],
    [undefined, 'EPSG:5773'],
    ['ellipsoidal', undefined],
    [undefined, undefined]
  ]) {
    expect(
      () =>
        new GPUTerrainPeakSnap({
          ...props,
          elevation: {...props.elevation, verticalDatum},
          candidateVerticalDatum
        })
    ).not.toThrow();
  }
  expect(() => new GPUTerrainPeakSnap({...props, candidateHeights: undefined})).not.toThrow();
});
