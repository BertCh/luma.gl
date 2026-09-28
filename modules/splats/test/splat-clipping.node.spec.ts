// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getSplatClipCoverage,
  isSplatClipRegionActive,
  MAXIMUM_SPLAT_CLIP_PLANES,
  packSplatClipUniforms,
  SPLAT_CLIP_UNIFORM_BYTE_LENGTH,
  type SplatClipRegion
} from '../src/splat-clipping';

/** One-sigma world axes of an isotropic Gaussian of the given radius. */
function getIsotropicAxes(radius: number): readonly (readonly [number, number, number])[] {
  return [
    [radius, 0, 0],
    [0, radius, 0],
    [0, 0, radius]
  ];
}

const HALF_SPACE_AT_ORIGIN: SplatClipRegion = {
  planes: [{normal: [1, 0, 0], distance: 0}]
};

it('a clip boundary is soft in units of the Gaussian it cuts', () => {
  const axes = getIsotropicAxes(1);

  expect(
    getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, [0, 0, 0], axes),
    'a Gaussian centered on the plane keeps half its opacity'
  ).toBeCloseTo(0.5, 6);
  expect(
    getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, [3, 0, 0], axes) > 0.99,
    'three sigma inside the region it is untouched'
  ).toBe(true);
  expect(
    getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, [-3, 0, 0], axes) < 0.01,
    'and three sigma outside it is gone'
  ).toBe(true);
});

it('the transition width follows each Gaussian, not a global distance', () => {
  // The same signed distance means something different to a 10 cm splat and a 10 m one, which is
  // exactly why a center-based clip looks ragged on a volumetric primitive.
  const offset: readonly [number, number, number] = [1, 0, 0];
  const small = getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, offset, getIsotropicAxes(0.1));
  const large = getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, offset, getIsotropicAxes(10));

  expect(small > 0.99, 'a splat much smaller than its offset is fully inside').toBe(true);
  expect(large > 0.5 && large < 0.6, 'a splat much larger than its offset is barely cut').toBe(
    true
  );
});

it('an anisotropic Gaussian is measured along the plane normal only', () => {
  // A disc lying in the clip plane is wide in Y and Z and thin in X, so an X-facing plane should
  // see its thin extent and cut it sharply.
  const disc: readonly (readonly [number, number, number])[] = [
    [0.05, 0, 0],
    [0, 5, 0],
    [0, 0, 5]
  ];

  expect(
    getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, [0.2, 0, 0], disc) > 0.99,
    'a thin disc a fifth of a unit inside is fully retained'
  ).toBe(true);
  expect(
    getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, [0.2, 0, 0], getIsotropicAxes(5)) < 0.6,
    'while a sphere of the same width across is barely retained at all'
  ).toBe(true);
});

it('softness scales the transition and approaches a hard cut at zero', () => {
  const axes = getIsotropicAxes(1);
  const hard: SplatClipRegion = {...HALF_SPACE_AT_ORIGIN, softness: 0.01};

  expect(getSplatClipCoverage(hard, [0.5, 0, 0], axes) > 0.99, 'just inside is kept').toBe(true);
  expect(getSplatClipCoverage(hard, [-0.5, 0, 0], axes) < 0.01, 'just outside is dropped').toBe(
    true
  );
  expect(
    getSplatClipCoverage({...HALF_SPACE_AT_ORIGIN, softness: 4}, [1, 0, 0], axes) < 0.7,
    'a wide band fades over several sigma'
  ).toBe(true);
});

it('planes combine as an intersection by default and as a union on request', () => {
  const slab: SplatClipRegion = {
    planes: [
      {normal: [1, 0, 0], distance: 1},
      {normal: [-1, 0, 0], distance: 1}
    ],
    softness: 0.01
  };
  const axes = getIsotropicAxes(0.01);

  expect(getSplatClipCoverage(slab, [0, 0, 0], axes) > 0.99, 'inside the slab is kept').toBe(true);
  expect(getSplatClipCoverage(slab, [5, 0, 0], axes) < 0.01, 'beyond either face is dropped').toBe(
    true
  );
  expect(
    getSplatClipCoverage({...slab, combine: 'union'}, [5, 0, 0], axes) > 0.99,
    'a union of the same half-spaces keeps everything'
  ).toBe(true);
  expect(
    getSplatClipCoverage({...slab, invert: true}, [0, 0, 0], axes) < 0.01,
    'and inverting the slab keeps only what it excluded'
  ).toBe(true);
});

it('clipping is skipped entirely when no plane can attenuate anything', () => {
  expect(isSplatClipRegionActive(undefined), 'no region').toBe(false);
  expect(isSplatClipRegionActive({planes: []}), 'no planes').toBe(false);
  expect(
    isSplatClipRegionActive({planes: [{normal: [0, 0, 0], distance: 1}]}),
    'a degenerate normal'
  ).toBe(false);
  expect(isSplatClipRegionActive(HALF_SPACE_AT_ORIGIN), 'a real plane').toBe(true);
  expect(
    getSplatClipCoverage(undefined, [0, 0, 0], getIsotropicAxes(1)),
    'an inactive region leaves opacity alone'
  ).toBe(1);
});

it('the packed uniform block normalizes plane normals and matches its declared size', () => {
  const packed = packSplatClipUniforms({
    planes: [{normal: [0, 0, 4], distance: 8}],
    combine: 'union',
    softness: 2,
    invert: true
  });
  const floatValues = new Float32Array(packed);
  const integerValues = new Uint32Array(packed);

  expect(packed.byteLength, 'the block is the size the shader layout declares').toBe(
    SPLAT_CLIP_UNIFORM_BYTE_LENGTH
  );
  expect(
    [floatValues[0], floatValues[1], floatValues[2], floatValues[3]],
    'the normal is unit length and the plane constant is scaled with it'
  ).toEqual([0, 0, 1, 2]);
  expect(integerValues[MAXIMUM_SPLAT_CLIP_PLANES * 4], 'one plane was accepted').toBe(1);
  expect(integerValues[MAXIMUM_SPLAT_CLIP_PLANES * 4 + 1], 'combining as a union').toBe(1);
  expect(floatValues[MAXIMUM_SPLAT_CLIP_PLANES * 4 + 2], 'with the requested softness').toBe(2);
  expect(integerValues[MAXIMUM_SPLAT_CLIP_PLANES * 4 + 3], 'and inverted').toBe(1);
});

it('the packed block skips degenerate planes and rejects overfull regions', () => {
  const packed = packSplatClipUniforms({
    planes: [
      {normal: [0, 0, 0], distance: 1},
      {normal: [1, 0, 0], distance: 0}
    ]
  });

  expect(
    new Uint32Array(packed)[MAXIMUM_SPLAT_CLIP_PLANES * 4],
    'only the usable plane is counted'
  ).toBe(1);
  expect(
    () =>
      packSplatClipUniforms({
        planes: Array.from({length: MAXIMUM_SPLAT_CLIP_PLANES + 1}, () => ({
          normal: [1, 0, 0] as const,
          distance: 0
        }))
      }),
    'more planes than the block can hold is an error, not a silent truncation'
  ).toThrow(/at most 8 planes/);
});

it('degenerate planes do not count toward the plane limit', () => {
  const planes = [
    ...Array.from({length: MAXIMUM_SPLAT_CLIP_PLANES}, (_, index) => ({
      normal: [1, 0, 0] as const,
      distance: index
    })),
    {normal: [0, 0, 0] as const, distance: 1},
    {normal: [0, 1, 0] as const, distance: Number.NaN}
  ];

  const packed = packSplatClipUniforms({planes});
  expect(
    new Uint32Array(packed)[MAXIMUM_SPLAT_CLIP_PLANES * 4],
    'eight usable planes plus two degenerate ones pack as eight'
  ).toBe(MAXIMUM_SPLAT_CLIP_PLANES);
});

it('CPU coverage skips the same non-finite planes the packer drops', () => {
  const axes = getIsotropicAxes(1);
  const position: [number, number, number] = [3, 0, 0];
  const reference = getSplatClipCoverage(HALF_SPACE_AT_ORIGIN, position, axes);

  for (const invalidPlane of [
    {normal: [0, 1, 0] as const, distance: Number.NaN},
    {normal: [0, 1, 0] as const, distance: Number.POSITIVE_INFINITY},
    {normal: [Number.POSITIVE_INFINITY, 0, 0] as const, distance: 0}
  ]) {
    const coverage = getSplatClipCoverage(
      {planes: [...HALF_SPACE_AT_ORIGIN.planes, invalidPlane]},
      position,
      axes
    );
    expect(coverage, `ignores ${JSON.stringify(invalidPlane)} rather than returning NaN`).toBe(
      reference
    );
  }
  expect(
    isSplatClipRegionActive({planes: [{normal: [0, 1, 0], distance: Number.NaN}]}),
    'a region of only non-finite planes is inactive'
  ).toBe(false);
});
