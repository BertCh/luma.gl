// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import assert from 'node:assert/strict';
import {test} from 'node:test';
import {findNonFiniteBitcasts} from './check-wgsl-identifiers.mjs';

test('rejects constant infinity/NaN bitcasts in scalar, vector and interpolated snippets', () => {
  for (const source of [
    'output[index] = bitcast<f32>(0x7f800000u);',
    'bitcast<f32>(0xff800000u)',
    'bitcast<f32>(0x7fc00000u)',
    'bitcast<f32>(0x7f800001u | 1u)',
    'bitcast<f32>(2139095040u)',
    'bitcast<vec2<f32>>(vec2<u32>(0u, 0xffc00000u))',
    'bitcast<vec4f>(vec4u(0x7f800000u))',
    'const INFINITY_BITS: u32 = 0x7f800000u; const ALIAS: u32 = INFINITY_BITS; bitcast<f32>(ALIAS)',
    'bitcast<f32>(${positive ? "0x7f800000u" : "0xff800000u"})',
    'const infinityBits = "0x7f800000u"; bitcast<f32>(${infinityBits})'
  ]) {
    assert.equal(findNonFiniteBitcasts(source).length, 1, source);
  }
});

test('allows runtime bitcasts, finite constants, integer stores and exponent comparisons', () => {
  for (const source of [
    'var bits = 0x7f800000u; bitcast<f32>(bits)',
    'let bits = 0x7fc00000u; bitcast<f32>(bits)',
    'bitcast<f32>(0x7fc00000u | (index & 0u))',
    'bitcast<vec2<f32>>(vec2<u32>(0u, input[index]))',
    'bitcast<f32>(0x7f7fffffu)',
    'bitcast<f32>(0x00800000u)',
    'bitcast<f32>(${word})',
    'bitcast<u32>(value) >= 0x7f800000u',
    'outputScores[index] = ${positive ? "0x7f800000u" : "0xff800000u"};',
    '// bitcast<f32>(0x7fc00000u)\n/* bitcast<f32>(0x7f800000u) */'
  ]) {
    assert.deepEqual(findNonFiniteBitcasts(source), [], source);
  }
});
