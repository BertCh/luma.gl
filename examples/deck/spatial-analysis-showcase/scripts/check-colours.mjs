// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Colour-system check for the showcase: reads the ramps, the exact class tables and the hue
 * registry and reports
 *
 * 1. luminance monotonicity (CIE L*) of every sequential ramp and sequential class table,
 * 2. the minimum Delta E (CIEDE2000) between ADJACENT classes (all pairs for qualitative sets)
 *    of every class table (3-9, 3-11 and published qualitative sizes) and every registry entry
 *    on each ground, under normal vision and simulated protanopia, deuteranopia and tritanopia
 *    (Machado, Oliveira and Fernandes 2009, severity 1.0, applied in linear sRGB),
 * 3. the contrast of the lowest class of dark-ground sequential tables against the dark ground
 *    `#13171C` (WCAG 3:1).
 *
 * Registry entries whose minimum Delta E is below {@link WARN_DELTA_E} (6) are WARN, not failures.
 * ColorBrewer tables are designed for 5-7 classes and colour-blind use only partly, so many
 * 8-11 class tables and the pale ends of sequential tables warn: read the table, not just the
 * count. The exit code is 1 only when a sequential ramp that the registry references is not
 * monotone in L* (isoluminant, diverging, cyclic and the hypsometric convention are exempt).
 *
 * Usage (from the app directory): `node scripts/check-colours.mjs [--verbose]`
 *   --verbose  also prints every class table row (default: worst case per scheme)
 */
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';

const WARN_DELTA_E = 6;
const MONOTONE_TOLERANCE = 0.6;
const DARK_GROUND = '#13171c';
const VERBOSE = process.argv.includes('--verbose');
const SOURCE = join(dirname(fileURLToPath(import.meta.url)), '../src');

// ---------------------------------------------------------------------------------------------
// Colour science
// ---------------------------------------------------------------------------------------------

const toLinear = value => {
  const x = value / 255;
  return x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
};
const fromLinear = value => {
  const x = Math.min(Math.max(value, 0), 1);
  return (x <= 0.0031308 ? x * 12.92 : 1.055 * x ** (1 / 2.4) - 0.055) * 255;
};

function parseHex(hex) {
  const digits = hex.replace('#', '');
  return [0, 2, 4].map(start => Number.parseInt(digits.slice(start, start + 2), 16));
}

function linearToLab([r, g, b]) {
  const x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
  const y = 0.2126729 * r + 0.7151522 * g + 0.072175 * b;
  const z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;
  const f = t => (t > 216 / 24389 ? Math.cbrt(t) : ((24389 / 27) * t + 16) / 116);
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}

const rgbToLab = rgb => linearToLab(rgb.map(toLinear));

// Machado, Oliveira, Fernandes 2009, severity 1.0, linear RGB.
const CVD_MATRICES = {
  protanopia: [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998]
  ],
  deuteranopia: [
    [0.367322, 0.860646, -0.227968],
    [0.280085, 0.672501, 0.047413],
    [-0.01182, 0.04294, 0.968881]
  ],
  tritanopia: [
    [1.255528, -0.076749, -0.178779],
    [-0.078411, 0.930809, 0.147602],
    [0.004733, 0.691367, 0.3039]
  ]
};

function simulate(rgb, matrix) {
  const linear = rgb.map(toLinear);
  return matrix.map(row => {
    const value = row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
    return fromLinear(value);
  });
}

/** CIEDE2000 (Sharma, Wu, Dalal 2005). */
function deltaE2000([l1, a1, b1], [l2, a2, b2]) {
  const rad = d => (d * Math.PI) / 180;
  const deg = r => (r * 180) / Math.PI;
  const c1 = Math.hypot(a1, b1);
  const c2 = Math.hypot(a2, b2);
  const cMean = (c1 + c2) / 2;
  const g = 0.5 * (1 - Math.sqrt(cMean ** 7 / (cMean ** 7 + 25 ** 7)));
  const a1p = (1 + g) * a1;
  const a2p = (1 + g) * a2;
  const c1p = Math.hypot(a1p, b1);
  const c2p = Math.hypot(a2p, b2);
  const h1p = c1p === 0 ? 0 : (deg(Math.atan2(b1, a1p)) + 360) % 360;
  const h2p = c2p === 0 ? 0 : (deg(Math.atan2(b2, a2p)) + 360) % 360;
  const dL = l2 - l1;
  const dC = c2p - c1p;
  let dh = 0;
  if (c1p * c2p !== 0) {
    dh = h2p - h1p;
    if (dh > 180) dh -= 360;
    else if (dh < -180) dh += 360;
  }
  const dH = 2 * Math.sqrt(c1p * c2p) * Math.sin(rad(dh / 2));
  const lMean = (l1 + l2) / 2;
  const cpMean = (c1p + c2p) / 2;
  let hMean = h1p + h2p;
  if (c1p * c2p !== 0) {
    if (Math.abs(h1p - h2p) > 180) hMean += h1p + h2p < 360 ? 360 : -360;
    hMean /= 2;
  }
  const t =
    1 -
    0.17 * Math.cos(rad(hMean - 30)) +
    0.24 * Math.cos(rad(2 * hMean)) +
    0.32 * Math.cos(rad(3 * hMean + 6)) -
    0.2 * Math.cos(rad(4 * hMean - 63));
  const sl = 1 + (0.015 * (lMean - 50) ** 2) / Math.sqrt(20 + (lMean - 50) ** 2);
  const sc = 1 + 0.045 * cpMean;
  const sh = 1 + 0.015 * cpMean * t;
  const rt =
    -2 *
    Math.sqrt(cpMean ** 7 / (cpMean ** 7 + 25 ** 7)) *
    Math.sin(rad(60 * Math.exp(-(((hMean - 275) / 25) ** 2))));
  return Math.sqrt(
    (dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2 + rt * (dC / sc) * (dH / sh)
  );
}

function contrastRatio(hexA, hexB) {
  const luminance = hex => {
    const [r, g, b] = parseHex(hex).map(toLinear);
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  const [high, low] = [luminance(hexA), luminance(hexB)].sort((a, b) => b - a);
  return (high + 0.05) / (low + 0.05);
}

const VISIONS = ['normal', 'protanopia', 'deuteranopia', 'tritanopia'];

function visionLabs(rgbList) {
  return {
    normal: rgbList.map(rgbToLab),
    protanopia: rgbList.map(rgb => rgbToLab(simulate(rgb, CVD_MATRICES.protanopia))),
    deuteranopia: rgbList.map(rgb => rgbToLab(simulate(rgb, CVD_MATRICES.deuteranopia))),
    tritanopia: rgbList.map(rgb => rgbToLab(simulate(rgb, CVD_MATRICES.tritanopia)))
  };
}

/** Minimum Delta E between adjacent entries (or all pairs) per vision; skips transparent ones. */
function minimumDeltaE(hexes, allPairs) {
  const visible = hexes.filter(hex => !(hex.length === 9 && hex.endsWith('00')));
  const labs = visionLabs(visible.map(parseHex));
  const result = {};
  for (const vision of VISIONS) {
    let minimum = Number.POSITIVE_INFINITY;
    const list = labs[vision];
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < (allPairs ? list.length : i + 2); j++) {
        if (j < list.length) minimum = Math.min(minimum, deltaE2000(list[i], list[j]));
      }
    }
    result[vision] = minimum;
  }
  return result;
}

/** L* of each entry; monotone direction summary. */
function lightnessOrder(hexes) {
  const lightness = hexes.map(hex => rgbToLab(parseHex(hex))[0]);
  let up = 0;
  let down = 0;
  for (let i = 1; i < lightness.length; i++) {
    const step = lightness[i] - lightness[i - 1];
    if (step > MONOTONE_TOLERANCE) up++;
    else if (step < -MONOTONE_TOLERANCE) down++;
  }
  return {
    lightness,
    monotone: up === 0 || down === 0,
    direction: up === 0 && down === 0 ? 'flat' : up === 0 ? 'darkening' : down === 0 ? 'lightening' : 'mixed'
  };
}

const toHex = ([r, g, b]) =>
  `#${[r, g, b].map(value => Math.round(value).toString(16).padStart(2, '0')).join('')}`;

// ---------------------------------------------------------------------------------------------
// Load the modules
// ---------------------------------------------------------------------------------------------

async function load() {
  const result = await build({
    stdin: {
      contents: `
        export * from './engine/ramps.ts';
        export * from './cartography/class-table.ts';
        export * from './cartography/hue-registry.ts';
      `,
      resolveDir: SOURCE,
      sourcefile: 'check-colours-entry.ts',
      loader: 'ts'
    },
    bundle: true,
    write: false,
    format: 'cjs',
    platform: 'node',
    logLevel: 'silent'
  });
  const module = {exports: {}};
  new Function('module', 'exports', 'require', result.outputFiles[0].text)(
    module,
    module.exports,
    () => ({})
  );
  return module.exports;
}

const formatDelta = value => (Number.isFinite(value) ? value.toFixed(1).padStart(4) : '   -');
const formatRow = delta => VISIONS.map(vision => formatDelta(delta[vision])).join(' ');
const worstOf = delta => Math.min(...VISIONS.map(vision => delta[vision]));

async function main() {
  const lib = await load();
  let failures = 0;
  let warnings = 0;

  // 1. Ramps -------------------------------------------------------------------------------
  const registryRamps = new Set(
    Object.values(lib.HUE_REGISTRY)
      .filter(entry => entry.kind === 'sequential' && entry.ramp)
      .map(entry => entry.ramp)
  );
  const EXEMPT = new Set(['isolum', 'hypsometric']);
  console.log('RAMPS: sequential lightness (CIE L*) monotonicity');
  console.log('name         dir         L* start -> end   status');
  for (const name of lib.ALL_RAMP_NAMES) {
    const info = lib.RAMP_INFO[name];
    if (info.kind !== 'sequential') continue;
    const hexes = lib.RAMP_STOPS[name].map(toHex);
    const order = lightnessOrder(hexes);
    const first = order.lightness[0].toFixed(0);
    const last = order.lightness[order.lightness.length - 1].toFixed(0);
    let status = 'ok';
    if (!order.monotone) {
      if (registryRamps.has(name)) {
        status = 'FAIL (in registry)';
        failures++;
      } else if (EXEMPT.has(name)) {
        status = 'exempt (isoluminant / convention)';
      } else {
        status = 'WARN non-monotone';
        warnings++;
      }
    }
    console.log(`${name.padEnd(12)} ${order.direction.padEnd(11)} ${first.padStart(3)} -> ${last.padStart(3)}         ${status}`);
  }

  // 2. Class tables ------------------------------------------------------------------------
  console.log(`\nCLASS TABLES (ColorBrewer, light): worst case over n = 3..7 (the recommended range), per vision`);
  console.log(`(dE2000 of adjacent classes; qualitative: all pairs. "all n" is the worst over every published n)`);
  console.log('scheme       kind         worst n (dE)   normal  prot  deut  trit   all n   notes');
  for (const scheme of lib.CLASS_SCHEME_NAMES) {
    const info = lib.getClassSchemeInfo(scheme);
    let worstCount = 0;
    let worst = Number.POSITIVE_INFINITY;
    let worstDelta = null;
    let overall = Number.POSITIVE_INFINITY;
    let note = '';
    for (let count = info.minimumClasses; count <= info.maximumClasses; count++) {
      const hexes = lib.getClassPaletteHex(scheme, count);
      const delta = minimumDeltaE(hexes, info.kind === 'qualitative');
      if (VERBOSE) console.log(`  ${scheme}-${count}`.padEnd(14) + formatRow(delta));
      overall = Math.min(overall, worstOf(delta));
      if (count <= 7 && worstOf(delta) < worst) {
        worst = worstOf(delta);
        worstCount = count;
        worstDelta = delta;
      }
      if (info.kind === 'sequential') {
        const order = lightnessOrder(hexes);
        if (!order.monotone) note = `non-monotone at n=${count}`;
      }
    }
    if (worst < WARN_DELTA_E) warnings++;
    console.log(
      `${scheme.padEnd(12)} ${info.kind.padEnd(12)} ${String(worstCount).padStart(2)} (${worst.toFixed(1).padStart(4)})    ${formatRow(worstDelta)}   ${overall.toFixed(1).padStart(4)}   ${note}${worst < WARN_DELTA_E ? 'WARN' : ''}`
    );
  }

  // 3. Registry ----------------------------------------------------------------------------
  console.log(`\nHUE REGISTRY: min adjacent dE2000 (qualitative: all pairs)  [${VISIONS.join(' ')}]; WARN below ${WARN_DELTA_E}`);
  console.log('id                 ground kind        n  normal  prot  deut  trit  L*dir       status');
  for (const [id, entry] of Object.entries(lib.HUE_REGISTRY)) {
    for (const ground of ['light', 'dark']) {
      const hexes = entry[ground];
      const delta = minimumDeltaE(hexes, entry.kind === 'qualitative');
      const visible = hexes.filter(hex => !(hex.length === 9 && hex.endsWith('00'))).map(hex => hex.slice(0, 7));
      const order = lightnessOrder(visible);
      const reasons = [];
      if (worstOf(delta) < WARN_DELTA_E) {
        const worstVision = VISIONS.find(vision => delta[vision] === worstOf(delta));
        reasons.push(`dE ${worstOf(delta).toFixed(1)} (${worstVision})`);
      }
      if (entry.kind === 'sequential' && !order.monotone) reasons.push('L* not monotone');
      if (entry.kind === 'sequential') {
        // "Strongest = darkest" on light grounds, "strongest = brightest" on dark ones; the
        // strongest class is the last one, or the first when the entry is `reverse` (access).
        const expected = (ground === 'dark') !== Boolean(entry.reverse) ? 'lightening' : 'darkening';
        if (order.monotone && order.direction !== expected) reasons.push(`runs ${order.direction}, strongest should be ${expected === 'lightening' ? 'brightest' : 'darkest'}`);
      }
      if (ground === 'dark' && entry.kind === 'sequential') {
        const weakest = entry.reverse ? visible[visible.length - 1] : visible[0];
        const ratio = contrastRatio(weakest, DARK_GROUND);
        if (ratio < 3) reasons.push(`weakest class ${ratio.toFixed(1)}:1 (ground-level: draw transparent or additive)`);
      }
      if (reasons.length) warnings++;
      console.log(
        `${id.padEnd(18)} ${ground.padEnd(6)} ${entry.kind.padEnd(11)} ${String(hexes.length).padStart(2)} ${formatRow(delta)}  ${order.direction.padEnd(11)} ${reasons.length ? `WARN ${reasons.join('; ')}` : 'ok'}`
      );
    }
  }

  // 4. Dark-ground class palettes of the sequential schemes ------------------------------------
  // YlOrRd, YlOrBr and YlGnBu use the authored magma / inferno samples of the registry, whose
  // lowest class is ground-level by design (omit it or draw it additively). Every other scheme
  // is reversed and lifted, and must keep 3:1 for its lowest class.
  const AUTHORED = new Set(['YlOrRd', 'YlOrBr', 'YlGnBu']);
  const derivedRatios = [];
  const authoredRatios = [];
  for (const scheme of lib.getClassSchemeNames('sequential')) {
    for (let count = 3; count <= 9; count++) {
      const dark = lib.getClassPaletteHex(scheme, count, {ground: 'dark'});
      const record = {scheme, count, ratio: contrastRatio(dark[0], DARK_GROUND), order: lightnessOrder(dark)};
      (AUTHORED.has(scheme) ? authoredRatios : derivedRatios).push(record);
    }
  }
  const worstRatio = derivedRatios.reduce((a, b) => (b.ratio < a.ratio ? b : a));
  const worstAuthored = authoredRatios.reduce((a, b) => (b.ratio < a.ratio ? b : a));
  const nonMonotone = [...derivedRatios, ...authoredRatios].filter(record => !record.order.monotone);
  console.log(
    `\nDARK sequential class palettes (n 3-9): derived schemes, lowest class vs ${DARK_GROUND}: min ${worstRatio.ratio.toFixed(2)}:1 (${worstRatio.scheme}-${worstRatio.count}), need >= 3`
  );
  console.log(
    `  authored magma/inferno schemes (YlOrRd, YlOrBr, YlGnBu): lowest class min ${worstAuthored.ratio.toFixed(2)}:1, ground-level by design`
  );
  console.log(`  non-monotone dark class palettes: ${nonMonotone.length}`);
  if (worstRatio.ratio < 3) warnings++;
  warnings += nonMonotone.length;

  console.log(`\n${failures} failure(s), ${warnings} warning(s). dE = CIEDE2000; CVD = Machado 2009 severity 1.0.`);
  process.exit(failures > 0 ? 1 : 0);
}

main().catch(error => {
  console.error(error);
  process.exit(2);
});
