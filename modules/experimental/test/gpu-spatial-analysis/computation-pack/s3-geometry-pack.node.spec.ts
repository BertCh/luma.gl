// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import {GEOMETRY_OPERATORS} from '../../../src/gpu-spatial-analysis/computation-pack/operators/geometry';
import {SPATIAL_ANALYSIS_STAGES} from '../../../src/gpu-spatial-analysis/computation-pack/stages';
import {SPATIAL_GUARDS} from '../../../src/gpu-spatial-analysis/computation-pack/guards';
import {SPATIAL_VARIABLE_KINDS} from '../../../src/gpu-spatial-analysis/computation-pack/variable-kinds';
import {listTexTerms} from '../../../src/computation-views/views/formula-tex';
import type {OperatorReduction, OperatorScope} from '../../../src/computation-views/types';

const ELEMENTS = [
  'GPUOutlineGeometry',
  'GPUOffsetCurve',
  'GPUVertexSnap',
  'GPUPolygonOverlay',
  'GPUBufferSurface',
  'GPUAffineTransform',
  'GPUGeometryOrientation',
  'GPUGeometryCleanup',
  'GPUMinimumBounds',
  'GPUPolygonTriangulation',
  'GPUDelaunayTessellation',
  'GPUVoronoiDiagram',
  'GPULabelPoint',
  'GPUShapeDescriptors',
  'GPULineDensity',
  'GPULineLengthPerPolygon',
  'GPUGridGenerator',
  'GPUShapeGenerator',
  'GPUHilbertKeys',
  'GPULonLatPolygonPreparation',
  'GPURectangleClip'
] as const;

type GeometryElement = (typeof ELEMENTS)[number];

const SOURCE_ROOT = 'modules/experimental/src/gpu-spatial-analysis';
const ELEMENT_KERNEL_SOURCES: Record<GeometryElement, string[]> = {
  GPUOutlineGeometry: [`${SOURCE_ROOT}/outline-geometry/gpu-outline-geometry.ts`],
  GPUOffsetCurve: [`${SOURCE_ROOT}/outline-geometry/gpu-offset-curve.ts`],
  GPUVertexSnap: [`${SOURCE_ROOT}/vertex-snap/gpu-vertex-snap.ts`],
  GPUPolygonOverlay: [
    `${SOURCE_ROOT}/polygon-overlay/gpu-polygon-overlay.ts`,
    `${SOURCE_ROOT}/polygon-overlay/overlay-face-labelling-wgsl.ts`
  ],
  GPUBufferSurface: [
    `${SOURCE_ROOT}/polygon-buffer/gpu-buffer-surface.ts`,
    `${SOURCE_ROOT}/outline-geometry/gpu-offset-curve.ts`,
    `${SOURCE_ROOT}/polygon-overlay/gpu-polygon-overlay.ts`
  ],
  GPUAffineTransform: [`${SOURCE_ROOT}/geometry-edit/geometry-edit-kernels.ts`],
  GPUGeometryOrientation: [`${SOURCE_ROOT}/geometry-edit/geometry-edit-kernels.ts`],
  GPUGeometryCleanup: [`${SOURCE_ROOT}/geometry-edit/geometry-edit-kernels.ts`],
  GPUMinimumBounds: [`${SOURCE_ROOT}/minimum-bounds/gpu-minimum-bounds.ts`],
  GPUPolygonTriangulation: [
    `${SOURCE_ROOT}/polygon-triangulation/gpu-polygon-triangulation.ts`,
    `${SOURCE_ROOT}/polygon-triangulation/polygon-triangulation-layout.ts`,
    `${SOURCE_ROOT}/polygon-triangulation/polygon-triangulation-monotone.ts`,
    `${SOURCE_ROOT}/polygon-triangulation/polygon-triangulation-monotone-wgsl.ts`
  ],
  GPUDelaunayTessellation: [`${SOURCE_ROOT}/delaunay-tessellation/gpu-delaunay-tessellation.ts`],
  GPUVoronoiDiagram: [`${SOURCE_ROOT}/delaunay-tessellation/gpu-voronoi-diagram.ts`],
  GPULabelPoint: [`${SOURCE_ROOT}/label-point/gpu-label-point.ts`],
  GPUShapeDescriptors: [
    `${SOURCE_ROOT}/shape-descriptors/shape-descriptors-kernels.ts`,
    `${SOURCE_ROOT}/shape-descriptors/convexity-hull.ts`,
    `${SOURCE_ROOT}/geometry-measures/geometry-measures-kernels.ts`
  ],
  GPULineDensity: [
    `${SOURCE_ROOT}/line-density/line-density-kernels.ts`,
    `${SOURCE_ROOT}/geometry-measures/geodesic-wgsl.ts`,
    'modules/experimental/src/utils/sorted-segment-sums.ts'
  ],
  GPULineLengthPerPolygon: [
    `${SOURCE_ROOT}/line-density/gpu-line-length-per-polygon.ts`,
    `${SOURCE_ROOT}/geometry-measures/geodesic-wgsl.ts`,
    'modules/experimental/src/utils/sorted-segment-sums.ts'
  ],
  GPUGridGenerator: [`${SOURCE_ROOT}/grid-generators/gpu-grid-generator.ts`],
  GPUShapeGenerator: [`${SOURCE_ROOT}/grid-generators/gpu-shape-generator.ts`],
  GPUHilbertKeys: [
    `${SOURCE_ROOT}/hilbert-keys/gpu-hilbert-keys.ts`,
    `${SOURCE_ROOT}/hilbert-keys/bounds-reduction.ts`
  ],
  GPULonLatPolygonPreparation: [
    `${SOURCE_ROOT}/lon-lat-polygons/lon-lat-polygon-preparation-kernels.ts`,
    `${SOURCE_ROOT}/line-segmentize/line-segmentize-kernels.ts`,
    `${SOURCE_ROOT}/rectangle-clip/rectangle-clip-kernels.ts`
  ],
  GPURectangleClip: [`${SOURCE_ROOT}/rectangle-clip/rectangle-clip-kernels.ts`]
};

const SCOPES: readonly OperatorScope[] = [
  'local',
  'focal',
  'block',
  'zonal',
  'global',
  'frontier',
  'pairwise',
  'solver'
];
const REDUCTIONS: readonly OperatorReduction[] = [
  'none',
  'count',
  'sum',
  'weighted-mean',
  'argmin',
  'histogram',
  'mode',
  'select',
  'scan-sort',
  'solve'
];
const DATUM_KINDS = [
  'feature',
  'vertex',
  'segment',
  'cell',
  'line',
  'track',
  'fix',
  'pair',
  'triangle',
  'ring'
];

describe('S3 geometry computation pack', () => {
  test('contains exactly the requested 21 geometry entries and unique curated formula IDs', () => {
    expect(Object.keys(GEOMETRY_OPERATORS).sort()).toEqual([...ELEMENTS].sort());
    const identifiers = ELEMENTS.flatMap(element => {
      const formula = GEOMETRY_OPERATORS[element].formula;
      return 'formula' in formula ? [] : [formula.id];
    });
    expect(new Set(identifiers).size).toBe(identifiers.length);
    expect(identifiers.length).toBeGreaterThan(0);
  });

  for (const element of ELEMENTS) {
    describe(element, () => {
      test('is exported and declares valid family, algebra, stages, data and guards', () => {
        expect(typeof spatialAnalysis[element]).toBe('function');
        const entry = GEOMETRY_OPERATORS[element];
        expect(entry).toBeDefined();
        expect(entry.family).toBe('geometry');
        expect(SCOPES).toContain(entry.descriptor.scope);
        if (entry.descriptor.scopeSecondary) {
          expect(SCOPES).toContain(entry.descriptor.scopeSecondary);
          expect(entry.descriptor.scopeSecondary).not.toBe(entry.descriptor.scope);
        }
        expect(REDUCTIONS).toContain(entry.descriptor.reduction);
        expect(entry.stages.length).toBeGreaterThan(0);
        let previousStage = -1;
        for (const stage of entry.stages) {
          const stageIndex = SPATIAL_ANALYSIS_STAGES.indexOf(stage);
          expect(stageIndex).toBeGreaterThan(previousStage);
          previousStage = stageIndex;
        }
        expect(entry.datumKinds.length).toBeGreaterThan(0);
        for (const datum of entry.datumKinds) {
          expect(DATUM_KINDS).toContain(datum.kind);
          expect(datum.label.trim().length).toBeGreaterThan(0);
          expect(['1d', 'csr', '2d', 'batch']).toContain(datum.indexSpace);
        }
        for (const guard of entry.guards) {
          expect(Object.hasOwn(SPATIAL_GUARDS, guard)).toBe(true);
        }
        for (const choice of entry.choices) {
          expect(SPATIAL_ANALYSIS_STAGES).toContain(choice.stage);
          expect(entry.stages).toContain(choice.stage);
          expect(choice.name.trim().length).toBeGreaterThan(0);
          expect(typeof choice.sweepable).toBe('boolean');
        }
      });

      test('declares primary variables with shared kinds', () => {
        const variables = GEOMETRY_OPERATORS[element].variables;
        expect(Object.keys(variables).length).toBeGreaterThan(0);
        for (const [name, kind] of Object.entries(variables)) {
          expect(name.trim().length).toBeGreaterThan(0);
          expect(Object.hasOwn(SPATIAL_VARIABLE_KINDS, kind), `${name}: ${kind}`).toBe(true);
        }
      });

      test('has an explicit procedural reason or balanced, tagged curated formula', () => {
        const formula = GEOMETRY_OPERATORS[element].formula;
        if ('formula' in formula) {
          expect(formula.formula).toBe('none');
          expect(formula.reason.trim().length).toBeGreaterThan(0);
          return;
        }
        expect(formula.id).toMatch(/^s3\.[a-z]+(?:-[a-z]+)*$/);
        expect(formula.derivation).toBe('curated');
        expect(formula.tex.trim().length).toBeGreaterThan(0);
        expect(formula.ascii.trim().length).toBeGreaterThan(0);
        const symbolIdentifiers = new Set(formula.symbols.map(symbol => symbol.id));
        expect(symbolIdentifiers.size).toBe(formula.symbols.length);
        const terms = listTexTerms(formula.tex);
        const termIdentifiers = new Set(terms.map(term => term.symbolId));
        expect(terms.length).toBeGreaterThan(0);
        for (const term of terms) {
          expect(symbolIdentifiers.has(term.symbolId), term.symbolId).toBe(true);
        }
        for (const symbol of formula.symbols) {
          if (['input', 'output', 'parameter'].includes(symbol.role) || symbol.binding) {
            expect(termIdentifiers.has(symbol.id), symbol.id).toBe(true);
          }
        }
        let braceDepth = 0;
        for (const character of formula.tex) {
          if (character === '{') braceDepth++;
          if (character === '}') braceDepth--;
          expect(braceDepth).toBeGreaterThanOrEqual(0);
        }
        expect(braceDepth).toBe(0);
        // A malformed term must not silently disappear from the parsed-term contract.
        expect(terms.length).toBe((formula.tex.match(/\\term\{/g) ?? []).length);
      });

      test('binds formula symbols to identifiers in existing kernel sources', () => {
        expect(ELEMENT_KERNEL_SOURCES[element].length).toBeGreaterThan(0);
        const source = ELEMENT_KERNEL_SOURCES[element]
          .map(path => {
            expect(existsSync(path), path).toBe(true);
            return readFileSync(path, 'utf8');
          })
          .join('\n');
        const formula = GEOMETRY_OPERATORS[element].formula;
        if ('formula' in formula) return;
        for (const symbol of formula.symbols) {
          if (!symbol.binding) continue;
          expect(symbol.binding).toMatch(/^wgsl:[A-Za-z_][A-Za-z_0-9]*$/);
          const identifier = symbol.binding.slice('wgsl:'.length);
          expect(source, `${element}: ${symbol.id} -> ${symbol.binding}`).toMatch(
            new RegExp(`\\b${identifier}\\b`)
          );
        }
      });
    });
  }

  test('keeps geographic approximations distinct from planar-only and metric-free operators', () => {
    expect(GEOMETRY_OPERATORS['GPUOutlineGeometry'].metadata?.['longitudeLatitudeModel']).toBe(
      'local-equirectangular'
    );
    expect(GEOMETRY_OPERATORS['GPUShapeGenerator'].metadata?.['longitudeLatitudeModel']).toBe(
      'spherical-destination'
    );
    expect(GEOMETRY_OPERATORS['GPULineDensity'].metadata?.['longitudeLatitudeAreaModel']).toBe(
      'authalic-sphere'
    );
    expect(GEOMETRY_OPERATORS['GPUOffsetCurve'].metadata?.['planarOnly']).toBe(true);
    expect(GEOMETRY_OPERATORS['GPUHilbertKeys'].guards).not.toContain('coordinate-model');
    expect(GEOMETRY_OPERATORS['GPULineDensity'].guards).toEqual(
      expect.arrayContaining(['coordinate-model', 'conservation', 'count-vs-rate'])
    );
    expect(GEOMETRY_OPERATORS['GPULineLengthPerPolygon'].guards).toContain('count-vs-rate');
  });

  test('describes iterative topology and approximate label search without claiming closed forms', () => {
    expect(GEOMETRY_OPERATORS['GPUDelaunayTessellation'].descriptor).toEqual({
      scope: 'global',
      scopeSecondary: 'solver',
      reduction: 'solve'
    });
    expect(GEOMETRY_OPERATORS['GPULabelPoint'].descriptor.scope).toBe('zonal');
    for (const element of [
      'GPUPolygonOverlay',
      'GPUPolygonTriangulation',
      'GPUDelaunayTessellation',
      'GPULabelPoint'
    ]) {
      expect(GEOMETRY_OPERATORS[element].formula).toMatchObject({formula: 'none'});
    }
  });
});
