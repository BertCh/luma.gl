// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Stream S2 element-card contract for the spatial families dynamics, density, accessibility, joins,
// validity and recipes: pack entry, formula bindings, registered scenario and docs card order.
// Run one element with `-t GPUKernelDensity`.

import {readdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join, relative} from 'node:path';
import {describe, expect, it} from 'vitest';
import '../../../src/computation-lenses/scenarios/index';
import {listLensScenarios} from '../../../src/computation-lenses/index';
import {LENS_RUNG_IDS} from '../../../src/computation-views/index';
import {
  getSpatialOperatorEntry,
  SPATIAL_VARIABLE_KINDS
} from '../../../src/gpu-spatial-analysis/computation-pack/index';
import {
  DOCS_ROOT,
  getLensElementId,
  parseDirectiveAttributes,
  REPOSITORY_ROOT,
  scanLensMarkers
} from '../../computation-docs/lens-directives';

/** S2 elements by family; the family is also the docs page `operators-<family>.md` (recipes: `recipes.md`). */
export const S2_ELEMENTS = {
  dynamics: [
    'GPUClassAssignment',
    'GPUClassificationFit',
    'GPUTransitionMatrix',
    'GPUTransitionDiagnostics',
    'GPUSpatialMarkov',
    'GPULISAMarkov',
    'GPUKnoxTest',
    'GPUMantelTest',
    'GPUSpatialScanStatistic'
  ],
  density: [
    'GPUPointDensity',
    'GPUStudyWindowPreparation',
    'GPUKernelDensity',
    'GPUInverseDistanceWeighting',
    'GPUKriging',
    'GPUDotDensity',
    'GPURandomPointsInPolygon',
    'GPURandomPointsOnLine'
  ],
  accessibility: [
    'GPUCatchmentAccessibility',
    'GPUHuffTradeAreas',
    'GPUArealInterpolation',
    'GPUPycnophylactic',
    'GPUSegregation'
  ],
  joins: [
    'GPUSpatialPredicateJoin',
    'GPUSpatialJoinPrepared',
    'GPUSpatialJoinCandidates',
    'GPUPointInPolygonJoin',
    'GPUNearestFeatureJoin',
    'GPUNearestFeatureWeights',
    'GPUPairGather',
    'GPUOffsetExpansion',
    'GPUBoundsFilter',
    'GPUBufferSelection'
  ],
  validity: [
    'GPUSegmentIntersection',
    'GPUGeometryValidity',
    'GPUGeometryPredicates',
    'GPUMinimumClearance',
    'GPUCoverageValidity'
  ],
  recipes: [
    'addClusterAndOutlineRecipe',
    'addFleetDwellRecipe',
    'addSpatialRegressionRecipe',
    'addStraightLineCatchmentsRecipe',
    'addHotSpotAnalysisRecipe',
    'addRateClusterMapRecipe',
    'addPeriodComparisonRecipe',
    'addChangeOfSupportRecipe',
    'addPointsInPolygonsChoroplethRecipe'
  ]
} as const;

type S2Family = keyof typeof S2_ELEMENTS;

const SOURCE_ROOT = join(REPOSITORY_ROOT, 'modules/experimental/src');
const SPATIAL_ROOT = join(SOURCE_ROOT, 'gpu-spatial-analysis');
const PAGES_ROOT = join(DOCS_ROOT, 'api-reference/experimental/gpu-spatial-analysis');

function listTypeScriptFiles(directory: string): string[] {
  return readdirSync(directory).flatMap(name => {
    const path = join(directory, name);
    if (statSync(path).isDirectory())
      return name === 'computation-pack' ? [] : listTypeScriptFiles(path);
    return name.endsWith('.ts') ? [path] : [];
  });
}

const SPATIAL_FILES = listTypeScriptFiles(SPATIAL_ROOT).map(path => ({
  path,
  text: readFileSync(path, 'utf8')
}));
const SHARED_SOURCE_TEXT = [
  ...listTypeScriptFiles(join(SOURCE_ROOT, 'utils')),
  ...listTypeScriptFiles(join(SOURCE_ROOT, 'analysis-core'))
]
  .map(path => readFileSync(path, 'utf8'))
  .join('\n');

/** Source text an element's formula may bind to: its own directory plus shared WGSL utilities. */
function getElementSourceText(element: string): string {
  const declaration = new RegExp(`export (?:class|function|async function) ${element}\\b`);
  const file = SPATIAL_FILES.find(candidate => declaration.test(candidate.text));
  if (!file) throw new Error(`${element}: declaration not found under gpu-spatial-analysis`);
  const directory = dirname(file.path);
  return `${SPATIAL_FILES.filter(candidate => candidate.path.startsWith(`${directory}/`))
    .map(candidate => candidate.text)
    .join('\n')}\n${SHARED_SOURCE_TEXT}`;
}

function getPagePath(family: S2Family): string {
  return join(PAGES_ROOT, family === 'recipes' ? 'recipes.md' : `operators-${family}.md`);
}

/** The `###` section containing the element's lens marker, as lines. */
function getCardSection(family: S2Family, element: string): string[] {
  const path = getPagePath(family);
  const source = readFileSync(path, 'utf8');
  const marker = scanLensMarkers(path, source).find(candidate => candidate.name === element);
  return marker ? source.split(/\r?\n/).slice(marker.section.start - 1, marker.section.end) : [];
}

function getDirectives(
  lines: readonly string[]
): {line: number; attributes: Record<string, string>}[] {
  return lines.flatMap((line, index) => {
    const match = /^::lens\{(.*)\}\s*$/.exec(line.trim());
    return match ? [{line: index, attributes: parseDirectiveAttributes(match[1])}] : [];
  });
}

const scenarios = listLensScenarios();

describe('S2 element cards', () => {
  for (const [family, elements] of Object.entries(S2_ELEMENTS) as [S2Family, readonly string[]][]) {
    for (const element of elements) {
      const scenarioId = `gpu-spatial-analysis.${getLensElementId(element)}`;

      it(`${element}: pack entry`, () => {
        const entry = getSpatialOperatorEntry(element);
        expect(entry, 'missing pack entry').toBeTruthy();
        expect(entry!.family).toBe(family);
        expect(Object.keys(entry!.variables).length).toBeGreaterThan(0);
        for (const kind of Object.values(entry!.variables)) {
          expect(Object.keys(SPATIAL_VARIABLE_KINDS)).toContain(kind);
        }
        expect(entry!.stages.length).toBeGreaterThan(0);
        expect(entry!.datumKinds.length).toBeGreaterThan(0);
      });

      it(`${element}: formula bound to real names`, () => {
        const formula = getSpatialOperatorEntry(element)?.formula;
        expect(formula, 'missing formula').toBeTruthy();
        if (!formula) return;
        if (!('tex' in formula)) {
          expect(formula.reason.trim().length).toBeGreaterThan(0);
          const directives = getDirectives(getCardSection(family, element));
          expect(
            directives.some(directive =>
              ['iterations', 'graph'].includes(directive.attributes['rung'])
            ),
            'formula: none needs an iterations or graph lens'
          ).toBe(true);
          return;
        }
        expect(formula.id).toMatch(/^gpu-spatial-analysis\./);
        expect(formula.ascii.trim().length).toBeGreaterThan(0);
        const symbolIds = formula.symbols.map(symbol => symbol.id);
        expect(new Set(symbolIds).size).toBe(symbolIds.length);
        for (const [, id] of formula.tex.matchAll(/\\term\{([^}]+)\}/g)) {
          expect(symbolIds, `\\term{${id}} has no symbol`).toContain(id);
        }
        const sourceText = getElementSourceText(element);
        const identifiers = new Set(sourceText.match(/[A-Za-z_][A-Za-z_0-9]*/g));
        for (const symbol of formula.symbols) {
          if (!symbol.binding) continue;
          const [kind, ...rest] = symbol.binding.split(':');
          const target = rest.join(':');
          if (kind === 'wgsl')
            expect(identifiers.has(target), `wgsl:${target} not in source`).toBe(true);
          else if (kind === 'resource')
            expect(sourceText.includes(target), `resource:${target} not in source`).toBe(true);
        }
      });

      it(`${element}: registered scenario with a result rung`, () => {
        const scenario = scenarios.find(candidate => candidate.id === scenarioId);
        expect(scenario, `missing scenario ${scenarioId}`).toBeTruthy();
        expect(scenario!.element).toBe(element);
        expect(scenario!.rungs[0]).toBe('result');
        expect(scenario!.defaultRung).toBe('result');
      });

      it(`${element}: card order in the docs`, () => {
        const section = getCardSection(family, element);
        const location = relative(REPOSITORY_ROOT, getPagePath(family));
        expect(section.length, `${location}: no marker for ${element}`).toBeGreaterThan(0);
        const markers = scanLensMarkers(getPagePath(family), section.join('\n'));
        const markerLine = (markers.find(marker => marker.name === element)?.line ?? 0) - 1;
        const markerLines = new Set(markers.map(marker => marker.line - 1));
        const directives = getDirectives(section).filter(
          directive => directive.attributes['scenario'] === scenarioId
        );
        const result = directives.find(directive => directive.attributes['rung'] === 'result');
        expect(result, `${location}: ${element} has no result lens`).toBeTruthy();
        // The marker sits under this element's own "How it computes" heading.
        let how = markerLine;
        while (how >= 0 && section[how].trim() !== '#### How it computes') how--;
        expect(
          how,
          `${location}: ${element} marker is not under #### How it computes`
        ).toBeGreaterThan(result!.line);
        expect(
          section
            .slice(how + 1, markerLine)
            .every((line, index) => !line.trim() || markerLines.has(how + 1 + index)),
          'only blank lines or markers between the heading and the marker'
        ).toBe(true);
        expect(result!.attributes['caption']?.trim().split(/\s+/).length ?? 0).toBeLessThanOrEqual(
          18
        );
        const scenario = scenarios.find(candidate => candidate.id === scenarioId);
        const howLenses = directives.filter(directive => directive.line > how);
        expect(howLenses.length, 'How it computes needs at least one lens').toBeGreaterThan(0);
        for (const directive of directives) {
          expect(LENS_RUNG_IDS).toContain(directive.attributes['rung']);
          if (scenario) expect(scenario.rungs).toContain(directive.attributes['rung']);
        }
      });
    }
  }
});
