// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readdirSync, readFileSync, statSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';

// Rule 18: one vocabulary across the analysis entries (quality roadmap "Canonical vocabulary").
// Every `export type GPU*Props` / `GPU*Settings` property, nested object properties included, is
// checked against the table below, and every Props, Output, Result and Port type is checked for
// status placement: GPU-written status words live in a `status` member. Coordinate components
// (`minX` ... `maxZ`) are exempt. The only allowed exceptions are the APIs of files that luma
// owns upstream (the raster base classes), listed in `vocabulary-allowlist.json` as
// `{"<Type>.<property path>": {"rule": "<rule ID>", "reason": "upstream-owned"}}`.
// A new violation fails, and so does an allowlist entry that no longer occurs or has no reason.
//
// Regenerate with `UPDATE_VOCABULARY_ALLOWLIST=1 npx vitest run --project node
// modules/experimental/test/analysis-core/analysis-vocabulary.node.spec.ts`; it keeps existing
// reasons and leaves new entries without one, so they fail until reviewed.

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const ALLOWLIST_PATH = resolve(
  dirname(fileURLToPath(import.meta.url)),
  'vocabulary-allowlist.json'
);

const ANALYSIS_LAYERS = [
  'analysis-core',
  'geospatial',
  'gpu-dataframe',
  'gpu-raster',
  'gpu-terrain',
  'gpu-spatial-analysis',
  'gpu-network'
] as const;

type VocabularyRule = {
  /** Canonical replacement, reported with each finding. */
  canonical: string;
  /** True when a property name (with its type text) is non-canonical. */
  matches: (name: string, typeText: string, layer: string) => boolean;
};

const PACKED_VIEW_TYPE = /GraphDataView|GraphVectorView|GPUParameterBuffer/;

/** Coordinate components of bounds and extents, exempt from the `max*` / `min*` rules. */
const COORDINATE_COMPONENT = /^(min|max)[XYZ]$/;

/** One-row GPU-written status scalars; per-row columns use plural names (`counts`). */
const STATUS_SCALAR_TYPE = /GraphDataView<'uint32'>|GPUStatusScalar/;

/**
 * Status words (quality roadmap "Canonical vocabulary"): cardinality, completeness, convergence,
 * diagnostics and their stage-prefixed forms (`pairOverflow`, `weightsRequiredCount`).
 */
const STATUS_WORD =
  /^(count|converged|approximation|iterationCount|iterationLimitReached|\w*[oO]verflow|\w*[rR]equired\w*Count|\w*[iI]nvalidCount|\w*[uU]ncertainCount)$/;

/** The canonical-vocabulary table as checks. The rule ID is the replaced spelling. */
const VOCABULARY_RULES: Record<string, VocabularyRule> = {
  settings: {
    canonical: 'parameters (packed per-frame view)',
    matches: (name, typeText) => name === 'settings' && PACKED_VIEW_TYPE.test(typeText)
  },
  wordParameters: {canonical: 'uint32Parameters', matches: name => name === 'wordParameters'},
  'max*': {
    canonical: 'maximum*',
    matches: name => /^max[A-Z0-9]/.test(name) && !COORDINATE_COMPONENT.test(name)
  },
  'min*': {
    canonical: 'minimum*',
    matches: name => /^min[A-Z0-9]/.test(name) && !COORDINATE_COMPONENT.test(name)
  },
  iterations: {
    canonical: 'maximumIterations (input), iterationCount (output)',
    matches: name => name === 'iterations' || name === 'maximumRounds' || name === 'roundCount'
  },
  '*Passes': {canonical: 'maximumIterations', matches: name => /[a-z]Passes$|^passes$/.test(name)},
  maximumCapacity: {canonical: '<noun>Capacity', matches: name => name === 'maximumCapacity'},
  permutations: {
    canonical: 'maximumPermutations (compile time), permutationCount (per frame)',
    matches: name => name === 'permutations'
  },
  'row selection': {
    canonical: 'mask (input), outputMask (output)',
    matches: name => ['keepMask', 'selection', 'valueMask', 'included'].includes(name)
  },
  polygons: {
    canonical: 'GPUPolygonGeometryPort four-level layout',
    matches: name => name === 'polygonPositions' || name === 'featureRingOffsets'
  },
  lines: {
    canonical: 'positions + lineOffsets / trackOffsets',
    matches: name => name === 'pathOffsets' || name === 'startIndices'
  },
  network: {
    canonical: 'topology: {offsets, neighbors, weights, nodePositions}; costBreaks, costLimit',
    // Class `breaks` of classifications and isobands are canonical; only cost bands are renamed.
    matches: (name, _typeText, layer) =>
      ['edgeTargets', 'edgeCosts', 'bandThresholds', 'bandBreaks'].includes(name) ||
      (layer === 'gpu-network' && name === 'breaks')
  },
  'raster grid': {
    canonical: 'grid: GPURasterMetadata',
    matches: (name, _typeText, layer) =>
      (layer === 'gpu-raster' || layer === 'gpu-terrain') &&
      ['width', 'height', 'cellSize'].includes(name)
  },
  'seed starts': {
    canonical: 'start* (graph starts); seed only for random seeds',
    // Distance-field and catchment seeds are sources, not graph starts.
    matches: (name, _typeText, layer) =>
      layer === 'gpu-network' && (name === 'seeds' || /^seed[A-Z]/.test(name))
  },
  results: {canonical: 'output', matches: name => name === 'result' || name === 'results'},
  geographic: {canonical: 'spatialContext', matches: name => name === 'geographic'},
  falseDiscoveryRate: {
    canonical: 'multipleTesting',
    matches: name => name === 'falseDiscoveryRate'
  }
};

/** The status-placement rule: GPU-written status words live in a `status` member. */
const STATUS_RULE_ID = 'status';
const STATUS_CANONICAL = 'status.<word> (for example output.status.overflow)';

/** Status ports themselves (`GPUCardinalityPort`, `GPUIterationStatusPort`, ...) define the words. */
const STATUS_PORT_TYPE = /^GPU\w*(Status|StatusPort|CardinalityPort)$/;

/** True when `path` (`<Type>.<member>...`) places a status word outside a `status` member. */
function isMisplacedStatus(
  typeName: string,
  path: string,
  name: string,
  typeText: string
): boolean {
  if (!STATUS_WORD.test(name) || !STATUS_SCALAR_TYPE.test(typeText)) {
    return false;
  }
  if (STATUS_PORT_TYPE.test(typeName)) {
    return false;
  }
  const segments = path.split('.').slice(1, -1);
  if (segments.includes('status')) {
    return false;
  }
  // A top-level `count` of Props is an input (the number of rows to read), not a status word.
  return !(segments.length === 0 && name === 'count' && /Props$/.test(typeName));
}

/** `{<layer>: {"<Type>.<property path>": rule ID}}`. */
type VocabularyFindings = Record<string, Record<string, string>>;

/** `{<layer>: {"<Type>.<property path>": {rule, reason}}}`. */
type VocabularyAllowlist = Record<string, Record<string, {rule: string; reason: string}>>;

function listSourceFiles(directory: string): string[] {
  const files: string[] = [];
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      files.push(...listSourceFiles(path));
    } else if (name.endsWith('.ts')) {
      files.push(path);
    }
  }
  return files;
}

function getTypeLiterals(node: typescript.TypeNode): typescript.TypeLiteralNode[] {
  if (typescript.isTypeLiteralNode(node)) {
    return [node];
  }
  if (typescript.isIntersectionTypeNode(node) || typescript.isUnionTypeNode(node)) {
    return node.types.flatMap(getTypeLiterals);
  }
  if (typescript.isParenthesizedTypeNode(node)) {
    return getTypeLiterals(node.type);
  }
  if (typescript.isTypeReferenceNode(node) && node.typeArguments) {
    // Partial<{...}>, Readonly<{...}> and recipe override helpers carry literal members.
    return node.typeArguments.flatMap(getTypeLiterals);
  }
  if (typescript.isArrayTypeNode(node)) {
    return getTypeLiterals(node.elementType);
  }
  return [];
}

function collectMembers(
  sourceFile: typescript.SourceFile,
  literal: typescript.TypeLiteralNode,
  path: string,
  visit: (path: string, name: string, typeText: string) => void
): void {
  for (const member of literal.members) {
    if (
      !member.name ||
      !(typescript.isPropertySignature(member) || typescript.isMethodSignature(member))
    ) {
      continue;
    }
    const name = member.name.getText(sourceFile).replace(/^['"]|['"]$/g, '');
    const typeNode = typescript.isPropertySignature(member) ? member.type : undefined;
    visit(`${path}.${name}`, name, typeNode?.getText(sourceFile) || '');
    if (typeNode) {
      for (const nested of getTypeLiterals(typeNode)) {
        collectMembers(sourceFile, nested, `${path}.${name}`, visit);
      }
    }
  }
}

function getVocabularyFindings(): VocabularyFindings {
  const findings: VocabularyFindings = {};
  for (const layer of ANALYSIS_LAYERS) {
    const layerFindings: Record<string, string> = {};
    for (const file of listSourceFiles(join(SOURCE_ROOT, layer))) {
      const sourceFile = typescript.createSourceFile(
        file,
        readFileSync(file, 'utf8'),
        typescript.ScriptTarget.ES2022,
        true
      );
      for (const statement of sourceFile.statements) {
        const isExported = typescript
          .getModifiers(statement as typescript.HasModifiers)
          ?.some(({kind}) => kind === typescript.SyntaxKind.ExportKeyword);
        if (!isExported) {
          continue;
        }
        let typeName: string | undefined;
        let literals: typescript.TypeLiteralNode[] = [];
        if (typescript.isTypeAliasDeclaration(statement)) {
          typeName = statement.name.text;
          literals = getTypeLiterals(statement.type);
        } else if (typescript.isInterfaceDeclaration(statement)) {
          typeName = statement.name.text;
          literals = [statement as unknown as typescript.TypeLiteralNode];
        }
        if (!typeName || !/^GPU\w+$/.test(typeName)) {
          continue;
        }
        const checksVocabulary = /(Props|Settings)$/.test(typeName);
        const name = typeName;
        for (const literal of literals) {
          collectMembers(sourceFile, literal, name, (path, memberName, typeText) => {
            if (checksVocabulary) {
              for (const [ruleId, rule] of Object.entries(VOCABULARY_RULES)) {
                if (rule.matches(memberName, typeText, layer)) {
                  layerFindings[path] = ruleId;
                }
              }
            }
            if (isMisplacedStatus(name, path, memberName, typeText)) {
              layerFindings[path] = STATUS_RULE_ID;
            }
          });
        }
      }
    }
    findings[layer] = Object.fromEntries(Object.entries(layerFindings).sort());
  }
  return findings;
}

it('uses the canonical vocabulary in every analysis Props and Settings type', () => {
  const findings = getVocabularyFindings();
  if (process.env['UPDATE_VOCABULARY_ALLOWLIST']) {
    const previous = JSON.parse(readFileSync(ALLOWLIST_PATH, 'utf8')) as VocabularyAllowlist;
    const next: VocabularyAllowlist = {};
    for (const [layer, entries] of Object.entries(findings)) {
      for (const [path, rule] of Object.entries(entries)) {
        const reason = previous[layer]?.[path]?.reason ?? '';
        next[layer] = {...next[layer], [path]: {rule, reason}};
      }
    }
    writeFileSync(ALLOWLIST_PATH, `${JSON.stringify(next, null, 2)}\n`);
  }
  const allowlist = JSON.parse(readFileSync(ALLOWLIST_PATH, 'utf8')) as VocabularyAllowlist;
  const added: string[] = [];
  const stale: string[] = [];
  const unexplained: string[] = [];
  for (const layer of ANALYSIS_LAYERS) {
    const found = findings[layer];
    const allowed = allowlist[layer] || {};
    for (const [path, ruleId] of Object.entries(found)) {
      if (allowed[path]?.rule !== ruleId) {
        const canonical =
          ruleId === STATUS_RULE_ID ? STATUS_CANONICAL : VOCABULARY_RULES[ruleId].canonical;
        added.push(`${layer} ${path}: '${ruleId}' -> ${canonical}`);
      }
    }
    for (const [path, {reason}] of Object.entries(allowed)) {
      if (!(path in found)) {
        stale.push(`${layer} ${path} (fixed: remove it from vocabulary-allowlist.json)`);
      } else if (!reason) {
        unexplained.push(`${layer} ${path}`);
      }
    }
  }
  expect(added, 'non-canonical names').toEqual([]);
  expect(stale, 'stale allowlist entries').toEqual([]);
  expect(unexplained, 'allowlist entries without a reason').toEqual([]);
});
