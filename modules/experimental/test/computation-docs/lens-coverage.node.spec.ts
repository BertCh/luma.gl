// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readdirSync, readFileSync} from 'node:fs';
import {basename, join, relative, resolve, sep} from 'node:path';
import {describe, expect, it} from 'vitest';
import type {GPUComputationFormulaRef} from '../../../gpgpu/src/index';
// Registers every scenario descriptor (lazy; no GPU code runs).
import '../../src/computation-lenses/scenarios/index';
import {
  fromRecordedLensFile,
  listLensScenarios,
  type LensScenarioDescriptor
} from '../../src/computation-lenses/index';
import {
  parseLensDirective,
  type LensRungId,
  type OperatorScope,
  type OperatorReduction,
  type ViewPackData,
  LENS_RUNG_IDS
} from '../../src/computation-views/index';
import {
  DOCS_ROOT,
  getLensElementId,
  REPOSITORY_ROOT,
  listDocsMarkdownFiles,
  listEntryElements,
  parseDirectiveAttributes,
  scanLensDirectives,
  scanLensMarkers,
  type LensMarker,
  type ScannedLensDirective
} from './lens-directives';

// Resume lists for the next round: element → what is still missing (from the stream handoffs).
// Element → what is missing, as `<stream>: <parts>`. Generated from the failing checks and cross-checked with the
// stream handoffs (2026-10-14); 'docs card' means scenario and pack exist but the doc section lacks lenses.
const GPU_GRAPH_PENDING: Readonly<Record<string, string>> = {};
const GPU_CORE_PENDING: Readonly<Record<string, string>> = {};
const GPU_DATAFRAME_PENDING: Readonly<Record<string, string>> = {};
const GPU_SPATIAL_ANALYSIS_PENDING: Readonly<Record<string, string>> = {};
const GPU_NETWORK_PENDING: Readonly<Record<string, string>> = {};
const GPU_RASTER_PENDING: Readonly<Record<string, string>> = {};
const GPU_TERRAIN_PENDING: Readonly<Record<string, string>> = {};

/**
 * Entries whose coordinators declared lens coverage complete. Wave-2 coordinators append their entry here
 * when their docs, pack and scenarios are in place; from then on every check below is enforced for it.
 */
export const COVERED_ENTRIES: readonly CoveredEntry[] = [
  {
    entry: 'gpu-graph',
    barrel: 'modules/gpgpu/src/gpu-graph/index.ts',
    docs: [
      'docs/api-reference/experimental/gpu-graph.md',
      'docs/api-reference/experimental/gpu-graph-topology.md',
      'docs/api-reference/experimental/gpu-graph-traversal.md',
      'docs/api-reference/experimental/gpu-graph-connectivity.md',
      'docs/api-reference/experimental/gpu-graph-metrics.md',
      'docs/api-reference/experimental/gpu-graph-layouts.md'
    ],
    // gpu-graph contributors add themselves with `addToGraph(commandGraph)`.
    contributorMethods: ['addToGraph'],
    loadPack: async () =>
      (await import('../../src/computation-lenses/scenarios/gpu-graph/computation-pack'))
        .GPU_GRAPH_COMPUTATION_PACK,
    pending: GPU_GRAPH_PENDING
  },
  {
    entry: 'gpu-core',
    barrel: 'modules/gpgpu/src/gpu-core/index.ts',
    // The documented primitives are the pack's operators; the barrel also exports the compiler.
    scope: 'pack-operators',
    docs: ['docs/api-reference/experimental/gpu-core'],
    loadPack: async () =>
      (await import('../../src/computation-views/packs/gpu-core/pack-data')).loadGPUCorePackData(),
    pending: GPU_CORE_PENDING
  },
  {
    entry: 'gpu-dataframe',
    barrel: [
      'modules/experimental/src/gpu-dataframe/index.ts',
      'modules/experimental/src/gpu-sql/index.ts'
    ],
    docs: [
      'docs/api-reference/experimental/gpu-dataframe.md',
      'docs/api-reference/experimental/gpu-dataframe-aggregation.md',
      'docs/api-reference/experimental/gpu-dataframe-analysis.md',
      'docs/api-reference/experimental/gpu-dataframe-expressions.md',
      'docs/api-reference/experimental/gpu-dataframe-indexes-joins.md',
      'docs/api-reference/experimental/gpu-dataframe-operations.md',
      'docs/api-reference/experimental/gpu-dataframe-sorting.md',
      'docs/api-reference/experimental/gpu-sql.md'
    ],
    loadPack: async () =>
      (await import('../../src/gpu-dataframe/computation-pack/index')).createDataFramePackData(),
    pending: GPU_DATAFRAME_PENDING
  },
  {
    entry: 'gpu-spatial-analysis',
    barrel: 'modules/experimental/src/gpu-spatial-analysis/index.ts',
    docs: ['docs/api-reference/experimental/gpu-spatial-analysis'],
    requireVariableKinds: true,
    loadPack: async () =>
      (
        await import('../../src/gpu-spatial-analysis/computation-pack/index')
      ).createSpatialPackData(),
    pending: GPU_SPATIAL_ANALYSIS_PENDING
  },
  {
    entry: 'gpu-network',
    barrel: 'modules/experimental/src/gpu-network/index.ts',
    docs: ['docs/api-reference/experimental/gpu-network'],
    loadPack: async () =>
      (await import('../../src/gpu-network/computation-pack/network-pack-data'))
        .GPU_NETWORK_PACK_DATA,
    pending: GPU_NETWORK_PENDING
  },
  {
    entry: 'gpu-raster',
    barrel: 'modules/experimental/src/gpu-raster/index.ts',
    docs: ['docs/api-reference/experimental/gpu-raster'],
    loadPack: async () =>
      (await import('../../src/gpu-raster/computation-pack/index')).getRasterViewPack(),
    pending: GPU_RASTER_PENDING
  },
  {
    entry: 'gpu-terrain',
    barrel: 'modules/experimental/src/gpu-terrain/index.ts',
    docs: ['docs/api-reference/experimental/gpu-terrain'],
    loadPack: async () =>
      (await import('../../src/gpu-terrain/computation-pack/index')).getTerrainViewPack(),
    pending: GPU_TERRAIN_PENDING
  }
];

/**
 * Recorded records are written only by the main session's headless capture run. Until it has run, missing
 * `docs/lens-records/<scenario>.json` files are allowed; set LENS_COVERAGE_REQUIRE_RECORDS=1 to enforce.
 */
const ALLOW_MISSING_RECORDS = process.env['LENS_COVERAGE_REQUIRE_RECORDS'] !== '1';

type CoveredEntry = {
  /** Entry id, also the scenario id prefix, e.g. `gpu-spatial-analysis`. */
  entry: string;
  /** Repo-relative barrel(s) whose exports are the elements. */
  barrel: string | readonly string[];
  /**
   * `pack-operators`: the elements are the pack's operator keys instead of every barrel contributor.
   * For entries whose barrel also exports infrastructure that is documented without lenses
   * (gpu-core command graphs and programs, gpu-raster filters and residency).
   */
  scope?: 'pack-operators';
  /** Repo-relative doc pages (directories allowed) that hold the element sections. */
  docs: readonly string[];
  /** Loads the domain pack data (plain data; must not import GPU code). */
  loadPack: () => Promise<PackCoverageData>;
  /** Require `variableKinds` on every operator descriptor (spatial pack). */
  requireVariableKinds?: boolean;
  /** Exported contributor classes intentionally without a lens, with the reason. */
  exempt?: Readonly<Record<string, string>>;
  /** Instance methods that make an exported class a contributor (default `getCommandNodes`). */
  contributorMethods?: readonly string[];
  /**
   * The resume list: element → what is still missing (from the stream handoffs). Pending elements
   * skip checks (b)–(e); (a) still applies unless the reason names the `pack`. A pending element
   * that passes every check fails the spec, so the list only shrinks.
   */
  pending?: Readonly<Record<string, string>>;
};
/** The pack fields the contract reads (ViewPackData plus `formulas`, see V.done). */
type PackCoverageData = Pick<ViewPackData, 'id' | 'operators'> & {
  formulas?: Readonly<Record<string, GPUComputationFormulaRef | {formula: 'none'; reason: string}>>;
};

// Wave 2 may author directives before scenarios land. Once any entry is covered, all must register.
const ALLOW_UNREGISTERED_SCENARIOS = COVERED_ENTRIES.length === 0;
const OPERATOR_SCOPES: readonly OperatorScope[] = [
  'local',
  'focal',
  'block',
  'zonal',
  'global',
  'frontier',
  'pairwise',
  'solver'
];
const OPERATOR_REDUCTIONS: readonly OperatorReduction[] = [
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
const DIRECTIVE_RUNGS: readonly LensRungId[] = LENS_RUNG_IDS;
const ATTRIBUTE_NAMES = ['scenario', 'rung', 'bind', 'datum', 'iteration', 'caption', 'view'];
const SLOT_ENTRIES = ['gpu-spatial-analysis', 'gpu-network', 'gpu-terrain'];

type ScannedPage = {file: string; markers: LensMarker[]; directives: ScannedLensDirective[]};

function isNonEmpty(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function collectDirectiveFailures(
  attributes: Record<string, string>,
  scenarios: readonly LensScenarioDescriptor[],
  allowUnregistered: boolean
): string[] {
  const failures: string[] = [];
  for (const name of Object.keys(attributes)) {
    if (!ATTRIBUTE_NAMES.includes(name)) failures.push(`unknown attribute ${name}`);
  }
  const scenario = attributes['scenario'];
  const rung = attributes['rung'];
  if (!scenario || !/^[a-z0-9-]+(\.[a-z0-9-]+){1,2}$/.test(scenario)) {
    failures.push(`invalid scenario ${scenario}`);
  }
  if (!DIRECTIVE_RUNGS.some(candidate => candidate === rung)) failures.push(`invalid rung ${rung}`);
  for (const name of ['datum', 'iteration']) {
    const value = attributes[name];
    if (value !== undefined && (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)))) {
      failures.push(`${name} must be a non-negative safe integer`);
    }
  }
  if (failures.length === 0) {
    try {
      parseLensDirective({...attributes, scenario, rung});
    } catch (error) {
      failures.push(String(error));
    }
  }
  const descriptor = scenarios.find(candidate => candidate.id === scenario);
  if (!descriptor && !allowUnregistered) failures.push(`unregistered scenario ${scenario}`);
  if (descriptor && !descriptor.rungs.some(candidate => candidate === rung)) {
    failures.push(`scenario ${scenario} does not list rung ${rung}`);
  }
  return failures;
}

function collectSlotFailures(
  elements: ReturnType<typeof listEntryElements>,
  pages: readonly ScannedPage[],
  conceptsFile: string
): string[] {
  const failures: string[] = [];
  const markers = pages.flatMap(page => page.markers.map(marker => ({...marker, file: page.file})));
  const required = [...elements.contributors, ...elements.recipes];
  const exported = new Set([...required, ...elements.otherClasses]);
  for (const name of required) {
    const count = markers.filter(marker => marker.name === name).length;
    if (count !== 1) failures.push(`${name}: expected exactly one marker, found ${count}`);
  }
  for (const marker of markers) {
    if (marker.name === 'execution-model') {
      if (marker.file !== conceptsFile) {
        failures.push(`${marker.file}:${marker.line}: execution-model belongs in concepts.md`);
      }
    } else if (!exported.has(marker.name)) {
      failures.push(`${marker.file}:${marker.line}: marker ${marker.name} is not exported`);
    }
  }
  const conceptsCount = markers.filter(
    marker => marker.name === 'execution-model' && marker.file === conceptsFile
  ).length;
  if (conceptsCount !== 1)
    failures.push(`execution-model: expected one concepts.md marker, found ${conceptsCount}`);
  return failures;
}

function collectEntryFailures(
  entry: CoveredEntry,
  elements: ReturnType<typeof listEntryElements>,
  pack: PackCoverageData,
  pages: readonly ScannedPage[],
  scenarios: readonly LensScenarioDescriptor[],
  allowMissingRecords: boolean,
  hasRecord: (scenario: string) => boolean
): string[] {
  const failures: string[] = [];
  const names = [...elements.contributors, ...elements.recipes];
  for (const [name, reason] of Object.entries(entry.exempt ?? {})) {
    if (!elements.contributors.includes(name) || !isNonEmpty(reason)) {
      failures.push(`${name}: invalid exemption (requires an exported contributor and a reason)`);
    }
  }
  for (const [name, reason] of Object.entries(entry.pending ?? {})) {
    if (!names.includes(name) || !isNonEmpty(reason)) {
      failures.push(`${name}: invalid pending entry (requires an element and what is missing)`);
    }
  }
  for (const name of names) {
    if (elements.contributors.includes(name) && isNonEmpty(entry.exempt?.[name])) continue;
    const elementFailures = collectElementFailures(
      entry,
      name,
      pack,
      pages,
      scenarios,
      allowMissingRecords,
      hasRecord
    );
    const pending = entry.pending?.[name];
    if (pending === undefined) {
      failures.push(...elementFailures);
    } else if (elementFailures.length === 0) {
      // The resume list must shrink honestly: a finished element leaves `pending`.
      failures.push(`${name}: listed as pending (${pending}) but complete; remove it from pending`);
    } else if (!/\bpack\b/.test(pending)) {
      // A pending element whose pack entry landed must keep a valid operator descriptor.
      failures.push(...elementFailures.filter(failure => failure.startsWith(`${name}: (a)`)));
    }
  }
  return failures;
}

/** Checks (a)–(e) for one element; every failure string starts with `<name>: (<check>)`. */
function collectElementFailures(
  entry: CoveredEntry,
  name: string,
  pack: PackCoverageData,
  pages: readonly ScannedPage[],
  scenarios: readonly LensScenarioDescriptor[],
  allowMissingRecords: boolean,
  hasRecord: (scenario: string) => boolean
): string[] {
  const failures: string[] = [];
  const operator = pack.operators?.[name];
  if (!operator) failures.push(`${name}: (a) missing operator`);
  else {
    if (!OPERATOR_SCOPES.includes(operator.scope)) failures.push(`${name}: (a) invalid scope`);
    if (!OPERATOR_REDUCTIONS.includes(operator.reduction))
      failures.push(`${name}: (a) invalid reduction`);
    if (entry.requireVariableKinds && Object.keys(operator.variableKinds ?? {}).length === 0) {
      failures.push(`${name}: (a) missing variableKinds`);
    }
  }
  const directives = pages.flatMap(page =>
    page.directives.filter(directive => directive.sectionClasses.includes(name))
  );
  const formula = pack.formulas?.[name];
  if (!formula) failures.push(`${name}: (b) missing formula`);
  else if ('formula' in formula && formula.formula === 'none') {
    if (!isNonEmpty(formula.reason)) failures.push(`${name}: (b) missing formula exemption reason`);
    if (
      !directives.some(directive => ['iterations', 'graph'].includes(directive.attributes['rung']))
    ) {
      failures.push(`${name}: (b) formula:none needs an iterations or graph directive`);
    }
  } else if (
    !('tex' in formula) ||
    !isNonEmpty(formula.tex) ||
    !isNonEmpty(formula.ascii) ||
    !Array.isArray(formula.symbols) ||
    !['curated', 'derived'].includes(formula.derivation)
  )
    failures.push(`${name}: (b) invalid formula`);
  // `element` is the exported name; the kebab form is only the scenario id's second segment.
  const elementScenarios = scenarios.filter(
    scenario => scenario.entry === entry.entry && scenario.element === name
  );
  if (elementScenarios.length === 0) {
    failures.push(`${name}: (c) missing registered scenario`);
  } else if (
    !elementScenarios.some(scenario => scenario.id.split('.')[1] === getLensElementId(name))
  ) {
    failures.push(`${name}: (c) no scenario id ${entry.entry}.${getLensElementId(name)}[.variant]`);
  }
  const markerCount = pages
    .flatMap(page => page.markers)
    .filter(marker => marker.name === name).length;
  if (markerCount !== 1)
    failures.push(`${name}: (d) expected exactly one marker, found ${markerCount}`);
  if (
    !directives.some(directive =>
      scenarios.some(
        scenario =>
          scenario.id === directive.attributes['scenario'] && scenario.entry === entry.entry
      )
    )
  )
    failures.push(`${name}: (d) missing directive for a registered ${entry.entry} scenario`);
  if (!allowMissingRecords) {
    for (const directive of directives) {
      const scenario = directive.attributes['scenario'];
      if (!hasRecord(scenario))
        failures.push(
          `${name}: (e) missing record ${scenario} (${directive.file}:${directive.line})`
        );
    }
  }
  return failures;
}

/** The entry's elements: barrel contributors and recipes, or the pack's operator keys. */
function getCoveredElements(
  entry: CoveredEntry,
  pack: PackCoverageData
): ReturnType<typeof listEntryElements> {
  if (entry.scope === 'pack-operators') {
    return {contributors: Object.keys(pack.operators ?? {}).sort(), recipes: [], otherClasses: []};
  }
  const barrels = typeof entry.barrel === 'string' ? [entry.barrel] : entry.barrel;
  const listed = barrels.map(barrel => listEntryElements(barrel, entry.contributorMethods));
  return {
    contributors: listed.flatMap(elements => elements.contributors).sort(),
    recipes: listed.flatMap(elements => elements.recipes).sort(),
    otherClasses: listed.flatMap(elements => elements.otherClasses).sort()
  };
}

function selectPages(pages: readonly ScannedPage[], paths: readonly string[]): ScannedPage[] {
  const absolutePaths = paths.map(path => resolve(REPOSITORY_ROOT, path));
  return pages.filter(page =>
    absolutePaths.some(path => page.file === path || page.file.startsWith(`${path}${sep}`))
  );
}

describe('lens coverage contract', () => {
  const files = listDocsMarkdownFiles();
  const scanFailures: string[] = [];
  const pages = files.map(file => {
    const text = readFileSync(file, 'utf8');
    let directives: ScannedLensDirective[] = [];
    try {
      directives = scanLensDirectives(file, text);
    } catch (error) {
      scanFailures.push(String(error));
    }
    return {file, markers: scanLensMarkers(file, text), directives};
  });

  it('validates every authored directive and its registered rungs', () => {
    expect(files.length).toBeGreaterThan(0);
    const failures = [...scanFailures];
    const scenarios = listLensScenarios();
    for (const page of pages) {
      for (const directive of page.directives) {
        const location = `${relative(REPOSITORY_ROOT, page.file)}:${directive.line}`;
        failures.push(
          ...collectDirectiveFailures(
            directive.attributes,
            scenarios,
            ALLOW_UNREGISTERED_SCENARIOS
          ).map(failure => `${location}: ${failure}`)
        );
      }
    }
    expect(failures).toEqual([]);
  });

  for (const entry of SLOT_ENTRIES) {
    it(`${entry} has exactly one slot for every contributor and recipe`, () => {
      const directory = join(DOCS_ROOT, 'api-reference/experimental', entry);
      const elements = listEntryElements(`modules/experimental/src/${entry}/index.ts`);
      expect(elements.contributors.length).toBeGreaterThan(0);
      expect(
        collectSlotFailures(
          elements,
          selectPages(pages, [directory]),
          join(directory, 'concepts.md')
        )
      ).toEqual([]);
    }, 120_000);
  }

  for (const entry of COVERED_ENTRIES) {
    it(`${entry.entry} covers every contributor and recipe with a pack, scenario and lens`, async () => {
      const failures = [...scanFailures];
      try {
        const pack = await entry.loadPack();
        const elements = getCoveredElements(entry, pack);
        failures.push(
          ...collectEntryFailures(
            entry,
            elements,
            pack,
            selectPages(pages, entry.docs),
            listLensScenarios(),
            ALLOW_MISSING_RECORDS,
            scenario => existsSync(join(DOCS_ROOT, 'lens-records', `${scenario}.json`))
          )
        );
      } catch (error) {
        failures.push(`${entry.entry}: pack coverage failed: ${String(error)}`);
      }
      expect(failures).toEqual([]);
    }, 120_000);
  }

  it('validates every captured record and its scenario identity', () => {
    const directory = join(DOCS_ROOT, 'lens-records');
    const failures: string[] = [];
    const scenarios = listLensScenarios();
    for (const file of existsSync(directory) ? readdirSync(directory, {withFileTypes: true}) : []) {
      if (!file.isFile() || !file.name.endsWith('.json')) continue;
      try {
        const record = fromRecordedLensFile(
          JSON.parse(readFileSync(join(directory, file.name), 'utf8'))
        );
        if (record.scenarioId !== basename(file.name, '.json'))
          failures.push(`${file.name}: scenarioId does not match filename`);
        if (
          !ALLOW_UNREGISTERED_SCENARIOS &&
          !scenarios.some(scenario => scenario.id === record.scenarioId)
        ) {
          failures.push(`${file.name}: unregistered scenario ${record.scenarioId}`);
        }
      } catch (error) {
        failures.push(`${file.name}: ${String(error)}`);
      }
    }
    expect(failures).toEqual([]);
  });
});

describe('coverage helpers', () => {
  const elements = {contributors: ['GPUExample'], recipes: [], otherClasses: []};
  const entry: CoveredEntry = {
    entry: 'test',
    barrel: 'unused.ts',
    docs: ['inline.md'],
    loadPack: async () => ({id: 'test', operators: {}})
  };
  const scenario: LensScenarioDescriptor = {
    id: 'test.example',
    entry: 'test',
    element: 'GPUExample',
    title: 'Example',
    audience: 'analyst',
    rungs: ['graph'],
    defaultRung: 'graph',
    load: async () => {
      throw new Error('Coverage must never load GPU code');
    }
  };
  function makePage(text: string, file = 'inline.md'): ScannedPage {
    return {file, markers: scanLensMarkers(file, text), directives: scanLensDirectives(file, text)};
  }

  it('reports a missing marker and rejects unknown or duplicate slots', () => {
    const concepts = makePage('## Execution\n<!-- lens:execution-model -->', 'concepts.md');
    expect(collectSlotFailures(elements, [concepts], 'concepts.md')).toEqual([
      'GPUExample: expected exactly one marker, found 0'
    ]);
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n<!-- lens:GPUExample -->\n<!-- lens:Unknown -->'
    );
    expect(collectSlotFailures(elements, [concepts, page], 'concepts.md')).toEqual([
      'GPUExample: expected exactly one marker, found 2',
      'inline.md:4: marker Unknown is not exported'
    ]);
  });

  it('rejects bad rungs, unknown attributes, invalid pins and unregistered scenarios', () => {
    expect(collectDirectiveFailures({scenario: 'test.example', rung: 'bogus'}, [], true)).toContain(
      'invalid rung bogus'
    );
    expect(
      collectDirectiveFailures(
        {scenario: 'test.example', rung: 'graph', datum: '-1', surprise: 'yes'},
        [],
        true
      )
    ).toEqual(['unknown attribute surprise', 'datum must be a non-negative safe integer']);
    expect(collectDirectiveFailures({scenario: 'test.example', rung: 'graph'}, [], false)).toEqual([
      'unregistered scenario test.example'
    ]);
    expect(
      collectDirectiveFailures({scenario: 'test.example', rung: 'datum'}, [scenario], false)
    ).toEqual(['scenario test.example does not list rung datum']);
  });

  it('reports all coverage failures for an empty pack', async () => {
    expect(
      collectEntryFailures(entry, elements, await entry.loadPack(), [], [], true, () => false)
    ).toEqual([
      'GPUExample: (a) missing operator',
      'GPUExample: (b) missing formula',
      'GPUExample: (c) missing registered scenario',
      'GPUExample: (d) expected exactly one marker, found 0',
      'GPUExample: (d) missing directive for a registered test scenario'
    ]);
  });

  it('enforces formula fallbacks, variable kinds and the capture gate', () => {
    const pack: PackCoverageData = {
      id: 'test',
      operators: {GPUExample: {scope: 'local', reduction: 'none'}},
      formulas: {GPUExample: {formula: 'none', reason: 'Structural operator'}}
    };
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=graph}'
    );
    expect(
      collectEntryFailures(entry, elements, pack, [page], [scenario], true, () => false)
    ).toEqual([]);
    expect(
      collectEntryFailures(
        {...entry, requireVariableKinds: true},
        elements,
        pack,
        [page],
        [scenario],
        false,
        () => false
      )
    ).toEqual([
      'GPUExample: (a) missing variableKinds',
      'GPUExample: (e) missing record test.example (inline.md:3)'
    ]);
    const noFallback = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=datum}'
    );
    expect(
      collectEntryFailures(entry, elements, pack, [noFallback], [scenario], true, () => false)
    ).toContain('GPUExample: (b) formula:none needs an iterations or graph directive');
  });

  it('skips pending elements honestly and fails once they are complete', async () => {
    const page = makePage(
      '### Example\n<!-- lens:GPUExample -->\n::lens{scenario=test.example rung=graph}'
    );
    const pack: PackCoverageData = {
      id: 'test',
      operators: {GPUExample: {scope: 'local', reduction: 'none'}},
      formulas: {GPUExample: {formula: 'none', reason: 'Structural operator'}}
    };
    const pending = (reason: string) => ({...entry, pending: {GPUExample: reason}});
    // Missing scenario, docs and formula are skipped; the landed operator is still checked.
    expect(
      collectEntryFailures(
        pending('docs card'),
        elements,
        await entry.loadPack(),
        [],
        [],
        true,
        () => false
      )
    ).toEqual(['GPUExample: (a) missing operator']);
    expect(
      collectEntryFailures(
        pending('pack + formula + scenario'),
        elements,
        await entry.loadPack(),
        [],
        [],
        true,
        () => false
      )
    ).toEqual([]);
    expect(
      collectEntryFailures(
        pending('docs card'),
        elements,
        pack,
        [page],
        [scenario],
        true,
        () => false
      )
    ).toEqual(['GPUExample: listed as pending (docs card) but complete; remove it from pending']);
    expect(
      collectEntryFailures(
        {...entry, pending: {GPUOther: 'docs card'}},
        elements,
        pack,
        [page],
        [scenario],
        true,
        () => false
      )
    ).toEqual(['GPUOther: invalid pending entry (requires an element and what is missing)']);
  });

  it('parses quoted attributes and rejects malformed lists', () => {
    expect(
      parseDirectiveAttributes(
        'scenario=test.example rung=graph caption="A quoted caption" bind=\'page\''
      )
    ).toEqual({scenario: 'test.example', rung: 'graph', caption: 'A quoted caption', bind: 'page'});
    for (const text of [
      'rung=graph trailing',
      'rung="graph',
      'rung=graph rung=datum',
      'caption=unquoted'
    ]) {
      expect(() => parseDirectiveAttributes(text)).toThrow();
    }
  });

  it.each([
    'html',
    'mdx'
  ])('ignores fences and associates %s markers before and after directives', format => {
    const marker = (name: string) =>
      format === 'html' ? `<!-- lens:${name} -->` : `{/* lens:${name} */}`;
    const text = [
      '```md',
      '### Fake',
      marker('Fake'),
      '::lens{rung=bad}',
      '```',
      '~~~md',
      '::lens{rung=bad}',
      '~~~',
      '### Shared',
      '::lens{scenario=test.example rung=graph}',
      marker('GPUExample'),
      '#### How it computes',
      marker('GPUOther'),
      '::lens{scenario=test.example rung=datum}',
      '## Boundary',
      '::lens{scenario=test.example rung=graph}',
      '### Next',
      marker('GPUNext'),
      '::lens{scenario=test.example rung=graph}'
    ].join('\n');
    expect(
      scanLensDirectives('inline.md', text).map(directive => ({
        line: directive.line,
        classes: directive.sectionClasses
      }))
    ).toEqual([
      {line: 10, classes: ['GPUExample', 'GPUOther']},
      {line: 14, classes: ['GPUExample', 'GPUOther']},
      {line: 16, classes: []},
      {line: 19, classes: ['GPUNext']}
    ]);
    expect(scanLensMarkers('inline.md', text)[0]).toEqual({
      name: 'GPUExample',
      line: 11,
      section: {start: 9, end: 14}
    });
  });
});
