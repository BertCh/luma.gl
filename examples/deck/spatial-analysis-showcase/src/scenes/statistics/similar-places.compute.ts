// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  GPU_SIMILAR_LOCATIONS_NO_RANK,
  GPUSimilarLocations,
  type GPUSimilarLocationsStandardization
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {hexToRgba, MAP_INK, NO_DATA_COLOR} from '../../cartography/hue-registry';
import {formatCount, formatDistance, formatOrdinal, liveText} from '../../cartography/live-text';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {getHairlineColor, getStateLineStyle} from '../weights/hot-spots.style';
import {createByteReader, getSortedFinite, getQuantile} from './b5-common';
import {REFERENCE_PRESETS, SIMILARITY_ATTRIBUTES} from './b5-variables';
import {createCountyGeometry} from './choropleth-classes.geometry';
import {
  ARC_SEGMENT_CAPACITY,
  createSegmentWriter,
  getArcSegments,
  getOutlineCapacity,
  getOutlineSegments
} from './similar-places.geometry';
import {
  type AttributeContribution,
  countShared,
  getContributions,
  getKilometers,
  getMedian,
  getNearestRows,
  getReferenceProfile,
  getStrongestCorrelation,
  rankRows,
  type Standardization,
  standardizeColumns
} from './similar-places.stats';
import {
  formatAttributeValue,
  getCountyName,
  getRankClass,
  getRankTable,
  getWeights,
  SOUTH_STATES,
  type SimilarityDirection
} from './similar-places.style';

/** Option state of the similar-places scene. Weights are `weight0` to `weight11`. */
export type SimilarPlacesOptions = {
  preset: string;
  /** What a click on the map does: inspect a county, set the reference, add to the reference set. */
  click: 'inspect' | 'set' | 'add';
  standardization: GPUSimilarLocationsStandardization;
  direction: SimilarityDirection;
  resultCount: number;
  excludeReference: boolean;
  showMatches: boolean;
  showArcs: boolean;
  showNeighbours: boolean;
  outlineSelected: boolean;
} & Record<`weight${number}`, number>;

/** Data shared with `legends(state, data)` through `ctx.setLegendData`. */
export type SimilarPlacesLegendData = {
  table: ClassTable;
  /** Counties per rank class (the match set first). */
  counts: number[];
  /** Counties without a rank that are not references. */
  missingCount: number;
  rankedCount: number;
  referenceName: string;
  referenceCount: number;
  groundIsDark: boolean;
};

const ATTRIBUTE_COUNT = SIMILARITY_ATTRIBUTES.length;
const MAXIMUM_RESULT_COUNT = 32;
const MAXIMUM_MARKED = 20;
const MAXIMUM_REFERENCES = 6;
const NEIGHBOUR_LABELS = 6;
const NO_RANK = GPU_SIMILAR_LOCATIONS_NO_RANK;

type Variant = {compiled: CompiledGPUCommandGraph<void>};

/** One readback of the GPU ranking, with the rows it ranked. */
type Latest = {
  ranks: Uint32Array;
  distances: Float32Array;
  topIds: number[];
  rankedCount: number;
};

/**
 * Feature-space search over US counties. `GPUSimilarLocations` ranks every county by weighted
 * Euclidean distance in standardised attribute space (the mean of the references when several are
 * chosen); a small kernel turns the ranks into five rank classes that the polygon layer reads
 * as categories. Weights, the result count, the direction and the references are buffer writes;
 * the standardisation is compile-time, so each rank graph compiles on first use.
 *
 * What the GPU does not return (the standardised values behind the profile chart, the baseline
 * the persist readout compares against) is computed on the CPU by `similar-places.stats.ts`,
 * which also cross-checks the GPU matches.
 */
export async function createSimilarPlaces(
  ctx: SceneContext<SimilarPlacesOptions>
): Promise<SceneInstance<SimilarPlacesOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'similar-places');
  const geometry = createCountyGeometry(resources, counties, ctx.datasets.get('us-states'));
  const n = geometry.features.length;
  const labelPoints = geometry.mesh.labelPoints;
  const nameCache = Array.from({length: n}, (_, row) =>
    getCountyName(geometry.features[row]?.properties ?? null)
  );
  const nameOf = (row: number): string => nameCache[row] ?? `County ${row}`;
  const stateOf = (row: number): string => String(geometry.features[row]?.properties?.['state']);
  const findCounty = (name: string, state: string): number =>
    geometry.features.findIndex(feature => {
      const properties = feature.properties;
      return (
        properties?.['name'] === name &&
        properties?.['state'] === state &&
        Number(properties?.['fips']) % 1000 < 500
      );
    });
  const labelPointOf = (row: number): LngLat => labelPoints[row] as LngLat;

  const displayColumns = SIMILARITY_ATTRIBUTES.map(attribute =>
    counties.column<Float32Array>(attribute.id)
  );
  // Heavy-tailed densities are compared on a log scale, as the table of attributes says.
  const modelColumns = SIMILARITY_ATTRIBUTES.map((attribute, k) =>
    attribute.transform === 'log10'
      ? Float32Array.from(displayColumns[k], value => Math.log10(Math.max(value, 1e-3)))
      : displayColumns[k]
  );
  const attributeMatrix = new Float32Array(n * ATTRIBUTE_COUNT);
  for (let row = 0; row < n; row++) {
    for (let k = 0; k < ATTRIBUTE_COUNT; k++) {
      attributeMatrix[row * ATTRIBUTE_COUNT + k] = modelColumns[k][row];
    }
  }
  const standardized: Record<Standardization, Float64Array[]> = {
    zscore: standardizeColumns(modelColumns, 'zscore', n),
    rank: standardizeColumns(modelColumns, 'rank', n)
  };

  const attributesBuffer = resources.createBuffer('attributes', attributeMatrix);
  const selectionBuffer = resources.createBuffer('selection', n * 4);
  const ranksBuffer = resources.createBuffer('ranks', n * 4);
  const distancesBuffer = resources.createBuffer('distances', n * 4);
  const topIdsBuffer = resources.createBuffer('top-ids', MAXIMUM_RESULT_COUNT * 4);
  const countBuffer = resources.createBuffer('count', 4);
  const classBuffer = resources.createBuffer('rank-classes', n * 4);
  const similarityParameters = resources.createParameterBuffer(
    'similarity-parameters',
    'float32',
    getGPUSimilarLocationsParameterLength(ATTRIBUTE_COUNT)
  );
  const displayParameters = resources.createParameterBuffer('display-parameters', 'float32', 4);

  const matchOutline = createSegmentWriter(
    resources,
    'match-outline',
    getOutlineCapacity(geometry, MAXIMUM_MARKED)
  );
  const referenceOutline = createSegmentWriter(
    resources,
    'reference-outline',
    getOutlineCapacity(geometry, MAXIMUM_REFERENCES)
  );
  const arcs = createSegmentWriter(resources, 'arcs', MAXIMUM_MARKED * ARC_SEGMENT_CAPACITY);
  let matchOutlineCount = 0;
  let referenceOutlineCount = 0;
  let arcCount = 0;
  let selectionCount = 0;

  const compileVariant = (standardization: GPUSimilarLocationsStandardization): Variant => {
    const graph = new GPUCommandGraph<void>(device, {id: `similar-${standardization}`});
    const ranks = importGraphBuffer(graph, 'ranks', ranksBuffer, 'uint32', n);
    const selection = importGraphBuffer(graph, 'selection', selectionBuffer, 'uint32', n);
    graph.add(
      new GPUSimilarLocations({
        id: 'similar-locations',
        attributes: importGraphBuffer(
          graph,
          'attributes',
          attributesBuffer,
          'float32',
          n * ATTRIBUTE_COUNT
        ),
        attributeCount: ATTRIBUTE_COUNT,
        selection,
        parameters: similarityParameters.importToGraph(graph),
        standardization,
        maximumResultCount: MAXIMUM_RESULT_COUNT,
        output: {
          ranks,
          distances: importGraphBuffer(graph, 'distances', distancesBuffer, 'float32', n),
          topIds: importGraphBuffer(graph, 'top-ids', topIdsBuffer, 'uint32', MAXIMUM_RESULT_COUNT),
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1)
        }
      })
    );
    // Rank classes for the layer: 0 the match set (and the references), 1-3 the bands after it,
    // 4 the rest, and a sentinel for rows that were not ranked.
    addKernelPass(graph, {
      id: 'rank-classes',
      invocationCount: n,
      bindings: [
        {
          name: 'parameters',
          view: displayParameters.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
        {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
        {name: 'selection', view: selection, type: 'u32', access: 'read'},
        {
          name: 'bands',
          view: importGraphBuffer(graph, 'rank-classes', classBuffer, 'uint32', n),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let rank = ranks[ranksOffset + index];
  var bandIndex = 0xffffffffu;
  if (selection[selectionOffset + index] != 0u) {
    bandIndex = 0u;
  } else if (rank != ${NO_RANK}u) {
    let matchLimit = u32(parameters[parametersOffset]);
    bandIndex = select(select(select(select(4u, 3u, rank < 600u), 2u, rank < 200u), 1u, rank < 50u), 0u, rank < matchLimit);
  }
  bands[bandsOffset + index] = bandIndex;`
    });
    return {compiled: resources.track(graph.compile())};
  };

  const variants = new Map<GPUSimilarLocationsStandardization, Variant>();
  const getVariant = (standardization: GPUSimilarLocationsStandardization): Variant => {
    let variant = variants.get(standardization);
    if (!variant) {
      variant = compileVariant(standardization);
      variants.set(standardization, variant);
    }
    return variant;
  };
  let active = getVariant(ctx.options.standardization);

  // ---------------------------------------------------------------------------------------------
  // References, selection and parameters
  // ---------------------------------------------------------------------------------------------

  const references: number[] = [];
  /** The county the reader clicked in "inspect" mode; -1 follows the first match. */
  let inspectedClick = -1;
  let latest: Latest | null = null;
  let legendHighlight: number[] | null = null;
  let table: ClassTable = getRankTable({
    direction: ctx.options.direction,
    matchCount: ctx.options.resultCount,
    rankedCount: n,
    ground: ctx.ground()
  });
  let dirty = true;

  const setPreset = () => {
    const preset = REFERENCE_PRESETS.find(entry => entry.value === ctx.options.preset);
    if (!preset) return;
    const row = findCounty(preset.name, preset.state);
    if (row >= 0) {
      references.length = 0;
      references.push(row);
    }
  };
  setPreset();
  if (!references.length) references.push(Math.max(0, findCounty('Loudoun', 'VA')));

  const getWeightValues = (): number[] =>
    Array.from({length: ATTRIBUTE_COUNT}, (_, k) =>
      Number(ctx.options[`weight${k}` as `weight${number}`] ?? 1)
    );

  const writeSelection = () => {
    const selection = new Uint32Array(n);
    for (const row of references) selection[row] = 1;
    selectionBuffer.write(selection);
    referenceOutlineCount = referenceOutline.write(getOutlineSegments(geometry, references));
    dirty = true;
    reader.markStale();
  };

  const writeParameters = () => {
    const options = ctx.options;
    similarityParameters.write(
      getGPUSimilarLocationsParameterValues(
        {
          resultCount: options.resultCount,
          direction: options.direction,
          excludeReference: options.excludeReference,
          weights: getWeightValues()
        },
        ATTRIBUTE_COUNT
      )
    );
    displayParameters.write(Float32Array.of(options.resultCount, 0, 0, 0));
    dirty = true;
    reader.markStale();
  };

  // ---------------------------------------------------------------------------------------------
  // Derived numbers (CPU), charts, annotations and legend data
  // ---------------------------------------------------------------------------------------------

  const getRankOf = (row: number): number =>
    latest && latest.ranks[row] !== NO_RANK ? latest.ranks[row] + 1 : Number.NaN;

  const getInspectedRow = (): number => {
    if (!latest) return -1;
    if (inspectedClick >= 0) return inspectedClick;
    return latest.topIds[0] ?? -1;
  };

  /** Mean of the references' raw (unstandardised) values of attribute `k`. */
  const getReferenceValue = (k: number): number =>
    references.reduce((sum, row) => sum + displayColumns[k][row], 0) / references.length;

  const getReferenceName = (): string =>
    references.length > 2
      ? `${references.length} references`
      : references.map(nameOf).join(' and ');

  const publishLegendData = () => {
    if (!latest) return;
    const options = ctx.options;
    const ground = ctx.ground();
    table = getRankTable({
      direction: options.direction,
      matchCount: options.resultCount,
      rankedCount: latest.rankedCount,
      ground
    });
    const counts = [0, 0, 0, 0, 0];
    let missingCount = 0;
    const referenceSet = new Set(references);
    for (let row = 0; row < n; row++) {
      const rank = latest.ranks[row];
      if (rank === NO_RANK) {
        if (!referenceSet.has(row)) missingCount++;
      } else counts[getRankClass(rank, options.resultCount)]++;
    }
    const legendData: SimilarPlacesLegendData = {
      table,
      counts,
      missingCount,
      rankedCount: latest.rankedCount,
      referenceName: getReferenceName(),
      referenceCount: references.length,
      groundIsDark: ground === 'dark'
    };
    ctx.setLegendData('similarPlaces', legendData);
  };

  const publishProfile = (contributions: AttributeContribution[], inspected: number) => {
    const options = ctx.options;
    if (inspected < 0 || references.includes(inspected) || !contributions.length) {
      ctx.setChart('profile', null);
      ctx.setChart('contribution', null);
      ctx.setReadout('selectedMatch', null);
      ctx.setReadout('topAttribute', null);
      return;
    }
    const mode = options.standardization;
    const zValues = standardized[mode];
    const profile = getReferenceProfile(zValues, references);
    const total = contributions.reduce((sum, term) => sum + term.contribution, 0);
    const referenceName = nameOf(references[0]);
    const limit = Math.max(
      3,
      Math.ceil(
        Math.max(
          ...contributions.map(term => Math.abs(profile[term.attribute]) + Math.abs(term.delta))
        )
      )
    );
    ctx.setChart('profile', {
      kind: 'dumbbell',
      title: `${nameOf(inspected)} against ${references.length > 1 ? 'the reference mean' : referenceName}`,
      rows: contributions.map((term, index) => ({
        label: SIMILARITY_ATTRIBUTES[term.attribute].label,
        a: profile[term.attribute],
        b: zValues[term.attribute][inspected],
        highlight: index === 0
      })),
      aLabel: references.length > 1 ? 'Reference mean' : referenceName,
      bLabel: nameOf(inspected),
      xLabel:
        mode === 'zscore'
          ? 'Standard deviations from the national mean'
          : 'Percentile (0 lowest, 1 highest)',
      xDomain: mode === 'zscore' ? [-limit, limit] : [0, 1],
      height: Math.max(120, contributions.length * 15 + 44),
      description: `Dumbbell chart: standardised values of ${nameOf(inspected)} and the reference for each attribute with weight, sorted by their share of the distance.`
    });
    ctx.setChart('contribution', {
      kind: 'bars',
      title: 'Share of the squared distance',
      values: contributions.map(term => (total > 0 ? (100 * term.contribution) / total : 0)),
      labels: contributions.map(term => SIMILARITY_ATTRIBUTES[term.attribute].label),
      horizontal: true,
      highlight: [0],
      xLabel: 'Percent of the squared distance',
      table: false,
      description: `Bars: each weighted attribute's share of the squared distance between ${nameOf(inspected)} and the reference.`
    });
    const top = contributions[0];
    ctx.setReadout(
      'selectedMatch',
      `${nameOf(inspected)}, rank ${formatCount(getRankOf(inspected))}`
    );
    ctx.setReadout(
      'topAttribute',
      `${SIMILARITY_ATTRIBUTES[top.attribute].label}, ${total > 0 ? Math.round((100 * top.contribution) / total) : 0}% of the squared distance`
    );
  };

  const publishAnnotations = (
    notes: {
      contributions: AttributeContribution[];
      inspected: number;
      medianKilometers: number;
      neighbours: {row: number; kilometers: number}[];
    } | null
  ) => {
    const options = ctx.options;
    const list: MapAnnotation[] = [];
    if (!latest || !notes) {
      ctx.setAnnotations('similar-places', null);
      return;
    }
    // The reference is a reader-chosen input: ring-and-dot in the signal colour.
    references.slice(0, 2).forEach(row => {
      list.push({
        kind: 'point',
        coordinate: labelPointOf(row),
        text: nameOf(row),
        marker: 'ring',
        rank: 'subject',
        tone: 'signal',
        priority: 10
      });
    });
    if (options.showMatches) {
      latest.topIds.slice(0, MAXIMUM_MARKED).forEach((row, index) => {
        list.push({
          kind: 'marker',
          coordinate: labelPointOf(row),
          number: index + 1,
          text: `${index + 1}. ${nameOf(row)}`
        });
      });
    }
    if (options.showArcs && references.length === 1 && latest.topIds.length) {
      list.push({
        kind: 'ring',
        coordinate: labelPointOf(references[0]),
        radiusMeters: notes.medianKilometers * 1000,
        geodesic: true,
        text: liveText('median match {distance:distance}', {
          distance: notes.medianKilometers * 1000
        }),
        tone: 'muted',
        priority: 3
      });
    }
    if (options.showNeighbours && references.length === 1) {
      notes.neighbours.slice(0, NEIGHBOUR_LABELS).forEach(({row}) => {
        list.push({
          kind: 'point',
          coordinate: labelPointOf(row),
          text: nameOf(row),
          detail: liveText('rank {rank:integer}', {rank: getRankOf(row)}),
          marker: 'dot',
          rank: 'context',
          priority: 4
        });
      });
    }
    if (options.direction === 'most' && !options.showNeighbours && options.showArcs) {
      // The farthest match, where the arc ends: a note with its distance.
      let farthest = -1;
      let farthestKilometers = 0;
      for (const row of latest.topIds) {
        const kilometers = Math.min(
          ...references.map(reference => getKilometers(labelPointOf(reference), labelPointOf(row)))
        );
        if (kilometers > farthestKilometers) {
          farthestKilometers = kilometers;
          farthest = row;
        }
      }
      if (farthest >= 0) {
        list.push({
          kind: 'note',
          coordinate: labelPointOf(farthest),
          title: liveText('{distance:distance} away', {distance: farthestKilometers * 1000}),
          text: `${nameOf(farthest)}, the farthest match`,
          priority: 6
        });
      }
    }
    if (options.direction === 'least') {
      // The three least similar counties, each with the attribute that separates it most.
      const profile = getReferenceProfile(standardized[options.standardization], references);
      latest.topIds.slice(0, 3).forEach(row => {
        const top = getContributions(
          standardized[options.standardization],
          profile,
          getWeightValues(),
          row
        )[0];
        if (!top) return;
        const attribute = SIMILARITY_ATTRIBUTES[top.attribute];
        list.push({
          kind: 'note',
          coordinate: labelPointOf(row),
          title: nameOf(row),
          text: liveText('{label}: {value} against {reference}', {
            label: attribute.label,
            value: formatAttributeValue(displayColumns[top.attribute][row]),
            reference: formatAttributeValue(getReferenceValue(top.attribute))
          }),
          priority: 6
        });
      });
    }
    if (options.outlineSelected && notes.inspected >= 0 && notes.contributions.length) {
      const total = notes.contributions.reduce((sum, term) => sum + term.contribution, 0);
      const top = notes.contributions[0];
      if (total > 0 && !references.includes(notes.inspected)) {
        list.push({
          kind: 'note',
          coordinate: labelPointOf(notes.inspected),
          title: liveText('{share:percent} of the distance', {
            share: top.contribution / total
          }),
          text: `${SIMILARITY_ATTRIBUTES[top.attribute].label} in ${nameOf(notes.inspected)}`,
          priority: 6
        });
      }
    }
    ctx.setAnnotations('similar-places', list.length ? list : null);
  };

  /** Everything derived from one readback: readouts, buffers, charts, annotations, legend data. */
  const publish = () => {
    if (!latest) return;
    const options = ctx.options;
    const weights = getWeightValues();
    const matchRows = latest.topIds;
    const matchCount = matchRows.length;
    const referenceSet = new Set(references);
    const excluded = options.excludeReference ? referenceSet : new Set<number>();
    const mode = options.standardization;
    const zValues = standardized[mode];
    const profile = getReferenceProfile(zValues, references);

    // CPU twin: it must agree with the GPU, and the equal-weight z-score ranking is the baseline.
    const cpu = rankRows(zValues, profile, weights, {direction: options.direction, excluded});
    const cpuTop = Array.from(cpu.order.slice(0, matchCount));
    const baseline = rankRows(
      standardized.zscore,
      getReferenceProfile(standardized.zscore, references),
      getWeights(ATTRIBUTE_COUNT),
      {direction: options.direction, excluded}
    );
    const baselineTop = Array.from(baseline.order.slice(0, matchCount));
    const persist = countShared(matchRows, baselineTop);

    // Geography of the matches.
    const kilometersOf = (row: number): number =>
      Math.min(
        ...references.map(reference => getKilometers(labelPointOf(reference), labelPointOf(row)))
      );
    const matchKilometers = matchRows.map(kilometersOf);
    const medianKilometers = getMedian(matchKilometers);
    const neighbours = getNearestRows(labelPoints, references, Math.max(matchCount, 1));
    const neighbourRanks = neighbours
      .map(({row}) => getRankOf(row))
      .filter(rank => Number.isFinite(rank));
    const neighbourMatches = countShared(
      neighbours.map(({row}) => row),
      matchRows
    );

    // Readouts.
    const ranked = getSortedFinite(latest.distances);
    ctx.setReadout('reference', getReferenceName());
    ctx.setReadout('ranked', latest.rankedCount);
    ctx.setReadout(
      'matchList',
      matchRows
        .map(
          (row, index) =>
            `${index + 1}. ${nameOf(row)}, ${formatDistance(kilometersOf(row) * 1000)}`
        )
        .join('\n') || '-'
    );
    ctx.setReadout('medianKm', matchCount ? formatDistance(medianKilometers * 1000) : '-');
    ctx.setReadout(
      'neighbourKm',
      neighbours.length
        ? formatDistance(getMedian(neighbours.map(entry => entry.kilometers)) * 1000)
        : '-'
    );
    ctx.setReadout(
      'neighbourRank',
      neighbourRanks.length ? formatCount(Math.round(getMedian(neighbourRanks))) : '-'
    );
    ctx.setReadout('neighbourHits', `${neighbourMatches} of ${neighbours.length}`);
    ctx.setReadout(
      'neighbourList',
      neighbours
        .slice(0, NEIGHBOUR_LABELS)
        .map(({row, kilometers}) => {
          const rank = getRankOf(row);
          return `${nameOf(row)}, ${formatDistance(kilometers * 1000)}: rank ${Number.isFinite(rank) ? formatCount(rank) : 'none'}`;
        })
        .join('\n') || '-'
    );
    ctx.setReadout('persist', `${persist} of ${matchCount}`);
    ctx.setReadout('topDistance', matchCount ? latest.distances[matchRows[0]].toFixed(2) : '-');
    ctx.setReadout(
      'farSouth',
      `${matchRows.filter(row => SOUTH_STATES.has(stateOf(row))).length} of ${matchCount}`
    );
    const strongest = getStrongestCorrelation(standardized.zscore, weights);
    ctx.setReadout(
      'strongestPair',
      strongest
        ? `${SIMILARITY_ATTRIBUTES[strongest.a].label} and ${SIMILARITY_ATTRIBUTES[strongest.b].label.toLowerCase()}, r = ${strongest.r.toFixed(2)}`
        : '-'
    );
    ctx.setReadout(
      'cpuCheck',
      `${countShared(matchRows, cpuTop)} of ${matchCount} matches agree with the CPU ranking`
    );
    ctx.setReadout(
      'spread',
      `nearest ${getQuantile(ranked, 0).toFixed(2)}, median ${getQuantile(ranked, 0.5).toFixed(2)}, farthest ${getQuantile(ranked, 1).toFixed(2)}`
    );

    // Buffers for the matches, the arcs and the inspected county.
    matchOutlineCount = matchOutline.write(
      getOutlineSegments(geometry, matchRows.slice(0, MAXIMUM_MARKED))
    );
    arcCount =
      references.length === 1 && matchCount
        ? arcs.write(
            getArcSegments(
              geometry,
              labelPointOf(references[0]),
              matchRows.slice(0, MAXIMUM_MARKED).map(labelPointOf)
            )
          )
        : 0;
    const inspected = getInspectedRow();
    selectionCount = inspected >= 0 ? geometry.setSelection(inspected) : 0;

    const contributions =
      inspected >= 0 && !referenceSet.has(inspected)
        ? getContributions(zValues, profile, weights, inspected)
        : [];
    publishLegendData();
    publishProfile(contributions, inspected);
    publishAnnotations({contributions, inspected, medianKilometers, neighbours});
    ctx.requestLayers();
  };

  const reader = new SummaryReader(
    resources,
    'similar-places',
    [
      {buffer: ranksBuffer, size: n * 4},
      {buffer: distancesBuffer, size: n * 4},
      {buffer: topIdsBuffer, size: MAXIMUM_RESULT_COUNT * 4},
      {buffer: countBuffer, size: 4}
    ],
    bytes => {
      const read = createByteReader(bytes);
      const ranks = read.words(n).slice();
      const distances = read.floats(n).slice();
      const topIdsAll = read.words(MAXIMUM_RESULT_COUNT);
      const count = read.words(1)[0];
      let rankedCount = 0;
      for (let row = 0; row < n; row++) if (ranks[row] !== NO_RANK) rankedCount++;
      latest = {
        ranks,
        distances,
        topIds: Array.from(topIdsAll.subarray(0, Math.min(count, MAXIMUM_RESULT_COUNT))),
        rankedCount
      };
      publish();
    }
  );

  ctx.setFurniture({
    title: {
      sample: `${formatCount(n)} counties of the contiguous US (Alaska and Hawaii not in the data)`
    }
  });
  ctx.setCost({records: n});
  writeSelection();
  writeParameters();

  // ---------------------------------------------------------------------------------------------
  // Layers
  // ---------------------------------------------------------------------------------------------

  const getLayers = (): Layer[] => {
    const options = ctx.options;
    const ground = ctx.ground();
    const ink = hexToRgba(MAP_INK[ground].ink);
    const halo = hexToRgba(MAP_INK[ground].halo);
    const origin: [number, number, number] = [geometry.origin[0], geometry.origin[1], 0];
    const layers: Layer[] = [
      new SpatialAnalysisPolygonLayer({
        id: 'similar-fill',
        coordinateOrigin: origin,
        triangles: geometry.buffers.triangles,
        features: geometry.buffers.triangleFeatures,
        vertexCount: geometry.buffers.vertexCount,
        // Opaque on the paper sheet; the class alphas carry the "rest" fade themselves.
        opacity: 1,
        values: classBuffer,
        valueFormat: 'uint32',
        colormap: 'category',
        palette: table.colors.map(color => [color[0], color[1], color[2], color[3] ?? 255]),
        noDataValue: 0xffffffff,
        noDataColor: NO_DATA_COLOR[ground],
        highlightClasses: legendHighlight
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'similar-county-hairlines',
        coordinateOrigin: origin,
        segments: geometry.buffers.outline,
        instanceCount: geometry.buffers.outlineCount,
        // Tier 3: a thin hairline over thousands of polygons, never a dark mesh.
        widthPixels: n > 2000 ? 0.4 : 0.5,
        color: getHairlineColor(ground)
      })
    ];
    if (geometry.stateLines) {
      // The zone-boundary tier, always on in county steps.
      const line = getStateLineStyle(ground);
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'similar-state-lines',
          coordinateOrigin: origin,
          segments: geometry.stateLines.buffer,
          instanceCount: geometry.stateLines.segmentCount,
          widthPixels: line.widthPixels,
          color: line.color,
          outlineColor: line.casing,
          outlineWidthPixels: (line.casingPixels - line.widthPixels) / 2
        })
      );
    }
    if (matchOutlineCount > 0 && options.showMatches) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'similar-match-outlines',
          coordinateOrigin: origin,
          segments: matchOutline.buffer,
          instanceCount: matchOutlineCount,
          widthPixels: 1.2,
          color: ink
        })
      );
    }
    if (arcCount > 0 && options.showArcs) {
      // Context tier: a thin dashed ink line at 0.45, under the markers.
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'similar-arcs',
          coordinateOrigin: origin,
          segments: arcs.buffer,
          instanceCount: arcCount,
          widthPixels: 1,
          color: [ink[0], ink[1], ink[2], 115],
          dashArray: [5, 4]
        })
      );
    }
    if (selectionCount > 0 && (options.outlineSelected || inspectedClick >= 0)) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'similar-inspected',
          coordinateOrigin: origin,
          segments: geometry.selectionBuffer,
          instanceCount: selectionCount,
          widthPixels: 2.5,
          color: ink,
          outlineColor: halo,
          outlineWidthPixels: 1
        })
      );
    }
    if (referenceOutlineCount > 0) {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'similar-reference',
          coordinateOrigin: origin,
          segments: referenceOutline.buffer,
          instanceCount: referenceOutlineCount,
          widthPixels: 2.5,
          color: ink,
          outlineColor: halo,
          outlineWidthPixels: 1
        })
      );
    }
    return layers;
  };

  // ---------------------------------------------------------------------------------------------
  // Tooltip
  // ---------------------------------------------------------------------------------------------

  const describeCounty = (row: number): TooltipContent | null => {
    if (!latest) return null;
    const options = ctx.options;
    const rings = geometry.getRings(row);
    const anchor = labelPointOf(row);
    const isReference = references.includes(row);
    const zValues = standardized[options.standardization];
    const profile = getReferenceProfile(zValues, references);
    const rows: TooltipRow[] = [];
    if (isReference) {
      const strongest = [...Array(ATTRIBUTE_COUNT).keys()]
        .filter(k => Number(options[`weight${k}` as `weight${number}`] ?? 1) > 0)
        .sort((a, b) => Math.abs(profile[b]) - Math.abs(profile[a]))
        .slice(0, 3);
      for (const k of strongest) {
        rows.push({
          label: SIMILARITY_ATTRIBUTES[k].label,
          value: formatAttributeValue(displayColumns[k][row]),
          unit: SIMILARITY_ATTRIBUTES[k].unit
        });
      }
      return {
        title: nameOf(row),
        subtitle: 'Reference county',
        rows,
        anchor,
        highlight: {kind: 'polygon', rings}
      };
    }
    const rank = getRankOf(row);
    if (!Number.isFinite(rank)) {
      return {
        title: nameOf(row),
        subtitle: 'County',
        note: 'Not ranked: an attribute is missing',
        anchor,
        highlight: {kind: 'polygon', rings}
      };
    }
    const classIndex = getRankClass(rank - 1, options.resultCount);
    const swatch = table.colors[classIndex];
    const kilometers = Math.min(
      ...references.map(reference => getKilometers(labelPointOf(reference), anchor))
    );
    rows.push(
      {
        label: options.direction === 'most' ? 'Similarity rank' : 'Rank from the least similar',
        value: formatCount(rank),
        unit: `of ${formatCount(latest.rankedCount)}`,
        swatch: [swatch[0], swatch[1], swatch[2], swatch[3] ?? 255],
        emphasis: true
      },
      {
        label: 'Distance in attribute space',
        value: latest.distances[row].toFixed(2),
        unit: options.standardization === 'zscore' ? 'standard deviations' : 'percentile units'
      },
      {label: 'From the reference', value: formatDistance(kilometers * 1000)}
    );
    const weights = getWeightValues();
    for (const term of getContributions(zValues, profile, weights, row).slice(0, 3)) {
      const attribute = SIMILARITY_ATTRIBUTES[term.attribute];
      rows.push({
        label: attribute.label,
        value: formatAttributeValue(displayColumns[term.attribute][row]),
        unit: `${attribute.unit}; reference ${formatAttributeValue(getReferenceValue(term.attribute))}`
      });
    }
    return {
      title: nameOf(row),
      subtitle: `County, ${formatOrdinal(rank)} in the ranking`,
      rows,
      anchor,
      highlight: {kind: 'polygon', rings}
    };
  };

  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.compiled),

    setOption(id) {
      if (id === 'standardization') {
        active = getVariant(ctx.options.standardization);
        dirty = true;
        reader.markStale();
      } else if (id === 'preset') {
        setPreset();
        inspectedClick = -1;
        writeSelection();
        writeParameters();
      } else if (id === 'click') {
        // A click mode is only a pointer behaviour: nothing to recompute.
      } else if (
        id === 'showMatches' ||
        id === 'showArcs' ||
        id === 'showNeighbours' ||
        id === 'outlineSelected'
      ) {
        publish();
      } else {
        writeParameters();
      }
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 2) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (frame.frameIndex >= 2) reader.flush(commandEncoder);
    },

    getLayers,

    onGroundChange() {
      publishLegendData();
      ctx.requestLayers();
    },

    onLegendFilter(_id, classes) {
      legendHighlight = classes ? [...classes] : null;
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const found = geometry.locator.find(event.coordinate);
      return found ? describeCounty(found.index) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const found = geometry.locator.find(event.coordinate);
      if (!found) return false;
      const row = found.index;
      const mode = ctx.options.click;
      if (mode === 'inspect') {
        inspectedClick = row === inspectedClick ? -1 : row;
        publish();
        return true;
      }
      const existing = references.indexOf(row);
      if (mode === 'add' && existing >= 0) {
        if (references.length > 1) references.splice(existing, 1);
      } else if (mode === 'add') {
        if (references.length < MAXIMUM_REFERENCES) references.push(row);
      } else {
        references.length = 0;
        references.push(row);
      }
      inspectedClick = -1;
      ctx.setOptions({preset: 'custom'});
      writeSelection();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      reader.stop();
      ctx.setAnnotations('similar-places', null);
      ctx.setChart('profile', null);
      ctx.setChart('contribution', null);
      resources.destroy();
    }
  };
}
