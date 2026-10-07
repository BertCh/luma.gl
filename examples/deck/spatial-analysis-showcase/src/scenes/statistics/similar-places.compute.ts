// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  getGPUSimilarLocationsParameterLength,
  getGPUSimilarLocationsParameterValues,
  GPU_SIMILAR_LOCATIONS_NO_RANK,
  GPUSimilarLocations,
  type GPUSimilarLocationsStandardization
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createByteReader,
  createChoroplethGeometry,
  formatNumber,
  getOutlineColor,
  getQuantile,
  getSortedFinite
} from './b5-common';
import {clearLegendData, formatCompact} from './b5-legend-bus';
import {REFERENCE_PRESETS, SIMILARITY_ATTRIBUTES} from './b5-variables';

/** Option state of the similar-places scene. Weights are `weight0` to `weight11`. */
export type SimilarPlacesOptions = {
  preset: string;
  standardization: GPUSimilarLocationsStandardization;
  direction: 'most' | 'least';
  resultCount: number;
  excludeReference: boolean;
  falloff: number;
  showMatches: boolean;
  multiSelect: boolean;
  outlines: boolean;
} & Record<`weight${number}`, number>;

const ATTRIBUTE_COUNT = SIMILARITY_ATTRIBUTES.length;
const MAXIMUM_RESULT_COUNT = 32;
const MAXIMUM_REFERENCES = 6;

type Variant = {compiled: CompiledGPUCommandGraph<void>};

/**
 * Feature-space search over US counties: click counties to set the reference, and
 * `GPUSimilarLocations` ranks every other county by weighted Euclidean distance in standardised
 * attribute space (the mean of the references when several are clicked). Weights, the result count,
 * the direction and the reference set are buffer writes; the standardisation is compile-time, so
 * the rank graph compiles on demand.
 */
export async function createSimilarPlaces(
  ctx: SceneContext<SimilarPlacesOptions>
): Promise<SceneInstance<SimilarPlacesOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'similar-places');
  const geometry = createChoroplethGeometry(resources, counties);
  const n = geometry.featureCount;
  const features = counties.geojson?.features ?? [];
  const nameOf = (row: number): string => {
    const properties = features[row]?.properties;
    return properties ? `${properties.name}, ${properties.state}` : `County ${row}`;
  };
  const findCounty = (name: string, state: string): number =>
    features.findIndex(feature =>
      feature.properties?.name === name || feature.properties?.name === `${name} County`
        ? feature.properties?.state === state
        : false
    );
  const centroids = counties.column<Float32Array>('centroid');

  const rawColumns = SIMILARITY_ATTRIBUTES.map(attribute => {
    const source = counties.column<Float32Array>(attribute.id);
    return attribute.transform === 'log10'
      ? Float32Array.from(source, value => Math.log10(Math.max(value, 1e-3)))
      : source;
  });
  const attributeMatrix = new Float32Array(n * ATTRIBUTE_COUNT);
  for (let row = 0; row < n; row++) {
    for (let k = 0; k < ATTRIBUTE_COUNT; k++) {
      attributeMatrix[row * ATTRIBUTE_COUNT + k] = rawColumns[k][row];
    }
  }
  const displayColumns = SIMILARITY_ATTRIBUTES.map(attribute =>
    counties.column<Float32Array>(attribute.id)
  );

  const attributesBuffer = resources.createBuffer('attributes', attributeMatrix);
  const selectionBuffer = resources.createBuffer('selection', n * 4);
  const ranksBuffer = resources.createBuffer('ranks', n * 4);
  const distancesBuffer = resources.createBuffer('distances', n * 4);
  const topIdsBuffer = resources.createBuffer('top-ids', MAXIMUM_RESULT_COUNT * 4);
  const countBuffer = resources.createBuffer('count', 4);
  const similarityBuffer = resources.createBuffer('similarity', n * 4);
  const referenceBuffer = resources.createBuffer('reference-positions', MAXIMUM_REFERENCES * 8);
  const centroidBuffer = resources.createBuffer('centroids', centroids);
  const parameterLength = getGPUSimilarLocationsParameterLength(ATTRIBUTE_COUNT);
  const similarityParameters = resources.createParameterBuffer(
    'similarity-parameters',
    'float32',
    parameterLength
  );
  const displayParameters = resources.createParameterBuffer('display-parameters', 'float32', 4);

  const compileVariant = (standardization: GPUSimilarLocationsStandardization): Variant => {
    const graph = new GPUCommandGraph<void>(device, {id: `similar-${standardization}`});
    const ranks = importGraphBuffer(graph, 'ranks', ranksBuffer, 'uint32', n);
    const distances = importGraphBuffer(graph, 'distances', distancesBuffer, 'float32', n);
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
        selection: importGraphBuffer(graph, 'selection', selectionBuffer, 'uint32', n),
        parameters: similarityParameters.importToGraph(graph),
        standardization,
        maximumResultCount: MAXIMUM_RESULT_COUNT,
        output: {
          ranks,
          distances,
          topIds: importGraphBuffer(graph, 'top-ids', topIdsBuffer, 'uint32', MAXIMUM_RESULT_COUNT),
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1)
        }
      })
    );
    // Similarity in [0, 1] for the layer: exp(-distance / falloff distance); unranked rows are NaN.
    addKernelPass(graph, {
      id: 'similarity-display',
      invocationCount: n,
      bindings: [
        {
          name: 'parameters',
          view: displayParameters.importToGraph(graph),
          type: 'f32',
          access: 'read'
        },
        {name: 'distances', view: distances, type: 'f32', access: 'read'},
        {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
        {
          name: 'similarity',
          view: importGraphBuffer(graph, 'similarity', similarityBuffer, 'float32', n),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let rank = ranks[ranksOffset + index];
  // Keeps the NaN pattern a runtime value because WGSL rejects a constant NaN.
  var value = bitcast<f32>(0x7fc00000u | (rank >> 31u));
  if (rank != ${GPU_SIMILAR_LOCATIONS_NO_RANK}u) {
    value = exp(-distances[distancesOffset + index] / max(parameters[parametersOffset], 1e-6));
  }
  similarity[similarityOffset + index] = value;`
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

  const references: number[] = [];
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

  let dirty = true;
  let latest: {ranks: Uint32Array; distances: Float32Array; topIds: number[]} | null = null;
  let falloffDistance = 3;
  let matchCount = 0;

  const writeSelection = () => {
    const selection = new Uint32Array(n);
    for (const row of references) selection[row] = 1;
    selectionBuffer.write(selection);
    const positions = new Float32Array(MAXIMUM_REFERENCES * 2);
    references.slice(0, MAXIMUM_REFERENCES).forEach((row, index) => {
      positions[index * 2] = centroids[row * 2];
      positions[index * 2 + 1] = centroids[row * 2 + 1];
    });
    referenceBuffer.write(positions);
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
          weights: Array.from({length: ATTRIBUTE_COUNT}, (_, k) =>
            Number(options[`weight${k}` as `weight${number}`] ?? 1)
          )
        },
        ATTRIBUTE_COUNT
      )
    );
    displayParameters.write(Float32Array.of(falloffDistance, 0, 0, 0));
    dirty = true;
    reader.markStale();
  };

  const referenceMean = SIMILARITY_ATTRIBUTES.map(() => 0);
  const updateReferenceMean = () => {
    SIMILARITY_ATTRIBUTES.forEach((_, k) => {
      referenceMean[k] = references.length
        ? references.reduce((sum, row) => sum + displayColumns[k][row], 0) / references.length
        : Number.NaN;
    });
  };

  const describeRow = (row: number): string => {
    const lines = [nameOf(row)];
    if (latest) {
      const rank = latest.ranks[row];
      if (references.includes(row)) lines.push('Reference county');
      else if (rank !== GPU_SIMILAR_LOCATIONS_NO_RANK)
        lines.push(
          `Rank ${formatNumber(rank + 1)} of ${formatNumber(n - references.length)}, distance ${latest.distances[row].toFixed(2)}`
        );
    }
    SIMILARITY_ATTRIBUTES.forEach((attribute, k) => {
      const weight = Number(ctx.options[`weight${k}` as `weight${number}`] ?? 1);
      if (weight <= 0) return;
      lines.push(
        `${attribute.label}: ${formatCompact(displayColumns[k][row])} (reference ${formatCompact(referenceMean[k])}) ${attribute.unit}`
      );
    });
    return lines.join('\n');
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
      const topIds = Array.from(topIdsAll.subarray(0, count));
      matchCount = count;
      latest = {ranks, distances, topIds};
      updateReferenceMean();
      // Falloff: distance at the chosen rank, so that many counties get a visible similarity.
      const ranked = getSortedFinite(distances);
      const target = Math.min(ctx.options.falloff, Math.max(ranked.length - 1, 0));
      const newFalloff = Math.max(ranked[target] ?? 3, 1e-3);
      if (Math.abs(newFalloff - falloffDistance) > 1e-4) {
        falloffDistance = newFalloff;
        displayParameters.write(Float32Array.of(falloffDistance, 0, 0, 0));
        dirty = true;
        reader.markStale();
      }
      ctx.setReadout(
        'reference',
        references.length
          ? references.map(nameOf).join(' + ')
          : 'click a county to choose a reference'
      );
      ctx.setReadout(
        'matches',
        topIds
          .slice(0, 8)
          .map(row => `${nameOf(row)} (${distances[row].toFixed(2)})`)
          .join('; ') || '-'
      );
      ctx.setReadout(
        'distances',
        `nearest ${formatCompact(getQuantile(ranked, 0))}, 10th ${formatCompact(ranked[Math.min(9, ranked.length - 1)] ?? 0)}, 100th ${formatCompact(ranked[Math.min(99, ranked.length - 1)] ?? 0)}, median ${formatCompact(getQuantile(ranked, 0.5))}, farthest ${formatCompact(getQuantile(ranked, 1))}`
      );
      ctx.setReadout(
        'ranked',
        `${formatNumber(ranked.length)} counties ranked, ${matchCount} listed`
      );
      ctx.requestLayers();
    }
  );

  writeSelection();
  writeParameters();

  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.compiled),

    setOption(id) {
      if (id === 'standardization') {
        active = getVariant(ctx.options.standardization);
        dirty = true;
        reader.markStale();
      } else if (id === 'preset') {
        setPreset();
        writeSelection();
        writeParameters();
      } else if (id === 'falloff') {
        reader.markStale();
        writeParameters();
      } else if (id === 'showMatches' || id === 'outlines' || id === 'multiSelect') {
        ctx.requestLayers();
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

    getLayers() {
      const {showMatches, outlines} = ctx.options;
      const layers: Layer[] = [
        geometry.createFillLayer('similarity-fill', {
          values: similarityBuffer,
          mode: 'ramp',
          ramp: 'viridis',
          valueRange: [0, 1],
          noDataColor: [128, 128, 128, 90],
          fillOpacity: 0.9
        }),
        geometry.createOutlineLayer(
          'similarity-outline',
          getOutlineColor(ctx.theme(), outlines ? 110 : 30),
          outlines ? 0.9 : 0.4
        )
      ];
      if (showMatches && matchCount > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'similarity-matches-halo',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: centroidBuffer,
            ids: topIdsBuffer,
            instanceCount: matchCount,
            radiusPixels: 5.5,
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'similarity-matches',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: centroidBuffer,
            ids: topIdsBuffer,
            instanceCount: matchCount,
            radiusPixels: 3.2,
            color: [220, 40, 100, 255]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'similarity-reference-halo',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          positions: referenceBuffer,
          instanceCount: Math.min(references.length, MAXIMUM_REFERENCES),
          radiusPixels: 10,
          color: [255, 255, 255, 255]
        }),
        new SpatialAnalysisPointLayer({
          id: 'similarity-reference',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          positions: referenceBuffer,
          instanceCount: Math.min(references.length, MAXIMUM_REFERENCES),
          radiusPixels: 6.5,
          color: [20, 24, 40, 255]
        })
      );
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      return row >= 0 ? describeRow(row) : null;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const row = geometry.locator.locate(event.coordinate[0], event.coordinate[1]);
      if (row < 0) return false;
      const existing = references.indexOf(row);
      if (existing >= 0) {
        if (references.length > 1) references.splice(existing, 1);
      } else if (ctx.options.multiSelect) {
        if (references.length < MAXIMUM_REFERENCES) references.push(row);
      } else {
        references.length = 0;
        references.push(row);
      }
      writeSelection();
      ctx.requestLayers();
      return true;
    },

    destroy() {
      reader.stop();
      clearLegendData('similar-places');
      resources.destroy();
    }
  };
}
