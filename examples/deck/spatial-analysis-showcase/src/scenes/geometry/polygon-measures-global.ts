// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import {
  getGPUAuthalicSphereRadius,
  getGPULonLatPolygonPreparationParameterValues,
  GPUGeometryMeasures,
  GPULonLatPolygonPreparation
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {addKernelPass} from '../../engine/mode-kernels';
import type {SceneContext, SceneInstance} from '../scene';
import {createGraphImporter} from './b3-common';
import {FeatureTriangleLayer, PairSegmentLayer} from './b3-layers';
import type {PolygonMeasuresOptions} from './polygon-measures.compute';

const SPATIAL_CONTEXT = {
  coordinateSpace: 'longitude-latitude',
  datum: 'EPSG:4326',
  metric: 'great-circle',
  units: 'meters',
  edgeModel: 'linear'
} as const;
const FEATURE_NAMES = ['Fiji box', 'Antarctica band', 'Arctic ring'];
const FEATURE_COUNT = 3;
// No densification for linear edges. This exceeds even 8 * (79 input vertices + 3 * 3 rings).
const POSITION_CAPACITY = 1024;

function createFixture() {
  // Same short-way shell as FIJI_TRANSMERIDIAN in the library fixtures (no test import).
  const fiji = [177, -19, -178, -19, -178, -16, 177, -16];
  const antarctica: number[] = [];
  for (let longitude = -180; longitude <= 180; longitude += 10) {
    antarctica.push(longitude, -70);
  }
  antarctica.push(180, -90, -180, -90);
  const arctic: number[] = [];
  for (let longitude = -180; longitude < 180; longitude += 10) {
    arctic.push(longitude, 80);
  }
  // Ring roles are positional, not winding-dependent: each polygon's first ring is its shell.
  // Antarctica retains the Natural-Earth pole-edge convention; Arctic closes without a pole edge.
  return {
    positions: Float32Array.from([...fiji, ...antarctica, ...arctic]),
    featureOffsets: new Uint32Array([0, 1, 2, 3]),
    polygonOffsets: new Uint32Array([0, 1, 2, 3]),
    ringOffsets: new Uint32Array([0, 4, 43, 79])
  };
}

function getKnownAreas(): number[] {
  // Display-only binary64 references, independent of fixture data. GPUGeometryMeasures uses
  // getGPUAuthalicSphereRadius (WGS84 authalic radius, ~6371007.18 m) for sphere-equal-area
  // with linear edges; getGPUSphereRadius is the mean radius used for spherical lengths.
  const radius = getGPUAuthalicSphereRadius(SPATIAL_CONTEXT);
  const radiansPerDegree = Math.PI / 180;
  const sphereAreaFactor = 2 * Math.PI * radius ** 2;
  return [
    radius ** 2 *
      5 *
      radiansPerDegree *
      Math.abs(Math.sin(-16 * radiansPerDegree) - Math.sin(-19 * radiansPerDegree)),
    sphereAreaFactor * (1 - Math.sin(70 * radiansPerDegree)),
    sphereAreaFactor * (1 - Math.sin(80 * radiansPerDegree))
  ];
}

/** Builds the inline global fixture's raw/prepared area comparison and GPU display geometry. */
export async function createGlobalPolygonMeasures(
  context: SceneContext<PolygonMeasuresOptions>
): Promise<SceneInstance<PolygonMeasuresOptions>> {
  const resources = new SpatialAnalysisResources(context.device, 'polygon-measures-global');
  try {
    const fixture = createFixture();
    const knownAreas = getKnownAreas();
    const input = {
      positions: resources.createBuffer('positions', fixture.positions),
      featureOffsets: resources.createBuffer('features', fixture.featureOffsets),
      polygonOffsets: resources.createBuffer('polygons', fixture.polygonOffsets),
      ringOffsets: resources.createBuffer('rings', fixture.ringOffsets)
    };
    const rawAreas = resources.createBuffer('raw-areas', FEATURE_COUNT * 4);
    const preparedAreas = resources.createBuffer('prepared-areas', FEATURE_COUNT * 4);
    const relativeErrors = resources.createBuffer('relative-errors', FEATURE_COUNT * 4);
    // Split the display-only binary64 constants so GPU comparison retains their low words.
    const referenceParts = Float32Array.from(
      knownAreas.flatMap(area => {
        const high = Math.fround(area);
        return [high, area - high];
      })
    );
    const referenceAreas = resources.createBuffer('reference-areas', referenceParts);
    const rawInvalidCount = resources.createBuffer('raw-invalid', 4);
    const preparedInvalidCount = resources.createBuffer('prepared-invalid', 4);
    const preparationInvalidCount = resources.createBuffer('preparation-invalid', 4);
    const overflow = resources.createBuffer('overflow', 4);
    const requiredCount = resources.createBuffer('required-count', 4);
    const ringFlags = resources.createBuffer('ring-flags', FEATURE_COUNT * 4);
    const preparedBuffers = {
      positions: resources.createBuffer('prepared-positions', POSITION_CAPACITY * 8),
      featureOffsets: resources.createBuffer('prepared-features', (FEATURE_COUNT + 1) * 4),
      polygonOffsets: resources.createBuffer('prepared-polygons', (2 * FEATURE_COUNT + 1) * 4),
      ringOffsets: resources.createBuffer('prepared-rings', (2 * FEATURE_COUNT + 1) * 4)
    };
    const corners = resources.createBuffer('corners', POSITION_CAPACITY * 6 * 4);
    const featureRows = resources.createBuffer('feature-rows', POSITION_CAPACITY * 4);
    const starts = resources.createBuffer('starts', POSITION_CAPACITY * 8);
    const ends = resources.createBuffer('ends', POSITION_CAPACITY * 8);
    const importInput = (importView: ReturnType<typeof createGraphImporter>) => ({
      positions: importView(
        'positions',
        input.positions,
        'float32x2',
        fixture.positions.length / 2
      ),
      featureOffsets: importView('features', input.featureOffsets, 'uint32', FEATURE_COUNT + 1),
      polygonOffsets: importView('polygons', input.polygonOffsets, 'uint32', FEATURE_COUNT + 1),
      ringOffsets: importView('rings', input.ringOffsets, 'uint32', FEATURE_COUNT + 1)
    });

    // Graph A measures the raw input, including NaN areas for rejected rings.
    const rawGraph = new GPUCommandGraph<void>(context.device, {id: 'global-raw-measures'});
    const importRaw = createGraphImporter(rawGraph);
    rawGraph.add(
      new GPUGeometryMeasures({
        ...importInput(importRaw),
        spatialContext: SPATIAL_CONTEXT,
        geometryType: 'polygons',
        output: {
          areas: importRaw('areas', rawAreas, 'float32', FEATURE_COUNT),
          status: {invalidCount: importRaw('invalid', rawInvalidCount, 'uint32', 1)}
        }
      })
    );
    const rawCompiled = resources.track(await rawGraph.compileAsync());

    // Graph B prepares before consuming. Topology sizes are F+1, 2P+1, 2R+1, including empty copies.
    const preparedGraph = new GPUCommandGraph<void>(context.device, {
      id: 'global-prepared-measures'
    });
    const importPrepared = createGraphImporter(preparedGraph);
    const prepared = {
      positions: importPrepared(
        'prepared-positions',
        preparedBuffers.positions,
        'float32x2',
        POSITION_CAPACITY
      ),
      featureOffsets: importPrepared(
        'prepared-features',
        preparedBuffers.featureOffsets,
        'uint32',
        FEATURE_COUNT + 1
      ),
      polygonOffsets: importPrepared(
        'prepared-polygons',
        preparedBuffers.polygonOffsets,
        'uint32',
        2 * FEATURE_COUNT + 1
      ),
      ringOffsets: importPrepared(
        'prepared-rings',
        preparedBuffers.ringOffsets,
        'uint32',
        2 * FEATURE_COUNT + 1
      )
    };
    const overflowView = importPrepared('overflow', overflow, 'uint32', 1);
    const parameters = resources.createParameterBuffer(
      'preparation-parameters',
      'float32',
      4,
      getGPULonLatPolygonPreparationParameterValues('linear')
    );
    const preparation = new GPULonLatPolygonPreparation({
      ...importInput(importPrepared),
      spatialContext: SPATIAL_CONTEXT,
      parameters: parameters.importToGraph(preparedGraph),
      output: {
        ...prepared,
        ringFlags: importPrepared('ring-flags', ringFlags, 'uint32', FEATURE_COUNT),
        status: {
          overflow: overflowView,
          requiredCount: importPrepared('required-count', requiredCount, 'uint32', 1),
          invalidCount: importPrepared('preparation-invalid', preparationInvalidCount, 'uint32', 1)
        }
      }
    });
    preparedGraph.add(preparation);
    preparedGraph.add(
      new GPUGeometryMeasures({
        ...prepared,
        spatialContext: {...preparation.spatialContext, metric: 'great-circle', units: 'meters'},
        geometryType: 'polygons',
        output: {
          areas: importPrepared('areas', preparedAreas, 'float32', FEATURE_COUNT),
          status: {
            invalidCount: importPrepared('measures-invalid', preparedInvalidCount, 'uint32', 1)
          }
        }
      })
    );

    addKernelPass(preparedGraph, {
      id: 'global-reference-comparison',
      invocationCount: FEATURE_COUNT,
      bindings: [
        {
          name: 'areas',
          view: importPrepared('areas', preparedAreas, 'float32', FEATURE_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'references',
          view: importPrepared('reference-areas', referenceAreas, 'float32', FEATURE_COUNT * 2),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'errors',
          view: importPrepared('relative-errors', relativeErrors, 'float32', FEATURE_COUNT),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let high = references[referencesOffset + index * 2u];
  let low = references[referencesOffset + index * 2u + 1u];
  errors[errorsOffset + index] = abs((areas[areasOffset + index] - high) - low) / high;`
    });

    // These three fixtures become convex, hole-free rings after preparation, so a triangle fan
    // suffices. The GPU expands the prepared topology; no host geometry readback or triangulation.
    addKernelPass(preparedGraph, {
      id: 'global-display',
      invocationCount: POSITION_CAPACITY,
      bindings: [
        {name: 'positions', view: prepared.positions, type: 'f32', access: 'read'},
        {name: 'rings', view: prepared.ringOffsets, type: 'u32', access: 'read'},
        {name: 'overflow', view: overflowView, type: 'u32', access: 'read'},
        {
          name: 'corners',
          view: importPrepared('corners', corners, 'float32', POSITION_CAPACITY * 6),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'features',
          view: importPrepared('feature-rows', featureRows, 'uint32', POSITION_CAPACITY),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'starts',
          view: importPrepared('starts', starts, 'float32', POSITION_CAPACITY * 2),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'ends',
          view: importPrepared('ends', ends, 'float32', POSITION_CAPACITY * 2),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: /* wgsl */ `
fn displayPosition(vertex: u32) -> vec2<f32> {
  // Web Mercator display clamp only. Measures consume the original prepared latitudes at ±90.
  return vec2<f32>(positions[positionsOffset + vertex * 2u],
    clamp(positions[positionsOffset + vertex * 2u + 1u], -85.0511, 85.0511));
}`,
      body: /* wgsl */ `
  var first = vec2<f32>(bitcast<f32>(0x7fc00000u));
  var start = first;
  var end = first;
  var feature = 0u;
  if (overflow[overflowOffset] == 0u) {
    for (var ring = 0u; ring < 6u; ring++) {
      let begin = rings[ringsOffset + ring];
      let finish = rings[ringsOffset + ring + 1u];
      if (index >= begin && index < finish && finish - begin >= 3u) {
        first = displayPosition(begin);
        start = displayPosition(index);
        var following = index + 1u;
        if (following == finish) { following = begin; }
        end = displayPosition(following);
        // One input polygon and shell per feature: main/east copies retain this fixed order.
        feature = ring / 2u;
      }
    }
  }
  features[featuresOffset + index] = feature;
  corners[cornersOffset + index * 6u] = first.x;
  corners[cornersOffset + index * 6u + 1u] = first.y;
  corners[cornersOffset + index * 6u + 2u] = start.x;
  corners[cornersOffset + index * 6u + 3u] = start.y;
  corners[cornersOffset + index * 6u + 4u] = end.x;
  corners[cornersOffset + index * 6u + 5u] = end.y;
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`
    });
    const preparedCompiled = resources.track(await preparedGraph.compileAsync());
    const reader = new SummaryReader(
      resources,
      'global-areas',
      [
        {buffer: rawAreas, size: 12},
        {buffer: preparedAreas, size: 12},
        {buffer: relativeErrors, size: 12},
        {buffer: rawInvalidCount, size: 4},
        {buffer: preparationInvalidCount, size: 4},
        {buffer: preparedInvalidCount, size: 4},
        {buffer: overflow, size: 4},
        {buffer: requiredCount, size: 4},
        {buffer: ringFlags, size: 12}
      ],
      bytes => {
        const raw = new Float32Array(bytes, 0, FEATURE_COUNT);
        const measured = new Float32Array(bytes, 12, FEATURE_COUNT);
        const errors = new Float32Array(bytes, 24, FEATURE_COUNT);
        const status = new Uint32Array(bytes, 36, 8);
        const areaText = (area: number) =>
          Number.isFinite(area)
            ? (area / 1e6).toLocaleString('en-US', {maximumFractionDigits: 1})
            : 'rejected';
        for (let feature = 0; feature < FEATURE_COUNT; feature++) {
          const relativeError = errors[feature];
          context.setReadout(
            `globalArea${feature}`,
            [
              `Known km²     ${areaText(knownAreas[feature])}`,
              `Unprepared    ${areaText(raw[feature])}`,
              `Prepared      ${status[3] ? 'overflow' : areaText(measured[feature])}`,
              `Relative error ${!status[3] && Number.isFinite(relativeError) ? `${(relativeError * 100).toFixed(5)}%` : '—'}`
            ].join('\n')
          );
        }
        context.setReadout(
          'globalRejected',
          `${status[0]} features rejected: ambiguous or over-wide rings`
        );
        context.setReadout(
          'globalPreparation',
          [
            `Preparation: ${status[1]} rejected; measures: ${status[2]} rejected`,
            ...FEATURE_NAMES.map((name, feature) => {
              const flags = status[5 + feature];
              return `${name}: ${[flags & 1 ? 'split' : '', flags & 2 ? 'pole-capped' : ''].filter(Boolean).join(', ') || 'kept'}`;
            }),
            `Overflow: ${status[3]}; required: ${status[4]} / ${POSITION_CAPACITY}`
          ].join('\n')
        );
        context.setStatus(status[3] ? `Preparation overflow; requires ${status[4]} positions` : '');
      }
    );
    let dirty = true;
    return {
      getCompiledGraphs: () => [rawCompiled, preparedCompiled],
      encode(encoder) {
        if (dirty) {
          rawCompiled.encode(encoder, {parameters: undefined});
          preparedCompiled.encode(encoder, {parameters: undefined});
          reader.request(encoder);
          dirty = false;
        } else reader.flush(encoder);
      },
      setOption() {
        context.requestLayers();
      },
      onThemeChange() {
        context.requestLayers();
      },
      getLayers() {
        const layers: Layer[] = [
          new FeatureTriangleLayer({
            id: 'global-polygon-fill',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            corners,
            featureRows,
            instanceCount: POSITION_CAPACITY,
            values: preparedAreas,
            valueMapping: 'ramp',
            valueRange: [0, 16e12],
            color: [255, 255, 255, Math.round(context.options.opacity * 255)]
          })
        ];
        if (context.options.showOutlines)
          layers.push(
            new PairSegmentLayer({
              id: 'global-polygon-outline',
              coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
              starts,
              ends,
              instanceCount: POSITION_CAPACITY,
              color: context.theme() === 'dark' ? [235, 240, 248, 220] : [40, 52, 70, 220],
              widthPixels: 1.1
            })
          );
        return layers;
      },
      destroy() {
        reader.stop();
        resources.destroy();
      }
    };
  } catch (error) {
    resources.destroy();
    throw error;
  }
}
