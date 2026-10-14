// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_EARTH_MEAN_RADIUS_METERS,
  type GPUSpatialContext
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {earcut} from '@math.gl/polygon';
import type {LoadedDataset} from '../../data/catalog';
import {fetchJson, getDataFileUrl} from '../../data/loaders';
import {
  ALBERS_CONUS_FRAME,
  type AnalysisFrame,
  createAnalysisFrameProjection
} from '../../engine/analysis-frame';
import {LocalMetricProjection} from '../../engine/projection';
import {projectRows} from '../regression/b6-polygons';
import type {ResolvedDatasets} from '../scene';

/** Geographies of the weights chapter. Both are polygon coverages with exact shared edges. */
export type GeographyId = 'us-counties' | 'chicago-tracts';

/** Variables every geography can map (counties from CDC PLACES / SAIPE, tracts from PLACES / ACS). */
export type VariableId =
  | 'diabetes'
  | 'obesity'
  | 'depression'
  | 'smoking'
  | 'inactivity'
  | 'income'
  | 'poverty'
  | 'uninsured'
  | 'minority';

/** One mappable variable. */
export type VariableInfo = {
  id: VariableId;
  label: string;
  unit: string;
  /** Fraction digits when a value is displayed. */
  digits: number;
  help: string;
};

/** Variable catalogue shared by the three weights scenes. */
export const VARIABLES: readonly VariableInfo[] = [
  {
    id: 'diabetes',
    label: 'Diabetes prevalence',
    unit: '% of adults',
    digits: 1,
    help: 'Diagnosed diabetes, CDC PLACES model-based estimate (age adjusted for counties).'
  },
  {
    id: 'obesity',
    label: 'Obesity prevalence',
    unit: '% of adults',
    digits: 1,
    help: 'Adults with a body mass index of 30 or more, CDC PLACES.'
  },
  {
    id: 'depression',
    label: 'Depression prevalence',
    unit: '% of adults',
    digits: 1,
    help: 'Ever told they have a depressive disorder, CDC PLACES.'
  },
  {
    id: 'smoking',
    label: 'Current smoking',
    unit: '% of adults',
    digits: 1,
    help: 'Adults who currently smoke, CDC PLACES.'
  },
  {
    id: 'inactivity',
    label: 'Physical inactivity',
    unit: '% of adults',
    digits: 1,
    help: 'No leisure-time physical activity, CDC PLACES.'
  },
  {
    id: 'income',
    label: 'Median household income',
    unit: 'USD',
    digits: 0,
    help: 'Counties: Census SAIPE 2022. Tracts: ACS 5-year via Census Reporter.'
  },
  {
    id: 'poverty',
    label: 'Poverty (below 150% of the line)',
    unit: '% of people',
    digits: 1,
    help: 'Share of residents below 150% of the federal poverty line, CDC/ATSDR SVI 2022.'
  },
  {
    id: 'uninsured',
    label: 'Without health insurance',
    unit: '% of people',
    digits: 1,
    help: 'Share of residents with no health insurance, SVI 2022 (ACS).'
  },
  {
    id: 'minority',
    label: 'Minority share',
    unit: '% of people',
    digits: 1,
    help: 'Residents who are not non-Hispanic White, SVI 2022 (ACS).'
  }
];

const COUNTY_COLUMNS: Record<VariableId, string> = {
  diabetes: 'places_diabetes_ageAdj',
  obesity: 'places_obesity_ageAdj',
  depression: 'places_depression_ageAdj',
  smoking: 'places_csmoking_ageAdj',
  inactivity: 'places_lpa_ageAdj',
  income: 'medianHouseholdIncome',
  poverty: 'poverty150',
  uninsured: 'noHealthInsurance',
  minority: 'minorityShare'
};

const TRACT_COLUMNS: Record<Exclude<VariableId, 'minority'>, string> = {
  diabetes: 'diabetes',
  obesity: 'obesity',
  depression: 'depression',
  smoking: 'smoking',
  inactivity: 'noLeisureActivity',
  income: 'medianHouseholdIncome',
  poverty: 'poverty150Pct',
  uninsured: 'uninsuredPct'
};

/** Everything the weights scenes need about one polygon coverage. */
export type Geography = {
  id: GeographyId;
  /** `county` or `tract`. */
  unit: string;
  unitPlural: string;
  count: number;
  /**
   * Origin of the DISPLAY frame (deck.gl `METER_OFFSETS`, Mercator true at the origin). The
   * polygon geometry (`triangles`, `outlineSegments`, `contiguityVertices`, `displayBounds`,
   * `pickMeters`, `project`) is in this frame. Never measure distance in it beyond city scale.
   */
  origin: [number, number];
  /** Dataset bounding box `[west, south, east, north]`. */
  bbox: readonly [number, number, number, number];
  /**
   * The ANALYSIS frame: what distance bands, kernels, correlogram bins and `medianSpacing` are
   * measured in. Counties use Albers equal-area conic (USGS CONUS parallels, distance scale error
   * at most 1.44% over the lower 48); tracts use AEQD about their centre.
   */
  analysisFrame: {
    name: string;
    /** Frame identity and metric contract of the analysis coordinates. */
    spatialContext: GPUSpatialContext;
    /** Largest local distance scale error of the frame over the dataset, as a fraction. */
    maxScaleError: number;
    project: (longitude: number, latitude: number) => [number, number];
    unproject: (x: number, y: number) => [number, number];
    /**
     * The `analysisFrame` layer prop that draws analysis-frame metres (`centroids` and what is
     * built from them, such as weight links) on the map: `ALBERS_CONUS_FRAME` for counties.
     * Tracts use their azimuthal-equidistant frame.
     */
    layerFrame: AnalysisFrame;
  };
  /** Metre bounds of the vertices in the analysis frame `[minX, minY, maxX, maxY]`. */
  bounds: [number, number, number, number];
  /** Metre bounds of the vertices in the display frame (the lattice of spatial-weights). */
  displayBounds: [number, number, number, number];
  /**
   * Centroid of each feature in analysis-frame metres, interleaved. Feeds the operators; draw it
   * with `analysisFrame.layerFrame`.
   */
  centroids: Float32Array;
  /** `[longitude, latitude]` of the centroid of a feature. */
  getCentroidLngLat: (feature: number) => [number, number];
  /** Contiguity layout: vertices in metres with closing duplicates removed. */
  contiguityVertices: Float32Array;
  /** Matching open-ring vertices in the analysis frame, for analysis operators. */
  analysisContiguityVertices: Float32Array;
  contiguityRingOffsets: Uint32Array;
  /** Feature to ring offsets (`count + 1`). */
  featureRingOffsets: Uint32Array;
  /** Feature to polygon part offsets (`count + 1`): a multipart feature owns several parts. */
  featurePolygonOffsets: Uint32Array;
  /** Polygon part to ring offsets (`partCount + 1`); the first ring of a part is its shell. */
  polygonRingOffsets: Uint32Array;
  /** Triangle list of every polygon in metres (3 vertices per triangle). */
  triangles: Float32Array;
  /** Feature of every triangle vertex. */
  triangleFeatures: Uint32Array;
  /** Ring outline segments `x0, y0, x1, y1` in metres. */
  outlineSegments: Float32Array;
  /** Group id of every feature (state for counties, community area for tracts), dense from 0. */
  groups: Uint32Array;
  groupCount: number;
  groupNames: string[];
  /** Population per feature, used by the subgraph mask. */
  population: Float32Array;
  /** Short display name of a feature. */
  getName: (feature: number) => string;
  /** Group label of a feature, for example the state or community area. */
  getGroupName: (feature: number) => string;
  /** Returns the variable column as float32. */
  getVariable: (id: VariableId) => Float32Array;
  /** Index of the feature containing a `[lng, lat]`, or -1. */
  pick: (longitude: number, latitude: number) => number;
  /** Index of the feature containing planar metres, or -1. */
  pickMeters: (x: number, y: number) => number;
  /** Median distance from a centroid to its nearest other centroid, analysis-frame metres. */
  medianSpacing: number;
  /** Projects `[lng, lat]` to display-frame metres around `origin`. */
  project: (longitude: number, latitude: number) => [number, number];
};

const STATE_NAMES: Record<number, string> = {
  1: 'AL',
  2: 'AK',
  4: 'AZ',
  5: 'AR',
  6: 'CA',
  8: 'CO',
  9: 'CT',
  10: 'DE',
  11: 'DC',
  12: 'FL',
  13: 'GA',
  15: 'HI',
  16: 'ID',
  17: 'IL',
  18: 'IN',
  19: 'IA',
  20: 'KS',
  21: 'KY',
  22: 'LA',
  23: 'ME',
  24: 'MD',
  25: 'MA',
  26: 'MI',
  27: 'MN',
  28: 'MS',
  29: 'MO',
  30: 'MT',
  31: 'NE',
  32: 'NV',
  33: 'NH',
  34: 'NJ',
  35: 'NM',
  36: 'NY',
  37: 'NC',
  38: 'ND',
  39: 'OH',
  40: 'OK',
  41: 'OR',
  42: 'PA',
  44: 'RI',
  45: 'SC',
  46: 'SD',
  47: 'TN',
  48: 'TX',
  49: 'UT',
  50: 'VT',
  51: 'VA',
  53: 'WA',
  54: 'WV',
  55: 'WI',
  56: 'WY'
};

type PartLayout = {
  featureCount: number;
  partFeature: Uint32Array;
  partRingOffsets: Uint32Array;
  ringOffsets: Uint32Array;
  featureRingOffsets: Uint32Array;
};

function readPartLayout(dataset: LoadedDataset): PartLayout {
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const partRingOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partCount = partRingOffsets.length - 1;
  let partFeature: Uint32Array;
  let featureCount: number;
  if (dataset.hasColumn('partFeature')) {
    partFeature = dataset.column<Uint32Array>('partFeature');
    featureCount = dataset.count;
  } else {
    // Counties: feature i owns parts [countyPolygonOffsets[i], countyPolygonOffsets[i + 1]).
    const featureOffsets = dataset.column<Uint32Array>('countyPolygonOffsets');
    featureCount = featureOffsets.length - 1;
    partFeature = new Uint32Array(partCount);
    for (let feature = 0; feature < featureCount; feature++) {
      for (let part = featureOffsets[feature]; part < featureOffsets[feature + 1]; part++) {
        partFeature[part] = feature;
      }
    }
  }
  // Parts of one feature are contiguous, so a feature owns a contiguous ring range.
  const featureRingOffsets = new Uint32Array(featureCount + 1);
  let part = 0;
  for (let feature = 0; feature < featureCount; feature++) {
    featureRingOffsets[feature] = partRingOffsets[part];
    while (part < partCount && partFeature[part] === feature) part++;
  }
  featureRingOffsets[featureCount] = partRingOffsets[partCount];
  return {featureCount, partFeature, partRingOffsets, ringOffsets, featureRingOffsets};
}

/** Point in ring (even-odd) test in planar coordinates. */
function isInsideRing(
  vertices: Float32Array,
  start: number,
  end: number,
  x: number,
  y: number
): boolean {
  let inside = false;
  for (let index = start, previous = end - 1; index < end; previous = index++) {
    const xi = vertices[index * 2];
    const yi = vertices[index * 2 + 1];
    const xj = vertices[previous * 2];
    const yj = vertices[previous * 2 + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Area-weighted centroids of the exterior rings of each feature, in the units of `positions`. */
function computeFeatureCentroids(
  positions: Float32Array,
  layout: PartLayout,
  isClosed: (ring: number) => boolean
): Float32Array {
  const {featureCount, partFeature, partRingOffsets, ringOffsets} = layout;
  const centroids = new Float32Array(featureCount * 2);
  const areaSums = new Float64Array(featureCount);
  const centroidSums = new Float64Array(featureCount * 2);
  for (let part = 0; part < partRingOffsets.length - 1; part++) {
    const feature = partFeature[part];
    const ring = partRingOffsets[part];
    const first = ringOffsets[ring];
    const end = ringOffsets[ring + 1] - (isClosed(ring) ? 1 : 0);
    // Relative to the first vertex, so the cross products stay small at continental coordinates.
    const baseX = positions[first * 2];
    const baseY = positions[first * 2 + 1];
    let signedArea = 0;
    let cx = 0;
    let cy = 0;
    for (let vertex = first; vertex < end; vertex++) {
      const next = vertex + 1 < end ? vertex + 1 : first;
      const x0 = positions[vertex * 2] - baseX;
      const y0 = positions[vertex * 2 + 1] - baseY;
      const x1 = positions[next * 2] - baseX;
      const y1 = positions[next * 2 + 1] - baseY;
      const cross = x0 * y1 - x1 * y0;
      signedArea += cross;
      cx += (x0 + x1) * cross;
      cy += (y0 + y1) * cross;
    }
    if (Math.abs(signedArea) > 0) {
      areaSums[feature] += signedArea / 2;
      centroidSums[feature * 2] += cx / 6 + (baseX * signedArea) / 2;
      centroidSums[feature * 2 + 1] += cy / 6 + (baseY * signedArea) / 2;
    }
  }
  for (let feature = 0; feature < featureCount; feature++) {
    const area = areaSums[feature];
    if (Math.abs(area) > 1e-6) {
      centroids[feature * 2] = centroidSums[feature * 2] / area;
      centroids[feature * 2 + 1] = centroidSums[feature * 2 + 1] / area;
    }
  }
  return centroids;
}

/**
 * Loads the polygon coverage of a weights scene: metres around the dataset origin, the contiguity
 * layout, a triangle mesh for the GPU choropleth, outline segments, groups and a CPU picker.
 */
export async function loadGeography(
  id: GeographyId,
  datasets: ResolvedDatasets,
  signal: AbortSignal
): Promise<Geography> {
  const dataset = datasets.get(id);
  const isCounty = id === 'us-counties';
  const origin = dataset.defaultOrigin;
  const meters = dataset.projectColumn('vertices', origin);
  const layout = readPartLayout(dataset);
  const {featureCount, partFeature, partRingOffsets, ringOffsets, featureRingOffsets} = layout;
  const partCount = partRingOffsets.length - 1;
  const ringCount = ringOffsets.length - 1;

  // Parts of one feature are contiguous, so a feature owns a contiguous part range.
  const featurePolygonOffsets = new Uint32Array(featureCount + 1);
  for (let part = 0; part < partCount; part++) featurePolygonOffsets[partFeature[part] + 1]++;
  for (let feature = 0; feature < featureCount; feature++) {
    featurePolygonOffsets[feature + 1] += featurePolygonOffsets[feature];
  }

  // Metre bounds.
  const bounds: [number, number, number, number] = [Infinity, Infinity, -Infinity, -Infinity];
  for (let index = 0; index < meters.length / 2; index++) {
    bounds[0] = Math.min(bounds[0], meters[index * 2]);
    bounds[1] = Math.min(bounds[1], meters[index * 2 + 1]);
    bounds[2] = Math.max(bounds[2], meters[index * 2]);
    bounds[3] = Math.max(bounds[3], meters[index * 2 + 1]);
  }

  // Contiguity layout: drop each ring's closing duplicate vertex.
  const contiguityRingOffsets = new Uint32Array(ringCount + 1);
  const contiguityKept: number[] = [];
  const isClosed = (ring: number) => {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1] - 1;
    return (
      last > first &&
      meters[first * 2] === meters[last * 2] &&
      meters[first * 2 + 1] === meters[last * 2 + 1]
    );
  };
  for (let ring = 0; ring < ringCount; ring++) {
    const first = ringOffsets[ring];
    const end = ringOffsets[ring + 1] - (isClosed(ring) ? 1 : 0);
    for (let vertex = first; vertex < end; vertex++) contiguityKept.push(vertex);
    contiguityRingOffsets[ring + 1] = contiguityKept.length;
  }
  const contiguityVertices = new Float32Array(contiguityKept.length * 2);
  contiguityKept.forEach((vertex, index) => {
    contiguityVertices[index * 2] = meters[vertex * 2];
    contiguityVertices[index * 2 + 1] = meters[vertex * 2 + 1];
  });

  // Analysis frame. Counties span the continent, where the Mercator-local display frame is off by
  // -14.7%..+20.7%, so the operators get Albers equal-area coordinates instead. The display
  // geometry above and below stays in the display frame (it is drawn with METER_OFFSETS).
  const displayProjection = new LocalMetricProjection(origin);
  // The Albers projection is bit-identical to `projectAlbers` (b6-polygons.ts).
  const albersProjection = createAnalysisFrameProjection(ALBERS_CONUS_FRAME);
  const tractProjection = createAnalysisFrameProjection({kind: 'azimuthal-equidistant', origin});
  const vertices = dataset.column<Float32Array>('vertices');
  let maximumAngularDistance = 0;
  if (!isCounty) {
    for (let vertex = 0; vertex < vertices.length; vertex += 2) {
      const [east, north] = tractProjection.project(vertices[vertex], vertices[vertex + 1]);
      maximumAngularDistance = Math.max(
        maximumAngularDistance,
        Math.hypot(east, north) / GPU_EARTH_MEAN_RADIUS_METERS
      );
    }
  }
  const analysisFrame: Geography['analysisFrame'] = isCounty
    ? {
        name: 'Albers equal-area conic (CONUS)',
        spatialContext: albersProjection.spatialContext,
        maxScaleError: 0.0144,
        project: albersProjection.project,
        unproject: albersProjection.unproject,
        layerFrame: ALBERS_CONUS_FRAME
      }
    : {
        name: 'azimuthal equidistant about the tract centre',
        spatialContext: tractProjection.spatialContext,
        maxScaleError:
          maximumAngularDistance === 0
            ? 0
            : maximumAngularDistance / Math.sin(maximumAngularDistance) - 1,
        project: tractProjection.project,
        unproject: tractProjection.unproject,
        layerFrame: tractProjection.frame
      };
  const analysisMeters = projectRows(vertices, analysisFrame.project);
  const analysisContiguityVertices = new Float32Array(contiguityVertices.length);
  contiguityKept.forEach((vertex, index) => {
    analysisContiguityVertices[index * 2] = analysisMeters[vertex * 2];
    analysisContiguityVertices[index * 2 + 1] = analysisMeters[vertex * 2 + 1];
  });
  const analysisBounds: [number, number, number, number] = [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ];
  for (let index = 0; index < analysisMeters.length / 2; index++) {
    analysisBounds[0] = Math.min(analysisBounds[0], analysisMeters[index * 2]);
    analysisBounds[1] = Math.min(analysisBounds[1], analysisMeters[index * 2 + 1]);
    analysisBounds[2] = Math.max(analysisBounds[2], analysisMeters[index * 2]);
    analysisBounds[3] = Math.max(analysisBounds[3], analysisMeters[index * 2 + 1]);
  }

  // Triangle mesh, outline segments and centroids.
  const triangleVertices: number[] = [];
  const triangleFeatureList: number[] = [];
  const centroids = new Float32Array(featureCount * 2);
  const areaSums = new Float64Array(featureCount);
  const centroidSums = new Float64Array(featureCount * 2);
  const outline: number[] = [];
  for (let part = 0; part < partCount; part++) {
    const feature = partFeature[part];
    const flat: number[] = [];
    const holes: number[] = [];
    for (let ring = partRingOffsets[part]; ring < partRingOffsets[part + 1]; ring++) {
      const first = ringOffsets[ring];
      const end = ringOffsets[ring + 1] - (isClosed(ring) ? 1 : 0);
      if (ring > partRingOffsets[part]) holes.push(flat.length / 2);
      for (let vertex = first; vertex < end; vertex++) {
        flat.push(meters[vertex * 2], meters[vertex * 2 + 1]);
      }
      // Outline segments (closed loop) and area-weighted centroid of the exterior ring.
      let signedArea = 0;
      let cx = 0;
      let cy = 0;
      for (let vertex = first; vertex < end; vertex++) {
        const next = vertex + 1 < end ? vertex + 1 : first;
        const x0 = meters[vertex * 2];
        const y0 = meters[vertex * 2 + 1];
        const x1 = meters[next * 2];
        const y1 = meters[next * 2 + 1];
        outline.push(x0, y0, x1, y1);
        const cross = x0 * y1 - x1 * y0;
        signedArea += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
      }
      // Centroid of the exterior rings only; holes are rare and tiny at this scale.
      if (ring === partRingOffsets[part] && Math.abs(signedArea) > 0) {
        areaSums[feature] += signedArea / 2;
        centroidSums[feature * 2] += cx / 6;
        centroidSums[feature * 2 + 1] += cy / 6;
      }
    }
    const indices = earcut(flat, holes.length ? holes : undefined, 2);
    for (const index of indices) {
      triangleVertices.push(flat[index * 2], flat[index * 2 + 1]);
      triangleFeatureList.push(feature);
    }
  }
  for (let feature = 0; feature < featureCount; feature++) {
    const area = areaSums[feature];
    if (Math.abs(area) > 1e-6) {
      centroids[feature * 2] = centroidSums[feature * 2] / area;
      centroids[feature * 2 + 1] = centroidSums[feature * 2 + 1] / area;
    }
  }
  // Compute centroids in the same frame used by distance operators.
  let analysisCentroids: Float32Array = centroids;
  if (isCounty && dataset.hasColumn('centroid')) {
    // The dataset ships equal-area centroids; use them so county points match the published ones.
    analysisCentroids = projectRows(
      dataset.column<Float32Array>('centroid'),
      analysisFrame.project
    );
  } else {
    analysisCentroids = computeFeatureCentroids(analysisMeters, layout, isClosed);
  }

  // Groups: state for counties, community area for tracts.
  const groups = new Uint32Array(featureCount);
  let groupNames: string[];
  let groupCount: number;
  let getName: (feature: number) => string;
  let getGroupName: (feature: number) => string;
  let population: Float32Array;
  if (isCounty) {
    const stateFips = dataset.column<Uint8Array>('stateFips');
    const stateIndex = new Map<number, number>();
    const stateCodes: number[] = [];
    for (let feature = 0; feature < featureCount; feature++) {
      const code = stateFips[feature];
      if (!stateIndex.has(code)) {
        stateIndex.set(code, stateCodes.length);
        stateCodes.push(code);
      }
      groups[feature] = stateIndex.get(code)!;
    }
    groupNames = stateCodes.map(code => STATE_NAMES[code] ?? String(code));
    groupCount = stateCodes.length;
    let names: {name: string[]; state: string[]} | null = null;
    try {
      names = await fetchJson<{name: string[]; state: string[]}>(
        getDataFileUrl('us-counties', 'names.json'),
        signal
      );
    } catch {
      names = null;
    }
    getName = feature => {
      const name = names?.name[feature] ?? `County ${feature}`;
      return /(County|Parish|Borough|city|City|Region)$/.test(name) ? name : `${name} County`;
    };
    getGroupName = feature => names?.state[feature] ?? groupNames[groups[feature]];
    population = dataset.column<Float32Array>('population');
  } else {
    const communityArea = dataset.column<Uint8Array>('communityArea');
    for (let feature = 0; feature < featureCount; feature++) {
      groups[feature] = communityArea[feature] > 0 ? communityArea[feature] - 1 : 77;
    }
    groupCount = 78;
    let areas: LoadedDataset['geojson'] = null;
    try {
      areas = datasets.get('chicago-community-areas').geojson;
    } catch {
      areas = null;
    }
    groupNames = Array.from({length: 78}, (_, index) => {
      const feature = areas?.features?.[index] as {properties?: {name?: string}} | undefined;
      return index === 77
        ? 'No community area'
        : (feature?.properties?.name ?? `Area ${index + 1}`);
    });
    const tractIds = (
      dataset.geojson?.features as {properties?: {GEOID?: string}}[] | undefined
    )?.map(feature => feature.properties?.GEOID ?? '');
    getName = feature => `Tract ${(tractIds?.[feature] ?? '').slice(5) || feature}`;
    getGroupName = feature => groupNames[groups[feature]];
    population = dataset.column<Float32Array>('population');
  }

  // Per-feature bounding boxes for the picker.
  const featureBounds = new Float32Array(featureCount * 4);
  for (let feature = 0; feature < featureCount; feature++) {
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        minimumX = Math.min(minimumX, meters[vertex * 2]);
        maximumX = Math.max(maximumX, meters[vertex * 2]);
        minimumY = Math.min(minimumY, meters[vertex * 2 + 1]);
        maximumY = Math.max(maximumY, meters[vertex * 2 + 1]);
      }
    }
    featureBounds.set([minimumX, minimumY, maximumX, maximumY], feature * 4);
  }
  const pickMeters = (x: number, y: number): number => {
    for (let feature = 0; feature < featureCount; feature++) {
      if (
        x < featureBounds[feature * 4] ||
        x > featureBounds[feature * 4 + 2] ||
        y < featureBounds[feature * 4 + 1] ||
        y > featureBounds[feature * 4 + 3]
      ) {
        continue;
      }
      // Even-odd over every ring of every part of the feature (holes cancel).
      let inside = false;
      for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
        if (isInsideRing(meters, ringOffsets[ring], ringOffsets[ring + 1], x, y)) inside = !inside;
      }
      if (inside) return feature;
    }
    return -1;
  };
  const pick = (longitude: number, latitude: number): number => {
    const [x, y] = displayProjection.project(longitude, latitude);
    return pickMeters(x, y);
  };

  // Median nearest-centroid distance (brute force over a coarse grid is unnecessary: n <= 3109).
  const nearest = new Float32Array(featureCount);
  for (let a = 0; a < featureCount; a++) {
    let best = Infinity;
    const ax = analysisCentroids[a * 2];
    const ay = analysisCentroids[a * 2 + 1];
    for (let b = 0; b < featureCount; b++) {
      if (a === b) continue;
      const dx = analysisCentroids[b * 2] - ax;
      const dy = analysisCentroids[b * 2 + 1] - ay;
      const squared = dx * dx + dy * dy;
      if (squared < best) best = squared;
    }
    nearest[a] = Math.sqrt(best);
  }
  const medianSpacing = [...nearest].sort((left, right) => left - right)[featureCount >> 1];

  const variableCache = new Map<VariableId, Float32Array>();
  const getVariable = (variable: VariableId): Float32Array => {
    let values = variableCache.get(variable);
    if (!values) {
      if (isCounty) {
        values = dataset.column<Float32Array>(COUNTY_COLUMNS[variable]);
      } else if (variable === 'minority') {
        const white = dataset.column<Float32Array>('nhWhite');
        const total = dataset.column<Float32Array>('population');
        values = Float32Array.from(white, (count, index) =>
          total[index] > 0 ? (100 * (total[index] - count)) / total[index] : Number.NaN
        );
      } else {
        values = dataset.column<Float32Array>(TRACT_COLUMNS[variable]);
      }
      variableCache.set(variable, values);
    }
    return values;
  };

  return {
    id,
    unit: isCounty ? 'county' : 'tract',
    unitPlural: isCounty ? 'counties' : 'tracts',
    count: featureCount,
    origin,
    bbox: dataset.manifest.bbox,
    analysisFrame,
    bounds: analysisBounds,
    displayBounds: bounds,
    centroids: analysisCentroids,
    getCentroidLngLat: feature =>
      analysisFrame.unproject(analysisCentroids[feature * 2], analysisCentroids[feature * 2 + 1]),
    contiguityVertices,
    analysisContiguityVertices,
    contiguityRingOffsets,
    featureRingOffsets,
    featurePolygonOffsets,
    polygonRingOffsets: partRingOffsets,
    triangles: Float32Array.from(triangleVertices),
    triangleFeatures: Uint32Array.from(triangleFeatureList),
    outlineSegments: Float32Array.from(outline),
    groups,
    groupCount,
    groupNames,
    population,
    getName,
    getGroupName,
    getVariable,
    pick,
    pickMeters,
    medianSpacing,
    project: (longitude, latitude) => displayProjection.project(longitude, latitude)
  };
}

/** Variable info by id. */
export function getVariableInfo(id: VariableId): VariableInfo {
  return VARIABLES.find(variable => variable.id === id) ?? VARIABLES[0];
}
