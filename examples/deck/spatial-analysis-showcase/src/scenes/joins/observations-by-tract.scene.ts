// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {RAMP_STOPS} from '../../engine/ramps';
import {defineScene} from '../scene';
import type {ObservationsByTractOptions} from './observations-by-tract.compute';

const OBSERVATION_GROUPS = [
  'Plants',
  'Birds',
  'Insects',
  'Fungi',
  'Mammals',
  'Spiders and kin',
  'Amphibians and reptiles',
  'Snails and mussels',
  'Fish',
  'Other life'
] as const;

/** Interpolates the ramp table for categorical legend swatches. */
function sampleRamp(name: keyof typeof RAMP_STOPS, t: number): [number, number, number] {
  const stops = RAMP_STOPS[name];
  const scaled = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const fraction = scaled - index;
  const from = stops[index];
  const to = stops[index + 1];
  return [
    Math.round(from[0] + (to[0] - from[0]) * fraction),
    Math.round(from[1] + (to[1] - from[1]) * fraction),
    Math.round(from[2] + (to[2] - from[2]) * fraction)
  ];
}

const STATISTIC_TITLES: Record<ObservationsByTractOptions['statistic'], string> = {
  count: 'Observations per tract',
  sum: 'Sum of the value per tract',
  mean: 'Mean of the value per tract',
  minimum: 'Minimum of the value per tract',
  maximum: 'Maximum of the value per tract',
  density: 'Observations per tract, normalised'
};

export default defineScene<ObservationsByTractOptions>({
  id: 'observations-by-tract',
  title: 'Chicago nature observations: counts by census tract',
  chapter: 'joins',
  order: 1,
  summary:
    'A GPU point-in-polygon join aggregates 43,557 iNaturalist records into 791 Chicago census tracts as counts, densities or shares. The output reflects observer effort and unstable small denominators.',
  contributors: [
    'GPUPointInPolygonJoin',
    'GPUSpatialJoinPrepared',
    'GPUZonalStatistics',
    'addPointsInPolygonsChoroplethRecipe'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'points to join'},
    {id: 'chicago-tracts', role: 'polygons, residents and areas'},
    {id: 'chicago-community-areas', role: 'names for tooltips'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10},
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Nature observations by tract', subtitle: 'Joined counts, densities and shares'},
    scaleBar: {units: 'metric'},
    credit: 'iNaturalist; U.S. Census Bureau; City of Chicago',
    caveat: 'Observation density measures reporting effort as well as ecological occurrence.'
  },

  options: [
    {
      kind: 'select',
      id: 'statistic',
      label: 'Statistic per tract',
      group: 'Statistic',
      apply: 'compile',
      default: 'count',
      help: 'Which per-tract number GPUZonalStatistics computes. The statistic decides which outputs the graph has, so changing it recompiles; the join work is the same.',
      options: [
        {value: 'count', label: 'Count of observations'},
        {
          value: 'density',
          label: 'Density (per km2 or per 1,000 residents)',
          help: 'counts / area, with the polygon area computed on the GPU or supplied by you.'
        },
        {
          value: 'mean',
          label: 'Mean of a value',
          help: 'sum(v) / valueCount. A mean of a 0/1 flag is a share.'
        },
        {value: 'sum', label: 'Sum of a value'},
        {
          value: 'maximum',
          label: 'Maximum of a value',
          help: 'For a 0/1 flag: 1 when any observation has it.'
        },
        {
          value: 'minimum',
          label: 'Minimum of a value',
          help: 'For a 0/1 flag: 1 when every observation has it.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'valueKind',
      label: 'Value per observation',
      group: 'Statistic',
      apply: 'param',
      default: 'introduced',
      disabledWhen: state => state.statistic === 'count' || state.statistic === 'density',
      help: 'The per-point value column. It is a buffer of 43,557 floats rewritten when you change it; the graph is not rebuilt.',
      options: [
        {value: 'researchGrade', label: 'Research grade, identification confirmed (0/1)'},
        {value: 'introduced', label: 'Introduced, non-native taxon (0/1)'},
        {value: 'animal', label: 'Animal rather than plant or fungus (0/1)'},
        {value: 'weekend', label: 'Made on a Saturday or Sunday (0/1)'},
        {value: 'category', label: 'Is the chosen group (0/1)'}
      ]
    },
    {
      kind: 'select',
      id: 'groupType',
      label: 'Group',
      group: 'Statistic',
      apply: 'param',
      default: 'Birds',
      disabledWhen: state =>
        state.valueKind !== 'category' ||
        state.statistic === 'count' ||
        state.statistic === 'density',
      help: 'Used when the value is "Is the chosen group". Sum gives its count per tract, mean its share of all observations.',
      options: OBSERVATION_GROUPS.map(group => ({value: group, label: group}))
    },
    {
      kind: 'select',
      id: 'denominator',
      label: 'Normalise density by',
      group: 'Statistic',
      apply: 'compile',
      default: 'area',
      disabledWhen: state => state.statistic !== 'density',
      help: 'Area uses the polygon area the GPU computes. Residents passes your own per-tract divisor (population / 1,000) through the `areas` input, which turns density into a per-resident rate; observations follow observers, not residents, so area is the meaningful divisor here.',
      options: [
        {value: 'population', label: 'Residents (per 1,000)'},
        {value: 'area', label: 'Area (per km2, GPU polygon area)'}
      ]
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Sum order',
      group: 'Statistic',
      apply: 'compile',
      default: 'sorted',
      help: 'Sorted points are reduced in a fixed tree order, so sums are reproducible on a device. Atomic float adds are faster to start but their order depends on GPU scheduling.',
      options: [
        {value: 'sorted', label: 'Sorted (reproducible)'},
        {value: 'atomic', label: 'Atomic (scheduling-dependent)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'includeBoundary',
      label: 'Count points on a tract edge',
      group: 'Join',
      apply: 'compile',
      default: true,
      help: 'Observation coordinates are precise, so very few sit exactly on a tract edge. On: they join the lowest-numbered tract whose edge they touch. Off: they join no tract.'
    },
    {
      kind: 'toggle',
      id: 'prepared',
      label: 'Prepared tract index',
      group: 'Join',
      apply: 'compile',
      default: false,
      help: 'GPUSpatialJoinPrepared builds the tract bounds and BVH once and reuses them while the observations are joined again. Off: the index is rebuilt in every join run.'
    },
    {
      kind: 'toggle',
      id: 'spatialSort',
      label: 'Sort tracts along a Hilbert curve',
      group: 'Join',
      apply: 'compile',
      default: true,
      help: 'Reorders features along a space-filling curve before the BVH build. Results are identical; only traversal cost changes. It pays off for thousands of shuffled features.'
    },
    {
      kind: 'toggle',
      id: 'rerunEveryFrame',
      label: 'Re-run the join every frame',
      group: 'Join',
      apply: 'param',
      default: false,
      help: 'By default the graphs run when an option changes. Turn this on to stream 43,557 points through the join at display rate; the readouts count index builds against join runs.'
    },
    {
      kind: 'toggle',
      id: 'invalidateEveryFrame',
      label: 'Invalidate the prepared index every frame',
      group: 'Join',
      apply: 'param',
      default: false,
      disabledWhen: state => !state.prepared,
      help: 'The contrast case: calling `prepared.invalidate()` each frame forces a rebuild, so the prepared index costs what the unprepared one does.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Colour scheme',
      group: 'Display',
      apply: 'compile',
      default: 'ramp',
      help: 'Continuous colours the statistic with the shared ramp over the GPU min and max. Classed runs addPointsInPolygonsChoroplethRecipe: class breaks and class colours are computed on the GPU.',
      options: [
        {value: 'ramp', label: 'Continuous ramp'},
        {value: 'classes', label: 'Classed (choropleth recipe)'}
      ]
    },
    {
      kind: 'select',
      id: 'backend',
      label: 'Recipe back end',
      group: 'Display',
      apply: 'compile',
      default: 'zonal',
      disabledWhen: state => state.display !== 'classes',
      help: 'Zonal aggregates with GPUZonalStatistics. Group runs GPUPointInPolygonJoin and then a dense GPUGroupStatistics. Same numbers, different kernels.',
      options: [
        {value: 'zonal', label: 'GPUZonalStatistics'},
        {value: 'group', label: 'Join + GPUGroupStatistics'}
      ]
    },
    {
      kind: 'select',
      id: 'classMethod',
      label: 'Classification method',
      group: 'Display',
      apply: 'param',
      default: 'quantile',
      disabledWhen: state => state.display !== 'classes',
      help: 'How GPUClassBreaks places the class edges. Quantile gives equal tract counts per class, equal interval equal value ranges, natural breaks (Jenks) the tightest classes.',
      options: [
        {value: 'quantile', label: 'Quantile'},
        {value: 'equal-interval', label: 'Equal interval'},
        {value: 'natural-breaks', label: 'Natural breaks (Jenks)'},
        {value: 'standard-deviation', label: 'Standard deviation'},
        {value: 'head-tail', label: 'Head/tail breaks'},
        {value: 'box-plot', label: 'Box plot (6 classes)'}
      ]
    },
    {
      kind: 'slider',
      id: 'classCount',
      label: 'Classes',
      group: 'Display',
      apply: 'param',
      min: 2,
      max: 7,
      step: 1,
      default: 5,
      disabledWhen: state => state.display !== 'classes' || state.classMethod === 'box-plot',
      help: 'Number of classes. Head/tail breaks can return fewer when the data runs out of heads.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'cividis',
      help: 'Perceptually uniform ramps. Classed maps sample the same ramp evenly.',
      options: [
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'sqrtScale',
      label: 'Square-root colour scale',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => state.display === 'classes',
      help: 'Lifts quiet tracts next to the Loop. Counts are heavy-tailed; a linear scale shows only the top few tracts.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.82,
      help: 'Lower it to read street names under the tracts.'
    },
    {
      kind: 'select',
      id: 'pointsMode',
      label: 'Observation points',
      group: 'Display',
      apply: 'param',
      default: 'off',
      help: 'Draw the points straight from the join output buffer: coloured by their tract, or only the observations that fall in no tract.',
      options: [
        {value: 'off', label: 'Hidden'},
        {value: 'joined', label: 'Coloured by joined tract'},
        {value: 'outside', label: 'Only observations outside every tract'}
      ]
    },
    {
      kind: 'toggle',
      id: 'outlines',
      label: 'Tract outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Thin outlines drawn from the same polygon buffer the join reads.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the join variants',
      group: 'Compare',
      help: 'Runs the assignment join as plain, Hilbert-sorted and prepared graphs outside the frame and reports GPU time per run.'
    }
  ],

  readouts: [
    {
      id: 'points',
      label: 'Observations',
      format: 'integer',
      help: 'Points uploaded once (iNaturalist, 2023).'
    },
    {id: 'tracts', label: 'Census tracts', format: 'integer'},
    {
      id: 'joined',
      label: 'Joined to a tract',
      help: 'Points whose pointFeatureIds is not NO_FEATURE.'
    },
    {
      id: 'outside',
      label: 'Outside every tract',
      help: 'Observations whose location falls in none of the 791 whole tracts (for example just outside the city limit).'
    },
    {
      id: 'agreement',
      label: 'Agrees with GeoPandas',
      help: 'Share of observations whose GPU tract equals the tract index the data pipeline computed with GeoPandas sjoin.'
    },
    {
      id: 'differenceNearEdge',
      label: 'Where the differences are',
      help: 'Many tract boundaries run along streets and park edges. A point within a metre of an edge can fall either side depending on coordinate rounding.'
    },
    {
      id: 'tractParity',
      label: 'Tract totals vs table',
      help: 'GPU per-tract counts compared with the natureObs2023 column built by the CPU pipeline.'
    },
    {
      id: 'candidates',
      label: 'Bounding-box candidates',
      help: 'Point-tract pairs the BVH produced before exact testing.'
    },
    {
      id: 'joinOverflow',
      label: 'Join overflow',
      help: 'Whether the candidate capacity was exceeded.'
    },
    {
      id: 'joinUncertain',
      label: 'Uncertain pairs',
      help: 'Pairs whose containment could not be proven; 0 means every decision is exact.'
    },
    {id: 'zonalOverflow', label: 'Zonal overflow'},
    {id: 'zonalUncertain', label: 'Zonal uncertain pairs'},
    {
      id: 'top',
      label: 'Highest tract (value · N)',
      help: 'Tract GEOID, community area and mapped value. Rates and shares keep their observation count N beside them.'
    },
    {id: 'cityRate', label: 'City density'},
    {
      id: 'unstableRates',
      label: 'Empty and crowded tracts',
      help: 'Observations follow observers, not residents or land: parks, preserves and the lakefront are crowded while many residential tracts have none. Per-resident rates would overstate the tracts with few residents and a park.'
    },
    {id: 'areaCheck', label: 'GPU polygon area'},
    {
      id: 'indexBuilds',
      label: 'Tract index builds',
      help: 'GPUSpatialJoinPrepared.encodedBuildCount against the number of join runs.'
    },
    {id: 'timePlain', label: 'Join: rebuilt index'},
    {id: 'timeSorted', label: 'Join: Hilbert sort'},
    {id: 'timePrepared', label: 'Join: prepared index'},
    {
      id: 'classBreaks',
      label: 'Class breaks',
      help: 'Edges chosen by GPUClassBreaks, read back after the map settles.'
    },
    {id: 'classesUsed', label: 'Classes'},
    {
      id: 'zoneRaster',
      label: 'Fill raster',
      help: 'Fills are painted from a zone raster produced once by GPUPolygonRasterization.'
    }
  ],

  legends: state => {
    if (state.display === 'classes') {
      const classCount = state.classMethod === 'box-plot' ? 6 : state.classCount;
      return [
        {
          kind: 'categories',
          title: `${STATISTIC_TITLES[state.statistic === 'density' ? 'count' : state.statistic]} (${state.classMethod})`,
          entries: Array.from({length: classCount}, (_, index) => ({
            color: [
              ...sampleRamp(state.ramp, classCount === 1 ? 0.5 : index / (classCount - 1)),
              255
            ] as const,
            label:
              index === 0
                ? 'Class 1 (lowest)'
                : index === classCount - 1
                  ? `Class ${classCount} (highest)`
                  : `Class ${index + 1}`
          })),
          note: 'Class edges are in the Class breaks readout.'
        }
      ];
    }
    const unit =
      state.statistic === 'density'
        ? state.denominator === 'area'
          ? 'observations / km2'
          : 'observations / 1,000 residents'
        : state.statistic === 'count'
          ? 'observations'
          : 'value';
    return [
      {
        kind: 'ramp',
        id: 'statistic',
        title: STATISTIC_TITLES[state.statistic],
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: state.sqrtScale,
        unit,
        format: value =>
          Math.abs(value) >= 100
            ? Math.round(value).toLocaleString('en-US')
            : value.toFixed(value < 10 ? 2 : 1)
      }
    ];
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUPointInPolygonJoin,
  GPUSpatialJoinPrepared,
  GPUZonalStatistics${state.display === 'classes' ? ',\n  addPointsInPolygonsChoroplethRecipe' : ''}
} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'observations-by-tract'});
${
  state.prepared
    ? `const prepared = new GPUSpatialJoinPrepared({geometry: tracts, spatialSort: ${state.spatialSort}});
graph.add(prepared);                          // built once, reused every run
`
    : ''
}graph.add(
  new GPUPointInPolygonJoin({
    points,                                   // float32x2 meters, uploaded once
    polygonPositions, featureOffsets, polygonOffsets, ringOffsets,
    candidateCapacity: points.length * ${8},
    includeBoundary: ${state.includeBoundary},${state.prepared ? '\n    prepared,' : `\n    spatialSort: ${state.spatialSort},`}
    pointFeatureIds,                          // per observation: tract row or NO_FEATURE
    featureCounts,                            // per tract: observations
    overflow, candidateCount, uncertainCount
  })
);
${
  state.display === 'classes'
    ? `addPointsInPolygonsChoroplethRecipe(graph, {
  backend: '${state.backend}',
  points, ${state.statistic === 'count' || state.statistic === 'density' ? '' : 'values, '}statistic: '${state.statistic === 'density' ? 'count' : state.statistic}',
  polygons: {polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity},
  color: {
    classBreaksParameters, maximumClassCount: 7, methods: ['${state.classMethod}', /* ... */],
    colorScaleParameters, palette, maximumPaletteCount: 7, colors
  }
});`
    : `graph.add(
  new GPUZonalStatistics({
    features: {kind: 'polygons', polygonPositions, featureOffsets, polygonOffsets, ringOffsets, candidateCapacity},
    points,${state.statistic === 'count' || state.statistic === 'density' ? '' : '\n    values,                                   // per-observation float: introduced, researchGrade, ...'}${state.statistic === 'density' && state.denominator === 'population' ? '\n    areas,                                    // population / 1000 per tract' : ''}
    sumOrder: '${state.sumOrder}',
    output: {
      ${state.statistic === 'count' ? 'counts' : state.statistic === 'density' ? 'densities' : `${state.statistic === 'sum' ? 'sums' : state.statistic === 'mean' ? 'means' : state.statistic === 'minimum' ? 'minima' : 'maxima'}`}, extent, extentStatistic: '${state.statistic}', overflow
    }
  })
);`
}
const compiled = graph.compile();             // once
// when an option changes, or every frame in streaming mode:
valuesBuffer.write(values);                   // parameter writes only
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUPointInPolygonJoin` finds, for every observation, the census tract that contains it. A BVH over the tract bounds yields candidates and exact point-in-polygon predicates decide them. `GPUZonalStatistics` then reduces counts, sums, means, extremes and densities per tract, and `addPointsInPolygonsChoroplethRecipe` chains the same join into class breaks and class colours.',
    why: 'Observations arrive as points but parks, programmes and neighbourhoods are compared by area. A fast point-in-polygon join turns 43,557 observations into tract rows that line up with census population, income and health data, so the question becomes "how much, per square kilometre?" rather than "how many?".',
    howToRead:
      'Brighter tracts have a higher value of the chosen statistic; the legend range is the real min and max from the GPU. Gray tracts have no data (for example a mean of no observations). Raw counts follow where observers go, not where wildlife is: normalise by area and read shares next to their counts before you compare tracts.'
  },

  create: async ctx =>
    (await import('./observations-by-tract.compute')).createObservationsByTract(ctx),

  story: [
    {
      id: 'the-question',
      headline: 'Lakefront tracts contain the highest observation counts',
      textAlternative:
        'A tract choropleth aggregates Chicago iNaturalist observations with source points optionally visible.',
      controls: ['statistic'],
      readouts: ['joined', 'top'],
      title: 'Where do Chicagoans find wildlife, tract by tract?',
      body: 'Volunteers logged **43,557** observations of wild plants, animals and fungi inside Chicago on iNaturalist in 2023. The app publishes them as points, but parks, programmes and neighbourhoods are compared by area. **`GPUPointInPolygonJoin`** asks, for every observation, which of the 791 census tracts contains it. **`GPUZonalStatistics`** then counts the observations per tract, all on the GPU.\n\nThe brightest tract is on the Uptown lakefront, around the Montrose Point Bird Sanctuary: 4,119 observations of 819 kinds of life. The legend range is the real minimum and maximum read back from the GPU. Hover a tract for its GEOID, community area and count. The **Statistic per tract** control below switches from the count to densities and shares in the next steps.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10},
      callout: {coordinate: [-87.6325, 41.9625], text: 'Montrose Point: the busiest tract'},
      highlight: {readout: 'joined'}
    },
    {
      id: 'per-area',
      headline: 'Area normalization changes the leading tract',
      textAlternative:
        'Observation density per square kilometre emphasizes compact tracts rather than large count totals.',
      controls: ['statistic', 'denominator'],
      readouts: ['top', 'unstableRates', 'cityRate', 'areaCheck'],
      title: 'Counts follow tract size. Normalise.',
      body: 'A count says nothing about how much ground a tract covers. Set **Statistic per tract** to *Density* with **Normalise density by** *Area*: the GPU computes each polygon\u2019s area itself, so `densities = counts / area` is observation records per square kilometre, and the **GPU polygon area** readout checks the total against the area stored in the data.\n\nThe map changes. By count the Montrose Point tract leads; per square kilometre a compact Lincoln Park tract (community area 7) leads, at about 3,900 records per km2. Area asks how intensively a place was recorded. *Residents (per 1,000)* asks a different question and is a poor effort denominator when visitors make the observations: it makes a tract in community area 13 with 1,545 residents and 2,523 records score 1,633 per 1,000 residents. In both cases the **Highest tract** readout keeps N beside the rate.',
      options: {statistic: 'density', denominator: 'area'},
      highlight: {readout: 'unstableRates'}
    },
    {
      id: 'share-introduced',
      headline: 'Introduced-species shares vary with tract sample size',
      textAlternative:
        'Tracts are colored by the fraction of observations classified as introduced taxa.',
      controls: ['statistic', 'valueKind', 'groupType'],
      readouts: ['top'],
      title: 'What share of observations are introduced species?',
      body: 'Zonal statistics also reduce a per-record **value**. With **Statistic per tract** on *Mean of a value* and **Value per observation** set to *Introduced, non-native taxon (0/1)*, the mean per tract is the share of records marked introduced. The mean is `sum(v) / valueCount`; tracts with no records have no mean and are drawn gray.\n\nCitywide 15.7% of N=43,557 records are marked introduced. Never read a share without its N: the **Highest tract** readout prints both, because one pigeon record makes a tract 100% introduced but not well sampled. Change **Value per observation** to *Research grade* (citywide 63.0%), *Made on a Saturday or Sunday*, or a chosen group. *Sum* gives its numerator; *Mean* divides by all tract records. The value buffer is rewritten without rebuilding the graph.',
      evidence:
        '**{{top}}** reports the mapped share and the record count supporting it in the same line.',
      caveat:
        'The denominator is uploaded records, not survey effort, visits, hours searched, residents or organisms present.',
      options: {statistic: 'mean', valueKind: 'introduced', sqrtScale: false, ramp: 'cividis'}
    },
    {
      id: 'classed-recipe',
      headline: 'Classification method changes the visible tract distribution',
      textAlternative:
        'The same tract values are grouped into quantile, equal-interval or natural-break classes.',
      controls: ['classMethod', 'classCount', 'backend'],
      readouts: ['classBreaks', 'classesUsed'],
      title: 'One recipe from points to classed colours',
      body: '**`addPointsInPolygonsChoroplethRecipe`** chains the join into `GPUClassBreaks` and `GPUColorScale`, so even the class edges and the per-tract colours are computed on the GPU. Choose a **Classification method** (quantile, equal interval or natural breaks) and **Classes**, and watch the same data tell three stories; the real edges appear in the **Class breaks** readout. Observation counts are very skewed, so equal interval puts nearly every tract in the lowest class while quantile spreads them out.\n\n**Recipe back end** switches between the zonal reduction and a join followed by a dense `GPUGroupStatistics`. Both give the same counts. The class count and method are parameter writes; the back end is compile-time.',
      options: {
        statistic: 'count',
        display: 'classes',
        classMethod: 'quantile',
        classCount: 5,
        ramp: 'cividis'
      }
    },
    {
      id: 'prepared-index',
      headline: 'Index reuse avoids repeated tract-tree construction',
      textAlternative:
        'Prepared and rebuilt spatial joins return the same tract aggregation with different setup work.',
      controls: ['prepared', 'rerunEveryFrame', 'invalidateEveryFrame', 'measure'],
      readouts: ['indexBuilds', 'timePlain', 'timePrepared'],
      title: 'Build the tract index once',
      body: 'Every join run builds bounds and a BVH over the tracts. They never move, so **`GPUSpatialJoinPrepared`** builds that index once and reuses it. Turn on **Prepared tract index** and **Re-run the join every frame**: **Tract index builds** stays at one while the join runs hundreds of times.\n\nThen turn on the invalidate contrast: calling `invalidate()` every frame forces a rebuild and the benefit disappears. Press **Time the join variants** for GPU times of the plain, Hilbert-sorted and prepared graphs. 791 coherent tracts and 43,557 points are small, so expect the gap to be modest; it grows with the feature count.',
      options: {
        display: 'ramp',
        prepared: true,
        rerunEveryFrame: true,
        statistic: 'count',
        sqrtScale: true
      },
      highlight: {readout: 'indexBuilds'}
    },
    {
      id: 'check-and-limits',
      headline: 'GPU tract totals agree away from boundary ambiguity',
      textAlternative:
        'Comparison readouts audit GPU assignments against GeoPandas and identify edge-adjacent differences.',
      controls: ['pointsMode', 'includeBoundary'],
      readouts: ['agreement', 'differenceNearEdge', 'tractParity'],
      title: 'Check against GeoPandas, then mind the limits',
      body: 'The data pipeline joined the same points with GeoPandas. **Agrees with GeoPandas** reports the share of records where the GPU tract equals that tract index, and **Tract totals vs table** compares every per-tract count. **Where the differences are** shows that any disagreement lies within 2 m of a tract edge, where float32 coordinates can put a point on either side. **Observation points** is set to *Only records outside every tract*: every record in this data lies inside a tract, so expect none.\n\n**Limits.** Tracts stabilize the denominator but introduce fixed administrative boundaries. Parks, the lakefront and enthusiastic users dominate, and the City Nature Challenge pulse from 28 April through 1 May moves counts without proving an ecological change. The next time story replaces tracts with fixed 500 m cells and keeps those boundaries constant through 52 weekly slices—then tests how much the result changes with neighbourhood and significance choices.',
      options: {
        pointsMode: 'outside',
        display: 'ramp',
        rerunEveryFrame: false,
        includeBoundary: true
      },
      highlight: {readout: 'agreement'}
    }
  ]
});
