// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CHICAGO, CITY_FRAMES, findPlace, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {formatDistance} from '../../cartography/live-text';
import type {ClassTable} from '../../cartography/types';
import {defineScene, type LegendSpec} from '../scene';
import {formatCategory} from './b1-nature-data';
import {
  getGhostColor,
  getParameterInk,
  getSecondColor,
  getSubjectColor,
  nextStoryLine,
  pointsCartouche,
  POINTS_CREDITS,
  type PointsGround
} from './b1-points-look';
import type {QuadratClasses} from './point-patterns-classes';
import type {PointPatternOptions} from './point-patterns.compute';

const GROUP_CATEGORIES = [
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
];

const PLACE_CATEGORIES: readonly [string, string][] = [
  ['restaurant_cafe', 'Restaurants and cafes'],
  ['bar_nightlife', 'Bars and nightlife'],
  ['grocery', 'Grocery and convenience'],
  ['health', 'Health care'],
  ['school_education', 'Schools and education'],
  ['park_recreation', 'Parks and recreation'],
  ['transit', 'Transit stops'],
  ['retail', 'Retail'],
  ['finance_business', 'Finance and business'],
  ['personal_services', 'Personal services'],
  ['arts_culture', 'Arts and culture'],
  ['worship_community', 'Worship and community'],
  ['lodging', 'Lodging'],
  ['other', 'Other']
];

/** The camera of one gazetteer place, at `zoom` (the gazetteer owns every coordinate). */
function getPlaceView(id: string, zoom: number) {
  const place = findPlace(CHICAGO, id);
  return place
    ? {longitude: place.lngLat[0], latitude: place.lngLat[1], zoom}
    : {...CITY_FRAMES.chicago};
}

/** The camera halfway between two gazetteer places, at `zoom`. */
function getMidpointView(firstId: string, secondId: string, zoom: number) {
  const first = findPlace(CHICAGO, firstId);
  const second = findPlace(CHICAGO, secondId);
  return first && second
    ? {
        longitude: (first.lngLat[0] + second.lngLat[0]) / 2,
        latitude: (first.lngLat[1] + second.lngLat[1]) / 2,
        zoom
      }
    : {...CITY_FRAMES.chicago};
}

const NIGHT = ground('night');

export default defineScene<PointPatternOptions>({
  id: 'point-patterns',
  title: 'Are the points clustered, random or regular?',
  chapter: 'points',
  order: 3,
  summary:
    'Is a pattern more clumped than chance? Ripley K and L, the nearest-neighbour functions G, F and J, Clark-Evans and quadrat counts for Chicago nature records, each drawn against a 39-pattern Monte Carlo envelope under two null models, with the study window and its lake drawn on the map.',
  contributors: ['GPURipley', 'GPURipleyDistanceFunctions', 'GPUPointPatternIndices'],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-places', role: 'Overture places, for comparison'},
    {id: 'chicago-boundary', role: 'city limit and lake: the study window and the null'}
  ],
  initialView: {...CITY_FRAMES.chicago},

  options: [
    {
      kind: 'select',
      id: 'subject',
      label: 'Point set',
      group: 'Pattern',
      apply: 'compile',
      default: 'observations',
      help: 'Nature observations or Overture places. Each has its own compiled graph (cached after first use).',
      options: [
        {value: 'observations', label: 'Nature observations (2023)'},
        {value: 'places', label: 'Places (restaurants, shops, schools, ...)'}
      ]
    },
    {
      kind: 'select',
      id: 'groupCategory',
      label: 'Observation group',
      group: 'Pattern',
      apply: 'param',
      default: 'Birds',
      disabledWhen: state => state.subject !== 'observations',
      help: 'Rows of other groups are masked out: a mask buffer write, no recompile. The chart keeps the previous selection as a dashed grey curve for comparison.',
      options: [
        {value: 'all', label: 'All groups'},
        ...GROUP_CATEGORIES.map(name => ({value: name, label: formatCategory(name)}))
      ]
    },
    {
      kind: 'select',
      id: 'placeCategory',
      label: 'Place type',
      group: 'Pattern',
      apply: 'param',
      default: 'restaurant_cafe',
      disabledWhen: state => state.subject !== 'places',
      help: 'Mask by Overture category. Also a buffer write.',
      options: [
        {value: 'all', label: 'All places'},
        ...PLACE_CATEGORIES.map(([value, label]) => ({value, label}))
      ]
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Radius r',
      group: 'Pattern',
      apply: 'param',
      min: 100,
      max: 5000,
      step: 50,
      default: 400,
      unit: 'm',
      describe: value => `a circle ${formatDistance(value * 2)} across`,
      autoSweep: {from: 100, to: 1500, durationMs: 9000, ease: 'in-out'},
      help: 'The radius the map ring and the chart marker show: K counts the neighbours inside this circle. Drag the marker on the chart, or the slider. The curves themselves always run up to the largest radius.'
    },
    {
      kind: 'slider',
      id: 'maximumDistance',
      label: 'Largest radius',
      group: 'Pattern',
      apply: 'param',
      min: 200,
      max: 5000,
      step: 100,
      default: 1500,
      unit: 'm',
      help: 'The curves are evaluated at evenly spaced radii up to this distance. Isotropic edge correction is only meaningful up to half the shorter side of the window. Changing it restarts the random patterns.'
    },
    {
      kind: 'select',
      id: 'window',
      label: 'Study window',
      group: 'Pattern',
      apply: 'param',
      default: 'city',
      help: 'The rectangle whose area A enters every estimator. The city window includes Lake Michigan, which inflates clustering (hatched on the map); the map-view window follows the camera, so you can analyse one neighbourhood.',
      options: [
        {value: 'city', label: 'City window'},
        {value: 'view', label: 'Current map view (follows pan and zoom)'}
      ]
    },
    {
      kind: 'select',
      id: 'nullModel',
      label: 'Null model',
      group: 'The null',
      apply: 'param',
      default: 'city',
      display: 'segmented',
      help: 'What "random" means. In the rectangle: points anywhere in the window, lake included. On city land: points only where a record could fall (rejection sampling with the city limit). The grey band on the chart is 39 patterns of this null; changing it restarts them.',
      options: [
        {value: 'window', label: 'Random in the rectangle'},
        {value: 'city', label: 'Random on city land'}
      ]
    },
    {
      kind: 'select',
      id: 'ripleyCorrection',
      label: 'Edge correction of K and L',
      group: 'Edge effects',
      apply: 'param',
      default: 'isotropic',
      help: 'Points near the window edge have fewer neighbours inside it. Isotropic (Ripley 1977) weights each pair by the inside fraction of its circle; border uses only points far enough from the edge; none ignores the problem.',
      options: [
        {value: 'isotropic', label: 'Isotropic (Ripley 1977)'},
        {value: 'border', label: 'Border (reduced sample)'},
        {value: 'none', label: 'None'}
      ]
    },
    {
      kind: 'toggle',
      id: 'compareCorrections',
      label: 'Compare the three corrections',
      group: 'Edge effects',
      apply: 'param',
      default: false,
      help: 'Draws the curve for none, border and isotropic together: the same compiled graph, run three times with a different edge-correction parameter.'
    },
    {
      kind: 'select',
      id: 'distanceCorrection',
      label: 'Edge correction of G, F and J',
      group: 'Edge effects',
      apply: 'param',
      default: 'border',
      help: 'The same idea for the nearest-neighbour and empty-space functions: border (reduced sample), Kaplan-Meier (censored distances), Hanisch / Chiu-Stoyan weights, or none (biased low near the edge).',
      options: [
        {value: 'border', label: 'Border (reduced sample)'},
        {value: 'kaplan-meier', label: 'Kaplan-Meier'},
        {value: 'hanisch', label: 'Hanisch (G) / Chiu-Stoyan (F)'},
        {value: 'none', label: 'None'}
      ]
    },
    {
      kind: 'slider',
      id: 'jitter',
      label: 'Jitter shared coordinates',
      group: 'Data quality',
      apply: 'param',
      min: 0,
      max: 60,
      step: 5,
      default: 0,
      unit: 'm',
      autoSweep: {durationMs: 6000, ease: 'in-out'},
      help: 'Many records share an exact coordinate with another (repeat visits to one spot, reused map pins). Jitter moves each point by a random offset up to this radius (a positions buffer write). Large-scale K and L barely change; nearest-neighbour measures change a lot.'
    },
    {
      kind: 'select',
      id: 'radiusCount',
      label: 'Number of radii',
      group: 'Resolution',
      apply: 'compile',
      default: '30',
      expert: true,
      help: 'How many distances the curves are sampled at (compile-time, 1 to 256).',
      options: [
        {value: '20', label: '20'},
        {value: '30', label: '30'},
        {value: '60', label: '60'}
      ]
    },
    {
      kind: 'select',
      id: 'quadratGrid',
      label: 'Quadrat grid',
      group: 'Resolution',
      apply: 'compile',
      default: '12',
      display: 'segmented',
      help: 'The window is cut into n x n quadrats and the points counted in each (compile-time). Coarse quadrats test large-scale variation; fine ones need fewer points per cell.',
      options: [
        {value: '6', label: '6 x 6'},
        {value: '12', label: '12 x 12'},
        {value: '24', label: '24 x 24'}
      ]
    },
    {
      kind: 'select',
      id: 'referenceGrid',
      label: 'F reference lattice',
      group: 'Resolution',
      apply: 'compile',
      default: '47',
      expert: true,
      help: 'The empty-space function F measures the distance from a regular lattice of reference locations to the nearest point. A finer lattice reduces sampling noise (compile-time).',
      options: [
        {value: '24', label: '24 x 24 locations'},
        {value: '47', label: '47 x 47 locations'},
        {value: '94', label: '94 x 94 locations'}
      ]
    },
    {
      kind: 'select',
      id: 'colorPoints',
      label: 'Color points by',
      group: 'Display',
      apply: 'param',
      default: 'selection',
      display: 'segmented',
      help: 'The selected points, or each point classed by the distance to its nearest neighbour as a multiple of the distance a random pattern would give (the GPUPointPatternIndices distance output).',
      options: [
        {value: 'selection', label: 'Selection'},
        {value: 'nearest', label: 'Nearest neighbour'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showRandom',
      label: 'Show a random pattern',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.colorPoints === 'nearest',
      help: 'Swaps the selected points for one random pattern of the same size under the chosen null model (the first of the 39 the envelope runs), so the comparison is fair.'
    },
    {
      kind: 'toggle',
      id: 'showRadius',
      label: 'Show the radius on the map',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the circle of radius r on the densest selected point, with the number of neighbours inside, and a tick on the scale bar.'
    },
    {
      kind: 'toggle',
      id: 'showQuadrats',
      label: 'Show quadrat counts',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Classes each quadrat by how many selected points it holds (natural breaks, fixed per selection).'
    },
    {
      kind: 'toggle',
      id: 'emphasizeEdge',
      label: 'Emphasise the window edge',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Strengthens the hatch over land where no record can fall and names how much of the window it covers.'
    },
    {
      kind: 'toggle',
      id: 'showOthers',
      label: 'Show unselected points',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws the masked-out points faintly for context. Off by default so the subject beats the context.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the pattern graph',
      group: 'Compare',
      help: 'GPU time of Ripley, the distance functions and the point-pattern indices together at the current radius.'
    }
  ],

  readouts: [
    {
      id: 'included',
      label: 'Points analysed',
      emphasis: 'tile',
      help: 'Selected points inside the window.'
    },
    {
      id: 'intensity',
      label: 'Intensity',
      help: 'Selected points per square kilometre of the whole window, water included.'
    },
    {id: 'windowArea', label: 'Study window area', hood: true},
    {
      id: 'windowLandShare',
      label: 'City land in the window',
      emphasis: 'tile',
      help: 'Share of the window rectangle that is inside the city limit; the rest is lake and suburbs.'
    },
    {id: 'outsideShare', label: 'Lake and suburbs in the window', hood: true},
    {id: 'nullName', label: 'Null model', hood: true},
    {
      id: 'lCurve',
      label: 'L(r) - r against random patterns',
      kind: 'chart',
      help: 'Besag L minus r. Zero under complete spatial randomness; above the grey band the pattern is clustered at that radius. Drag the marker to set the radius.'
    },
    {id: 'lPeak', label: 'Strongest clustering'},
    {id: 'peakR', label: 'Radius of the peak', hood: true},
    {
      id: 'lAt1km',
      label: 'L(r) - r at 1 km',
      help: 'Barely moves when the shared coordinates are jittered.'
    },
    {id: 'withinRadius', label: 'At the densest point'},
    {id: 'simulations', label: 'Random patterns run'},
    {
      id: 'gfjCurves',
      label: 'G, F and J',
      kind: 'chart',
      help: 'G: fraction of points whose nearest neighbour is within r (climbing early means clustering). F: fraction of a reference lattice within r of a point (climbing late means gaps). J = (1-G)/(1-F): below 1 clustering, above 1 regularity.'
    },
    {
      id: 'clarkEvansGauge',
      label: 'Clark-Evans gauge',
      kind: 'chart',
      help: 'R below 1 is clustered, above 1 dispersed; the shaded band is where a random pattern falls.'
    },
    {id: 'clarkEvansR', label: 'Clark-Evans R', emphasis: 'tile'},
    {id: 'clarkEvans', label: 'Clark-Evans index', hood: true},
    {
      id: 'expectedNN',
      label: 'Expected nearest-neighbour distance',
      help: 'The mean distance to the nearest neighbour of a random pattern of the same intensity.'
    },
    {id: 'observedNN', label: 'Observed nearest-neighbour distance'},
    {
      id: 'quadratHistogram',
      label: 'Quadrat counts',
      kind: 'chart',
      help: 'How many quadrats hold each number of records; the marker is the count a random pattern expects.'
    },
    {
      id: 'vmr',
      label: 'Quadrat variance-to-mean ratio',
      help: 'About 1 for a random pattern; above 1 for clustering.'
    },
    {id: 'chiSquare', label: 'Quadrat chi-square', hood: true},
    {
      id: 'duplicateShare',
      label: 'Records on a shared coordinate',
      help: 'Share of the selected records whose exact coordinate another selected record also has.'
    },
    {id: 'timing', label: 'Pattern graph timing', hood: true},
    {id: 'isotropicCap', label: 'Numerical notes', hood: true},
    {id: 'graphs', label: 'Compiled graphs and runs', hood: true}
  ],

  pipeline: [
    {
      id: 'lattice',
      label: 'Lattice',
      detail: 'Rows are sorted into a 96 x 96 grid of cells at least as wide as the largest radius'
    },
    {
      id: 'pairs',
      label: 'Pairs',
      detail:
        'Each row scans the 3 x 3 cells around it and adds integer fixed-point sums, so the result is bitwise reproducible'
    },
    {
      id: 'curves',
      label: 'Curves',
      detail: 'K, L, G, F and J from the pair counts and nearest-neighbour distances'
    },
    {
      id: 'envelope',
      label: 'Envelope',
      detail: '39 random patterns: the same compiled graph, a new positions buffer each time'
    },
    {id: 'draw', label: 'Draw', detail: 'Layers read the same GPU buffers'}
  ],

  legends: (state, data) => {
    const darkGround = ((data['ground'] as PointsGround | undefined) ?? 'dark') === 'dark';
    const legendGround: PointsGround = darkGround ? 'dark' : 'light';
    const legends: LegendSpec[] = [];
    const nearestTable = data['nearestTable'] as ClassTable | undefined;
    const nearestExpected = data['nearestExpected'] as number | undefined;
    if (state.colorPoints === 'nearest' && nearestTable) {
      legends.push(
        getClassTableLegend(nearestTable, {
          title: 'Distance to nearest neighbour (m)',
          layout: 'list',
          note:
            nearestExpected !== undefined
              ? `Expected under a random pattern: ${formatDistance(nearestExpected)}`
              : undefined
        })
      );
    } else {
      const groupName =
        state.subject === 'observations'
          ? state.groupCategory === 'all'
            ? 'All observations'
            : formatCategory(state.groupCategory)
          : (PLACE_CATEGORIES.find(([value]) => value === state.placeCategory)?.[1] ??
            'All places');
      const subjectColor =
        state.subject === 'observations'
          ? getSubjectColor(legendGround, 255)
          : getSecondColor(legendGround, 255);
      legends.push({
        kind: 'categories',
        title: 'Points and window',
        layout: 'list',
        entries: [
          {color: subjectColor, label: groupName, shape: 'dot'},
          ...(state.showRandom
            ? [
                {
                  color:
                    state.subject === 'observations'
                      ? getSecondColor(legendGround, 255)
                      : ([233, 236, 240, 255] as const),
                  label: 'One random pattern, same size',
                  shape: 'dot' as const
                }
              ]
            : []),
          ...(state.showOthers
            ? [
                {
                  color: getGhostColor(legendGround, 160),
                  label: 'Other records',
                  shape: 'dot' as const
                }
              ]
            : []),
          {
            color: getParameterInk(legendGround, 160),
            label: 'Not city land: no record can fall here',
            shape: 'hatch'
          },
          {color: getParameterInk(legendGround, 235), label: 'Study window', shape: 'line'}
        ]
      });
    }
    const quadrat = data['quadrat'] as QuadratClasses | undefined;
    if (state.showQuadrats && quadrat) {
      legends.push(
        getClassTableLegend(quadrat.table, {
          title: 'Records per quadrat',
          layout: 'list',
          counts: quadrat.counts
        })
      );
    }
    return legends;
  },

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURipley, GPURipleyDistanceFunctions, GPUPointPatternIndices,
  getGPURipleyParameterValues, getGPURipleyDistanceParameterValues,
  getGPUPointPatternIndicesParameterValues
} from '@luma.gl/experimental/gpu-dataframe';

// The observed pattern: compiled once.
const graph = new GPUCommandGraph(device, {id: 'point-pattern'});
graph.add(new GPURipley({
  positions, mask,                       // mask: 1 for the selected group
  parameters: ripleyParameters.importToGraph(graph),
  gridSize: [96, 96], radiusCount: ${state.radiusCount},
  k, l, lMinusR, radii
}));
graph.add(new GPURipleyDistanceFunctions({
  positions, mask, parameters: distanceParameters.importToGraph(graph),
  gridSize: [96, 96], referenceGrid: [${state.referenceGrid}, ${state.referenceGrid}], radiusCount: ${state.radiusCount},
  g, f, j
}));
graph.add(new GPUPointPatternIndices({
  positions, mask, parameters: indicesParameters.importToGraph(graph),
  gridSize: [96, 96], quadratGrid: [${state.quadratGrid}, ${state.quadratGrid}],
  nearestNeighborDistances, clarkEvans, quadratCounts, quadratStatistics
}));
const compiled = graph.compile();        // once

// window, radii and edge correction are parameter writes
ripleyParameters.write(getGPURipleyParameterValues({
  bounds, maximumDistance: ${state.maximumDistance}, edgeCorrection: '${state.ripleyCorrection}'
}));
distanceParameters.write(getGPURipleyDistanceParameterValues({
  bounds, maximumDistance: ${state.maximumDistance}, edgeCorrection: '${state.distanceCorrection}'
}));
compiled.encode(commandEncoder, {parameters: undefined});

// The envelope: the same graph shape, compiled once over a second positions buffer.
// Each of 39 random patterns is a buffer write and one encode, never a recompile.
for (let run = 0; run < 39; run++) {
  fillRandomPositions(scratch, n, bounds, ${state.nullModel === 'city' ? 'cityLand' : 'null'}, createSeededRandom(seed + run));
  simulationPositions.write(scratch);
  simulationGraph.encode(commandEncoder, {parameters: undefined});
  // read L - r, G, F, J back; the band is the pointwise min and max of the 39 runs
}`,

  about: {
    what: "`GPURipley` counts pairs of points within growing radii to estimate K(r) and Besag's L(r) = sqrt(K/pi), with edge corrections. `GPURipleyDistanceFunctions` adds the nearest-neighbour function G, the empty-space function F and J = (1-G)/(1-F). `GPUPointPatternIndices` gives each point's nearest-neighbour distance, the Clark-Evans ratio and quadrat dispersion. The story reruns the same compiled graph over 39 random patterns for a Monte Carlo envelope. Parity target: spatstat `Kest`, `Lest`, `Gest`, `Fest`, `Jest` (rectangle windows) and pointpats.",
    why: 'Before modelling a point pattern you need to know how it departs from complete spatial randomness: at what distance do events attract, and is the clustering real or an artefact of the window? The curves answer that scale by scale, and the envelope says what chance alone would draw.',
    howToRead:
      'L(r) - r above zero is clustering at that radius; it is only convincing above the grey band of random patterns. G rising early and F rising late both indicate clustering; J below 1 agrees. Clark-Evans R below 1 and a quadrat variance-to-mean ratio above 1 are the single-number versions. A hatched area is inside the window but not city land: no record can fall there, which is why "random in the rectangle" is an unfair null.'
  },

  // Night ground: additive amber points. Step 5 switches to paper for a classed fill.
  basemap: NIGHT,
  furniture: {
    title: pointsCartouche(
      'Clustered, or just crowded?',
      'Bird records and a random pattern of the same size'
    ),
    scaleBar: {units: 'metric'},
    credit: POINTS_CREDITS.natureAndPlaces
  },
  annotations: labelsFor(CHICAGO, ['lake-michigan']),

  create: async ctx => (await import('./point-patterns.compute')).createPointPatterns(ctx),

  story: [
    {
      id: 'the-pile-up',
      title: 'Is this more clumped than chance?',
      headline: 'Birds pile up where a random scatter would not',
      textAlternative:
        'Dark map of Chicago with amber dots for bird records, densest along the lakefront, inside a hatched study window that includes the lake.',
      body: 'Each dot is a bird record: **{{included}}** in a window of {{windowArea}}, {{intensity}}. "Clustered" only means more than a null model gives, and the null here is complete spatial randomness: the same number of points thrown down at random. Tick **Show a random pattern** to swap the birds for one such scatter. Clark-Evans R, {{clarkEvansR}}, is the one-number verdict: 1 is random.',
      optionsMode: 'fresh',
      options: {showRandom: false},
      controls: ['showRandom'],
      readouts: ['included', 'intensity', 'clarkEvansR'],
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      basemap: NIGHT,
      furniture: {
        title: pointsCartouche(
          'Clustered, or just crowded?',
          'Bird records and a random pattern of the same size'
        )
      },
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park'])
    },
    {
      id: 'k-and-l',
      title: 'Count neighbours within r',
      headline: 'How clumped depends on the radius you ask',
      textAlternative:
        'Map of Montrose Point with a dashed ring on the densest bird record, and a chart of L(r) minus r rising far above a grey band of random patterns.',
      body: 'Draw a circle of radius r round each sighting and count the others inside: that is K. L(r) − r subtracts what chance gives, so zero is random. Slide **Radius r**: the ring sits on the densest point, {{withinRadius}}, and the marker rides the curve. Above the grey band of {{simulations}} the clumping is not chance; it peaks at {{peakR}}. **Largest radius** sets how far the curve runs.',
      optionsMode: 'fresh',
      options: {showRadius: true, radius: 400, maximumDistance: 1500},
      controls: ['radius', 'maximumDistance'],
      readouts: ['lCurve', 'lPeak', 'simulations'],
      stage: 'pairs',
      camera: {...getPlaceView('montrose-point', 12.5), transitionMs: 1600},
      basemap: NIGHT,
      furniture: {
        title: pointsCartouche(
          'How far do neighbours reach?',
          'L(r) − r for bird records, with 39 random patterns'
        )
      },
      annotations: labelsFor(CHICAGO, ['montrose-point'])
    },
    {
      id: 'window-and-edges',
      title: 'The window sets the null',
      headline: 'Half the window can never hold a sighting',
      textAlternative:
        'City map with the lake and suburbs hatched inside a rectangular frame, and a chart where the grey band of random patterns lifts when the null is confined to city land.',
      body: '{{outsideShare}} of the study window is lake and suburbs, hatched: no record can fall there. Under **Null model** "Random in the rectangle" the birds look hugely clustered; "Random on city land" lifts the grey band and leaves their own excess. **Edge correction of K and L** fixes a smaller problem: missing neighbours at the edge.\n\n*A pattern is clustered only relative to a window and a null.*',
      optionsMode: 'fresh',
      options: {nullModel: 'window', emphasizeEdge: true, compareCorrections: true},
      controls: ['nullModel', 'ripleyCorrection'],
      readouts: ['lCurve', 'windowLandShare', 'lPeak'],
      stage: 'envelope',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      basemap: NIGHT,
      furniture: {
        title: pointsCartouche(
          'Clustered against what window?',
          'L(r) − r under two nulls and three edge corrections'
        )
      },
      annotations: labelsFor(CHICAGO, ['loop'], {loop: {minZoom: 10.2}})
    },
    {
      id: 'nearest-neighbours',
      title: 'Nearest neighbours tell the small-scale story',
      headline: 'Nearest neighbours tell the small-scale story',
      textAlternative:
        'Map of the north lakefront with each bird record coloured by its distance to its nearest neighbour, bright for short, and a ring on the densest point; charts of G, F and J and a Clark-Evans gauge.',
      body: 'Each dot is classed by its distance to the nearest neighbour, in multiples of E, the {{expectedNN}} a random scatter would give; the birds average {{observedNN}}. Clustering shows as G climbing early, F late and J below 1; Clark-Evans R is {{clarkEvansR}}. Choose **Edge correction of G, F and J**, or switch **Color points by** back to the selection.',
      optionsMode: 'fresh',
      options: {colorPoints: 'nearest', showRadius: true},
      controls: ['distanceCorrection', 'colorPoints'],
      readouts: ['gfjCurves', 'clarkEvansGauge', 'expectedNN', 'observedNN'],
      stage: 'curves',
      camera: {...getMidpointView('lincoln-park', 'montrose-point', 11.4), transitionMs: 1600},
      basemap: NIGHT,
      furniture: {
        title: pointsCartouche(
          'How close is the nearest neighbour?',
          'Distance to the nearest record, in multiples of random'
        )
      },
      annotations: labelsFor(CHICAGO, ['lincoln-park', 'montrose-point'])
    },
    {
      id: 'stacks-and-quadrats',
      title: 'Stacks and quadrat counts',
      headline: 'Many records share one exact spot',
      textAlternative:
        'Paper-coloured map of Chicago with small dark dots over a blue classed grid of quadrat counts, the lake and suburbs hatched, and a histogram of counts per quadrat.',
      body: 'Cut the window into **Quadrat grid** cells and count: a random pattern has a variance equal to its mean, here the ratio is {{vmr}}. {{duplicateShare}} of records share an exact coordinate. Slide **Jitter shared coordinates**: Clark-Evans R, {{clarkEvansR}}, moves while L at a kilometre, {{lAt1km}}, barely does. A classed fill reads on paper. The grid is a choice, as in the [density story](#/story/nature-density).',
      optionsMode: 'fresh',
      options: {showQuadrats: true, quadratGrid: '12', jitter: 0},
      controls: ['quadratGrid', 'jitter'],
      readouts: ['quadratHistogram', 'duplicateShare', 'clarkEvansR', 'lAt1km'],
      stage: 'curves',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      basemap: ground('paperCity'),
      furniture: {
        title: pointsCartouche('Do the quadrats agree?', 'Records per quadrat, natural breaks')
      },
      annotations: labelsFor(CHICAGO, ['loop'], {loop: {minZoom: 10.2}})
    },
    {
      id: 'birds-vs-restaurants',
      title: 'Restaurants and birds',
      headline: 'Restaurants cluster on corridors, birds on the lake',
      textAlternative:
        'Dark map of Chicago with sky-blue dots for restaurants and cafes along commercial corridors, a ring on the densest point, and a chart of L(r) minus r with the bird curve as a dashed grey ghost.',
      body:
        'Both patterns follow people: observers for birds, customers for restaurants; Overture coverage is thinner on the South and West Sides. Pick a **Point set** and **Place type**; the dashed grey ghost is the birds. Set **Study window** to the map view to test one neighbourhood.\n\n*Numerical note: isotropic weights are capped at 100; Kaplan-Meier and Hanisch exist for G, F and J only; positions are float32 metres.*\n\n' +
        nextStoryLine('point-patterns'),
      optionsMode: 'fresh',
      options: {subject: 'places', placeCategory: 'restaurant_cafe', showRadius: true},
      controls: ['subject', 'placeCategory', 'window'],
      readouts: ['lCurve', 'lPeak', 'included'],
      stage: 'draw',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      basemap: NIGHT,
      furniture: {
        title: pointsCartouche(
          'Same test, different pattern',
          'Restaurants and cafes, with the birds as a ghost',
          {effort: false}
        )
      },
      annotations: labelsFor(CHICAGO, ['loop', 'lincoln-park'], {loop: {minZoom: 10.2}})
    }
  ]
});
