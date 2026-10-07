// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {CHICAGO_VIEW, formatCategory} from './b1-nature-data';
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

export default defineScene<PointPatternOptions>({
  id: 'point-patterns',
  title: 'Are the points clustered, random or regular?',
  chapter: 'points',
  order: 3,
  summary:
    'Ripley K, L, G, F and J curves, Clark-Evans and quadrat tests for one group of Chicago nature observations or one place type, with edge corrections, a movable window and a jitter control for observations that share a coordinate.',
  contributors: ['GPURipley', 'GPURipleyDistanceFunctions', 'GPUPointPatternIndices'],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023)'},
    {id: 'chicago-places', role: 'Overture places, for comparison'}
  ],
  initialView: {...CHICAGO_VIEW},

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
      help: 'Rows of other groups are masked out: a mask buffer write, no recompile. The curves keep the previous selection beside the new one for comparison.',
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
      id: 'maximumDistance',
      label: 'Largest radius',
      group: 'Pattern',
      apply: 'param',
      min: 200,
      max: 5000,
      step: 100,
      default: 1500,
      unit: 'm',
      help: 'The curves are evaluated at evenly spaced radii up to this distance. Isotropic edge correction is only meaningful up to half the shorter side of the window.'
    },
    {
      kind: 'select',
      id: 'window',
      label: 'Study window',
      group: 'Pattern',
      apply: 'param',
      default: 'city',
      help: 'The rectangle whose area A enters every estimator. The city bounding box includes Lake Michigan, which inflates clustering; the map-view window follows the camera, so you can analyse one neighbourhood.',
      options: [
        {value: 'city', label: 'City bounding box'},
        {value: 'view', label: 'Current map view (follows pan and zoom)'}
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
      max: 120,
      step: 5,
      default: 0,
      unit: 'm',
      help: 'About one observation in six shares its exact coordinate with another (repeat visits to one spot, reused map pins). Jitter moves each point by a random offset up to this radius (a positions buffer write). Large-scale K and L barely change; nearest-neighbour measures change a lot.'
    },
    {
      kind: 'select',
      id: 'radiusCount',
      label: 'Number of radii',
      group: 'Resolution',
      apply: 'compile',
      default: '30',
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
      help: 'The selected points, or each point colored by the distance to its nearest neighbour (the GPUPointPatternIndices distance output).',
      options: [
        {value: 'selection', label: 'Selection'},
        {value: 'nearest', label: 'Nearest-neighbour distance'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showQuadrats',
      label: 'Show quadrat counts',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Shades each quadrat by how many selected points it holds.'
    },
    {
      kind: 'toggle',
      id: 'showWindow',
      label: 'Show study window',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Outline of the analysis rectangle.'
    },
    {
      kind: 'toggle',
      id: 'showOthers',
      label: 'Show unselected points',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the masked-out points faintly for context.'
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
    {id: 'included', label: 'Points analysed', help: 'Selected points inside the window.'},
    {id: 'area', label: 'Window area and intensity'},
    {
      id: 'lCurve',
      label: 'L(r) - r',
      help: 'Besag L minus r from the smallest to the largest radius. Zero under complete spatial randomness; above zero means clustering at that scale.'
    },
    {id: 'lPeak', label: 'Strongest clustering'},
    {id: 'lReading', label: 'Reading'},
    {
      id: 'previousCurve',
      label: 'Previous selection L(r) - r',
      help: 'The curve of the selection you just changed from, scaled on its own.'
    },
    {
      id: 'gCurve',
      label: 'G(r) nearest neighbour',
      help: 'Fraction of points whose nearest neighbour is within r (0 to 1). Climbing early means clustering.'
    },
    {
      id: 'fCurve',
      label: 'F(r) empty space',
      help: 'Fraction of reference locations within r of a point (0 to 1). Climbing late means gaps.'
    },
    {
      id: 'jCurve',
      label: 'J(r) = (1-G)/(1-F)',
      help: 'Below 1 clustering, above 1 regularity, 1 under randomness. Drawn between 0 and 2.'
    },
    {id: 'clarkEvans', label: 'Clark-Evans index'},
    {id: 'nearest', label: 'Mean nearest-neighbour distance'},
    {
      id: 'quadrat',
      label: 'Quadrat dispersion',
      help: 'Variance-to-mean ratio of the quadrat counts; 1 for a random pattern, above 1 for clustering.'
    },
    {id: 'timing', label: 'Pattern graph timing'}
  ],

  legends: state => [
    state.colorPoints === 'nearest'
      ? {
          kind: 'ramp' as const,
          id: 'nearest',
          title: 'Distance to nearest neighbour',
          ramp: 'viridis' as const,
          extent: 'gpu' as const,
          sqrtScale: true,
          unit: 'm',
          format: (value: number) => `${value.toFixed(0)}`
        }
      : {
          kind: 'categories' as const,
          title: state.subject === 'observations' ? 'Nature observations' : 'Places',
          entries: [
            {
              color: (state.subject === 'observations'
                ? [96, 200, 110, 230]
                : [64, 196, 255, 230]) as [number, number, number, number],
              label: 'Selected group'
            },
            {
              color: [120, 130, 150, 120] as [number, number, number, number],
              label: 'Other groups (faint)'
            }
          ]
        },
    ...(state.showQuadrats
      ? [
          {
            kind: 'ramp' as const,
            id: 'quadrat',
            title: 'Points per quadrat',
            ramp: 'magma' as const,
            extent: 'gpu' as const,
            sqrtScale: true,
            format: (value: number) => `${value.toFixed(0)}`
          }
        ]
      : [])
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURipley, GPURipleyDistanceFunctions, GPUPointPatternIndices,
  getGPURipleyParameterValues, getGPURipleyDistanceParameterValues,
  getGPUPointPatternIndicesParameterValues
} from '@luma.gl/experimental/gpu-dataframe';

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

// window, radius and edge correction are parameter writes
ripleyParameters.write(getGPURipleyParameterValues({
  bounds, maximumDistance: ${state.maximumDistance}, edgeCorrection: '${state.ripleyCorrection}'
}));
distanceParameters.write(getGPURipleyDistanceParameterValues({
  bounds, maximumDistance: ${state.maximumDistance}, edgeCorrection: '${state.distanceCorrection}'
}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: "`GPURipley` counts pairs of points within growing radii to estimate K(r) and Besag's L(r) = sqrt(K/pi), with edge corrections. `GPURipleyDistanceFunctions` adds the nearest-neighbour function G, the empty-space function F and J = (1-G)/(1-F). `GPUPointPatternIndices` gives each point's nearest-neighbour distance, the Clark-Evans ratio and quadrat dispersion. Parity target: spatstat `Kest`, `Lest`, `Gest`, `Fest`, `Jest` (rectangle windows) and pointpats.",
    why: 'Before modelling a point pattern you need to know how it departs from complete spatial randomness: at what distance do events attract, and is the clustering real or an artefact of the window? The curves answer that scale by scale.',
    howToRead:
      'Curves are drawn as small bar charts from the smallest to the largest radius. L(r) - r above zero is clustering at that radius. G rising early and F rising late both indicate clustering; J below 1 agrees. Clark-Evans R below 1 and a quadrat variance-to-mean ratio above 1 are the single-number versions.'
  },

  create: async ctx => (await import('./point-patterns.compute')).createPointPatterns(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['groupCategory', 'maximumDistance'],
      readouts: ['lCurve', 'lReading'],
      title: 'Are bird sightings clustered, and at what scale?',
      body: "A map of 11,198 bird observations looks clumped, but clumped compared with what? **`GPURipley`** counts, for every sighting, how many other sightings lie within each radius, and compares the total with what a completely random scatter of the same density would give. The result is Besag's **L(r) − r**: zero for randomness, positive for clustering.\n\nThe bar chart under **L(r) − r** runs from the smallest to the largest radius. Tall bars everywhere means sightings are clustered at every scale up to 1.5 km (change the **Observation group** or the **Largest radius** below to test it). Part of that is simply the city's shape: the study window is a rectangle that includes Lake Michigan, where nobody logs a bird.",
      options: {
        subject: 'observations',
        groupCategory: 'Birds',
        maximumDistance: 1500,
        window: 'city',
        jitter: 0
      },
      highlight: {readout: 'lCurve'}
    },
    {
      id: 'k-and-l',
      controls: ['maximumDistance', 'groupCategory'],
      readouts: ['lPeak', 'lCurve'],
      title: 'Reading K and L: the scale of clustering',
      body: 'K(r) is the average number of other points within distance r of a point, divided by the intensity; L(r) = √(K/π) turns it into a distance so that "L(r) − r = 0" is easy to read. The readout **Strongest clustering** gives the radius where the excess is largest.\n\nDrag **Largest radius** to 4 km: the excess is already huge at under a kilometre, the scale of a park or a stretch of lakefront, and then levels off, because beyond that the city simply runs out of places where anyone looks. Switch **Observation group** to *Mammals* (1,089 points): fewer pairs, noisier curve, same message.',
      options: {maximumDistance: 4000},
      highlight: {readout: 'lPeak'}
    },
    {
      id: 'edge-correction',
      controls: ['ripleyCorrection', 'maximumDistance'],
      readouts: ['lCurve'],
      title: 'Edges bias the estimate',
      body: 'Points near the window edge have neighbours missing outside it, which lowers the count and biases K. **Isotropic** correction weights every pair by the inverse fraction of its circle inside the window; **border** discards points closer to the edge than r; **none** ignores the problem.\n\nHere **Edge correction of K and L** is set to *None*: compare the curve at large radii with *Isotropic* or *Border*. Choose the correction before trusting the largest radii, and keep the radius below half the shorter side of the window.',
      options: {ripleyCorrection: 'none', maximumDistance: 4000}
    },
    {
      id: 'distance-functions',
      controls: ['distanceCorrection', 'colorPoints'],
      readouts: ['gCurve', 'fCurve', 'jCurve'],
      title: 'G, F and J: nearest-neighbour views',
      body: '**`GPURipleyDistanceFunctions`** looks at distances to the *nearest* point rather than counting all pairs. G(r) is the fraction of sightings whose nearest other sighting is within r; F(r) is the fraction of a regular lattice of locations that lie within r of any sighting. Clustered patterns have G rising early and F rising late; **J = (1 − G)/(1 − F)** below 1 summarises that.\n\nThe map now colors each point by its **nearest-neighbour distance** from `GPUPointPatternIndices`. The distance functions have their own edge corrections: try Kaplan-Meier, Hanisch and None under **Edge correction of G, F and J** below, and switch **Color points by** to compare.',
      options: {
        ripleyCorrection: 'isotropic',
        maximumDistance: 1500,
        distanceCorrection: 'kaplan-meier',
        colorPoints: 'nearest'
      },
      highlight: {readout: 'gCurve'}
    },
    {
      id: 'clark-evans-and-stacking',
      controls: ['showQuadrats', 'jitter'],
      readouts: ['clarkEvans', 'quadrat', 'nearest'],
      title: 'Clark-Evans, quadrats and shared coordinates',
      body: '**Clark-Evans** compares the mean nearest-neighbour distance with the one expected for random points, R = observed / expected; below 1 is clustered. **Quadrat counts** cut the window into a grid and compare the variance of the counts with their mean (VMR, above 1 clustered). Turn on **Show quadrat counts** to see the grid.\n\nAbout one observation in six (7,033 of 43,557) shares its exact coordinate with another, so some nearest-neighbour distances are exactly 0 and pull R down. Slide **Jitter shared coordinates** up to 30 m and watch R and the nearest-neighbour map change while L(r) at 1 km barely moves.',
      options: {
        colorPoints: 'selection',
        showQuadrats: true,
        jitter: 30,
        distanceCorrection: 'border'
      }
    },
    {
      id: 'wildlife-vs-places',
      controls: ['subject', 'placeCategory'],
      readouts: ['lCurve', 'previousCurve'],
      title: 'Compare with restaurants and cafes',
      body: 'Switch **Point set** to places and the same graph logic runs on 11,600 restaurants and cafes. They cluster along commercial streets at a different scale from birds, which cluster on the lakefront and in parks: compare the shape of **L(r) − r** with the **Previous selection L(r) - r** curve, which keeps the last selection for comparison. The jitter is switched off here.\n\nChange **Place type** to *Grocery and convenience* (1,255 places) for a sparser, more regular pattern.',
      options: {
        subject: 'places',
        placeCategory: 'restaurant_cafe',
        showQuadrats: true,
        jitter: 0,
        colorPoints: 'selection'
      }
    },
    {
      id: 'limits',
      controls: ['window', 'maximumDistance', 'groupCategory', 'measure'],
      readouts: ['timing'],
      title: 'Limits and things to try',
      body: 'Ripley\'s functions assume a stationary pattern in a rectangle: a city with a lake, parks and industrial land violates that, so "clustered" often just means "where people go". For a fair test, shrink the window to a neighbourhood by setting **Study window** to *Current map view* and panning around. Use the **Largest radius** with care: the isotropic weights are capped, and the largest radii use the fewest pairs.\n\nTry, with **Observation group**: Plants against Fungi; with **Point set**, Transit stops (a nearly regular pattern); the map-view window over Lincoln Park; **Time the pattern graph** with *All groups* (43,557 points).',
      options: {
        subject: 'observations',
        groupCategory: 'Fungi',
        window: 'city',
        showQuadrats: false,
        colorPoints: 'selection'
      }
    }
  ]
});
