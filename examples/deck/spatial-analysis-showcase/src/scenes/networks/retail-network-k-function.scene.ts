// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {defineScene} from '../scene';
import type {RetailNetworkKOptions} from './retail-network-k-function.compute';

const CATEGORY_LABELS: Record<RetailNetworkKOptions['category'], string> = {
  retail: 'retail',
  restaurant_cafe: 'restaurants and cafés',
  grocery: 'groceries',
  combined: 'shops, cafés and groceries'
};
const LOAD_CLASSES = makeClassTable({
  breaks: [1, 3, 8, 20, 50],
  scheme: 'YlOrRd',
  labels: ['0–1', '1–3', '3–8', '8–20', '20–50', '>50'],
  unit: 'mapped Overture places per km',
  noData: {color: [105, 112, 125, 255], label: 'no mapped places'},
  method:
    'Six fixed first-order street-load classes; segments shorter than 50 m use the guard denominator.'
});
const SEARCH_DISTANCE_CLASSES = makeClassTable({
  breaks: [0.25, 0.5, 0.75],
  colors: [
    [215, 240, 255, 255],
    [145, 205, 245, 255],
    [70, 155, 220, 255],
    [20, 95, 175, 255]
  ],
  labels: ['0–¼ d', '¼–½ d', '½–¾ d', '¾–d'],
  unit: 'selected-place network distance',
  noData: {color: [105, 112, 125, 255], label: 'outside selected reach'},
  method: 'CPU provenance display; classes are fractions of the selected maximum network distance.'
});

/** A second-order network pattern analysis of Overture places. */
export default defineScene<RetailNetworkKOptions>({
  id: 'retail-network-k-function',
  title: 'Do Chicago shops bunch along streets?',
  chapter: 'networks',
  order: 6,
  summary:
    'Network-constrained Ripley K compares mapped Chicago retail places with length-weighted random locations on the same street graph.',
  contributors: ['GPUNetworkKFunction'],
  datasets: [
    {id: 'chicago-roads', role: 'street graph (CSR, lengths in metres)'},
    {id: 'chicago-places', role: 'Overture retail places (position and category)'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10.1},
  basemap: ground('night'),
  furniture: {
    title: {title: 'Retail places on the street network'},
    credit: 'Overture Maps Foundation · OpenStreetMap contributors (ODbL)'
  },
  options: [
    {
      kind: 'select',
      id: 'category',
      label: 'Place category',
      group: 'Places',
      apply: 'param',
      default: 'retail',
      help: 'Chooses the mapped Overture places that become events and the street load.',
      options: [
        {value: 'retail', label: 'Retail'},
        {value: 'restaurant_cafe', label: 'Restaurants and cafés'},
        {value: 'grocery', label: 'Groceries'},
        {value: 'combined', label: 'Combined retail and food'}
      ]
    },
    {
      kind: 'slider',
      id: 'eventCount',
      label: 'Places in the sample',
      group: 'Places',
      apply: 'param',
      min: 20,
      max: 256,
      step: 4,
      default: 192,
      help: 'K uses a reproducible sample; the 256-place cap keeps observed and simulated network searches interactive.'
    },
    {
      kind: 'slider',
      id: 'sampleSeed',
      label: 'Sample seed',
      group: 'Places',
      apply: 'param',
      min: 1,
      max: 30,
      step: 1,
      default: 1,
      help: 'Changes the deterministic sample of eligible places, not the street-length null.'
    },
    {
      kind: 'slider',
      id: 'maxDistance',
      label: 'Maximum network distance',
      group: 'K function',
      apply: 'param',
      min: 200,
      max: 2000,
      step: 100,
      default: 1200,
      unit: 'm',
      help: 'The largest network distance d. K is cumulative from zero through this value.'
    },
    {
      kind: 'slider',
      id: 'maxSnapDistance',
      label: 'Maximum snap distance',
      group: 'K function',
      apply: 'param',
      min: 20,
      max: 200,
      step: 10,
      default: 60,
      unit: 'm',
      help: 'Places farther than this from a graph segment are excluded rather than forced onto a street.'
    },
    {
      kind: 'slider',
      id: 'simulations',
      label: 'Random street patterns',
      group: 'Envelope',
      apply: 'param',
      min: 0,
      max: 19,
      step: 1,
      default: 19,
      help: 'Length-weighted random places on the same streets form a pointwise min–max envelope; 19 is the compiled maximum.'
    },
    {
      kind: 'slider',
      id: 'envelopeSeed',
      label: 'Envelope seed',
      group: 'Envelope',
      apply: 'param',
      min: 1,
      max: 50,
      step: 1,
      default: 1,
      help: 'Selects a reproducible set of simulated street patterns.'
    },
    {
      kind: 'select',
      id: 'bandCount',
      label: 'Distance bands',
      group: 'Compile-time',
      apply: 'compile',
      default: '24',
      help: 'More thresholds smooth the cumulative curve and rebuild the compiled graph.',
      options: [
        {value: '12', label: '12 bands'},
        {value: '24', label: '24 bands'},
        {value: '48', label: '48 bands'},
        {value: '96', label: '96 bands'}
      ]
    },
    {
      kind: 'select',
      id: 'rowsPerBlock',
      label: 'Searches per block',
      group: 'Compile-time',
      apply: 'compile',
      default: '128',
      help: 'Controls concurrent shortest-path scratch memory and rebuilds the graph.',
      options: [
        {value: '32', label: '32 rows'},
        {value: '64', label: '64 rows'},
        {value: '128', label: '128 rows'}
      ]
    },
    {
      kind: 'toggle',
      id: 'spatialSort',
      label: 'Morton-sort edges for snapping',
      group: 'Compile-time',
      apply: 'compile',
      default: false,
      help: 'Changes BVH construction order for snapping; the analytical result is unchanged.'
    },
    {
      kind: 'select',
      id: 'roadStyle',
      label: 'Street measure',
      group: 'Display',
      apply: 'param',
      default: 'perKm',
      help: 'Maps first-order place intensity separately from the second-order K statistic.',
      options: [
        {value: 'perKm', label: 'Mapped places per km'},
        {value: 'plain', label: 'Plain street graph'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showEvents',
      label: 'Show sampled places',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws observed sampled places; the envelope represents the simulated patterns.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time the K function',
      group: 'Timing',
      help: 'Runs the observed pattern and active simulations outside the frame and reports the median time.'
    }
  ],
  legends: state => [
    ...(state.roadStyle === 'perKm'
      ? [
          getClassTableLegend(LOAD_CLASSES, {
            id: 'placesPerKm',
            title: `Mapped ${CATEGORY_LABELS[state.category]} places per km`,
            note: LOAD_CLASSES.method
          })
        ]
      : []),
    {
      kind: 'categories' as const,
      title: 'K-function sample',
      entries: [{color: [0, 120, 200, 235] as const, label: 'Observed mapped place'}],
      note: 'The envelope, not a planar random-dot map, is the street-length null.'
    },
    getClassTableLegend(SEARCH_DISTANCE_CLASSES, {
      id: 'searchBall',
      title: 'Selected-place network reach',
      note: SEARCH_DISTANCE_CLASSES.method
    })
  ],
  readouts: [
    {id: 'network', label: 'Street graph'},
    {id: 'length', label: 'Network length L'},
    {id: 'eligible', label: 'Eligible mapped places', format: 'integer'},
    {id: 'eventsUsed', label: 'Places requested', format: 'integer'},
    {id: 'events', label: 'Places snapped to the network'},
    {id: 'pairs', label: 'Pairs within maximum distance', format: 'integer'},
    {id: 'kCurve', label: 'Observed and null K curve', kind: 'chart'},
    {id: 'firstOutside', label: 'First outside pointwise envelope'},
    {id: 'sampleFraction', label: 'Eligible-place sample fraction'},
    {id: 'kAtMaximum', label: 'Observed K at maximum distance', format: 'decimal'},
    {id: 'envelopeRange', label: 'Simulated range at maximum'},
    {id: 'ratio', label: 'Observed K over mean simulated K'},
    {id: 'bands', label: 'Envelope comparison'},
    {id: 'verdict', label: 'Pointwise reading'},
    {id: 'converged', label: 'Searches converged'},
    {id: 'rows', label: 'Active search rows'},
    {id: 'blocks', label: 'Search blocks'},
    {id: 'time', label: 'Measured time'}
  ],
  snippet: state =>
    `const k = new GPUNetworkKFunction({points, nodePositions, offsets, neighbors, weights, bandCount: ${state.bandCount}, simulationCount: 19});\nparameters.write(getGPUNetworkKFunctionParameterValues({seed: ${state.envelopeSeed}, activeSimulations: ${state.simulations}}));`,
  about: {
    what: '`GPUNetworkKFunction` snaps sampled mapped places to a street graph, measures shortest-path distances, and counts unordered pairs strictly closer than each distance band.',
    why: 'A street-bound dataset looks clustered against a uniform plane. This null scatters points along streets in proportion to street length; it does not control footfall, zoning, or Overture coverage.',
    howToRead:
      'Street colours are first-order mapped places per km. K is second-order: values above the simulated mean indicate more close network pairs than the street-length null. The min–max envelope is pointwise, not a global multiple-comparison test.'
  },
  create: async ctx =>
    (await import('./retail-network-k-function.compute')).createRetailNetworkKFunction(ctx),
  story: [
    {
      id: 'retail-load',
      headline: 'Street load is not yet a clustering result',
      textAlternative:
        'Six fixed street-load classes show mapped Overture places per kilometre and solid observed selection-blue rings.',
      optionsMode: 'fresh',
      title: 'Mapped places are a first-order pattern',
      body: 'The classed street map shows **mapped Overture places per km**, not a clustering test. Change **Place category** below to see how the measurement changes; longer segments do not automatically look busier.',
      controls: ['category', 'roadStyle'],
      readouts: ['eligible', 'events']
    },
    {
      id: 'choose-null',
      headline: 'The null must preserve the street network',
      textAlternative:
        'Observed places are contrasted with hollow neutral street-length random rings; any planar rectangle illustration is explicitly rejected.',
      optionsMode: 'fresh',
      title: 'Random must live on streets',
      body: 'A rectangle-wide random pattern would make every street-bound category look clustered. `GPUNetworkKFunction` instead samples random locations in proportion to street length, then compares the same cumulative pair count.',
      controls: ['simulations', 'envelopeSeed'],
      readouts: ['network', 'length']
    },
    {
      id: 'network-distance',
      headline: 'A close pair is measured along streets, not through blocks',
      textAlternative:
        'A selected mapped place has a bounded cool network-distance search and a dashed straight-line radius ring.',
      optionsMode: 'fresh',
      title: 'Distance follows the network',
      body: 'Slide **Maximum network distance** below. The GPU runs bounded shortest-path searches from sampled places; pairs are counted only when their graph distance is strictly below each threshold.',
      controls: ['maxDistance', 'maxSnapDistance'],
      readouts: ['pairs', 'kAtMaximum', 'converged']
    },
    {
      id: 'envelope',
      headline: 'Only a simulated reference gives the curve meaning',
      textAlternative:
        'A K chart has observed and simulated mean lines, a grey pointwise min-max envelope, a null guide and outside markers.',
      optionsMode: 'fresh',
      title: 'A curve needs a yardstick',
      body: 'Use **Random street patterns** below to build a pointwise envelope from up to 19 length-weighted simulations. It is a diagnostic at each distance, not a single global significance test.',
      controls: ['simulations', 'envelopeSeed'],
      readouts: ['envelopeRange', 'ratio', 'bands']
    },
    {
      id: 'categories',
      headline: 'Categories can differ without proving a cause',
      textAlternative:
        'The category comparison preserves valid fixed first-order and K scales without making a causal claim.',
      optionsMode: 'fresh',
      title: 'Business types make different patterns',
      body: 'Switch **Place category** while retaining the same network null. Differences can reflect commercial geography, the Overture source mix, or both; this map does not establish a cause.',
      controls: ['category', 'sampleSeed'],
      readouts: ['eligible', 'verdict']
    },
    {
      id: 'limits',
      headline: 'Street length is a deliberately limited null model',
      textAlternative:
        'Live sample fraction, pair count and coverage caveats describe a bounded network K calculation.',
      optionsMode: 'fresh',
      title: 'Street length is only one control',
      body: 'Try a different **Sample seed** or more **Distance bands**. The analysis caps samples at 256, uses cumulative bands, clips at the graph boundary, and does not control footfall, zoning, or incomplete business coverage.',
      controls: ['sampleSeed', 'bandCount', 'rowsPerBlock'],
      readouts: ['rows', 'blocks', 'time'],
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10.1}
    }
  ]
});
