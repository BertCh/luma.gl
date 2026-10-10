// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {CITY_FRAMES, labelsFor, MONTREAL} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './bixi-communities.md?raw';
import type {BixiCommunitiesOptions} from './bixi-communities.compute';
import {getCommunityLegends} from './bixi-communities-style';
import {FLOW_CREDITS} from './flows-style';

/** Mirrors the compute module without importing it (scene files stay light). */
const REPLAY_LAST_ROUND = 32;

const CREDIT = joinCredits(
  FLOW_CREDITS.bixi,
  FLOW_CREDITS.montrealBoroughs,
  CREDITS.openStreetMap,
  CREDITS.okabeIto
);

/** The cartouche of a step: line 1 here, the subtitle and sample line come from the compute module. */
const cartouche = (title: string) => ({title: {title}});

function getSnippet(state: BixiCommunitiesOptions): string {
  return `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGraph, GPUGraphTopology, GPUGraphLabelPropagation,
  GPUGraphModularityOptimization, GPUGraphModularity
} from '@luma.gl/gpgpu/gpu-graph';

// Caller-owned vectors: edge columns (rides as weights), CSR, and every output.
// Edges below ${state.minRides} rides/day, beyond ${state.maxDistanceKm} km or outside the
// strongest-${state.neighbors} neighborhood get an out-of-domain source: filters are buffer writes.
const graph = new GPUGraph({vertexCount: 905, sourceVertices, targetVertices,
  edgeWeights: ${state.weighting === 'rides' ? 'ridesPerDay' : state.weighting === 'sqrt' ? 'sqrtRidesPerDay' : 'ones'}, directed: false});
const topology = new GPUGraphTopology({graph, forward, invalidEdgeCount});

const commandGraph = new GPUCommandGraph(device, {id: 'communities'});
topology.addToGraph(commandGraph);
new GPUGraphLabelPropagation({
  topology, output: proposal, iterations: ${state.propagationRounds}, converged   // unweighted majority vote
}).addToGraph(commandGraph);
new GPUGraphModularityOptimization({
  topology, output: refined, modularity: refinedScore,
  initialCommunities: proposal,                    // warm start
  resolution: ${state.resolution}, iterations: ${state.optimizeRounds}, minimumGain: ${state.minimumGain},
  converged, valid
}).addToGraph(commandGraph);
new GPUGraphModularity({graph, communities: boroughs, output: boroughScore,
  resolution: ${state.resolution}}).addToGraph(commandGraph);   // same gamma for the borough partition

const compiled = commandGraph.compile();            // once
compiled.encode(commandEncoder, {parameters: undefined});

// on the CPU, from the readback (a few KB): hues, hulls, seams
const groups = assignGroupHues(refinedLabels, stationLngLat, previousHues);   // matchByOverlap`;
}

export default defineScene<BixiCommunitiesOptions>({
  id: 'bixi-communities',
  title: 'Do riders follow the borough map?',
  chapter: 'flows',
  order: 10,
  summary:
    'Label propagation and modularity optimization on the BIXI station graph (August 2024), with identity-stable hues, a CPU replay of the vote, the boroughs of the agglomeration swiped against the riding groups, and modularity against resolution.',
  contributors: [
    'GPUGraphLabelPropagation',
    'GPUGraphModularityOptimization',
    'GPUGraphModularity'
  ],
  datasets: [
    {id: 'bixi-flows', role: 'station pairs, August 2024'},
    {id: 'montreal-boroughs', role: 'borough outlines and the borough partition key'}
  ],
  initialView: {...CITY_FRAMES.montreal},

  options: [
    {
      kind: 'select',
      id: 'partition',
      label: 'Partition shown',
      group: 'Display',
      apply: 'param',
      display: 'segmented',
      default: 'optimized',
      help: 'Which grouping of stations colours the map: the refined riding groups, the label-propagation proposal, or the boroughs published with the data. All three are always scored.',
      options: [
        {value: 'optimized', label: 'Riding groups'},
        {value: 'propagation', label: 'Propagation'},
        {value: 'boroughs', label: 'Boroughs'}
      ]
    },
    {
      kind: 'slider',
      id: 'replayRound',
      label: 'Voting round',
      group: 'Display',
      apply: 'param',
      min: 0,
      max: REPLAY_LAST_ROUND,
      step: 1,
      default: REPLAY_LAST_ROUND,
      display: 'stepper',
      autoSweep: {durationMs: 9000, from: 0, to: REPLAY_LAST_ROUND},
      disabledWhen: state => state.partition !== 'propagation',
      describe: (value, state) =>
        value >= state.propagationRounds
          ? 'The GPU result: the last round'
          : value === 0
            ? 'Every station is its own group'
            : `Round ${value} of ${state.propagationRounds}, replayed on the CPU`,
      help: 'Shows the label-propagation partition after this many synchronous voting rounds. It is a CPU replay of the same vote; at the last round it is checked against the labels the GPU wrote.'
    },
    {
      kind: 'toggle',
      id: 'showHulls',
      label: 'Group wash',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A pale convex hull and outline around each group that has a hue, after dropping stations far from the group.'
    },
    {
      kind: 'toggle',
      id: 'showBoroughs',
      label: 'Borough outlines',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'The boroughs and linked cities of the agglomeration, from the Ville de Montréal open data. Stations in Laval, Longueuil and the south shore have no outline.'
    },
    {
      kind: 'toggle',
      id: 'showSeams',
      label: 'Seam stations',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Rings the stations whose riding group is mostly in another borough than their own.'
    },
    {
      kind: 'toggle',
      id: 'showBetween',
      label: 'Links between groups',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draw the faint grey links that cross a group boundary, under the links inside groups.'
    },
    {
      kind: 'slider',
      id: 'edges',
      label: 'Links drawn',
      group: 'Display',
      apply: 'param',
      min: 200,
      max: 20000,
      step: 100,
      default: 3000,
      help: 'How many of the busiest station pairs are drawn, busiest first. The graph itself always uses every pair that passes the filters.'
    },
    {
      kind: 'slider',
      id: 'neighbors',
      label: 'Strongest links per station',
      group: 'Graph',
      apply: 'param',
      min: 2,
      max: 60,
      step: 1,
      default: 8,
      help: 'A pair stays in the graph when it is among this many strongest links of either of its stations. A dense graph makes label propagation collapse into one blob; a sparse backbone keeps the groups apart. A buffer write: no rebuild.'
    },
    {
      kind: 'slider',
      id: 'minRides',
      label: 'Minimum rides per day',
      group: 'Graph',
      apply: 'param',
      min: 0,
      max: 50,
      step: 0.5,
      default: 1,
      unit: 'rides/day',
      help: 'Drops pairs below this average-day rate. Weekdays divide by 22 days and weekends by nine before this threshold, so changing Day type compares rates rather than unequal August totals.'
    },
    {
      kind: 'slider',
      id: 'maxDistanceKm',
      label: 'Maximum link distance',
      group: 'Graph',
      apply: 'param',
      min: 0.5,
      max: 50,
      step: 0.5,
      default: 50,
      unit: 'km',
      help: 'Drops station pairs farther apart than this straight-line distance. Lower it to test how much of the apparent community structure comes from geographic proximity alone.'
    },
    {
      kind: 'select',
      id: 'weighting',
      label: 'Edge weighting',
      group: 'Graph',
      apply: 'param',
      display: 'segmented',
      default: 'rides',
      help: 'How a pair counts in the modularity objective and its score: by average rides per day, by their square root (evens out the downtown giants) or equally. Label propagation ignores weights.',
      options: [
        {value: 'rides', label: 'Rides'},
        {value: 'sqrt', label: 'Square root'},
        {value: 'equal', label: 'Equal'}
      ]
    },
    {
      kind: 'select',
      id: 'dayType',
      label: 'Day type',
      group: 'Graph',
      apply: 'param',
      display: 'segmented',
      default: 'all',
      help: 'Build the graph from average daily rates for all days, weekdays or weekend days. Weekday and weekend graphs use pairs with at least ten August rides from the hourly rows, so their eligible topology is narrower than the all-days graph.',
      options: [
        {value: 'all', label: 'All days'},
        {value: 'weekday', label: 'Weekdays'},
        {value: 'weekend', label: 'Weekends'}
      ]
    },
    {
      kind: 'slider',
      id: 'propagationRounds',
      label: 'Voting rounds',
      group: 'Algorithm',
      apply: 'compile',
      min: 1,
      max: 100,
      step: 1,
      default: 32,
      help: 'Synchronous voting rounds, all encoded without early exit. A compile-time constant, so changing it rebuilds the graph.'
    },
    {
      kind: 'slider',
      id: 'optimizeRounds',
      label: 'Refinement rounds',
      group: 'Algorithm',
      apply: 'compile',
      min: 0,
      max: 1024,
      step: 32,
      default: 512,
      help: 'Rounds of modularity optimization; each accepts only the single best station move. Zero keeps the propagation partition. Compile-time.'
    },
    {
      kind: 'slider',
      id: 'resolution',
      label: 'Resolution',
      group: 'Algorithm',
      apply: 'compile',
      min: 0.25,
      max: 3,
      step: 0.25,
      default: 1,
      marks: [{value: 1, label: 'default'}],
      danger: [2.5, 3],
      describe: value =>
        value >= 2.5
          ? 'Fine: many small groups, most of them grey'
          : value <= 0.5
            ? 'Coarse: a few large groups'
            : 'Groups the size of a few neighbourhoods',
      help: 'Modularity resolution gamma. Below 1 favours few large groups, above 1 many small ones. It scales the null-model term, so it is a shader constant: the graph rebuilds once the slider rests.'
    },
    {
      kind: 'select',
      id: 'minimumGain',
      label: 'Minimum gain',
      group: 'Algorithm',
      apply: 'compile',
      display: 'segmented',
      default: '0',
      expert: true,
      help: 'A move must raise modularity by more than this to be accepted. Larger values stop earlier and keep the partition closer to the propagation.',
      options: [
        {value: '0', label: '0 (any)'},
        {value: '0.0001', label: '0.0001'},
        {value: '0.001', label: '0.001'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showSweep',
      label: 'Chart Q against resolution',
      group: 'Algorithm',
      apply: 'param',
      default: false,
      help: 'Runs eight more analyses, one per resolution from 0.25 to 3, one after the other (a compiled graph each, run once), and charts the modularity of the refined partition and of the boroughs.'
    }
  ],

  readouts: [
    {
      id: 'communityCount',
      label: 'Groups',
      format: 'integer',
      help: 'Distinct groups in the partition shown, of any size. Only the seven largest groups of eight or more stations take a hue.'
    },
    {id: 'largestCommunity', label: 'Largest group', help: 'Named after its main boroughs.'},
    {
      id: 'modularityRefined',
      label: 'Modularity of the riding groups',
      format: 'decimal',
      help: 'Weighted modularity Q of the refined partition: rides inside groups minus what a random network with the same degrees would keep inside. Zero is chance; 0.3 to 0.7 is typical of real structure.'
    },
    {
      id: 'modularityPropagation',
      label: 'Modularity of propagation',
      format: 'decimal',
      help: 'Q of the label-propagation proposal at the same resolution.'
    },
    {
      id: 'modularityBoroughs',
      label: 'Modularity of the boroughs',
      format: 'decimal',
      help: 'Q of the borough partition at the same resolution.'
    },
    {
      id: 'qualityPair',
      label: 'Modularity, boroughs and riding groups',
      help: 'Both scored with the same resolution and weights.'
    },
    {
      id: 'withinShare',
      label: 'Rides inside a group',
      format: 'percent',
      help: 'Share of the rides on the kept links whose two stations share a group in the partition shown. It rises with bigger groups for no reason of behaviour: modularity subtracts that.'
    },
    {
      id: 'withinBoroughShare',
      label: 'Rides inside a borough',
      format: 'percent',
      help: 'The same share for the borough partition.'
    },
    {
      id: 'seamStations',
      label: 'Seam stations',
      format: 'integer',
      help: 'Stations whose riding group is mostly in another borough than their own.'
    },
    {
      id: 'agreement',
      label: 'Agreement with boroughs',
      format: 'decimal',
      help: 'Normalised mutual information between the partition shown and the boroughs: 1 identical, 0 unrelated. It also depends on how many groups there are.'
    },
    {
      id: 'medianLink',
      label: 'Median ride distance',
      help: 'Ride-weighted median straight-line distance between the two stations of the kept links.'
    },
    {
      id: 'replayMatch',
      label: 'CPU replay matches GPU',
      help: 'The CPU replay of the synchronous vote, at its last round, compared with the labels the GPU wrote.'
    },
    {
      id: 'propagationStatus',
      label: 'Propagation',
      help: 'Whether the vote reached a fixed point.'
    },
    {
      id: 'refinementStatus',
      label: 'Refinement',
      help: 'Whether the single-move refinement converged.'
    },
    {
      id: 'changesChart',
      label: 'Stations that changed label, by round',
      kind: 'chart',
      help: 'Stations whose label changed in each voting round. Click a bar to show that round.'
    },
    {
      id: 'qualityChart',
      label: 'Modularity of three partitions',
      kind: 'chart',
      help: 'Weighted modularity of the boroughs, of label propagation and of the refined partition. Zero is chance; 0.3 to 0.7 is typical.'
    },
    {
      id: 'sweepChart',
      label: 'Modularity against resolution',
      kind: 'chart',
      help: 'Q of the refined partition and of the boroughs at eight resolutions. Click to set the resolution.'
    },
    {id: 'sweepStatus', label: 'Resolution scan', hood: true},
    {id: 'selected', label: 'Selected station', layout: 'block'},
    {id: 'stations', label: 'Stations', format: 'integer', hood: true},
    {id: 'edgesKept', label: 'Edges in the graph', hood: true},
    {
      id: 'graphBasis',
      label: 'Eligible-pair basis',
      hood: true,
      help: 'All days start from month pairs with at least three rides. Weekday and weekend graphs start from hourly pairs with at least ten August rides. Every weight is converted to rides per average day.'
    },
    {
      id: 'candidateCoverage',
      label: 'Source rides represented before filters',
      format: 'percent',
      hood: true,
      help: 'Share of the exact station-to-station ride rate represented by the eligible pair table, before the neighbor, rate and distance filters.'
    },
    {
      id: 'ridesKept',
      label: 'Source rides kept after filters',
      format: 'percent',
      hood: true,
      help: 'Share of the exact station-to-station ride rate carried by the graph after the neighbor, rate and distance filters.'
    },
    {
      id: 'isolatedStations',
      label: 'Stations isolated by filters',
      hood: true,
      help: 'Stations with no surviving graph edge. Their singleton labels are an input/filter outcome, not a discovered riding community.'
    },
    {id: 'validity', label: 'Graph status', hood: true}
  ],

  pipeline: [
    {
      id: 'csr',
      label: 'Graph',
      detail: 'Station pairs become a weighted graph; filters are a buffer write'
    },
    {
      id: 'vote',
      label: 'Vote',
      detail: 'Label propagation: synchronous majority votes among neighbours'
    },
    {
      id: 'move',
      label: 'Move',
      detail: 'Modularity optimisation: the best single station move per round'
    },
    {id: 'score', label: 'Score', detail: 'Modularity of propagation, refinement and the boroughs'},
    {id: 'draw', label: 'Draw', detail: 'Hues, hulls and seams on the CPU from a few KB of labels'}
  ],

  legends: getCommunityLegends,

  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Do riders follow the borough map?'},
    scaleBar: {units: 'metric'},
    credit: CREDIT,
    caveat: 'A spatial network also clusters by distance: read the groups with care.'
  },

  snippet: getSnippet,

  about: {
    what: 'Previously: communities of airports (airline-network). Next: bundles of rides (bixi-bundles).\n\n`GPUGraphLabelPropagation` groups stations by the label most common among themselves and their neighbours, in synchronous rounds with ties to the lowest label and no use of weights. `GPUGraphModularityOptimization` improves that partition by moving one station at a time to the neighbouring group that raises weighted modularity most, and `GPUGraphModularity` scores any partition, here also the published boroughs at the same resolution. The hues, hulls, seams and the round-by-round replay of the vote are CPU work on the few KB read back.',
    why: 'Service areas, rebalancing routes and station placement should follow how people ride, not how a map is drawn. A riding-group partition scored against the borough partition tells you where the two disagree, with a number instead of an impression. How many groups you get is a resolution choice, not a fact (the resolution limit of modularity: Fortunato and Barthélemy, PNAS 2007), and a spatial network clusters by distance alone, so a group is not by itself a behaviour (Austwick, O’Brien, Strano and Viana, PLoS ONE 2013). Weekday and weekend weights are rates per average day—22 weekdays versus nine weekend days—not incomparable August totals.',
    howToRead:
      'Dots and lines take the hue of their riding group, west to east on first load and inherited afterwards, so a hue is an identity, not a rank. Only the seven largest groups of eight or more stations have a hue; the rest are grey. Pale washes are the hulls of groups; grey lines cross a boundary. Rings are seam stations. Higher modularity means more rides inside groups than a random network with the same degrees would keep there. Stations outside the agglomeration (Laval, the south shore) have no borough outline.'
  },

  create: async ctx => (await import('./bixi-communities.compute')).createBixiCommunities(ctx),

  story: storyFromMarkdown<BixiCommunitiesOptions>(narrative, {
    'the-question': {
      headline: 'Riders form regional groups',
      textAlternative:
        'Map of Montreal with BIXI stations coloured by riding group in seven hues, each group washed with a pale hull and linked by lines of its own hue.',
      optionsMode: 'fresh',
      options: {partition: 'optimized'},
      controls: ['edges', 'showBetween'],
      readouts: ['communityCount', 'largestCommunity', 'modularityRefined'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1400},
      furniture: cartouche('Do riders follow the borough map?'),
      annotations: labelsFor(MONTREAL, [
        'plateau',
        'downtown',
        'verdun',
        'villeray',
        'hochelaga',
        'mount-royal'
      ]),
      stage: 'move'
    },
    'neighbours-vote': {
      headline: 'Each station copies its neighbours',
      textAlternative:
        'The same map after a few voting rounds: stations start grey and merge into coloured groups as the round slider moves.',
      optionsMode: 'fresh',
      options: {partition: 'propagation', replayRound: 0, showBetween: false},
      controls: ['replayRound', 'neighbors'],
      readouts: ['changesChart', 'replayMatch', 'communityCount'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1200},
      furniture: cartouche('Each station copies its neighbours'),
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'villeray', 'verdun']),
      stage: 'vote'
    },
    climb: {
      headline: 'Moving single stations raises modularity',
      textAlternative:
        'The refined riding groups on the map and a bar chart of modularity for the boroughs, label propagation and the refined partition, with guides at chance and at the typical range.',
      optionsMode: 'fresh',
      options: {partition: 'optimized', showBetween: false},
      controls: ['optimizeRounds'],
      readouts: ['qualityChart', 'modularityPropagation', 'modularityRefined', 'refinementStatus'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1200},
      furniture: cartouche('Moving stations raises modularity'),
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'verdun', 'hochelaga']),
      stage: 'score'
    },
    seams: {
      headline: 'Boroughs and riding groups disagree at the seams',
      textAlternative:
        'A swipe map: on the left the stations coloured by borough, on the right by riding group, with borough outlines and rings on the stations whose group is mostly in another borough.',
      optionsMode: 'fresh',
      options: {partition: 'optimized', showBoroughs: true, showSeams: true, showBetween: false},
      controls: ['showBoroughs', 'showSeams'],
      readouts: ['seamStations', 'withinShare', 'withinBoroughShare', 'qualityPair'],
      compare: {mode: 'swipe', labels: ['Boroughs', 'Communities']},
      camera: {...CITY_FRAMES.montreal, zoom: 11.3, transitionMs: 1400},
      furniture: cartouche('Boroughs and riding groups disagree'),
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'mile-end', 'verdun', 'hochelaga']),
      stage: 'score'
    },
    resolution: {
      headline: 'Resolution decides how many groups exist',
      textAlternative:
        'The riding groups at a chosen resolution next to a line chart of modularity against resolution for the riding groups and the boroughs.',
      optionsMode: 'fresh',
      options: {partition: 'optimized', showBetween: false, showSweep: true},
      controls: ['resolution'],
      readouts: ['communityCount', 'sweepChart', 'largestCommunity'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1200},
      furniture: cartouche('How many groups is the right number?'),
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'verdun', 'hochelaga']),
      stage: 'move'
    },
    'read-with-care': {
      headline: 'Distance alone would make clusters too',
      textAlternative:
        'The riding groups again, with controls for edge weighting, average weekday or weekend rates, strongest links, minimum rate and maximum link distance.',
      optionsMode: 'fresh',
      options: {partition: 'optimized'},
      controls: ['weighting', 'dayType', 'neighbors', 'minRides', 'maxDistanceKm'],
      readouts: ['medianLink', 'agreement', 'candidateCoverage', 'ridesKept'],
      camera: {...CITY_FRAMES.montreal, transitionMs: 1200},
      furniture: cartouche('Read communities with care'),
      annotations: labelsFor(MONTREAL, ['downtown', 'plateau', 'villeray', 'verdun', 'hochelaga']),
      stage: 'csr'
    }
  })
});
