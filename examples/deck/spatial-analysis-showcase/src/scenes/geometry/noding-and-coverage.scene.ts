// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import type {NodingAndCoverageOptions} from './b3-noding-options';
import {B3_PALETTE} from './b3-palette';

const CITY = {longitude: -87.68, latitude: 41.85, zoom: 9.9};
const LOOP = {longitude: -87.635, latitude: 41.882, zoom: 13};
const US = {longitude: -96, latitude: 38.2, zoom: 3.45};

const KIND_LEGEND = [
  {color: [255, 70, 70, 255], label: 'Proper crossing (cross without sharing a vertex)'},
  {color: [255, 200, 40, 255], label: 'Touch (shared end point or vertex)'},
  {color: [255, 90, 220, 255], label: 'Collinear touch'},
  {color: [70, 205, 150, 255], label: 'Overlap (shared stretch)'},
  {color: [255, 255, 255, 255], label: 'Uncertain'}
] as const;

const logDegrees = (value: number) => {
  const degrees = 10 ** value;
  return `${degrees < 0.01 ? degrees.toExponential(1) : degrees.toFixed(2)}° (~${Math.round(degrees * 95)} km)`;
};

/** Noding, crossings, dissolve and coverage simplification. GPU work in `noding-and-coverage.compute.ts`. */
export default defineScene<NodingAndCoverageOptions>({
  id: 'noding-and-coverage',
  title: 'Crossings, networks and gap-free maps',
  chapter: 'geometry',
  order: 4,
  summary:
    'Find where Chicago streets cross the L, turn CTA routes into a routable network, dissolve US counties into states, and simplify the county map without opening a single gap.',
  contributors: [
    'GPUSegmentIntersection',
    'GPULineSplit',
    'GPULineMerge',
    'GPUNetworkNoding',
    'GPUSegmentRingAssembly',
    'GPUCoverageSimplification',
    'GPULineSimplification'
  ],
  datasets: [
    {id: 'chicago-roads', role: 'street polylines'},
    {id: 'cta-transit', role: 'L and bus route shapes'},
    {id: 'us-counties', role: 'polygon coverage'}
  ],
  initialView: CITY,
  basemap: ground('paperCity'),
  furniture: {
    title: {title: 'Noding and coverage', subtitle: 'Crossings, graph nodes and shared borders'},
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.cta, CREDITS.openStreetMap, CREDITS.usCensus)
  },

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Tool',
      group: 'Tool',
      apply: 'compile',
      default: 'crossings',
      help: 'Each tool has its own compiled graphs, built the first time you open it.',
      options: [
        {value: 'crossings', label: 'Crossings (GPUSegmentIntersection)'},
        {value: 'noding', label: 'Noding (split, merge, network)'},
        {value: 'dissolve', label: 'Dissolve counties (ring assembly)'},
        {value: 'generalise', label: 'Generalise a coverage'}
      ]
    },

    {
      kind: 'select',
      id: 'crossMode',
      label: 'Intersect',
      group: 'Crossings',
      apply: 'compile',
      default: 'streets-rail',
      disabledWhen: s => s.view !== 'crossings',
      help: 'Two-sided: street segments against the L routes. Self: streets against streets. Compile-time: the second side changes the graph.',
      options: [
        {value: 'streets-rail', label: 'Streets × L routes'},
        {value: 'streets-self', label: 'Streets × streets (self)'}
      ]
    },
    {
      kind: 'select',
      id: 'crossKinds',
      label: 'Show',
      group: 'Crossings',
      apply: 'param',
      default: 'proper',
      disabledWhen: s => s.view !== 'crossings',
      help: 'Which kinds of intersection get a marker. Self mode has tens of thousands of touches (the street junctions): the proper crossings are the interesting ones.',
      options: [
        {value: 'proper', label: 'Proper crossings'},
        {value: 'touches', label: 'Touches (junctions)'},
        {value: 'overlaps', label: 'Overlaps'},
        {value: 'all', label: 'All kinds'}
      ]
    },
    {
      kind: 'toggle',
      id: 'sameFeatureOnly',
      label: 'Same street only',
      group: 'Crossings',
      apply: 'compile',
      default: false,
      disabledWhen: s => s.view !== 'crossings' || s.crossMode !== 'streets-self',
      help: 'Self mode: keep only pairs from the same polyline (a street crossing itself), the per-feature validity question. Compile-time.'
    },
    {
      kind: 'toggle',
      id: 'crossSpatialSort',
      label: 'Morton sort the segments',
      group: 'Crossings',
      apply: 'compile',
      default: true,
      disabledWhen: s => s.view !== 'crossings',
      help: 'Reorders the right-hand segments along a Z-order curve before the BVH build. The result is identical; only the cost changes. Compile-time.'
    },

    {
      kind: 'select',
      id: 'nodingInput',
      label: 'Lines to node',
      group: 'Noding',
      apply: 'compile',
      default: 'rail',
      disabledWhen: s => s.view !== 'noding',
      help: 'The eight L routes, or every CTA route inside a 7 km window around the Loop (exact duplicates removed). Compile-time: the input size changes.',
      options: [
        {value: 'rail', label: 'The eight L routes'},
        {value: 'rail-bus', label: 'All CTA routes downtown'}
      ]
    },
    {
      kind: 'select',
      id: 'nodingOutput',
      label: 'Show',
      group: 'Noding',
      apply: 'param',
      default: 'pieces',
      disabledWhen: s => s.view !== 'noding',
      help: 'Pieces: GPULineSplit. Chains: GPULineMerge of those pieces. Network: GPUNetworkNoding edges and nodes.',
      options: [
        {
          value: 'pieces',
          label: 'Split pieces',
          help: 'Every line cut at each crossing, touch and overlap end.'
        },
        {
          value: 'merged',
          label: 'Merged chains',
          help: 'Pieces joined where exactly two ends meet.'
        },
        {
          value: 'network',
          label: 'Network: edges and nodes',
          help: 'Nodes coloured by degree; red is a dead end.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'nodingTolerance',
      label: 'Snap tolerance',
      group: 'Noding',
      apply: 'param',
      min: 0,
      max: 60,
      step: 1,
      default: 0,
      unit: ' m',
      disabledWhen: s => s.view !== 'noding',
      help: "End points in the same tolerance cell share a node. 0 merges only identical end points. A per-frame parameter; lines that stop near another line's middle are not connected."
    },
    {
      kind: 'select',
      id: 'nodingCapacity',
      label: 'Intersection capacity',
      group: 'Noding',
      apply: 'compile',
      default: '65536',
      disabledWhen: s => s.view !== 'noding',
      help: 'Size of the internal list of intersecting segment pairs. Too small and the output is cut short with the overflow flag set. Compile-time.',
      options: [
        {value: '1024', label: '1,024 (overflows downtown)'},
        {value: '16384', label: '16,384'},
        {value: '65536', label: '65,536'},
        {value: '524288', label: '524,288'}
      ]
    },
    {
      kind: 'toggle',
      id: 'nodingSpatialSort',
      label: 'Morton sort the segments',
      group: 'Noding',
      apply: 'compile',
      default: true,
      disabledWhen: s => s.view !== 'noding',
      help: 'Passed to the intersection step; changes cost only. Compile-time.'
    },

    {
      kind: 'select',
      id: 'dissolveBy',
      label: 'Dissolve counties by',
      group: 'Dissolve',
      apply: 'param',
      default: 'state',
      disabledWhen: s => s.view !== 'dissolve',
      help: 'The group label of every boundary segment: rings never mix groups. A buffer write; the graph is not recompiled.',
      options: [
        {value: 'state', label: 'State'},
        {value: 'rucc', label: 'Rural-urban continuum code (USDA, 9 classes)'},
        {value: 'none', label: 'Nothing (one nation)'}
      ]
    },
    {
      kind: 'select',
      id: 'dissolveColor',
      label: 'Outline colour',
      group: 'Dissolve',
      apply: 'param',
      default: 'group',
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Shells and holes are told apart by the ringIsHole output.',
      options: [
        {value: 'group', label: 'One colour'},
        {value: 'hole', label: 'Holes in red'}
      ]
    },
    {
      kind: 'toggle',
      id: 'cancelOpposing',
      label: 'Cancel opposing segments',
      group: 'Dissolve',
      apply: 'compile',
      default: true,
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Removes a to b and b to a pairs within a group: the shared edges between counties of one group vanish and only the outer boundary remains. Turn it off and every county stays its own ring. Compile-time.'
    },
    {
      kind: 'toggle',
      id: 'splitTouching',
      label: 'Split rings that touch',
      group: 'Dissolve',
      apply: 'compile',
      default: true,
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Where a hole touches its shell at one corner, split into two rings (the usual simple-feature convention). Compile-time.'
    },
    {
      kind: 'select',
      id: 'vertexTolerance',
      label: 'Vertex matching distance',
      group: 'Dissolve',
      apply: 'compile',
      default: '0.00005',
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Chebyshev distance in degrees within which two vertices count as one. Must be at least four times the f32 spacing and well below the shortest edge. Compile-time.',
      options: [
        {value: '0.00005', label: '0.00005° (about 5 m)'},
        {value: '0.0005', label: '0.0005° (about 50 m)'},
        {value: '0.005', label: '0.005° (about 500 m)'}
      ]
    },
    {
      kind: 'select',
      id: 'interiorSide',
      label: 'Interior side',
      group: 'Dissolve',
      apply: 'compile',
      default: 'left',
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Which side of each directed segment is filled. County rings put the interior on the left; choosing right makes shells and holes swap roles. Compile-time.',
      options: [
        {value: 'left', label: 'Left (counter-clockwise shells)'},
        {value: 'right', label: 'Right'}
      ]
    },
    {
      kind: 'toggle',
      id: 'normalizeWinding',
      label: 'Normalise winding',
      group: 'Dissolve',
      apply: 'compile',
      default: false,
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Reverse rings of right-side input so shells are counter-clockwise and holes clockwise (RFC 7946). Compile-time.'
    },
    {
      kind: 'toggle',
      id: 'showFill',
      label: 'Fill counties by group',
      group: 'Dissolve',
      apply: 'param',
      default: true,
      disabledWhen: s => s.view !== 'dissolve',
      help: 'Counties coloured by the group they are dissolved into.'
    },

    {
      kind: 'slider',
      id: 'coverageTolerance',
      label: 'Tolerance',
      group: 'Generalise',
      apply: 'param',
      min: -3.5,
      max: -0.3,
      step: 0.05,
      default: -1.5,
      format: logDegrees,
      disabledWhen: s => s.view !== 'generalise',
      help: 'Douglas-Peucker tolerance in degrees (log scale). Only the selection re-runs when it moves.'
    },
    {
      kind: 'select',
      id: 'coverageOutline',
      label: 'Outlines',
      group: 'Generalise',
      apply: 'param',
      default: 'both',
      disabledWhen: s => s.view !== 'generalise',
      help: 'Blue: GPUCoverageSimplification (shared edges simplified once). Red: every ring simplified on its own with GPULineSimplification.',
      options: [
        {value: 'both', label: 'Both'},
        {value: 'coverage', label: 'Coverage simplification'},
        {value: 'independent', label: 'Independent rings'},
        {value: 'original', label: 'Original'}
      ]
    },
    {
      kind: 'select',
      id: 'topologyRounds',
      label: 'Topology repair rounds',
      group: 'Generalise',
      apply: 'compile',
      default: '4',
      disabledWhen: s => s.view !== 'generalise',
      help: 'Rounds that find simplified segments crossing without a shared point and restore the original vertex farthest from each. 0 skips detection and repair. Compile-time.',
      options: [
        {value: '0', label: '0 (plain)'},
        {value: '1', label: '1'},
        {value: '2', label: '2'},
        {value: '4', label: '4'},
        {value: '8', label: '8'}
      ]
    },
    {
      kind: 'select',
      id: 'coverageSnap',
      label: 'Snap tolerance',
      group: 'Generalise',
      apply: 'compile',
      default: '0',
      disabledWhen: s => s.view !== 'generalise',
      help: 'Vertices that snap to the same point are one point of the coverage. 0 compares exact f32 values. Compile-time.',
      options: [
        {value: '0', label: 'Exact'},
        {value: '0.0001', label: '0.0001°'},
        {value: '0.001', label: '0.001°'}
      ]
    },
    {
      kind: 'select',
      id: 'coverageRounds',
      label: 'Maximum rounds',
      group: 'Generalise',
      apply: 'compile',
      default: '64',
      disabledWhen: s => s.view !== 'generalise',
      help: 'Cap on Douglas-Peucker rounds. A low cap leaves more vertices than the exact result. Compile-time.',
      options: [
        {value: '8', label: '8'},
        {value: '16', label: '16'},
        {value: '64', label: '64'},
        {value: '256', label: '256'}
      ]
    }
  ],

  story: [
    {
      id: 'crossings',
      title: 'Where do streets cross the L?',
      headline: 'A crossing has several geometric kinds',
      body: `This card reports geometric intersections between the loaded street and rail linework. \`GPUSegmentIntersection\` emits candidate pairs within its fixed capacity and classifies the segment relationship; uncertainty and overflow remain visible outputs.

Red marks are **proper crossings**: a street segment cuts an L segment without sharing a vertex. Each has a point and a kind; the readout counts the pairs. Set **Intersect** below to *Streets × streets (self)* to see where streets cross each other without a shared node.`,
      camera: {...CITY, transitionMs: 1000},
      options: {view: 'crossings', crossMode: 'streets-rail', crossKinds: 'proper'},
      highlight: {readout: 'crossPairs'},
      controls: ['crossMode', 'crossKinds'],
      readouts: ['crossPairs']
    },
    {
      id: 'network-node',
      title: 'Crossing does not guarantee connection',
      headline: 'Crossing does not guarantee connection',
      body: `A geometric crossing is not yet a source-network node. Compare each reported crossing with source endpoints: an endpoint match is a candidate shared node; other crossings are simply crossings without a shared source node. The archive has no structural-level attributes, so this card makes no construction claim.

Switch **Show** below between proper crossings and touches. The uncertainty and capacity readouts remain part of the interpretation.`,
      camera: {longitude: -87.65, latitude: 41.87, zoom: 11, transitionMs: 1500},
      options: {crossMode: 'streets-self', crossKinds: 'proper'},
      highlight: {readout: 'crossKinds'},
      controls: ['crossKinds', 'sameFeatureOnly'],
      readouts: ['crossKinds']
    },
    {
      id: 'noding',
      title: 'Turn routes into a network',
      headline: 'Split first, then merge chains',
      body: `A routing engine needs **edges between nodes**, not overlapping polylines. \`GPULineSplit\` cuts every route at each place it crosses, touches or shares a stretch with another; each piece is coloured. Six L lines share the Loop and trunk tracks, so shared sections split at both ends.

Set **Show** to *Merged chains*: \`GPULineMerge\` joins pieces where exactly two ends meet. Then *Network*: \`GPUNetworkNoding\` numbers the end points into nodes (red is a dead end: a terminal) and builds an edge list and a routing-ready adjacency. Raise **Snap tolerance** below to merge end points that nearly coincide.`,
      camera: {...CITY, transitionMs: 1200},
      options: {view: 'noding', nodingInput: 'rail', nodingOutput: 'pieces'},
      highlight: {readout: 'nodingPieces'},
      controls: ['nodingOutput', 'nodingTolerance'],
      readouts: ['nodingPieces', 'nodingMerged', 'nodingEdges']
    },
    {
      id: 'capacity',
      title: 'The messy case: every CTA route downtown',
      headline: 'A fixed list can overflow',
      body: `Now the bus routes too: every CTA route inside a 7 km window of the Loop, hundreds of polylines that overlap along the same streets. Splitting finds each overlap's two ends. Look at the readouts: how many pieces, how many nodes, and what the degrees say about the downtown grid.

Set **Lines to node** to *All CTA routes downtown* (already on). **Intersection capacity** is the size of the internal pair list. With 1,024 it overflows: the output is a prefix and the overflow flag is raised, never a silent loss. Pick 65,536 or more in **Intersection capacity** and the network is complete.`,
      camera: {...LOOP, transitionMs: 1500},
      options: {nodingInput: 'rail-bus', nodingOutput: 'network', nodingCapacity: '524288'},
      highlight: {readout: 'nodingEdges'},
      controls: ['nodingInput', 'nodingCapacity'],
      readouts: ['nodingPieces', 'nodingEdges', 'nodingDegrees']
    },
    {
      id: 'dissolve',
      title: 'Dissolve counties into states',
      headline: 'Opposing shared edges cancel',
      body: `Dissolving is the overlay step behind every "county to state" map. \`GPUSegmentRingAssembly\` takes **directed boundary segments** and chains them into closed rings with holes. Here the input is every edge of 3,109 counties, labelled with the state; **Cancel opposing segments** removes the edges two counties share, leaving only each state's outer boundary.

Change **Dissolve counties by** below to the USDA rural-urban continuum code: the regions are now metro and rural belts, not states, and the ring and hole counts change. The readouts count rings, holes, cancelled segments and segments on no ring.`,
      camera: {...US, transitionMs: 1600},
      options: {view: 'dissolve', dissolveBy: 'state'},
      highlight: {readout: 'dissolveRings'},
      controls: ['cancelOpposing', 'dissolveBy'],
      readouts: ['dissolveRings', 'dissolveSegments', 'dissolveFlags']
    },
    {
      id: 'coverage',
      title: 'Simplify without opening gaps',
      headline: 'A shared border is one fact',
      body: `Simplifying every county on its own is the classic trap: two neighbours decide differently about their shared boundary and a **gap or overlap** opens. \`GPUCoverageSimplification\` simplifies every shared arc once, so neighbours keep identical vertices. Blue is the coverage result; red is each ring simplified independently with \`GPULineSimplification\`.

Use **Outlines** to show one result at a time, and zoom into a state border and raise **Tolerance** below: the blue line is single, the red one splits into two lines that disagree. **Topology repair rounds** (below) also find crossings and restore vertices (see the readout). Compare the kept percentage of the two.`,
      camera: {longitude: -80.5, latitude: 38.5, zoom: 6.4, transitionMs: 1600},
      options: {view: 'generalise', coverageTolerance: -1.0, coverageOutline: 'both'},
      highlight: {readout: 'coverageTopology'},
      controls: ['coverageTolerance', 'coverageOutline', 'topologyRounds'],
      readouts: ['coverageTopology', 'coverageKept', 'independentKept']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      headline: 'Topology depends on tolerance and capacity',
      body: `**Limits.** Capacities are fixed at compile time: a too-small intersection or ring capacity truncates and sets a flag. Noding snaps end points only: a line that stops near the middle of another is not connected, and a bridge cannot be excluded. Ring assembly leaves open chains unwritten. Coverage simplification uses a distance, not an area, criterion, and a huge arc is processed serially.

**Try it.** Set **Topology repair rounds** to 0 and watch the crossings reported. Then switch **Tool** to *Dissolve counties (ring assembly)* and turn **Cancel opposing segments** off, or make **Vertex matching distance** 0.005° (about 500 m) and see rings merge. Under *Crossings*, set **Intersect** to *Streets × streets (self)* and **Show** to *Overlaps* to find duplicated street geometry.`,
      options: {view: 'generalise', coverageTolerance: -0.7, topologyRounds: '0'},
      controls: ['topologyRounds', 'view'],
      readouts: ['coverageTopology']
    }
  ],

  about: {
    what: 'Exact segment intersection (GPUSegmentIntersection), splitting and merging lines (GPULineSplit, GPULineMerge), turning lines into a routable network (GPUNetworkNoding), chaining boundary segments into rings (GPUSegmentRingAssembly) and gap-free simplification of polygon coverages (GPUCoverageSimplification).',
    why: 'Overlay, routing and map generalisation all start from the same topology steps. On the GPU they run on whole cities and countries while parameters change.',
    howToRead:
      'Markers are coloured by intersection kind; pieces and chains by index; nodes by degree. For the dissolve, colour is the group; for generalisation, blue is the coverage result and red the independent one.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    if (state.view === 'crossings') {
      legends.push({kind: 'categories', title: 'Intersection kind', entries: KIND_LEGEND});
    } else if (state.view === 'noding') {
      if (state.nodingOutput === 'network') {
        legends.push({
          kind: 'categories',
          title: 'Node degree',
          entries: [
            {color: [226, 96, 80, 255], label: '1 (dead end)'},
            {color: [150, 160, 180, 255], label: '2'},
            {color: [255, 200, 40, 255], label: '3'},
            {color: [70, 205, 150, 255], label: '4'},
            {color: [78, 168, 222, 255], label: '5'},
            {color: [190, 120, 255, 255], label: '6 or more'}
          ]
        });
      } else {
        legends.push({
          kind: 'categories',
          title:
            state.nodingOutput === 'pieces' ? 'Pieces (colour cycles)' : 'Chains (colour cycles)',
          entries: B3_PALETTE.slice(0, 4).map((color, index) => ({
            color: [...color, 255] as const,
            label: `piece ${index}, ${index + 8}...`
          }))
        });
      }
    } else if (state.view === 'dissolve') {
      legends.push({
        kind: 'categories',
        title: 'Dissolved outline',
        entries: [
          {color: [10, 20, 40, 255], label: 'Ring written by GPUSegmentRingAssembly'},
          ...(state.dissolveColor === 'hole'
            ? [{color: [226, 96, 80, 255] as const, label: 'Hole'}]
            : [])
        ],
        note: state.showFill ? 'Fill colour is the group the county dissolves into.' : undefined
      });
    } else {
      legends.push({
        kind: 'categories',
        title: 'Outlines',
        entries: [
          {color: [30, 130, 190, 255], label: 'Coverage simplification (shared edges once)'},
          {color: [226, 96, 80, 255], label: 'Each ring simplified independently'},
          {color: [150, 160, 180, 120], label: 'Original'}
        ]
      });
    }
    return legends;
  },

  readouts: [
    {id: 'crossPairs', label: 'Intersecting pairs'},
    {id: 'crossKinds', label: 'By kind'},
    {id: 'crossFlags', label: 'Overflow and uncertain'},
    {id: 'nodingInput', label: 'Input lines'},
    {id: 'nodingPieces', label: 'Split pieces'},
    {id: 'nodingMerged', label: 'Merged chains'},
    {id: 'nodingEdges', label: 'Network'},
    {id: 'nodingDegrees', label: 'Node degrees'},
    {id: 'nodingUncertain', label: 'Uncertain pairs'},
    {id: 'dissolveRings', label: 'Dissolved rings'},
    {id: 'dissolveSegments', label: 'Boundary segments'},
    {id: 'dissolveFlags', label: 'Segment flags'},
    {id: 'coverageInput', label: 'Coverage'},
    {id: 'coverageTolerance', label: 'Tolerance'},
    {id: 'coverageKept', label: 'Kept by coverage simplification'},
    {id: 'independentKept', label: 'Kept by independent rings'},
    {id: 'coverageConverged', label: 'Converged'},
    {
      id: 'coverageTopology',
      label: 'Topology repair',
      help: 'Crossings found before repair, remaining, vertices restored.'
    }
  ],

  snippet: state => {
    if (state.view === 'crossings') {
      return `import {GPUSegmentIntersection} from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUSegmentIntersection({
  left: {kind: 'lines', positions: streets, lineOffsets: streetOffsets},${state.crossMode === 'streets-rail' ? "\n  right: {kind: 'lines', positions: rail, lineOffsets: railOffsets}," : `\n  sameFeatureOnly: ${state.sameFeatureOnly},`}
  spatialSort: ${state.crossSpatialSort},
  pairs: {leftIds, rightIds, count, overflow, totalCount},
  kinds, points, uncertainCount          // 1 proper, 2 touch, 3 collinear touch, 4 overlap
}));`;
    }
    if (state.view === 'noding') {
      return `import {GPULineSplit, GPULineMerge} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUNetworkNoding} from '@luma.gl/experimental/gpu-network';

graph.add(new GPULineSplit({
  lines: {kind: 'lines', positions, lineOffsets},
  intersectionCapacity: ${state.nodingCapacity}, spatialSort: ${state.nodingSpatialSort},
  pieces: {lineIds, offsets, positions: pieces, count, vertexCount, overflow}
}));
graph.add(new GPULineMerge({positions: pieces, lineOffsets: offsets, output: {chainOffsets, positions: chains, count}}));

graph.add(new GPUNetworkNoding({
  lines, intersectionCapacity: ${state.nodingCapacity},
  tolerance: tolerance.importToGraph(graph),   // ${state.nodingTolerance} m, per frame
  pieces, nodes: {positions: nodes, count}, edges: {fromNodes, toNodes, lengths},
  csr: {offsets, neighbors, weights}
}));`;
    }
    if (state.view === 'dissolve') {
      return `import {GPUSegmentRingAssembly} from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUSegmentRingAssembly({
  endpoints,                  // float32x4: x0, y0, x1, y1 of every directed ring edge
  groups,                     // dissolve label per segment (${state.dissolveBy})
  vertexTolerance: ${state.vertexTolerance},
  interiorSide: '${state.interiorSide}',
  cancelOpposingSegments: ${state.cancelOpposing},
  splitTouchingRings: ${state.splitTouching},
  output: {ringOffsets, positions, ringAreas, ringIsHole, ringGroups, count, overflow}
}));`;
    }
    return `import {GPUCoverageSimplification, getGPULineSimplificationParameterValues}
  from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUCoverageSimplification({
  positions, ringOffsets, polygonOffsets,
  snapTolerance: ${state.coverageSnap},
  topologyRounds: ${state.topologyRounds}, maximumRounds: ${state.coverageRounds},
  parameters: tolerance.importToGraph(graph),
  output: {positions: outPositions, ringOffsets: outRings, keepMask, overflow, topologyStats}
}));
tolerance.write(getGPULineSimplificationParameterValues({tolerance: ${(10 ** state.coverageTolerance).toFixed(4)}}));`;
  },

  create: async ctx => (await import('./noding-and-coverage.compute')).createNodingAndCoverage(ctx)
});
