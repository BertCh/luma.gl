// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {REGION_PALETTE} from './b6-colors';
import type {RegionalizationOptions} from './regionalization.compute';

const ATTRIBUTE_HELP = (text: string) =>
  `${text} Switching it off writes a constant column into the attribute table; the compiled graph is simply encoded again.`;

/** Regionalization of Chicago tracts: SKATER regions versus aspatial k-means. */
export default defineScene<RegionalizationOptions>({
  id: 'regionalization',
  title: 'Can Chicago be divided into similar neighbourhood regions?',
  chapter: 'regression',
  order: 3,
  summary:
    'Group 780-odd census tracts into contiguous regions of similar socioeconomic and health character with SKATER, by cutting a minimum spanning tree, and score the result against k-means, which ignores geography.',
  contributors: [
    'GPUSpatialWeightsMinimumSpanningTree',
    'GPUSkaterRegions',
    'GPURegionPartitionEvaluation',
    'GPUKMeans',
    'GPUContiguityWeights'
  ],
  datasets: [{id: 'chicago-tracts', role: 'tract polygons with ACS, SVI and PLACES attributes'}],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 9.9},

  options: [
    {
      kind: 'toggle',
      id: 'poverty',
      label: 'Poverty rate',
      group: 'Attributes',
      apply: 'param',
      default: true,
      help: ATTRIBUTE_HELP('Share of residents below 150 percent of the poverty line.')
    },
    {
      kind: 'toggle',
      id: 'income',
      label: 'ln income per capita',
      group: 'Attributes',
      apply: 'param',
      default: true,
      help: ATTRIBUTE_HELP('Log of ACS per-capita income.')
    },
    {
      kind: 'toggle',
      id: 'uninsured',
      label: 'Uninsured rate',
      group: 'Attributes',
      apply: 'param',
      default: true,
      help: ATTRIBUTE_HELP('Share of residents without health insurance.')
    },
    {
      kind: 'toggle',
      id: 'age65',
      label: 'Age 65+ share',
      group: 'Attributes',
      apply: 'param',
      default: true,
      help: ATTRIBUTE_HELP('Share of residents aged 65 or older.')
    },
    {
      kind: 'toggle',
      id: 'noVehicle',
      label: 'No-vehicle households',
      group: 'Attributes',
      apply: 'param',
      default: true,
      help: ATTRIBUTE_HELP('Share of households without a vehicle.')
    },
    {
      kind: 'toggle',
      id: 'diabetes',
      label: 'Diabetes prevalence',
      group: 'Attributes',
      apply: 'param',
      default: true,
      help: ATTRIBUTE_HELP('CDC PLACES model-based prevalence.')
    },
    {
      kind: 'toggle',
      id: 'black',
      label: 'Black share',
      group: 'Attributes',
      apply: 'param',
      default: false,
      help: ATTRIBUTE_HELP(
        'Non-Hispanic Black share of residents. Off by default: Chicago’s segregation would otherwise dominate the regions.'
      )
    },
    {
      kind: 'toggle',
      id: 'hispanic',
      label: 'Hispanic share',
      group: 'Attributes',
      apply: 'param',
      default: false,
      help: ATTRIBUTE_HELP('Hispanic share of residents.')
    },
    {
      kind: 'slider',
      id: 'regions',
      label: 'Regions',
      group: 'SKATER',
      apply: 'param',
      min: 1,
      max: 20,
      step: 1,
      default: 8,
      help: 'Target number of regions, islands included. Partitions are nested: the k-region result is the (k+1)-region result with its last cut undone, so scrubbing costs nothing.'
    },
    {
      kind: 'slider',
      id: 'minimumSize',
      label: 'Minimum region size',
      group: 'SKATER',
      apply: 'param',
      min: 1,
      max: 60,
      step: 1,
      default: 1,
      unit: 'tracts',
      help: 'A cut is only allowed if both sides keep at least this many tracts. Raise it for balanced regions; when it binds, fewer regions than the target are produced.'
    },
    {
      kind: 'toggle',
      id: 'standardize',
      label: 'Standardize attributes',
      group: 'Method',
      apply: 'compile',
      default: true,
      help: 'Compile-time option of the spanning tree: z-score every column before dissimilarity is taken. Off, columns with big numbers (poverty in points) outweigh small ones (ln income).'
    },
    {
      kind: 'select',
      id: 'criterion',
      label: 'Contiguity',
      group: 'Method',
      apply: 'compile',
      default: 'queen',
      help: 'Which tracts are adjacent: sharing a boundary point (queen) or an edge (rook). Rook leaves more tracts connected by a corner only, which can split the tree into several trees.',
      options: [
        {value: 'queen', label: 'Queen'},
        {value: 'rook', label: 'Rook'}
      ]
    },
    {
      kind: 'select',
      id: 'kmeansClusters',
      label: 'k-means clusters (for comparison)',
      group: 'Method',
      apply: 'compile',
      default: '8',
      help: 'Number of aspatial clusters, run on the first two principal components of the same attributes (k is compile-time: each size compiles its graph once). Set it equal to the region count for a fair comparison.',
      options: ['3', '4', '6', '8', '10', '12', '16'].map(value => ({
        value,
        label: `${value} clusters`
      }))
    },
    {
      kind: 'select',
      id: 'map',
      label: 'Fill',
      group: 'Display',
      apply: 'param',
      default: 'skater',
      help: 'Which partition colors the tracts. Neighbouring regions never share a color.',
      options: [
        {value: 'skater', label: 'SKATER regions (contiguous)'},
        {value: 'kmeans', label: 'k-means clusters (aspatial)'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showTree',
      label: 'Show the spanning tree',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Draws every edge of the minimum spanning tree between tract centroids.'
    },
    {
      kind: 'toggle',
      id: 'treeByCost',
      label: 'Color tree edges by dissimilarity',
      group: 'Display',
      apply: 'param',
      default: true,
      disabledWhen: state => !state.showTree,
      help: 'Edge color is the squared attribute distance between the two tracts: dark links join similar tracts, bright links join tracts that differ most. Off draws plain white edges.'
    },
    {
      kind: 'toggle',
      id: 'showCuts',
      label: 'Show the SKATER cuts',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws the tree edges SKATER removed, in red.'
    }
  ],

  readouts: [
    {
      id: 'rows',
      label: 'Tracts',
      help: 'Tracts with every attribute present, in the largest contiguous group.'
    },
    {id: 'regions', label: 'SKATER regions'},
    {
      id: 'explained',
      label: 'Explained variance',
      help: 'Between-region share of the total sum of squares of the attribute table, the same metric for both partitions (GPURegionPartitionEvaluation).'
    },
    {
      id: 'within',
      label: 'Within-region SSD',
      help: 'Sum of squared deviations from region means. Lower is more homogeneous.'
    },
    {id: 'sizes', label: 'Region sizes'},
    {
      id: 'boundary',
      label: 'Boundary links',
      help: 'Share of neighbour links whose two tracts lie in different regions. A compact partition has few.'
    },
    {
      id: 'pieces',
      label: 'Connected pieces',
      help: 'Number of connected pieces after counting each region’s tracts through the contiguity graph. SKATER regions are contiguous by construction; k-means clusters usually are not.'
    },
    {
      id: 'gains',
      label: 'Variance removed by each cut',
      help: 'Sum-of-squares reduction of the first cuts, in order: the elbow tells you where more regions stop paying off.'
    },
    {id: 'tree', label: 'Spanning tree'},
    {id: 'kmeans', label: 'k-means run'},
    {id: 'capacity', label: 'Weights capacity'}
  ],

  legends: state => {
    const entries: LegendSpec[] = [
      {
        kind: 'categories',
        title: state.map === 'skater' ? 'SKATER region' : 'k-means cluster',
        entries: REGION_PALETTE.slice(0, 6).map((color, index) => ({
          color,
          label: `Region color ${index + 1}`
        })),
        note: 'Colors only distinguish neighbouring groups and repeat across the map.'
      }
    ];
    if (state.showTree) {
      entries.push(
        state.treeByCost
          ? {
              kind: 'ramp',
              id: 'tree',
              title: 'Tree edge dissimilarity (squared attribute distance)',
              ramp: 'inferno',
              extent: 'gpu',
              sqrtScale: true,
              labels: ['similar', 'different'],
              format: value => value.toFixed(1)
            }
          : {
              kind: 'categories',
              title: 'Lines',
              entries: [{color: [255, 255, 255, 240], label: 'Minimum spanning tree edge'}]
            }
      );
    }
    if (state.showCuts) {
      entries.push({
        kind: 'categories',
        title: 'Cuts',
        entries: [{color: [230, 40, 40, 255], label: 'Tree edge removed by SKATER'}]
      });
    }
    return entries;
  },

  snippet: state => `import {
  GPUContiguityWeights, GPUSpatialWeightsMinimumSpanningTree, GPUSkaterRegions,
  GPURegionPartitionEvaluation, GPUKMeans
} from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(new GPUContiguityWeights({criterion: '${state.criterion}', positions: vertices, ringOffsets, polygonOffsets, weights, overflow}));

// Cheapest tree over attribute dissimilarity (Boruvka)
graph.add(new GPUSpatialWeightsMinimumSpanningTree({
  weights, values, columnCount: 8, standardize: ${state.standardize},
  treeEdgeFlags, componentLabels, standardizedValues,
  edges: {ids, count, overflow}, edgeCosts
}));

// Greedy cuts of that tree. Region count and minimum size are a parameter buffer.
graph.add(new GPUSkaterRegions({
  weights, treeEdgeFlags, componentLabels, values: standardizedValues, columnCount: 8,
  maximumRegionCount: 20, parameters,          // [regions, minimumSize]
  labels, regionCount, cutEdges, cutGains
}));
skaterParameters.write(new Uint32Array([${state.regions}, ${state.minimumSize}]));

// Score any partition on the same attributes, SKATER or k-means
graph.add(new GPURegionPartitionEvaluation({values: standardizedValues, columnCount: 8, labels, weights, summary}));
graph.add(new GPUKMeans({positions: principalPlane, k: ${state.kmeansClusters}, iterations: 24, initialization: 'kmeans++', labels: kLabels, centers}));`,

  about: {
    what: '`GPUSpatialWeightsMinimumSpanningTree` joins neighbouring tracts into the cheapest tree where an edge costs the squared difference of their attributes. `GPUSkaterRegions` makes greedy cuts: each removes the tree edge whose removal reduces the within-region sum of squares most, so every region stays connected. `GPURegionPartitionEvaluation` scores any partition (SKATER or k-means) on the same attributes.',
    why: 'Neighbourhood typologies for planning, service territories and sampling need regions that are both internally similar and geographically contiguous. Ordinary clustering finds similar tracts wherever they are; SKATER only joins tracts that touch.',
    howToRead:
      'Each color is a region. White-to-yellow lines are tree edges (bright = dissimilar neighbours); red lines are the cuts. Compare the readouts for SKATER and k-means: k-means usually explains more variance because it may group tracts that are far apart, but its clusters shatter into many disconnected pieces.'
  },

  create: async ctx => (await import('./regionalization.compute')).createRegionalization(ctx),

  story: [
    {
      id: 'question',
      title: 'Where are Chicago’s distinct neighbourhood types?',
      body: 'Chicago has 780-odd census tracts and dozens of attributes. A planner wants a handful of **regions**: each internally similar (poverty, income, insurance, age, car access and diabetes), but also **contiguous**, so a region can be a service district or a sampling stratum. That is *regionalization*.\n\nThe map shows the finished answer, eight regions from **`GPUSkaterRegions`**, colored so neighbouring regions differ. The next steps build it from the neighbour graph.',
      options: {regions: 8, showTree: false, showCuts: false},
      controls: ['regions'],
      readouts: ['regions']
    },
    {
      id: 'tree',
      title: 'Step 1: the cheapest tree through the neighbour graph',
      body: '**`GPUContiguityWeights`** finds which tracts touch (queen: share a boundary point). **`GPUSpatialWeightsMinimumSpanningTree`** then keeps just enough links to connect everything at the lowest total cost, where the cost of a link is the **squared distance between the two tracts’ attribute vectors** (SKATER’s metric, spopt `SpanningForest`). It uses Borůvka rounds on the GPU.\n\nWith **Regions** set to 1 the whole city is one region and the tree is drawn: dark lines join similar neighbours, bright ones join neighbours that differ most. Those bright edges are the natural place to cut.',
      options: {regions: 1, showTree: true, treeByCost: true, showCuts: false},
      controls: ['regions', 'showTree', 'treeByCost'],
      readouts: ['tree']
    },
    {
      id: 'cuts',
      title: 'Step 2: cut the tree where it hurts least',
      body: '**SKATER** (Spatial “K”luster Analysis by Tree Edge Removal) greedily removes the tree edge whose cut most reduces the total within-region sum of squares, then repeats. Every piece of a tree is connected, so every region is **contiguous by construction**. Red lines (**Show the SKATER cuts**) are the cuts for eight regions.\n\nThe readout **Variance removed by each cut** lists the gain of the first cuts. Early cuts separate the South and West sides from the North; later ones carve smaller pockets.',
      options: {regions: 8, showTree: true, showCuts: true},
      controls: ['regions', 'showCuts'],
      readouts: ['gains']
    },
    {
      id: 'region-count',
      title: 'How many regions? Scrub it',
      body: 'Move **Regions** from 2 to 20. The partition with k regions is the one with k+1 plus its last cut undone, so the compiled graph holds the whole cut log and the slider is only a parameter write (**Under the hood** shows no rebuild). Watch **Explained variance** and the gains: when each extra cut explains little, you have found the elbow.\n\nRaise **Minimum region size** to forbid tiny regions: a cut is refused when either side would be smaller, so the region count can fall short of the target.',
      options: {regions: 12, minimumSize: 12, showTree: false},
      controls: ['regions', 'minimumSize'],
      readouts: ['explained', 'gains']
    },
    {
      id: 'standardize',
      title: 'Units matter unless you standardize',
      body: 'Tree costs add up squared differences across attributes. Turn **Standardize attributes** off and the columns keep their own units: poverty and uninsured rates in points, car-free households in points, diabetes in points, but **ln income** in tiny log units. Large-unit variables now dominate and income barely matters. This is a compile-time option of the spanning tree, so the graph variant compiles once.\n\nTurn it back on before you trust a result: z-scoring gives every attribute an equal voice.',
      options: {regions: 8, standardize: false},
      controls: ['standardize'],
      readouts: ['explained']
    },
    {
      id: 'kmeans',
      title: 'What if geography is ignored? k-means',
      body: 'Plain **`GPUKMeans`** groups tracts by attribute similarity alone (here on the first two principal components of the same attributes). Switch **Fill** to *k-means clusters (aspatial)* and set **k-means clusters (for comparison)** to match **Regions**. Every cluster is labelled, but look at the map and the **Connected pieces** readout: a cluster is scattered across many tracts that do not touch.\n\n**`GPURegionPartitionEvaluation`** scores both partitions with one metric. k-means usually explains slightly more variance because it is unconstrained; SKATER pays a small price for contiguity and keeps **Boundary links** far lower. Choose by purpose: typologies for statistics (k-means), territories for action (SKATER).',
      options: {standardize: true, regions: 8, map: 'kmeans', showCuts: false},
      controls: ['map', 'kmeansClusters', 'regions'],
      readouts: ['pieces', 'explained', 'boundary']
    },
    {
      id: 'try',
      title: 'Try it, and know the limits',
      body: 'Switch on **Black share** and **Hispanic share**: the regions follow Chicago’s well-known segregation pattern. Switch **Contiguity** to rook, change the attribute set, or compare **k-means clusters (for comparison)** with the same count.\n\n**Limits:** SKATER is greedy, so it can miss the best partition; a tree keeps only n-1 links, so some good regions that need a different tree are out of reach; attributes at tract level carry sampling error (ACS, modelled PLACES); and the tool does not decide the right number of regions for you. Reference: spopt `Skater`, ArcGIS Spatially Constrained Multivariate Clustering.',
      options: {map: 'skater', black: true, hispanic: true, regions: 8},
      controls: ['black', 'hispanic', 'criterion', 'kmeansClusters'],
      readouts: ['regions', 'explained']
    }
  ]
});
