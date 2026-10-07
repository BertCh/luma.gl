// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import {CHICAGO_VIEW, formatCategory} from './b1-nature-data';
import type {NatureClusterOptions} from './nature-clusters.compute';

const CATEGORIES = [
  'Birds',
  'Plants',
  'Insects',
  'Fungi',
  'Mammals',
  'Spiders and kin',
  'Amphibians and reptiles'
];

const CLUSTER_COLORS = [
  [78, 201, 255, 255],
  [255, 148, 72, 255],
  [189, 122, 255, 255],
  [87, 235, 168, 255],
  [255, 105, 168, 255],
  [245, 220, 87, 255],
  [107, 158, 255, 255],
  [255, 92, 92, 255]
] as const;

export default defineScene<NatureClusterOptions>({
  id: 'nature-clusters',
  title: 'Where do sightings form hot spots?',
  chapter: 'points',
  order: 2,
  summary:
    'DBSCAN and k-means clusters of one group of Chicago wildlife observations, each outlined with a convex hull, ellipse, centre and medoid, plus the centre of gravity and standard deviational ellipse of the whole pattern.',
  contributors: [
    'GPUSpatialClustering',
    'GPUKMeans',
    'GPUGroupGeometry',
    'GPUGroupConvexHull',
    'GPUGeographicDistribution',
    'addClusterAndOutlineRecipe'
  ],
  datasets: [{id: 'chicago-nature', role: 'iNaturalist observations (2023), one group at a time'}],
  initialView: {...CHICAGO_VIEW},

  options: [
    {
      kind: 'select',
      id: 'category',
      label: 'Group',
      group: 'Data',
      apply: 'compile',
      default: 'Birds',
      help: 'Which iNaturalist group to cluster. The points are gathered into a new buffer, so this rebuilds the graphs. Sparse groups (mammals, reptiles) need a bigger radius and fewer minimum points: try 400 m / 8 or 800 m / 5.',
      options: CATEGORIES.map(name => ({value: name, label: formatCategory(name)}))
    },
    {
      kind: 'select',
      id: 'weight',
      label: 'Weight observations by',
      group: 'Data',
      apply: 'param',
      default: 'none',
      help: 'Weighted centres, ellipses and the overall distribution counts a flagged observation five times (weight 5 instead of 1). Rewrites a weights buffer; no rebuild.',
      options: [
        {value: 'none', label: 'Nothing (every observation counts once)'},
        {value: 'researchGrade', label: 'Research grade (identification confirmed)'},
        {value: 'introduced', label: 'Introduced (non-native) taxon'}
      ]
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Method',
      group: 'Clustering',
      apply: 'compile',
      default: 'dbscan',
      help: 'DBSCAN finds dense groups of any shape and leaves sparse points as noise. K-means partitions every point into k compact groups.',
      options: [
        {value: 'dbscan', label: 'DBSCAN (density-based)'},
        {value: 'kmeans', label: 'K-means (partition)'}
      ]
    },
    {
      kind: 'slider',
      id: 'epsilon',
      label: 'Neighbour radius (epsilon)',
      group: 'DBSCAN',
      apply: 'param',
      min: 50,
      max: 1500,
      step: 10,
      default: 150,
      unit: 'm',
      disabledWhen: state => state.method !== 'dbscan',
      help: 'Two observations are neighbours when closer than this. Written into a parameter buffer: dragging it reruns the graphs without recompiling.'
    },
    {
      kind: 'slider',
      id: 'minimumPoints',
      label: 'Minimum points (core threshold)',
      group: 'DBSCAN',
      apply: 'param',
      min: 2,
      max: 80,
      step: 1,
      default: 20,
      disabledWhen: state => state.method !== 'dbscan',
      help: 'An observation is a core point when at least this many (including itself) lie within epsilon. Higher values keep only dense hot spots.'
    },
    {
      kind: 'toggle',
      id: 'denseBoxShortcut',
      label: 'Dense-box shortcut',
      group: 'DBSCAN',
      apply: 'compile',
      default: false,
      disabledWhen: state => state.method !== 'dbscan' || state.pipeline === 'recipe',
      help: 'Same labels, fewer distance tests when epsilon is large compared with point spacing. Compile-time; use the timing button to see when it pays off.'
    },
    {
      kind: 'select',
      id: 'sumOrder',
      label: 'Centroid summation',
      group: 'DBSCAN',
      apply: 'compile',
      default: 'sorted',
      disabledWhen: state => state.method !== 'dbscan' || state.pipeline === 'recipe',
      help: 'Sorted sums are bitwise reproducible; atomic float adds are a little faster when a few clusters hold most points but may differ in the last bits. Affects cluster centroids only, never labels.',
      options: [
        {value: 'sorted', label: 'Sorted (reproducible)'},
        {value: 'atomic', label: 'Atomic float adds'}
      ]
    },
    {
      kind: 'select',
      id: 'pipeline',
      label: 'Outline pipeline',
      group: 'DBSCAN',
      apply: 'compile',
      default: 'assembled',
      disabledWhen: state => state.method !== 'dbscan',
      help: 'Assembled: clustering, group geometry and hulls as separate contributors. Recipe: one addClusterAndOutlineRecipe call that also measures every hull area and perimeter (no weighted centres, ellipses or medoids).',
      options: [
        {value: 'assembled', label: 'Assembled contributors'},
        {value: 'recipe', label: 'One call: addClusterAndOutlineRecipe'}
      ]
    },
    {
      kind: 'slider',
      id: 'k',
      label: 'Number of clusters (k)',
      group: 'K-means',
      apply: 'compile',
      min: 2,
      max: 64,
      step: 1,
      default: 12,
      disabledWhen: state => state.method !== 'kmeans',
      help: 'Compile-time: k sets buffer sizes. The graph rebuilds shortly after you stop dragging.'
    },
    {
      kind: 'select',
      id: 'initialization',
      label: 'Initial centres',
      group: 'K-means',
      apply: 'compile',
      default: 'kmeans++',
      disabledWhen: state => state.method !== 'kmeans',
      help: 'The first k valid points (fast, depends on data order) or seeded k-means++ (spread-out starts, usually better).',
      options: [
        {value: 'first-valid', label: 'First k valid points'},
        {value: 'kmeans++', label: 'Seeded k-means++'}
      ]
    },
    {
      kind: 'slider',
      id: 'seed',
      label: 'k-means++ seed',
      group: 'K-means',
      apply: 'compile',
      min: 0,
      max: 31,
      step: 1,
      default: 0,
      disabledWhen: state => state.method !== 'kmeans' || state.initialization !== 'kmeans++',
      help: 'Different seeds give different local optima; the same seed always gives the same result on one device.'
    },
    {
      kind: 'slider',
      id: 'iterations',
      label: 'Iteration cap',
      group: 'K-means',
      apply: 'compile',
      min: 4,
      max: 64,
      step: 4,
      default: 48,
      disabledWhen: state => state.method !== 'kmeans',
      help: 'The graph holds this many Lloyd iterations and stops moving centres once converged. Lower it to see a partition that has not settled.'
    },
    {
      kind: 'slider',
      id: 'tolerance',
      label: 'Convergence tolerance',
      group: 'K-means',
      apply: 'compile',
      min: 0,
      max: 200,
      step: 5,
      default: 5,
      unit: 'm',
      disabledWhen: state => state.method !== 'kmeans',
      help: 'Centres that move less than this count as settled. 0 demands an exact fixed point.'
    },
    {
      kind: 'select',
      id: 'prefilterLevels',
      label: 'Hull prefilter levels',
      group: 'Shapes',
      apply: 'compile',
      default: '2',
      disabledWhen: state => state.pipeline === 'recipe' && state.method === 'dbscan',
      help: 'GPUGroupConvexHull discards points that cannot be on the hull in parallel passes before the exact chain. 0 turns it off; the result is identical, the time is not.',
      options: [
        {value: '0', label: '0 (no prefilter)'},
        {value: '1', label: '1'},
        {value: '2', label: '2 (default)'},
        {value: '3', label: '3'},
        {value: '4', label: '4'}
      ]
    },
    {
      kind: 'select',
      id: 'standardDeviations',
      label: 'Ellipse size',
      group: 'Shapes',
      apply: 'param',
      default: '1',
      help: 'How many standard deviations the standard distance circle and the ellipses span (ArcGIS offers 1, 2 and 3).',
      options: [
        {value: '1', label: '1 standard deviation (about 68%)'},
        {value: '2', label: '2 standard deviations (about 95%)'},
        {value: '3', label: '3 standard deviations (about 99%)'}
      ]
    },
    {
      kind: 'select',
      id: 'ellipseConvention',
      label: 'Ellipse convention',
      group: 'Shapes',
      apply: 'param',
      default: 'standard',
      help: 'ArcGIS scales each axis by the square root of 2; standard uses the plain standard deviation along each principal axis.',
      options: [
        {value: 'standard', label: 'Standard (plain sigma)'},
        {value: 'arcgis', label: 'ArcGIS Directional Distribution'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showHulls',
      label: 'Convex hulls',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'The smallest convex polygon around each cluster (GPUGroupConvexHull).'
    },
    {
      kind: 'toggle',
      id: 'showClusterEllipses',
      label: 'Cluster ellipses',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Standard deviational ellipse per cluster, centred on its weighted centre (GPUGroupGeometry).'
    },
    {
      kind: 'toggle',
      id: 'showBounds',
      label: 'Bounding boxes',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Axis-aligned box of each cluster.'
    },
    {
      kind: 'toggle',
      id: 'showMeans',
      label: 'Mean centres (rings)',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Average position of the members of each cluster.'
    },
    {
      kind: 'toggle',
      id: 'showWeighted',
      label: 'Weighted centres (diamonds)',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Centre pulled toward heavier observations. Only differs from the mean when a weight is chosen.'
    },
    {
      kind: 'toggle',
      id: 'showMedoids',
      label: 'Medoids (dots)',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'The member closest to all the others: a real observation that represents the cluster. Skipped for clusters over 1,024 members.'
    },
    {
      kind: 'toggle',
      id: 'showCore',
      label: 'Highlight core points',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Ringed points have at least the minimum number of neighbours; the others are border points of a cluster.'
    },
    {
      kind: 'toggle',
      id: 'showNoise',
      label: 'Show noise points',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Grey points belong to no DBSCAN cluster.'
    },
    {
      kind: 'toggle',
      id: 'showDistribution',
      label: 'Overall distribution',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Mean centre (yellow ring), median centre (red diamond), standard distance circle and standard deviational ellipse of the whole selected group (GPUGeographicDistribution).'
    },
    {
      kind: 'button',
      id: 'measureShortcut',
      label: 'Time DBSCAN plain vs dense-box',
      group: 'Compare',
      help: 'Compiles both variants into scratch buffers and times them at the current epsilon and minimum points.'
    },
    {
      kind: 'button',
      id: 'measureShapes',
      label: 'Time the shape graph',
      group: 'Compare',
      help: 'GPU time of group geometry plus convex hulls at the current prefilter level. Change the level and press again.'
    }
  ],

  readouts: [
    {id: 'points', label: 'Observations clustered'},
    {id: 'parameters', label: 'Method and parameters'},
    {id: 'clusters', label: 'Clusters'},
    {
      id: 'clustered',
      label: 'Clustered / noise',
      help: 'Observations inside a cluster and observations left as noise.'
    },
    {id: 'largest', label: 'Largest cluster (observations)'},
    {id: 'meanSize', label: 'Mean cluster size'},
    {id: 'convergence', label: 'K-means convergence'},
    {id: 'hulls', label: 'Hulls'},
    {id: 'hullArea', label: 'Hull area (recipe)'},
    {id: 'hullPerimeter', label: 'Largest hull perimeter (recipe)'},
    {id: 'weightedShift', label: 'Weighted minus mean centre', help: 'For the largest cluster.'},
    {id: 'medoid', label: 'Medoid vs mean centre', help: 'For the largest cluster.'},
    {id: 'medianGap', label: 'Median centre vs mean centre'},
    {id: 'standardDistance', label: 'Standard distance'},
    {id: 'ellipse', label: 'Standard ellipse (axes, rotation)'},
    {id: 'timing', label: 'DBSCAN timing'},
    {id: 'shapeTiming', label: 'Shape graph timing'}
  ],

  legends: state => [
    {
      kind: 'categories',
      title: state.method === 'dbscan' ? 'DBSCAN cluster' : 'K-means cluster',
      entries: [
        ...CLUSTER_COLORS.map((color, index) => ({
          color,
          label: index === 7 ? 'cluster 7, 15, 23, ...' : `cluster ${index}`
        })),
        ...(state.method === 'dbscan'
          ? [{color: [128, 138, 156, 200] as const, label: 'noise'}]
          : [])
      ],
      note: 'Cluster colors repeat every eight labels. Hulls, ellipses and boxes take their cluster color.'
    },
    {
      kind: 'categories',
      title: 'Markers',
      entries: [
        {color: [235, 235, 245, 255], label: 'Mean centre of a cluster (ring)'},
        {color: [255, 214, 64, 255], label: 'Weighted centre (diamond) / overall mean centre'},
        {color: [90, 90, 110, 255], label: 'Medoid (dot)'},
        {color: [255, 70, 70, 255], label: 'Overall median centre (diamond)'}
      ]
    }
  ],

  snippet: state => {
    if (state.method === 'kmeans') {
      return `const graph = new GPUCommandGraph(device, {id: 'clusters'});
graph.add(new GPUKMeans({
  positions,                               // float32x2 meters of the chosen group
  k: ${state.k},
  iterations: ${state.iterations},                       // compile-time cap, stops early on convergence
  tolerance: ${state.tolerance},                         // meters a centre may still move
  initialization: '${state.initialization}',${state.initialization === 'kmeans++' ? `\n  seed: ${state.seed},` : ''}
  convergence,                             // [iterationsUsed, converged]
  labels, centers, sizes
}));
const compiled = graph.compile();
compiled.encode(commandEncoder, {parameters: undefined});`;
    }
    if (state.pipeline === 'recipe') {
      return `const graph = new GPUCommandGraph(device, {id: 'cluster-outline'});
addClusterAndOutlineRecipe(graph, {
  positions,
  clusteringParameters: clusteringParameters.importToGraph(graph),  // bounds, epsilon, minimumPoints
  gridSize: [256, 256],
  geometryParameters: geometryParameters.importToGraph(graph),
  maximumClusterCount: 2048,
  maximumVerticesPerHull: 256,
  hullCapacity: 1 << 14,
  labels, counts, bounds, meanCenters,
  hullPositions, hullOffsets, hullCounts, hullOverflow,
  areas, perimeters, centroids              // hull measures, per cluster
});
const compiled = graph.compile();
// each change of epsilon / minimumPoints:
clusteringParameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: ${state.epsilon}, minimumPoints: ${state.minimumPoints}}));
compiled.encode(commandEncoder, {parameters: undefined});`;
    }
    return `const graph = new GPUCommandGraph(device, {id: 'clusters'});
graph.add(new GPUSpatialClustering({
  positions, parameters: clusteringParameters.importToGraph(graph),
  gridSize: [256, 256],${state.denseBoxShortcut ? '\n  denseBoxShortcut: true,' : ''}${state.sumOrder === 'atomic' ? "\n  sumOrder: 'atomic'," : ''}
  labels, coreFlags, clusterCount, clusterSizes, clusterCentroids,
  clusters: {ids, count, overflow}
}));
const shapes = new GPUCommandGraph(device, {id: 'shapes'});
shapes.add(new GPUGroupGeometry({positions, labels, groupCount: 2048, weights,
  parameters: geometryParameters.importToGraph(shapes),
  output: {counts, bounds, meanCenters, weightedCenters, medoidIndices, ellipses}}));
shapes.add(new GPUGroupConvexHull({positions, labels, groupCount: 2048,
  maximumVerticesPerGroup: 256, totalCapacity: 1 << 14, prefilterLevels: ${state.prefilterLevels},
  output: {vertexIndices, vertexPositions, offsets, counts, overflow}}));
const overall = new GPUCommandGraph(device, {id: 'distribution'});
overall.add(new GPUGeographicDistribution({positions, weights, polygonVertexCount: 64,
  parameters: distributionParameters.importToGraph(overall),
  output: {meanCenters, medianCenters, standardDistances, ellipses, ellipseVertices, circleVertices}}));
// when epsilon / minimumPoints change:
clusteringParameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: ${state.epsilon}, minimumPoints: ${state.minimumPoints}}));
clustersCompiled.encode(commandEncoder, {parameters: undefined});
shapesCompiled.encode(commandEncoder, {parameters: undefined});`;
  },

  about: {
    what: '`GPUSpatialClustering` runs DBSCAN on the GPU: an observation with enough neighbours within epsilon is a core point, clusters are the connected core points plus their borders, and everything else is noise. `GPUKMeans` assigns every observation to the nearest of k moving centres. Then `GPUGroupGeometry` and `GPUGroupConvexHull` describe every cluster, and `GPUGeographicDistribution` summarises the whole pattern with centres, a standard distance and a standard deviational ellipse.',
    why: 'Clustering turns a cloud of sightings into places you can name, visit and compare: hull areas, sizes and centres per hot spot. DBSCAN suits irregular patterns such as a park, a pond or a path and tells you what is not a hot spot; k-means is for dividing a city into balanced survey areas.',
    howToRead:
      "Each color is one cluster; grey points are noise. White rings are cluster centres, hulls and ellipses outline the shape. In the overall view the yellow ellipse is the pattern's footprint and its long axis is the direction the sightings are stretched along."
  },

  create: async ctx => (await import('./nature-clusters.compute')).createNatureClusters(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['category', 'epsilon', 'minimumPoints'],
      readouts: ['clusters', 'clustered'],
      title: 'Hot spots, not just a haze',
      body: 'A citywide heat map says birds are logged everywhere along the lake. **`GPUSpatialClustering`** asks a sharper question: which groups of sightings are dense enough to count as a place? DBSCAN calls an observation a *core point* when at least 20 observations (including itself) lie within 150 m, then joins touching core points into clusters. Pick the **Group** below, and change **Neighbour radius (epsilon)** and **Minimum points (core threshold)** to see what counts as a place.\n\nEach color is a cluster; grey points are noise, scattered sightings that belong to no hot spot. The largest bird cluster, around Montrose Point, holds roughly 2,400 observations. The white outline around each cluster is its convex hull, computed by **`GPUGroupConvexHull`**.',
      options: {category: 'Birds', method: 'dbscan', epsilon: 150, minimumPoints: 20}
    },
    {
      id: 'radius-and-density',
      controls: ['epsilon', 'minimumPoints', 'showCore', 'measureShortcut'],
      readouts: ['clusters', 'timing'],
      title: 'Epsilon and minimum points decide what is a place',
      body: 'Drag **Neighbour radius (epsilon)** up and clusters merge into larger blobs; drag **Minimum points (core threshold)** up and only the densest cores survive. Both are written into a parameter buffer, so the graphs re-run in a few milliseconds and nothing recompiles.\n\nHere the radius is widened to 400 m with 60 minimum points, so a park with several busy paths becomes one place. Turn on **Highlight core points** to see which observations anchor a cluster and which are just attached to its edge. Press **Time DBSCAN plain vs dense-box** to compare the two compile-time strategies, which return identical labels.',
      options: {epsilon: 400, minimumPoints: 60, showCore: true, showMeans: false},
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.8}
    },
    {
      id: 'describe-clusters',
      controls: ['weight', 'showMedoids', 'showClusterEllipses'],
      readouts: ['weightedShift', 'medoid'],
      title: 'Describe every cluster',
      body: '**`GPUGroupGeometry`** reads the labels and writes a count, bounding box, mean centre, weighted centre, medoid and standard deviational ellipse per cluster. The ring is the mean centre; the dot is the **medoid**, the real observation closest to all the others; the ellipse shows how stretched the cluster is.\n\nSwitch **Weight observations by** to *Research grade*: the weighted centre (diamond) leans toward where identifications were confirmed. The readouts report the shift for the largest cluster.',
      options: {
        epsilon: 150,
        minimumPoints: 20,
        showCore: false,
        showMeans: true,
        showMedoids: true,
        showWeighted: true,
        showClusterEllipses: true,
        weight: 'researchGrade'
      },
      camera: {longitude: -87.645, latitude: 41.93, zoom: 12.3},
      highlight: {readout: 'weightedShift'}
    },
    {
      id: 'kmeans',
      controls: ['k', 'iterations', 'initialization', 'seed'],
      readouts: ['convergence'],
      title: 'K-means: every observation gets a group',
      body: "**`GPUKMeans`** takes a different view: choose **Number of clusters (k)**, start from k centres, then repeatedly assign each observation to the nearest centre and move the centres to the mean until nothing moves (Lloyd's algorithm). There is no noise: even a lone sighting in a vacant lot is assigned somewhere.\n\nThat suits balanced survey areas, not hot spots. The graph holds up to **Iteration cap** passes and stops moving once the shift is below the tolerance; the readout says how many it needed. Compare **Initial centres** (*First k valid points* against *Seeded k-means++*) and change the **k-means++ seed** below: different starts settle into different partitions, which is the method's main weakness.",
      options: {
        method: 'kmeans',
        k: 12,
        initialization: 'kmeans++',
        seed: 0,
        showMeans: true,
        showWeighted: false,
        showMedoids: false,
        showClusterEllipses: false,
        weight: 'none'
      },
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.8}
    },
    {
      id: 'overall-pattern',
      controls: ['standardDeviations', 'ellipseConvention', 'showDistribution'],
      readouts: ['medianGap', 'standardDistance', 'ellipse'],
      title: 'The centre of gravity of a group',
      body: '**`GPUGeographicDistribution`** summarises the whole pattern, not each cluster. For the 1,089 mammal observations of 2023 it gives the mean centre (yellow ring), the geometric **median** centre (red diamond), the *standard distance* circle and the *standard deviational ellipse*.\n\nThe mean centre lands near North Avenue in Lincoln Park, and the median sits about two kilometres further north: a long tail of sightings on the South and Southwest Sides drags the mean south. The ellipse is about 2.6 times longer than it is wide and runs north to south, following the lakefront and the shape of the city. Below, **Ellipse size** sets 1, 2 or 3 standard deviations and **Ellipse convention** switches between plain and ArcGIS scaling.',
      options: {
        category: 'Mammals',
        method: 'dbscan',
        epsilon: 400,
        minimumPoints: 8,
        showDistribution: true,
        showMeans: false,
        showHulls: false,
        showNoise: true,
        standardDeviations: '1'
      },
      camera: {longitude: -87.68, latitude: 41.835, zoom: 9.8}
    },
    {
      id: 'recipe',
      controls: ['pipeline', 'epsilon', 'minimumPoints'],
      readouts: ['hullArea', 'hullPerimeter'],
      title: 'The same pipeline in one call',
      body: "**`addClusterAndOutlineRecipe`** wires clustering, group geometry, convex hulls and **`GPUGeometryMeasures`** into one graph. Besides the hulls it measures each hull's area, perimeter and centroid, so you can rank hot spots by footprint: the readouts report the largest hull and the total.\n\nThe recipe is the DBSCAN-plus-outline chain only: no weighted centres, ellipses or medoids. With minimum points set to 1 it behaves like PostGIS `ST_ClusterWithin`.",
      options: {
        category: 'Fungi',
        pipeline: 'recipe',
        epsilon: 250,
        minimumPoints: 20,
        showDistribution: false,
        showMeans: true,
        showHulls: true
      },
      highlight: {readout: 'hullArea'}
    },
    {
      id: 'limits',
      controls: ['category', 'prefilterLevels', 'measureShapes'],
      readouts: ['shapeTiming'],
      title: 'Limits and things to try',
      body: 'Observations follow observers: a cluster marks where people looked and logged, so one popular trail can outweigh a wild but unvisited preserve. DBSCAN with one global epsilon struggles when density varies a lot, as at Montrose Point against the rest of the city; k-means assumes round, similar-sized groups; a convex hull can swallow open water or empty lots.\n\nTry, with **Group**: Insects at 200 m / 25; Plants with *Cluster ellipses*; Amphibians and reptiles at 800 m / 5; and Fungi with the **Hull prefilter levels** at 0 and then 4 (press **Time the shape graph** each time).',
      options: {
        category: 'Insects',
        pipeline: 'assembled',
        epsilon: 200,
        minimumPoints: 25,
        showClusterEllipses: true
      }
    }
  ]
});
