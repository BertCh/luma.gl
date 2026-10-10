// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassTableLegend} from '../../cartography/class-table';
import {CHICAGO, CITY_FRAMES, labelsFor} from '../../cartography/gazetteer';
import {ground} from '../../cartography/grounds';
import {formatCategory} from './b1-nature-data';
import {getSubjectColor, nextStoryLine, POINTS_CREDITS, pointsCartouche} from './b1-points-look';
import {defineScene, type ClassTable, type LegendSpec} from '../scene';
import type {NatureClusterOptions} from './nature-clusters.compute';
import {
  getInk,
  getPartitionPalette,
  getSignal,
  ITERATION_LADDER,
  makeSizeTable,
  type ClusterGround
} from './nature-clusters-classes';

const CATEGORIES = [
  'Birds',
  'Plants',
  'Insects',
  'Fungi',
  'Mammals',
  'Spiders and kin',
  'Amphibians and reptiles'
];

/** Places the story frames, read from the gazetteer (never typed). */
const MONTROSE = CHICAGO.places['montrose-point'].lngLat;
const UPTOWN = CHICAGO.places.uptown.lngLat;

/** Names that orient the reader in every step (drawn above the data). */
const ORIENTATION = labelsFor(CHICAGO, ['lake-michigan', 'loop'], {loop: {minZoom: 10.2}});

export default defineScene<NatureClusterOptions>({
  id: 'nature-clusters',
  title: 'Where do sightings form hot spots?',
  chapter: 'points',
  order: 2,
  summary:
    'DBSCAN turns a haze of bird records into named hot spots coloured by size, and shows that its radius is a scale you can draw. Convex hulls, centres and a water check describe the winner; k-means and the centre of gravity answer different questions.',
  contributors: [
    'GPUSpatialClustering',
    'GPUKMeans',
    'GPUGroupGeometry',
    'GPUGroupConvexHull',
    'GPUGeographicDistribution',
    'addClusterAndOutlineRecipe'
  ],
  datasets: [
    {id: 'chicago-nature', role: 'iNaturalist observations (2023), one group at a time'},
    {id: 'chicago-boundary', role: 'city limit, for the share of a hull that is not city land'}
  ],
  initialView: {...CITY_FRAMES.chicago},

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
      help: 'Weighted centres, ellipses and the overall distribution count a flagged observation five times (weight 5 instead of 1). Rewrites a weights buffer; no rebuild.',
      options: [
        {value: 'none', label: 'Nothing (every observation counts once)'},
        {value: 'researchGrade', label: 'Research grade (identification confirmed)'},
        {value: 'introduced', label: 'Introduced (non-native) taxon'}
      ]
    },
    {
      kind: 'select',
      id: 'view',
      label: 'Map shows',
      group: 'Data',
      apply: 'param',
      display: 'segmented',
      default: 'hotspots',
      help: 'The raw records (one dot each, before any clustering) or the clustered map. Switching is a layer change; nothing is recompiled.',
      options: [
        {value: 'records', label: 'Records'},
        {value: 'hotspots', label: 'Hot spots'}
      ]
    },
    {
      kind: 'select',
      id: 'method',
      label: 'Method',
      group: 'Clustering',
      apply: 'compile',
      display: 'segmented',
      default: 'dbscan',
      help: 'DBSCAN finds dense groups of any shape and leaves sparse points as noise. K-means partitions every point into k compact groups. Each method is its own compiled graph.',
      options: [
        {value: 'dbscan', label: 'DBSCAN'},
        {value: 'kmeans', label: 'k-means'}
      ]
    },
    {
      kind: 'slider',
      id: 'epsilon',
      label: 'Neighbour radius (ε)',
      group: 'DBSCAN',
      apply: 'param',
      min: 50,
      max: 1500,
      step: 10,
      default: 150,
      unit: 'm',
      disabledWhen: state => state.method !== 'dbscan',
      marks: [{value: 150, label: 'default'}],
      // Too small: nothing is dense enough to be a place.
      danger: [50, 90],
      describe: value => `${value} m is about ${(value / 201).toFixed(1)} Chicago blocks`,
      autoSweep: {from: 100, to: 800, durationMs: 9000, ease: 'in-out'},
      help: 'Two observations are neighbours when closer than this. Written into a parameter buffer: dragging it reruns the graphs without recompiling. Play sweeps 100 to 800 m.'
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
      expert: true,
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
      expert: true,
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
      expert: true,
      disabledWhen: state => state.method !== 'dbscan',
      help: 'Assembled: clustering, group geometry and hulls as separate contributors. Recipe: one addClusterAndOutlineRecipe call that also measures every hull area and perimeter (no weighted centres, ellipses, medoids or core flags).',
      options: [
        {value: 'assembled', label: 'Assembled contributors'},
        {value: 'recipe', label: 'One call: addClusterAndOutlineRecipe'}
      ]
    },
    {
      kind: 'slider',
      id: 'k',
      label: 'Number of groups (k)',
      group: 'K-means',
      apply: 'compile',
      display: 'stepper',
      min: 2,
      max: 16,
      step: 1,
      default: 8,
      format: value => `k = ${value}`,
      disabledWhen: state => state.method !== 'kmeans',
      help: 'Compile-time: k sets buffer sizes. At most 16, because a nominal map can tell only a few hues apart; colours are assigned so neighbours differ.'
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
      id: 'iterationStep',
      label: 'Iteration cap',
      group: 'K-means',
      apply: 'compile',
      display: 'stepper',
      min: 0,
      max: ITERATION_LADDER.length - 1,
      step: 1,
      default: ITERATION_LADDER.length - 1,
      format: value => `${ITERATION_LADDER[Math.round(value)]} iterations`,
      autoSweep: {from: 0, to: ITERATION_LADDER.length - 1, durationMs: 9000},
      disabledWhen: state => state.method !== 'kmeans',
      help: 'The graph holds this many Lloyd iterations (1, 2, 4, 8, 16 or 48) and stops moving centres once converged. Each cap is its own compiled graph, so Play recompiles at every step.'
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
      expert: true,
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
      expert: true,
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
      help: 'The smallest convex polygon around each cluster (GPUGroupConvexHull), filled in the cluster colour. A hull outlines a cluster; it does not measure its footprint.'
    },
    {
      kind: 'toggle',
      id: 'labelHotSpots',
      label: 'Name the biggest hot spots',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.method !== 'dbscan' || state.view !== 'hotspots',
      help: 'Notes on the three biggest hot spots: record count and the nearest named place, read back from the GPU labels.'
    },
    {
      kind: 'toggle',
      id: 'zoomToWinner',
      label: 'Zoom to the biggest hot spot',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.method !== 'dbscan' || state.view !== 'hotspots',
      help: 'Flies to the biggest hot spot and measures its hull area, its density and how much of the hull is outside the city limit (2,000 random points tested against the city polygon).'
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
      default: false,
      help: 'Average position of the members of each cluster.'
    },
    {
      kind: 'toggle',
      id: 'showWeighted',
      label: 'Weighted centres (diamonds)',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'Centre pulled toward heavier observations. Only drawn when a weight is chosen.'
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
      label: 'Core, border and noise',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state => state.method !== 'dbscan' || state.pipeline === 'recipe',
      help: 'Core records are filled, border records are hollow rings of the same colour, noise is small and grey.'
    },
    {
      kind: 'toggle',
      id: 'showEpsilonDiscs',
      label: 'Epsilon discs',
      group: 'Display',
      apply: 'param',
      default: false,
      disabledWhen: state =>
        state.method !== 'dbscan' || state.pipeline === 'recipe' || state.view !== 'hotspots',
      help: 'Draws the neighbour radius true to scale around one core, one border and one noise record near Montrose Point, with the live count of records inside each.'
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
      help: 'Mean centre (ring), median centre (diamond), standard distance circle and standard deviational ellipse of the whole selected group (GPUGeographicDistribution).'
    },
    {
      kind: 'button',
      id: 'measureShortcut',
      label: 'Time DBSCAN plain vs dense-box',
      group: 'Compare',
      expert: true,
      help: 'Compiles both variants into scratch buffers and times them at the current epsilon and minimum points.'
    },
    {
      kind: 'button',
      id: 'measureShapes',
      label: 'Time the shape graph',
      group: 'Compare',
      expert: true,
      help: 'GPU time of group geometry plus convex hulls at the current prefilter level. Change the level and press again.'
    }
  ],

  readouts: [
    {id: 'points', label: 'Records on the map'},
    {
      id: 'stacked',
      label: 'Records stacked on another',
      help: 'Share of records that sit at exactly the same coordinate as another record: a dot map draws them as one.'
    },
    {id: 'clusters', label: 'Hot spots', emphasis: 'tile'},
    {
      id: 'clusteredShare',
      label: 'Records in a hot spot',
      emphasis: 'tile',
      help: 'The rest are noise: fewer neighbours than the core threshold, and no core point nearby.'
    },
    {
      id: 'coreCount',
      label: 'Core records',
      help: 'At least the minimum number of records within epsilon.'
    },
    {
      id: 'borderCount',
      label: 'Border records',
      help: 'Within epsilon of a core record, but not core themselves.'
    },
    {id: 'noiseCount', label: 'Noise records', help: 'In no hot spot.'},
    {id: 'winnerSize', label: 'Biggest hot spot (records)', emphasis: 'tile'},
    {id: 'hullArea', label: 'Hull area', help: 'Area of the convex hull of the biggest hot spot.'},
    {id: 'density', label: 'Records per hull area'},
    {
      id: 'hullWaterShare',
      label: 'Hull outside the city limit',
      help: 'Share of 2,000 random points in the hull that fall outside the city polygon (lake and suburbs).'
    },
    {
      id: 'convergence',
      label: 'K-means convergence',
      help: 'Iterations used against the compiled cap.'
    },
    {
      id: 'farthest',
      label: 'Farthest record from its group',
      help: 'k-means gives every record a group, even a lone sighting far from the rest.'
    },
    {id: 'medianGap', label: 'Median centre vs mean centre'},
    {id: 'standardDistance', label: 'Standard distance'},
    {id: 'ellipse', label: 'Standard ellipse (axes, rotation)'},
    {
      id: 'epsilonCurve',
      label: 'Hot spots against epsilon',
      kind: 'chart',
      help: 'Eight epsilon values run through one compiled graph by parameter writes; click the curve to set epsilon.'
    },
    {
      id: 'sizeBars',
      label: 'Hot spot sizes',
      kind: 'chart',
      help: 'Records per hot spot, largest first, log scale; click a bar to fly to it.'
    },
    {id: 'parameters', label: 'Method and parameters', hood: true},
    {id: 'clustered', label: 'Clustered / noise', hood: true},
    {id: 'largest', label: 'Largest cluster (records)', hood: true},
    {id: 'meanSize', label: 'Mean cluster size', hood: true},
    {id: 'hulls', label: 'Hulls', hood: true},
    {id: 'weightedShift', label: 'Weighted minus mean centre', hood: true},
    {id: 'medoid', label: 'Medoid vs mean centre', hood: true},
    {id: 'recipeHulls', label: 'Recipe hull measures', hood: true},
    {id: 'numerics', label: 'Numerical notes', hood: true},
    {id: 'timing', label: 'DBSCAN timing', hood: true},
    {id: 'shapeTiming', label: 'Shape graph timing', hood: true}
  ],

  pipeline: [
    {
      id: 'lattice',
      label: 'Lattice',
      detail:
        'Records are binned into cells at least epsilon wide, so 3 x 3 cells hold every neighbour',
      show: {option: 'view', value: 'records'}
    },
    {
      id: 'core',
      label: 'Core',
      detail: 'Each record counts its neighbours within epsilon; enough makes it a core',
      show: {option: 'showCore', value: true}
    },
    {
      id: 'union',
      label: 'Union',
      detail: 'Cores within epsilon of each other are joined, lock-free'
    },
    {id: 'labels', label: 'Labels', detail: 'Borders attach to a cluster; ids are made canonical'},
    {
      id: 'shapes',
      label: 'Shapes',
      detail: 'Group geometry and a convex hull per hot spot',
      show: {option: 'showHulls', value: true}
    },
    {id: 'draw', label: 'Draw', detail: 'Layers read the class and hull buffers'}
  ],

  legends: (state, data) => {
    const groundName = (data['ground'] as ClusterGround | undefined) ?? 'light';
    const table = (data['table'] as ClassTable | undefined) ?? makeSizeTable(groundName);
    const counts = data['classCounts'] as number[] | undefined;
    const ink = getInk(groundName);
    const legends: LegendSpec[] = [];
    if (state.view === 'records') {
      legends.push({
        kind: 'categories',
        title: 'Records',
        entries: [
          {color: getSubjectColor(groundName, 160), label: 'One bird record', shape: 'dot'}
        ],
        note: 'Records at the same coordinate draw as one dot.'
      });
    } else if (state.method === 'kmeans') {
      legends.push({
        kind: 'categories',
        title: 'k-means groups',
        entries: [
          {
            color: getPartitionPalette(groundName)[0],
            label: 'Each colour is one k-means group (colours only separate neighbours)',
            shape: 'swatch'
          }
        ],
        note: 'Group identity is nominal: no colour is more or less than another.'
      });
    } else {
      if (state.showCore) {
        const noise = table.noData?.color ?? [150, 156, 166, 160];
        legends.push({
          kind: 'categories',
          title: 'Role of a record',
          entries: [
            {color: table.colors[2], label: 'Core: enough neighbours within ε', shape: 'dot'},
            {color: table.colors[2], label: 'Border: reached by a core', shape: 'ring'},
            {color: noise, label: 'Noise: in no hot spot', shape: 'dot'}
          ]
        });
      }
      legends.push(
        getClassTableLegend(table, {
          title: 'Hot spot size',
          id: 'hot-spot-size',
          basis: 'records',
          counts,
          layout: 'list',
          interactive: true,
          note: 'Fixed breaks. Points and hull fills share the classes.'
        })
      );
    }
    const markers: {
      color: readonly [number, number, number, number];
      label: string;
      shape: 'dot' | 'ring';
    }[] = [];
    if (state.view === 'hotspots' && state.showMeans) {
      markers.push({color: ink, label: 'Mean centre of a hot spot', shape: 'ring'});
    }
    if (state.view === 'hotspots' && state.showMedoids) {
      markers.push({color: ink, label: 'Medoid: the most central record', shape: 'dot'});
    }
    if (state.view === 'hotspots' && state.showWeighted && state.weight !== 'none') {
      markers.push({color: getSignal(groundName), label: 'Weighted centre', shape: 'dot'});
    }
    if (state.showDistribution) {
      markers.push(
        {color: ink, label: 'Mean centre of all records', shape: 'ring'},
        {color: getSignal(groundName), label: 'Median centre of all records', shape: 'dot'}
      );
    }
    if (markers.length) legends.push({kind: 'categories', title: 'Centres', entries: markers});
    return legends;
  },

  snippet: state => {
    const cap = ITERATION_LADDER[Math.round(state.iterationStep)];
    if (state.method === 'kmeans') {
      return `const graph = new GPUCommandGraph(device, {id: 'clusters'});
graph.add(new GPUKMeans({
  positions,                               // float32x2 meters of the chosen group
  k: ${state.k},
  iterations: ${cap},                       // compile-time cap: each cap is its own graph
  tolerance: ${state.tolerance},                         // meters a centre may still move
  initialization: '${state.initialization}',${state.initialization === 'kmeans++' ? `\n  seed: ${state.seed},` : ''}
  convergence,                             // [iterationsUsed, converged]
  labels, centers, sizes
}));
const compiled = graph.compile();
compiled.encode(commandEncoder, {parameters: undefined});
// Colours are nominal: map-colour the centres so neighbours differ.
const slots = colourPartition(centres, k, previousSlots, 7);`;
    }
    if (state.pipeline === 'recipe') {
      return `const graph = new GPUCommandGraph(device, {id: 'cluster-outline'});
addClusterAndOutlineRecipe(graph, {
  positions,
  spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
  clusteringParameters: clusteringParameters.importToGraph(graph),  // bounds, epsilon, minimumPoints
  gridSize: [256, 256],
  geometryParameters: geometryParameters.importToGraph(graph),
  maximumClusterCount: 2048,
  maximumVerticesPerHull: 256,
  hullCapacity: 1 << 14,
  outputs: {labels, hullPositions, hullOffsets, hullOverflow, areas, perimeters, centroids},
  scratch: {counts, bounds, meanCenters, hullCounts}
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
// Compiled once. Epsilon and minimumPoints are parameter words:
clusteringParameters.write(getGPUSpatialClusteringParameterValues({bounds, epsilon: ${state.epsilon}, minimumPoints: ${state.minimumPoints}}));
clustersCompiled.encode(commandEncoder, {parameters: undefined});
shapesCompiled.encode(commandEncoder, {parameters: undefined});
// When the labels settle, read them back once: a class per record (size breaks 50 / 200 / 1,000).
const sizeClass = size => breaks.filter(boundary => size >= boundary).length;`;
  },

  about: {
    what: '`GPUSpatialClustering` runs DBSCAN on the GPU in five stages: a lattice of cells at least epsilon wide, a neighbour count that flags core records, a lock-free union-find that joins cores, borders that attach to the smallest root, and canonical labels. Epsilon and the minimum points are parameter words, so they never recompile. `GPUKMeans` assigns every record to the nearest of k moving centres. `GPUGroupGeometry` and `GPUGroupConvexHull` describe every cluster, and `GPUGeographicDistribution` summarises the whole pattern with a mean, a median, a standard distance and an ellipse.',
    why: 'Clustering turns a cloud of sightings into places you can name, visit and compare. DBSCAN suits irregular patterns such as a park, a pond or a path and tells you what is not a hot spot; k-means is for dividing a city into balanced areas and never answers "nothing here". Epsilon is a scale: change it and places merge or vanish.',
    howToRead:
      'Hot spots are coloured by their size in four fixed classes (darker is bigger), never by an arbitrary id. Grey points are noise. A convex hull outlines a cluster and often swallows water; its area is not the footprint of the place. In the k-means view colours are nominal and only keep neighbouring groups apart. Counts follow observers, not only birds.'
  },

  basemap: ground('paperCity'),
  furniture: {
    title: pointsCartouche('Where are the bird hot spots?', 'iNaturalist records, 2023'),
    scaleBar: {units: 'metric'},
    credit: POINTS_CREDITS.nature
  },
  annotations: ORIENTATION,

  create: async ctx => (await import('./nature-clusters.compute')).createNatureClusters(ctx),

  story: [
    {
      id: 'the-haze',
      title: 'A haze of dots',
      headline: 'Bird records pile up in a few places',
      textAlternative:
        'Paper map of Chicago covered in blue-ink dots, one per bird record, thick along the lakefront and in the large parks.',
      body: 'Each dot is one bird record: **{{points}}** of them, and **{{stacked}}** sit exactly on top of another, so a dot map hides its own piles. Where do enough records gather to call the spot a *place*? Observers, not birds, draw this map.\n\nThe **Group** control below swaps the animals.',
      optionsMode: 'fresh',
      options: {view: 'records'},
      controls: ['category'],
      readouts: ['points', 'stacked'],
      stage: 'lattice',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1400},
      furniture: {
        title: pointsCartouche(
          'Where could a place be?',
          'One dot per bird record, iNaturalist 2023'
        )
      },
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park', 'jackson-park'])
    },
    {
      id: 'core-border-noise',
      title: 'Core, border, noise',
      headline: 'A place is enough neighbours within ε',
      textAlternative:
        'Zoomed map of the north lakefront with three dashed circles of equal radius, each labelled with the number of records inside and its role: core, border or noise.',
      body: 'DBSCAN asks each record one question: how many records lie within **Neighbour radius (ε)**, drawn here true to scale? Enough, set by **Minimum points (core threshold)**, makes a *core*. A record that only a core reaches is a *border*; the rest is *noise*. This map holds **{{coreCount}}** core, **{{borderCount}}** border and **{{noiseCount}}** noise records.\n\n*Epsilon is a distance you can draw.*',
      optionsMode: 'fresh',
      options: {
        epsilon: 150,
        minimumPoints: 20,
        showCore: true,
        showEpsilonDiscs: true,
        showHulls: false
      },
      controls: ['epsilon', 'minimumPoints'],
      readouts: ['coreCount', 'borderCount', 'noiseCount'],
      stage: 'core',
      camera: {longitude: MONTROSE[0], latitude: MONTROSE[1], zoom: 12.8, transitionMs: 1600},
      furniture: {
        title: pointsCartouche(
          'What makes a record part of a place?',
          'DBSCAN roles: core, border, noise',
          {
            effort: false
          }
        )
      },
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park', 'uptown'])
    },
    {
      id: 'scale-changes-places',
      title: 'The scale changes the places',
      headline: 'Widen ε and hot spots merge',
      textAlternative:
        'Map of the north side with hot spots drawn as filled hulls in four blue size classes, beside a curve of the number of hot spots against epsilon.',
      body: 'Colour is size, in four fixed classes, so a hot spot keeps its colour as the scale moves. Drag **Neighbour radius (ε)** or press Play: **{{clusters}}** hot spots hold **{{clusteredShare}}** of the records at this radius, and the curve shows eight radii run through one compiled graph. Raise **Minimum points (core threshold)** to keep only dense cores; **Name the biggest hot spots** adds notes.',
      optionsMode: 'fresh',
      options: {labelHotSpots: true},
      controls: ['epsilon', 'minimumPoints', 'labelHotSpots'],
      readouts: ['clusters', 'clusteredShare', 'epsilonCurve', 'sizeBars'],
      stage: 'union',
      camera: {longitude: UPTOWN[0], latitude: UPTOWN[1], zoom: 10.9, transitionMs: 1600},
      furniture: {
        title: pointsCartouche('How big is a place at each scale?', 'Hot spots by size class', {
          effort: false
        })
      },
      annotations: labelsFor(CHICAGO, ['montrose-point', 'lincoln-park'])
    },
    {
      id: 'describe-the-winner',
      title: 'Describe the winner',
      headline: 'The biggest hot spot: size, area, water',
      textAlternative:
        'Close map of the biggest hot spot: a filled convex hull around dense blue dots, with a ring at the mean centre, a dot at the medoid and a diamond for the weighted centre.',
      body: '**{{winnerSize}}** records fill a hull of **{{hullArea}}**, **{{density}}**, and **{{hullWaterShare}}** of that hull lies outside the city limit, where no record can fall. Ring: mean centre; dot: medoid. Turn on **Medoids (dots)** and **Cluster ellipses**; **Weight observations by** research grade pulls the diamond toward confirmed identifications.\n\n*A hull outlines a cluster; it does not measure it.*',
      optionsMode: 'fresh',
      options: {
        showMeans: true,
        showMedoids: true,
        showClusterEllipses: true,
        showWeighted: true,
        weight: 'researchGrade',
        zoomToWinner: true,
        labelHotSpots: true
      },
      controls: ['weight', 'showMedoids', 'showClusterEllipses'],
      readouts: ['winnerSize', 'hullArea', 'hullWaterShare', 'density'],
      stage: 'shapes',
      camera: {longitude: MONTROSE[0], latitude: MONTROSE[1], zoom: 13.4, transitionMs: 1600},
      furniture: {
        title: pointsCartouche(
          'What does the biggest hot spot hold?',
          'Hull, centres and weighted centre',
          {
            effort: false
          }
        )
      }
    },
    {
      id: 'kmeans',
      title: 'k-means never says no',
      headline: 'k-means divides the city; it never says no',
      textAlternative:
        'Map of Chicago split into eight coloured groups of dots, each outlined by a hull; neighbouring groups have different hues and every record belongs to one.',
      body: 'Same records, a different question. Under **Method**, k-means splits the city into **Number of groups (k)** groups and every record gets one, even a lone sighting ({{farthest}}). Colours only separate neighbours. Step **Iteration cap** from 1 to 48 and watch the centres settle: each cap is its own compiled graph, and **{{convergence}}**. A new **k-means++ seed** finds another partition.',
      optionsMode: 'fresh',
      options: {
        method: 'kmeans',
        k: 8,
        initialization: 'kmeans++',
        seed: 0,
        iterationStep: ITERATION_LADDER.length - 1,
        showMeans: true
      },
      controls: ['method', 'k', 'iterationStep', 'seed'],
      readouts: ['convergence', 'clusters', 'farthest'],
      stage: 'draw',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      furniture: {
        title: pointsCartouche(
          'What if every record had to belong?',
          'k-means partition, not density',
          {
            effort: false
          }
        )
      }
    },
    {
      id: 'centre-of-gravity',
      title: 'The centre of gravity',
      headline: 'Mammal records centre on the North Side',
      textAlternative:
        'Map of Chicago with blue dots for mammal records, a standard deviational ellipse, a dashed standard-distance circle, a ring at the mean centre and a diamond at the median centre.',
      body: `**\`GPUGeographicDistribution\`** summarises the whole pattern: the mean centre, the median centre **{{medianGap}}** away (outliers drag the mean, not the median), a standard distance of **{{standardDistance}}** and an ellipse of **{{ellipse}}**. **Ellipse size** and **Ellipse convention** change its coverage; **Group** swaps the animals, and the expert controls unlock the rest.\n\n${nextStoryLine('nature-clusters')}`,
      optionsMode: 'fresh',
      options: {
        category: 'Mammals',
        epsilon: 400,
        minimumPoints: 8,
        view: 'records',
        showDistribution: true,
        standardDeviations: '1'
      },
      controls: ['standardDeviations', 'ellipseConvention', 'category'],
      readouts: ['medianGap', 'standardDistance', 'ellipse'],
      stage: 'shapes',
      camera: {...CITY_FRAMES.chicago, transitionMs: 1600},
      furniture: {
        title: pointsCartouche(
          'Where is the middle of the mammal records?',
          'Mean, median and standard distance',
          {
            effort: false
          }
        )
      }
    }
  ]
});
