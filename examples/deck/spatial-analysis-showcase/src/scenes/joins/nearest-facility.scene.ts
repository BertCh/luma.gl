// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import type {NearestFacilityOptions} from './nearest-facility.compute';

const PLACE_CATEGORIES: readonly {value: string; label: string}[] = [
  {value: 'all', label: 'All places'},
  {value: 'restaurant_cafe', label: 'Restaurants and cafes'},
  {value: 'bar_nightlife', label: 'Bars and nightlife'},
  {value: 'grocery', label: 'Grocery and convenience'},
  {value: 'health', label: 'Health'},
  {value: 'school_education', label: 'Schools and education'},
  {value: 'park_recreation', label: 'Parks and recreation'},
  {value: 'transit', label: 'Transit-tagged places'},
  {value: 'retail', label: 'Retail'},
  {value: 'finance_business', label: 'Finance and business'},
  {value: 'personal_services', label: 'Personal services'},
  {value: 'arts_culture', label: 'Arts and culture'},
  {value: 'worship_community', label: 'Worship and community'},
  {value: 'lodging', label: 'Lodging'},
  {value: 'other', label: 'Other'}
];

const FEATURE_LABELS: Record<NearestFacilityOptions['featureSet'], string> = {
  hospital: 'hospital',
  library: 'library',
  school: 'school',
  fire: 'fire station',
  rail: 'L station',
  bus: 'bus stop',
  'rail-lines': 'L line'
};

export default defineScene<NearestFacilityOptions>({
  id: 'nearest-facility',
  title: 'How far is the nearest hospital?',
  chapter: 'joins',
  order: 2,
  summary:
    'Snap 105,808 Chicago places to the nearest hospital, library, school, fire station, L station, bus stop or L line within a radius, find their k nearest with foot points, turn kNN into spatial weights and see the bounding-box candidates behind a distance join.',
  contributors: ['GPUNearestFeatureJoin', 'GPUNearestFeatureWeights', 'GPUSpatialJoinCandidates'],
  datasets: [
    {id: 'chicago-places', role: 'places to join (queries)'},
    {id: 'chicago-facilities', role: 'hospitals, libraries, schools, fire stations'},
    {id: 'cta-transit', role: 'bus stops, L stations and L lines'},
    {id: 'chicago-tracts', role: 'polygons for the candidate stage'}
  ],
  initialView: {longitude: -87.68, latitude: 41.84, zoom: 10},

  options: [
    {
      kind: 'select',
      id: 'mode',
      label: 'Join mode',
      group: 'Join',
      apply: 'compile',
      default: 'nearest',
      help: 'The same contributor runs in two modes. Nearest snaps each place to one feature inside a radius; k nearest returns an ordered list per place with the exact foot point. The other two modes build on them.',
      options: [
        {
          value: 'nearest',
          label: 'Nearest within a radius',
          help: 'GPUNearestFeatureJoin, nearest-feature mode.'
        },
        {
          value: 'neighbors',
          label: 'k nearest with foot points',
          help: 'GPUNearestFeatureJoin, neighbors mode: branch-and-bound over the BVH.'
        },
        {
          value: 'weights',
          label: 'Facility-to-facility KNN weights',
          help: 'GPUNearestFeatureWeights turns the neighbours into a CSR weights matrix.'
        },
        {
          value: 'candidates',
          label: 'Bounding-box candidates',
          help: 'GPUSpatialJoinCandidates: the pair table that a distance join refines.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'featureSet',
      label: 'Nearest what?',
      group: 'Join',
      apply: 'compile',
      default: 'hospital',
      help: 'The feature layer places snap to. The BVH is built over it, so a new layer is a compile-time change.',
      options: [
        {value: 'hospital', label: 'Hospitals (53)'},
        {value: 'fire', label: 'Fire stations (92)'},
        {value: 'library', label: 'Public libraries (82)'},
        {value: 'school', label: 'CPS schools (649)'},
        {value: 'rail', label: 'L stations (135)'},
        {value: 'bus', label: 'Bus stops (10,466)'},
        {
          value: 'rail-lines',
          label: 'L lines (route shapes)',
          help: 'Line features: distance to the nearest segment, with a foot point on the track. Weights and candidates use the stations instead.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'radius',
      label: 'Snap radius',
      group: 'Join',
      apply: 'param',
      min: 100,
      max: 8000,
      step: 100,
      default: 3000,
      unit: 'm',
      disabledWhen: state => state.mode !== 'nearest',
      help: 'Per-frame buffer write. Places with no feature inside it are unmatched (red). The candidate buffer is sized per feature layer, so the radius is capped at 1 km for bus stops and 4 to 8 km for the others.'
    },
    {
      kind: 'slider',
      id: 'maxDistance',
      label: 'Search limit',
      group: 'Join',
      apply: 'param',
      min: 200,
      max: 8000,
      step: 100,
      default: 4000,
      unit: 'm',
      disabledWhen: state => state.mode !== 'neighbors',
      help: 'The k nearest join needs no candidate buffer; this per-frame limit only decides which neighbours count.'
    },
    {
      kind: 'slider',
      id: 'kShown',
      label: 'Neighbours drawn (k)',
      group: 'Join',
      apply: 'param',
      min: 1,
      max: 8,
      step: 1,
      default: 3,
      disabledWhen: state => state.mode !== 'neighbors',
      help: 'The join is compiled for k = 8; drawing the first k slots is exactly what a smaller k would return, so the slider costs nothing.'
    },
    {
      kind: 'select',
      id: 'ties',
      label: 'Ties at the k-th distance',
      group: 'Join',
      apply: 'compile',
      default: 'lowest-id',
      disabledWhen: state => state.mode !== 'neighbors',
      help: 'Lowest id keeps exactly k neighbours. All keeps every feature at the k-th distance (pandas sjoin_nearest semantics) and needs spare slots.',
      options: [
        {value: 'lowest-id', label: 'Keep the lowest feature rows'},
        {value: 'all', label: 'Keep all tied features'}
      ]
    },
    {
      kind: 'toggle',
      id: 'spatialSort',
      label: 'Hilbert-sort the features',
      group: 'Join',
      apply: 'compile',
      default: true,
      disabledWhen: state => state.mode === 'weights' || state.mode === 'candidates',
      help: 'Reorders features along a space-filling curve before the BVH build. Results are identical; only traversal cost changes. It matters for thousands of shuffled features such as bus stops.'
    },
    {
      kind: 'slider',
      id: 'weightK',
      label: 'Neighbours per facility (k)',
      group: 'Weights',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 4,
      disabledWhen: state => state.mode !== 'weights',
      help: 'libpysal KNN k. The join runs with k + 1 slots so that the facility itself can be dropped (excludeSelf).'
    },
    {
      kind: 'select',
      id: 'weightType',
      label: 'Weight rule',
      group: 'Weights',
      apply: 'compile',
      default: 'binary',
      disabledWhen: state => state.mode !== 'weights',
      help: 'Binary gives every neighbour weight 1. Inverse distance gives 1 / max(d, floor)^power, so nearer neighbours count more.',
      options: [
        {value: 'binary', label: 'Binary (1 per neighbour)'},
        {value: 'inverse-distance', label: 'Inverse distance'}
      ]
    },
    {
      kind: 'slider',
      id: 'weightPower',
      label: 'Distance decay power',
      group: 'Weights',
      apply: 'compile',
      min: 0.5,
      max: 3,
      step: 0.5,
      default: 1,
      disabledWhen: state => state.mode !== 'weights' || state.weightType !== 'inverse-distance',
      help: 'Exponent of the inverse-distance rule. Higher values make the nearest neighbour dominate.'
    },
    {
      kind: 'select',
      id: 'candidateDistance',
      label: 'Box expansion distance',
      group: 'Candidates',
      apply: 'compile',
      default: '1000',
      disabledWhen: state => state.mode !== 'candidates',
      help: 'Each facility box is grown by this planar distance before it probes the tract bounding boxes, which is what a dwithin join needs. Compile-time, so each value rebuilds.',
      options: [
        {value: '0', label: '0 m (boxes that overlap)'},
        {value: '250', label: '250 m'},
        {value: '500', label: '500 m'},
        {value: '1000', label: '1 km'},
        {value: '2000', label: '2 km'}
      ]
    },
    {
      kind: 'select',
      id: 'placeCategory',
      label: 'Places shown',
      group: 'Display',
      apply: 'param',
      default: 'all',
      help: 'A per-place mask buffer. All 105,808 places are always joined; the mask only decides which ones are drawn and counted.',
      options: PLACE_CATEGORIES
    },
    {
      kind: 'slider',
      id: 'distanceRange',
      label: 'Colour scale maximum',
      group: 'Display',
      apply: 'param',
      min: 250,
      max: 10000,
      step: 250,
      default: 5000,
      unit: 'm',
      disabledWhen: state => state.mode === 'weights' || state.mode === 'candidates',
      help: 'Distance that maps to the top of the ramp.'
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      disabledWhen: state => state.mode === 'weights' || state.mode === 'candidates',
      help: 'Perceptually uniform ramps. The legend and the map share one ramp table.',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'cividis', label: 'Cividis (colour-blind optimised)'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'}
      ]
    },
    {
      kind: 'toggle',
      id: 'showLinks',
      label: 'Draw links',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Lines from places to their match, drawn straight from the join output buffers.'
    },
    {
      kind: 'slider',
      id: 'linkStride',
      label: 'Link every n-th place',
      group: 'Display',
      apply: 'param',
      min: 1,
      max: 800,
      step: 1,
      default: 250,
      disabledWhen: state =>
        !state.showLinks || state.mode === 'weights' || state.mode === 'candidates',
      help: 'A stride keeps 105,808 links readable. Drag it down at high zoom to see every link.'
    },
    {
      kind: 'slider',
      id: 'pointSize',
      label: 'Place size',
      group: 'Display',
      apply: 'param',
      min: 0.5,
      max: 4,
      step: 0.5,
      default: 1.5,
      unit: 'px',
      help: 'Disc radius of every place.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Place opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Lower it where places overlap.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time Hilbert sort on vs off',
      group: 'Compare',
      help: 'Builds the nearest or k nearest graph both ways and times it outside the frame.'
    }
  ],

  readouts: [
    {
      id: 'places',
      label: 'Places joined',
      format: 'integer',
      help: 'Overture places inside Chicago, uploaded once.'
    },
    {id: 'features', label: 'Features'},
    {id: 'radiusUsed', label: 'Search limit in use'},
    {id: 'shown', label: 'Places shown'},
    {id: 'matched', label: 'With a match'},
    {id: 'median', label: 'Distance to nearest'},
    {
      id: 'walkable',
      label: 'Within walking distance',
      help: '400 m and 800 m are the common 5 and 10 minute walk thresholds.'
    },
    {
      id: 'catchment',
      label: 'Largest catchment',
      help: 'The feature that is nearest to the most places (nearest mode).'
    },
    {id: 'overflow', label: 'Overflow'},
    {id: 'candidates', label: 'Bounding-box candidates'},
    {id: 'weightsRows', label: 'Weights matrix'},
    {id: 'weightsRange', label: 'Weight values'},
    {id: 'weightsSymmetry', label: 'Reciprocity'},
    {id: 'candidatePairs', label: 'Candidate pairs'},
    {
      id: 'candidateExact',
      label: 'Exact within distance',
      help: 'Candidate pairs checked on the CPU against the true point-to-polygon distance.'
    },
    {id: 'candidateTracts', label: 'Tracts probed'},
    {id: 'timeSort', label: 'Join timing'}
  ],

  legends: state => {
    if (state.mode === 'weights') {
      return [
        {
          kind: 'categories',
          title: `${FEATURE_LABELS[state.featureSet === 'rail-lines' ? 'rail' : state.featureSet]} to its ${state.weightK} nearest`,
          entries: [
            {color: [255, 160, 40, 255], label: `Link to a neighbour (${state.weightType})`}
          ],
          note: 'Links fade with neighbour rank. Not every link is reciprocal.'
        }
      ];
    }
    if (state.mode === 'candidates') {
      return [
        {
          kind: 'categories',
          title: 'Candidate stage',
          entries: [
            {color: [255, 170, 60, 200], label: 'Tract whose box meets an expanded facility box'},
            {color: [90, 100, 120, 200], label: 'Line: facility to tract centroid'}
          ]
        }
      ];
    }
    const label = FEATURE_LABELS[state.featureSet];
    const legends: LegendSpec[] = [
      {
        kind: 'ramp',
        title: `Distance from a place to the nearest ${label}`,
        ramp: state.ramp,
        extent: [0, state.distanceRange],
        unit: 'm',
        format: value => `${Math.round(value).toLocaleString('en-US')}`
      },
      {
        kind: 'categories',
        title: 'No match',
        entries: [{color: [210, 60, 50, 255], label: `No ${label} within the limit`}]
      }
    ];
    if (state.mode === 'nearest' && state.featureSet !== 'rail-lines') {
      return [
        ...legends,
        {
          kind: 'ramp',
          id: 'catchment',
          title: `Places nearest to each ${label}`,
          ramp: 'magma',
          extent: 'gpu',
          sqrtScale: true,
          unit: 'places',
          format: value => Math.round(value).toLocaleString('en-US')
        }
      ];
    }
    return legends;
  },

  snippet: state => {
    const join =
      state.mode === 'nearest'
        ? `import {GPUNearestFeatureJoin} from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(
  new GPUNearestFeatureJoin({
    points: places,                          // float32x2 meters, 105,808 rows
    features: ${state.featureSet === 'rail-lines' ? "{kind: 'segments', starts, ends}" : "{kind: 'points', positions: facilities}"},
    radius: radius.importToGraph(graph),     // one float32, rewritten per frame
    candidateCapacity: places.length * 32,
    spatialSort: ${state.spatialSort},
    nearestFeatureIds, nearestDistances,     // per place
    featureCounts,                           // per feature: catchment size
    overflow
  })
);`
        : state.mode === 'neighbors'
          ? `graph.add(
  new GPUNearestFeatureJoin({
    points: places,
    features: ${state.featureSet === 'rail-lines' ? "{kind: 'lines', positions, lineOffsets}" : "{kind: 'points', positions: facilities}"},
    k: 8, neighborCapacity: ${state.ties === 'all' ? 16 : 8}, ties: '${state.ties}',
    maxDistance: maxDistance.importToGraph(graph),  // per frame
    spatialSort: ${state.spatialSort},
    neighborIds, neighborCounts,             // ordered by (distance, feature row)
    neighborDistances, neighborFootPoints,   // nearest point on each feature
    overflow
  })
);
// draw the first ${state.kShown} slots of every place: no recompile`
          : state.mode === 'weights'
            ? `graph.add(new GPUNearestFeatureJoin({points: facilities, features: {kind: 'points', positions: facilities},
  k: ${state.weightK + 1}, neighborIds, neighborCounts, neighborDistances, overflow}));
graph.add(
  new GPUNearestFeatureWeights({
    neighborIds, neighborCounts, neighborDistances,
    slotCapacity: ${state.weightK + 1}, k: ${state.weightK},
    excludeSelf: true,                       // libpysal KNN never lists a row as its own neighbour
    weightType: '${state.weightType}'${state.weightType === 'inverse-distance' ? `, power: ${state.weightPower}` : ''},
    weights: {offsets, neighbors, weights, distances},   // CSR
    overflow
  })
);`
            : `import {GPUSpatialJoinCandidates} from '@luma.gl/experimental/gpu-spatial-analysis';

graph.add(
  new GPUSpatialJoinCandidates({
    left: {kind: 'points', positions: facilities},
    right: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},
    distance: ${state.candidateDistance},                      // grows each left box (compile-time)
    pairs: {leftIds, rightIds, count, overflow, totalCount}  // sorted by (left, right)
  })
);`;
    return `${join}
const compiled = graph.compile();            // once
compiled.encode(commandEncoder, {parameters: undefined});`;
  },

  about: {
    what: '`GPUNearestFeatureJoin` finds, for every place, the nearest feature. In nearest-feature mode a BVH over the features produces candidates inside a per-frame radius and exact point-to-segment distances pick the winner. In neighbors mode every place walks the BVH nearest child first and keeps its k best features with exact distances and the foot point. `GPUNearestFeatureWeights` converts those neighbours into a CSR weights matrix, and `GPUSpatialJoinCandidates` exposes the bounding-box stage of a distance join.',
    why: 'Access questions are nearest-neighbour questions: how far is the nearest hospital, which fire station would respond, who is within a walk of a bus stop. Doing it for every place at once, and for any layer, turns a one-off query into a map you can steer.',
    howToRead:
      'Each place is coloured by the straight-line distance to its nearest feature; red places have none within the limit. White-bordered dots are the features, shaded by how many places choose them. Distances are planar and straight-line, not travel time.'
  },

  create: async ctx => (await import('./nearest-facility.compute')).createNearestFacility(ctx),

  story: [
    {
      id: 'the-question',
      controls: ['featureSet', 'radius'],
      readouts: ['median', 'catchment'],
      title: 'How far is every place from the nearest hospital?',
      body: 'Overture Maps lists **105,808** open places in Chicago. **`GPUNearestFeatureJoin`** gives each one its nearest of **53 hospitals**, measured exactly to the point, on the GPU. Places are coloured by that distance: yellow is far, purple is close. **Nearest what?** below swaps the facility layer, and **Snap radius** bounds the search.\n\nThe dots with a white border are the hospitals, shaded by their **catchment**, the number of places that are closer to them than to any other hospital. Hover a hospital for its name. The hospital list is Overture-derived and approximate.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10},
      options: {
        mode: 'nearest',
        featureSet: 'hospital',
        radius: 8000,
        distanceRange: 6000,
        linkStride: 400
      },
      highlight: {readout: 'median'}
    },
    {
      id: 'snap-radius',
      controls: ['radius', 'showLinks'],
      readouts: ['matched', 'overflow'],
      title: 'Snap only within a radius',
      body: 'Nearest-feature mode asks a bounded question: "is there one within *r* metres?". The radius is a one-float parameter buffer, so dragging **Snap radius** re-runs the join with no rebuild. Places with nothing inside it turn **red**.\n\nSet **Snap radius** to 2,500 m and read **With a match**: that is the share of places with a hospital within about a 30-minute walk. The candidate buffer is sized once per layer; **Overflow** would turn to YES if it were too small.',
      options: {radius: 2500, showLinks: true, linkStride: 300}
    },
    {
      id: 'walk-to-a-stop',
      controls: ['featureSet', 'radius', 'spatialSort', 'measure'],
      readouts: ['walkable', 'timeSort'],
      title: 'Change the question: a bus stop within a walk',
      body: 'The feature layer is just another buffer set. Set **Nearest what?** to *Bus stops (10,466)* and **Snap radius** to 400 m, the usual five-minute walk. The BVH is built over a new layer, so the panel marks the change as a rebuild.\n\nRead **Within walking distance**, the share of places within 400 m of a stop; the red places are the gaps. Toggle **Hilbert-sort the features** below and press **Time Hilbert sort on vs off**: with ten thousand features the traversal cost visibly changes while the result stays identical.',
      camera: {longitude: -87.65, latitude: 41.88, zoom: 11.2},
      options: {featureSet: 'bus', radius: 400, distanceRange: 600, linkStride: 60, pointSize: 1.2}
    },
    {
      id: 'k-nearest',
      controls: ['maxDistance', 'kShown', 'ties'],
      title: 'k nearest, with the foot point',
      body: '**Join mode** *k nearest with foot points* returns an ordered list of the k nearest features per place with **exact distances and the foot point**: the closest point on the feature, here a spot on an L track. **Search limit** is the outer bound on how far it looks. Features can be lines or polygons, not only points. Inside that bound the traversal prunes branches farther than the current k-th distance, so the limit only has to be generous, not tuned to the answer.\n\nThe join is compiled for k = 8 and **Neighbours drawn (k)** just shows the first slots. Compare **Ties at the k-th distance** set to *Keep all tied features*: that is pandas `sjoin_nearest` semantics and needs spare slots.',
      camera: {longitude: -87.66, latitude: 41.9, zoom: 11},
      options: {
        mode: 'neighbors',
        featureSet: 'rail-lines',
        maxDistance: 3000,
        kShown: 3,
        distanceRange: 3000,
        linkStride: 90,
        pointSize: 1.2
      }
    },
    {
      id: 'knn-weights',
      controls: ['weightK', 'weightType', 'weightPower'],
      readouts: ['weightsRows', 'weightsSymmetry'],
      title: 'Turn nearest neighbours into spatial weights',
      body: '**`GPUNearestFeatureWeights`** converts the neighbours of a self-join into a CSR weights matrix, the structure the spatial statistics use: libpysal `KNN` for any geometry. Here every fire station links to its k nearest stations. `excludeSelf` drops the station itself.\n\nSwitch **Weight rule** to *Inverse distance* and the weights fall off with distance; **Distance decay power** sets how fast, and **Neighbours per facility (k)** how many neighbours each station gets. The **Reciprocity** readout shows why KNN weights are directed: if station A is among B’s four nearest, B need not be among A’s. Symmetrise before using Moran’s I.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10},
      options: {
        mode: 'weights',
        featureSet: 'fire',
        weightK: 4,
        weightType: 'inverse-distance',
        weightPower: 1
      }
    },
    {
      id: 'candidate-stage',
      controls: ['candidateDistance'],
      readouts: ['candidatePairs', 'candidateExact', 'candidateTracts'],
      title: 'What a distance join looks at first',
      body: 'A distance join never measures every pair. **`GPUSpatialJoinCandidates`** is its first stage on its own: facility boxes, grown by a distance, probe a BVH of tract boxes and return every pair whose boxes meet, sorted by (left, right). Orange tracts are probed; lines run to their centroids.\n\nBoxes are conservative, so many candidates are not truly within the distance. **Exact within distance** checks every pair on the CPU against the true point-to-polygon distance. Raise **Box expansion distance** to 2 km and the gap widens: that is the work the exact stage removes.',
      camera: {longitude: -87.66, latitude: 41.87, zoom: 10.4},
      options: {mode: 'candidates', featureSet: 'hospital', candidateDistance: '1000'}
    },
    {
      id: 'limits',
      controls: ['featureSet', 'radius', 'placeCategory'],
      readouts: ['matched'],
      title: 'Straight lines are not travel time',
      body: 'Everything here is **planar, straight-line distance** to points Overture and the City list. Rivers, rail corridors and highways make a nearby hospital far away on foot, and Overture coverage is uneven, so a place-poor tract is partly a data gap. Hospital locations are approximate and should be verified before citing.\n\nTry it: set **Nearest what?** to fire stations with a 1.5 km **Snap radius**, set **Places shown** to grocery and read the walking share of food access, or compare the distance to the nearest school and to the nearest library for the same places.',
      camera: {longitude: -87.68, latitude: 41.84, zoom: 10},
      options: {
        mode: 'nearest',
        featureSet: 'fire',
        radius: 1500,
        placeCategory: 'grocery',
        distanceRange: 1500,
        pointSize: 2,
        linkStride: 40
      }
    }
  ]
});
