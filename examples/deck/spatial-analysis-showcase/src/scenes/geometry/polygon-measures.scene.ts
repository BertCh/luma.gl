// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene, type LegendSpec} from '../scene';
import {B3_PALETTE} from './b3-palette';
import type {PolygonMeasuresOptions, PolygonMetric} from './polygon-measures.compute';

const VALIDITY_COLORS = [B3_PALETTE[3], B3_PALETTE[5], B3_PALETTE[7]] as const;

const METRIC_TITLES: Record<PolygonMetric, string> = {
  area: 'Area',
  areaDistortion: 'Planar area / ellipsoidal area',
  perimeter: 'Perimeter',
  vertices: 'Vertices per polygon',
  groupArea: 'Area of the whole group',
  compactness: 'Polsby-Popper compactness',
  schwartzberg: 'Schwartzberg compactness',
  elongation: 'Elongation',
  convexity: 'Convexity (area / hull area)',
  sliver: 'Sliver flag',
  validity: 'Validity',
  mapColor: 'Map color'
};

const formatKm2 = (value: number) =>
  `${(value / 1e6).toLocaleString('en-US', {maximumFractionDigits: value < 5e7 ? 1 : 0})} km²`;
const formatKm = (value: number) =>
  `${(value / 1000).toLocaleString('en-US', {maximumFractionDigits: value < 1e5 ? 1 : 0})} km`;

function formatMetricValue(metric: PolygonMetric, value: number): string {
  switch (metric) {
    case 'area':
    case 'groupArea':
      return formatKm2(value);
    case 'perimeter':
      return formatKm(value);
    case 'areaDistortion':
      return `${value.toFixed(2)}×`;
    case 'vertices':
      return Math.round(value).toLocaleString('en-US');
    default:
      return value.toFixed(2);
  }
}

/**
 * Polygon measures, shape descriptors, label points, validity and map coloring on real polygon
 * layers. The GPU work is in `polygon-measures.compute.ts`.
 */
export default defineScene<PolygonMeasuresOptions>({
  id: 'polygon-measures',
  title: 'How big, how round, how clean?',
  chapter: 'geometry',
  order: 1,
  summary:
    'Measure every US county, Chicago community area and census tract on the GPU: area on a plane, a sphere and the WGS84 ellipsoid, compactness, label points, validity and a four-to-six colouring.',
  contributors: [
    'GPUGeometryMeasures',
    'GPUShapeDescriptors',
    'GPULabelPoint',
    'GPUGeometryValidity',
    'GPUMapColoring',
    'GPUContiguityWeights'
  ],
  datasets: [
    {id: 'us-counties', role: 'continental polygon coverage'},
    {id: 'chicago-community-areas', role: 'city-scale polygons'},
    {id: 'chicago-tracts', role: 'many small polygons'}
  ],
  initialView: {longitude: -96, latitude: 38.2, zoom: 3.45},

  options: [
    {
      kind: 'select',
      id: 'dataset',
      label: 'Polygon layer',
      group: 'Data',
      apply: 'compile',
      default: 'us-counties',
      help: 'Rebuilds the graphs for the chosen layer. Fly the map to it, or use the story steps, which move the camera for you.',
      options: [
        {
          value: 'us-counties',
          label: 'US counties (3,109)',
          help: 'Continental scale: planar distortion is large here.'
        },
        {
          value: 'chicago-community-areas',
          label: 'Chicago community areas (77)',
          help: 'City scale, a few multi-part areas.'
        },
        {
          value: 'chicago-tracts',
          label: 'Chicago census tracts (791)',
          help: 'Small polygons; rings wind clockwise.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'areaSystem',
      label: 'Coordinate system',
      group: 'Measures',
      apply: 'param',
      default: 'wgs84',
      help: 'GPUGeometryMeasures compiles one node per system; this picks which output colours the map. Planar = Web Mercator metres (counties) or local metres (Chicago).',
      options: [
        {value: 'planar', label: 'Planar', help: 'Shoelace area of projected coordinates.'},
        {
          value: 'spherical',
          label: 'Spherical',
          help: 'Chamberlain-Duquette on a sphere (turf area parity).'
        },
        {
          value: 'wgs84',
          label: 'WGS84 ellipsoid',
          help: 'Authalic sphere on the ellipsoid; the reference here.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'holeRule',
      label: 'Hole rule',
      group: 'Measures',
      apply: 'compile',
      default: 'winding',
      help: 'How rings of one feature combine. Compile-time: rebuilds the measures and shape graphs.',
      options: [
        {
          value: 'winding',
          label: 'Winding',
          help: 'Signed ring areas add up, so holes subtract and extra parts add (RFC 7946).'
        },
        {
          value: 'first-ring-exterior',
          label: 'First ring is the shell',
          help: 'Every later ring of a feature is a hole, whatever its winding: a second island is subtracted by mistake.'
        }
      ]
    },
    {
      kind: 'slider',
      id: 'sliverThreshold',
      label: 'Sliver threshold',
      group: 'Shape descriptors',
      apply: 'param',
      min: 0,
      max: 0.4,
      step: 0.01,
      default: 0.1,
      help: 'Polsby-Popper below this flags a sliver. A parameter-buffer write; the graph is not recompiled.'
    },
    {
      kind: 'toggle',
      id: 'showAxes',
      label: 'Long-axis glyphs',
      group: 'Shape descriptors',
      apply: 'param',
      default: false,
      help: "A line along each polygon's principal axis, longer for more elongated shapes (orientation and elongation outputs)."
    },
    {
      kind: 'toggle',
      id: 'showLabels',
      label: 'Label points',
      group: 'Label points',
      apply: 'param',
      default: false,
      help: 'The pole of inaccessibility: the interior point farthest from every boundary.'
    },
    {
      kind: 'toggle',
      id: 'showInscribed',
      label: 'Inscribed circles',
      group: 'Label points',
      apply: 'param',
      default: false,
      help: 'A circle around each label point with the distance output as its radius.'
    },
    {
      kind: 'toggle',
      id: 'showCentroids',
      label: 'Centroids',
      group: 'Label points',
      apply: 'param',
      default: false,
      help: 'Area-weighted centroids from GPUGeometryMeasures. Concave shapes put them outside the polygon.'
    },
    {
      kind: 'toggle',
      id: 'showBounds',
      label: 'Bounding boxes',
      group: 'Label points',
      apply: 'param',
      default: false,
      help: 'The bounds output of GPUGeometryMeasures (WGS84 node).'
    },
    {
      kind: 'slider',
      id: 'initialGridSize',
      label: 'Initial search grid',
      group: 'Label points',
      apply: 'compile',
      min: 4,
      max: 32,
      step: 4,
      default: 16,
      unit: ' per axis',
      help: 'Samples per axis of the first grid over each polygon. Coarser grids are faster and can miss narrow interiors. Compile-time.'
    },
    {
      kind: 'slider',
      id: 'refinementCandidates',
      label: 'Refinement candidates',
      group: 'Label points',
      apply: 'compile',
      min: 1,
      max: 16,
      step: 1,
      default: 4,
      help: 'How many of the best first-grid cells are refined. More candidates help multi-lobed shapes. Compile-time.'
    },
    {
      kind: 'select',
      id: 'orientation',
      label: 'Ring orientation convention',
      group: 'Validity',
      apply: 'compile',
      default: 'counter-clockwise-shell',
      help: 'Which winding counts as correct. This is a convention, not a topology error. Compile-time.',
      options: [
        {
          value: 'counter-clockwise-shell',
          label: 'Counter-clockwise shells',
          help: 'RFC 7946, OGC. Chicago files fail this: they wind clockwise.'
        },
        {
          value: 'clockwise-shell',
          label: 'Clockwise shells',
          help: 'Shapefile and Mapbox vector tile convention.'
        },
        {value: 'ignore', label: 'Ignore orientation'}
      ]
    },
    {
      kind: 'select',
      id: 'ringClosure',
      label: 'Ring closure',
      group: 'Validity',
      apply: 'compile',
      default: 'implicit',
      help: 'Whether rings must repeat their first vertex. With explicit closure, an unclosed ring is a defect.',
      options: [
        {value: 'implicit', label: 'Implicit', help: 'A repeated first vertex is tolerated.'},
        {
          value: 'explicit',
          label: 'Explicit (GeoJSON)',
          help: 'A ring must end on its first vertex.'
        }
      ]
    },
    {
      kind: 'toggle',
      id: 'injectDefects',
      label: 'Inject six test defects',
      group: 'Validity',
      apply: 'param',
      default: false,
      help: 'Writes a bow-tie, a NaN, a repeated vertex, an unclosed ring, a spike and a reversed ring into six features of a copy of the data, then re-runs the check. The clean data stays untouched.'
    },
    {
      kind: 'select',
      id: 'contiguity',
      label: 'Adjacency for colouring',
      group: 'Map colouring',
      apply: 'compile',
      default: 'rook',
      help: 'Rook: shared edge. Queen: shared vertex too, which needs more colours. Compile-time.',
      options: [
        {value: 'rook', label: 'Rook'},
        {value: 'queen', label: 'Queen'}
      ]
    },
    {
      kind: 'slider',
      id: 'colorSeed',
      label: 'Priority seed',
      group: 'Map colouring',
      apply: 'compile',
      min: 0,
      max: 99,
      step: 1,
      default: 0,
      help: 'Different seeds give different valid colourings (and sometimes a different colour count). Compile-time.'
    },
    {
      kind: 'slider',
      id: 'maxRounds',
      label: 'Maximum rounds',
      group: 'Map colouring',
      apply: 'compile',
      min: 1,
      max: 64,
      step: 1,
      default: 64,
      help: 'Cap on parallel rounds. Too few leaves polygons uncoloured; the readout reports convergence. Compile-time.'
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Colour polygons by',
      group: 'Display',
      apply: 'param',
      default: 'area',
      help: 'Every choice is a GPU output column bound straight into the fill layer; switching is a bind, not a recompute.',
      options: [
        {
          value: 'area',
          label: 'Area',
          help: 'GPUGeometryMeasures areas in the chosen coordinate system.'
        },
        {
          value: 'areaDistortion',
          label: 'Planar / ellipsoidal area',
          help: 'How much a flat Web Mercator (or local planar) area overstates the true area.'
        },
        {
          value: 'perimeter',
          label: 'Perimeter',
          help: 'Total length of every ring, holes included.'
        },
        {
          value: 'vertices',
          label: 'Vertex count',
          help: 'The vertexCounts output: how detailed each outline is.'
        },
        {
          value: 'groupArea',
          label: 'Group total (state / community area)',
          help: 'The group output: areas summed per state (counties) or community area (tracts).'
        },
        {
          value: 'compactness',
          label: 'Polsby-Popper compactness',
          help: '4πA / P²: 1 for a circle, towards 0 for ragged or thin shapes.'
        },
        {
          value: 'schwartzberg',
          label: 'Schwartzberg compactness',
          help: 'Perimeter of the equal-area circle over the actual perimeter (sqrt of Polsby-Popper).'
        },
        {
          value: 'elongation',
          label: 'Elongation',
          help: '0 round, 1 a line; from the second moments of the polygon.'
        },
        {
          value: 'convexity',
          label: 'Convexity',
          help: 'Area over convex hull area: 1 for convex, lower for notched shapes.'
        },
        {
          value: 'sliver',
          label: 'Slivers',
          help: 'Flags polygons whose compactness is below the sliver threshold.'
        },
        {
          value: 'validity',
          label: 'Validity',
          help: 'The validity bitmask collapsed to valid / orientation only / structural defect.'
        },
        {
          value: 'mapColor',
          label: 'Map colouring',
          help: 'GPUMapColoring: touching polygons never share a colour.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Colour ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      options: [
        {value: 'viridis', label: 'Viridis'},
        {value: 'magma', label: 'Magma'},
        {value: 'inferno', label: 'Inferno'},
        {value: 'cividis', label: 'Cividis'}
      ],
      help: 'Used by the continuous metrics. The legend uses the same ramp table.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'Lower it to see the basemap and the overlays through the fill.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Polygon outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Every ring edge, drawn from the same vertex buffers the contributors read.'
    }
  ],

  story: [
    {
      id: 'question',
      title: 'How big is each county, really?',
      body: `Which county is the largest in the contiguous United States, and by how much? Answering needs an **area**, and area depends on the coordinate system you measure in.

\`GPUGeometryMeasures\` computes the area of all 3,109 counties in one pass. The map shows the **WGS84 ellipsoid** areas (the reference): large counties in the arid West (San Bernardino, CA) dominate and the eastern counties are tiny. Click any county to see its numbers, or change **Coordinate system** below to compare how the area is measured.`,
      camera: {longitude: -96, latitude: 38.2, zoom: 3.45, transitionMs: 1200},
      options: {dataset: 'us-counties', metric: 'area', areaSystem: 'wgs84'},
      highlight: {readout: 'areaWgs84'},
      controls: ['areaSystem'],
      readouts: ['areaWgs84']
    },
    {
      id: 'flat-map-lies',
      title: 'A flat map overstates northern counties',
      body: `Measure the same polygons as **planar Web Mercator metres** and divide by the ellipsoidal area: the colour now shows how much a flat web map inflates each county. Mercator stretches area by about 1 / cos²(latitude): Whatcom County, Washington is overstated 2.3 times, Florida's counties about 1.5, and the whole lower 48 measures 13.3 million km² on the flat plane against 7.84 million on the ellipsoid.

Read the legend as a ratio: 1.0 means the flat measure is honest. Switch **Coordinate system** below to *Spherical*: the sphere is within about half a percent of the ellipsoid, which is why \`GPUGeometryMeasures\` treats it as a turf-compatible shortcut. The planar column is the one that is wrong at this scale.`,
      options: {metric: 'areaDistortion'},
      highlight: {readout: 'distortion'},
      controls: ['areaSystem', 'metric'],
      readouts: ['distortion', 'areaPlanar', 'areaWgs84']
    },
    {
      id: 'how-round',
      title: 'How round, how elongated?',
      body: `Now the **shape** of each community area in Chicago. \`GPUShapeDescriptors\` turns every polygon into compactness (Polsby-Popper \`4πA/P²\`, Schwartzberg), elongation, convexity and a principal-axis direction, the measures behind gerrymandering tests and QGIS compactness.

Bright areas are compact; dark ones are ragged or thin. Turn on **Long-axis glyphs** below to draw each polygon's principal axis, longer where the shape is more elongated: Chicago's lakefront areas line up with the shore. The **Sliver threshold** slider flags the weakest shapes (set **Colour polygons by** to *Slivers* to see them).`,
      camera: {longitude: -87.68, latitude: 41.83, zoom: 9.4, transitionMs: 1600},
      options: {dataset: 'chicago-community-areas', metric: 'compactness', showAxes: true},
      highlight: {readout: 'meanCompactness'},
      controls: ['showAxes', 'sliverThreshold', 'metric'],
      readouts: ['meanCompactness', 'slivers']
    },
    {
      id: 'label-points',
      title: 'Where should the label go?',
      body: `The centroid is not always inside the polygon. \`GPULabelPoint\` finds the **pole of inaccessibility**, the point farthest from every edge (the polylabel idea; turf \`pointOnFeature\`), and its \`distances\` output is the radius of the largest circle around it.

Turn on **Label points**, **Inscribed circles** and **Centroids** below: orange label points sit inside every area while pink centroids can fall outside a concave shape. Raise **Initial search grid** or **Refinement candidates** to see the trade between cost and accuracy on the long O'Hare corridor in the northwest.`,
      camera: {zoom: 9.55, longitude: -87.7, latitude: 41.84, transitionMs: 1200},
      options: {
        showLabels: true,
        showInscribed: true,
        showCentroids: true,
        showAxes: false,
        metric: 'area'
      },
      highlight: {readout: 'centroidOutside'},
      controls: [
        'showLabels',
        'showInscribed',
        'showCentroids',
        'initialGridSize',
        'refinementCandidates'
      ],
      readouts: ['centroidOutside']
    },
    {
      id: 'validity',
      title: 'Is the layer clean?',
      body: `Before any overlay or join, check the geometry. \`GPUGeometryValidity\` writes a per-polygon bitmask (self-intersection, repeated vertices, crossing rings, holes outside the shell, orientation) with exact predicates and nothing read back.

Chicago's census tracts wind their shells **clockwise**. Under the RFC 7946 convention every one is flagged amber (orientation only): a convention, not a topology error. Here **Ring orientation convention** is set to *Clockwise shells*, so the layer is all green (try *Counter-clockwise shells*). Switch **Inject six test defects** on: a bow-tie and a spike (self-intersection), a NaN and a repeated vertex turn red, a reversed ring turns amber, and the unclosed ring only shows with **Ring closure** set to *Explicit*. The real US counties layer trips one check on its own: six counties repeat a vertex.`,
      camera: {longitude: -87.68, latitude: 41.83, zoom: 9.9, transitionMs: 1400},
      options: {
        dataset: 'chicago-tracts',
        metric: 'validity',
        orientation: 'clockwise-shell',
        showLabels: false,
        showInscribed: false,
        showCentroids: false
      },
      highlight: {readout: 'validity'},
      controls: ['orientation', 'injectDefects', 'ringClosure'],
      readouts: ['validity']
    },
    {
      id: 'map-colouring',
      title: 'Colour the map so neighbours differ',
      body: `A categorical map needs touching polygons to have different colours. \`GPUContiguityWeights\` finds the neighbours and \`GPUMapColoring\` runs a Jones-Plassmann parallel greedy colouring over them: in each round every polygon that out-ranks its uncoloured neighbours picks the lowest free colour.

The readout shows how many colours the result needed, how many conflicts remain (0 means a proper colouring) and how many rounds ran. Try another **Priority seed** below: each seed is a different valid colouring, and the count (at most max-degree + 1, typically 5 to 7) can change. Setting **Adjacency for colouring** to *Queen* also separates polygons that only meet at a corner.`,
      camera: {longitude: -96, latitude: 38.2, zoom: 3.45, transitionMs: 1600},
      options: {dataset: 'us-counties', metric: 'mapColor', orientation: 'counter-clockwise-shell'},
      highlight: {readout: 'coloring'},
      controls: ['colorSeed', 'contiguity'],
      readouts: ['coloring']
    },
    {
      id: 'limits',
      title: 'Limits and things to try',
      body: `**Limits.** Planar and spherical measures are only as good as the system you choose: pick the planar projection for your region. Spherical areas treat edges as straight in an equal-area projection, so very long edges and rings that enclose a pole are not supported. Validity does not check interior connectivity or nested holes; repair stays on the CPU. Map colouring is greedy, so the colour count is an upper bound, not the minimum. Tooltips and clicks use a small readback taken when the controls settle.

**Try it.** Set **Hole rule** to *First ring is the shell* on US counties and watch the total area drop: second islands are subtracted. Switch **Adjacency for colouring** to *Queen* and compare the colour count. Set **Polygon layer** to *Chicago census tracts* and drag **Sliver threshold** until a few percent are flagged.`,
      options: {dataset: 'us-counties', metric: 'area', holeRule: 'winding'},
      highlight: {readout: 'areaWgs84'},
      controls: ['holeRule', 'contiguity', 'dataset', 'sliverThreshold'],
      readouts: ['areaWgs84', 'coloring', 'slivers']
    }
  ],

  about: {
    what: 'Per-polygon measurement on the GPU: area, perimeter, centroid, bounds and vertex count in planar, spherical and WGS84 modes, with group totals; compactness, elongation, convexity and orientation; label points; a validity bitmask; and a parallel greedy map colouring.',
    why: 'These are the numbers behind every area-based rate, compactness test and cartographic label, and the checks you want before an overlay or join. Doing them on the GPU keeps a 3,000-polygon layer interactive while you change coordinate system, hole rule or thresholds.',
    howToRead:
      'The fill is a GPU output column read directly by the shader. The legend shows the colour range (2nd to 98th percentile for continuous metrics). Click a polygon to outline it and read its numbers in the panel. Orange marks are label points, pink dots are centroids, the blue boxes are bounds.'
  },

  legends: state => {
    const legends: LegendSpec[] = [];
    const metric = state.metric;
    if (metric === 'sliver') {
      legends.push({
        kind: 'categories',
        title: 'Slivers',
        entries: [
          {
            color: [226, 96, 80, 255],
            label: `Polsby-Popper below ${state.sliverThreshold.toFixed(2)}`
          },
          {color: [196, 204, 214, 200], label: 'Compact enough'}
        ]
      });
    } else if (metric === 'validity') {
      legends.push({
        kind: 'categories',
        title: 'Validity',
        entries: [
          {color: [...VALIDITY_COLORS[0], 255], label: 'Valid'},
          {color: [...VALIDITY_COLORS[1], 255], label: 'Orientation only'},
          {color: [...VALIDITY_COLORS[2], 255], label: 'Structural defect'}
        ],
        note: 'Structural: self-intersection, repeated vertex, NaN, crossing rings, hole outside shell, unclosed ring.'
      });
    } else if (metric === 'mapColor') {
      legends.push({
        kind: 'categories',
        title: 'Colour index',
        entries: B3_PALETTE.map((color, index) => ({
          color: [...color, 255] as const,
          label: `Colour ${index + 1}`
        })),
        note: 'Touching polygons never share a colour. More than eight colours would wrap around.'
      });
    } else {
      legends.push({
        kind: 'ramp',
        id: 'value',
        title: METRIC_TITLES[metric],
        ramp: state.ramp,
        extent: 'gpu',
        sqrtScale: metric === 'area' || metric === 'vertices',
        format: value => formatMetricValue(metric, value),
        labels: undefined
      });
    }
    const overlays: {color: [number, number, number, number]; label: string}[] = [];
    if (state.showLabels)
      overlays.push({color: [226, 150, 40, 255], label: 'Label point (pole of inaccessibility)'});
    if (state.showCentroids)
      overlays.push({color: [232, 90, 140, 255], label: 'Area-weighted centroid'});
    if (state.showInscribed)
      overlays.push({color: [226, 150, 40, 255], label: 'Inscribed circle (distance output)'});
    if (state.showAxes)
      overlays.push({
        color: [0, 130, 190, 255],
        label: 'Principal axis (length grows with elongation)'
      });
    if (state.showBounds) overlays.push({color: [78, 168, 222, 255], label: 'Bounding box'});
    if (overlays.length) legends.push({kind: 'categories', title: 'Overlays', entries: overlays});
    return legends;
  },

  readouts: [
    {id: 'features', label: 'Polygons', format: 'integer'},
    {id: 'vertices', label: 'Vertices', format: 'integer'},
    {id: 'areaWgs84', label: 'Total area, WGS84', help: 'Sum of the ellipsoidal areas.'},
    {id: 'areaSphere', label: 'Total area, sphere'},
    {
      id: 'areaPlanar',
      label: 'Total area, planar',
      help: 'Web Mercator metres for counties, local metres for Chicago.'
    },
    {
      id: 'distortion',
      label: 'Worst planar overstatement',
      help: 'Largest planar / ellipsoid area ratio and where it occurs.'
    },
    {id: 'meanCompactness', label: 'Mean compactness'},
    {id: 'slivers', label: 'Slivers', help: 'Polygons below the sliver threshold.'},
    {
      id: 'centroidOutside',
      label: 'Centroids outside polygon',
      help: 'Concave shapes whose area-weighted centroid is not inside them.'
    },
    {
      id: 'validity',
      label: 'Validity bits set',
      help: 'Features with each bit of the validity mask.'
    },
    {id: 'coloring', label: 'Map colouring'},
    {
      id: 'adjacencies',
      label: 'Neighbour slots',
      format: 'integer',
      help: 'Directed adjacency entries of the contiguity graph.'
    },
    {id: 'selected', label: 'Selected polygon', help: 'Click a polygon to select it.'}
  ],

  snippet: state => {
    const measures = `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUGeometryMeasures, GPUShapeDescriptors, GPULabelPoint,
  GPUGeometryValidity, GPUContiguityWeights, GPUMapColoring
} from '@luma.gl/experimental/gpu-spatial-analysis';

const graph = new GPUCommandGraph(device, {id: 'polygons'});
// positions: lng/lat float32x2; ringOffsets/featureRingOffsets: GeoArrow offsets (uint32).
graph.add(new GPUGeometryMeasures({
  positions, ringOffsets, featureRingOffsets,
  geometryType: 'polygons',
  coordinateSystem: '${state.areaSystem}',   // planar | spherical | wgs84
  holeRule: '${state.holeRule}',
  output: {areas, lengths, centroids, bounds, vertexCounts}
}));
`;
    const shape = `graph.add(new GPUShapeDescriptors({
  positions: planarPositions, ringOffsets, featureRingOffsets,
  holeRule: '${state.holeRule}',
  parameters: sliverParameters.importToGraph(graph),   // sliverThreshold = ${state.sliverThreshold}
  output: {polsbyPopper, schwartzberg, elongation, orientation, convexity, clockwise, sliver}
}));
graph.add(new GPULabelPoint({
  positions: planarPositions, ringOffsets, featureRingOffsets,
  initialGridSize: ${state.initialGridSize}, refinementCandidates: ${state.refinementCandidates},
  output: {points, distances, degenerate}
}));
`;
    const validity = `graph.add(new GPUGeometryValidity({
  polygons: {kind: 'polygons', positions, featureOffsets, polygonOffsets, ringOffsets},
  mask, overflow, intersectionCapacity: 4096,
  orientation: '${state.orientation}', ringClosure: '${state.ringClosure}'
}));
graph.add(new GPUContiguityWeights({criterion: '${state.contiguity}', positions, ringOffsets, polygonOffsets, weights, overflow: weightOverflow}));
graph.add(new GPUMapColoring({weights, colors, colorCount, seed: ${state.colorSeed}, maximumRounds: ${state.maxRounds}}));

const compiled = graph.compile();
compiled.encode(commandEncoder, {parameters: undefined});   // outputs stay on the GPU`;
    return measures + '\n' + shape + '\n' + validity;
  },

  create: async ctx => (await import('./polygon-measures.compute')).createPolygonMeasures(ctx)
});
