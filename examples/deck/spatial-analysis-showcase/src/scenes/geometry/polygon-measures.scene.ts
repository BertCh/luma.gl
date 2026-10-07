// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {getClassTableLegend, makeClassTable} from '../../cartography/class-table';
import {ground} from '../../cartography/grounds';
import {defineScene, type LegendSpec} from '../scene';
import {B3_PALETTE} from './b3-palette';
import type {PolygonMeasuresOptions, PolygonMetric} from './polygon-measures.compute';

// Keep this declarative scene loadable by static story gates; compute owns matching GPU tables.
const POLYGON_AREA_CLASSES = makeClassTable({
  breaks: [500e6, 1e9, 2e9, 5e9, 10e9],
  scheme: 'YlOrBr',
  unit: 'km²',
  labels: ['0–500', '500–1k', '1k–2k', '2k–5k', '5k–10k', '10k+'],
  noData: {label: 'No area'}
});
const POLYGON_DISTORTION_CLASSES = makeClassTable({
  breaks: [1.05, 1.2, 1.5, 2],
  scheme: 'PuRd',
  unit: '×',
  labels: ['1.00–1.05×', '1.05–1.20×', '1.20–1.50×', '1.50–2.00×', '2.00×+']
});
const POLYGON_COMPACTNESS_CLASSES = makeClassTable({
  breaks: [0.1, 0.25, 0.4, 0.6],
  scheme: 'BuGn',
  unit: 'Polsby-Popper',
  labels: ['< 0.10', '0.10–0.25', '0.25–0.40', '0.40–0.60', '0.60–1.00']
});
const POLYGON_VALIDITY_CLASSES = makeClassTable({
  breaks: [0.5, 1.5],
  colors: [
    [140, 149, 160, 255],
    [230, 159, 0, 255],
    [213, 94, 0, 255]
  ],
  labels: ['No bits set', 'Orientation convention only', 'Structural defect']
});

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
    {id: 'us-states', role: 'matching dissolved state-outline context'},
    {id: 'chicago-community-areas', role: 'city-scale polygons'},
    {id: 'chicago-tracts', role: 'many small polygons'}
  ],
  initialView: {longitude: -96, latitude: 38.2, zoom: 3.45},
  basemap: ground('paperSheet'),
  furniture: {
    title: {
      title: 'Polygon measures',
      subtitle: 'Area, form and validity depend on representation'
    },
    credit: joinCredits(CREDITS.usCensus, CREDITS.cityOfChicago)
  },

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
      id: 'areaClassification',
      label: 'Compare area classes',
      group: 'Measures',
      apply: 'param',
      default: 'fixed-log',
      display: 'segmented',
      options: [
        {value: 'fixed-log', label: 'Fixed log'},
        {value: 'quantile', label: 'Quantile'},
        {value: 'equal-interval', label: 'Equal interval'}
      ],
      help: 'The published fixed YlOrBr table remains the reference while this compares common alternatives.'
    },
    {
      kind: 'toggle',
      id: 'showDeflate',
      label: 'Deflate about GPU centroid',
      group: 'Display',
      apply: 'param',
      default: false
    },
    {
      kind: 'toggle',
      id: 'showLabelSearch',
      label: 'Show search grid',
      group: 'Label points',
      apply: 'param',
      default: false
    },
    {
      kind: 'select',
      id: 'coloringMethod',
      label: 'Colour assignment',
      group: 'Map colouring',
      apply: 'param',
      default: 'algorithm',
      display: 'segmented',
      options: [
        {value: 'algorithm', label: 'GPU greedy'},
        {value: 'naive-fips-mod', label: 'Naive FIPS mod'}
      ]
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
      id: 'area-classes',
      title: 'Area classes',
      headline: 'Area is skewed before it is mapped',
      textAlternative: 'WGS84 county areas use a fixed class table with state-outline context.',
      body: `WGS84 is the reference area. The fixed YlOrBr km² table is shared by the layer, legend and histogram; compare log, quantile and equal-interval breaks without silently changing that reference. {{areaClasses}}`,
      camera: {longitude: -96, latitude: 38.2, zoom: 3.45, transitionMs: 1200},
      optionsMode: 'fresh',
      options: {
        dataset: 'us-counties',
        metric: 'area',
        areaSystem: 'wgs84',
        areaClassification: 'fixed-log'
      },
      highlight: {readout: 'areaWgs84'},
      controls: ['areaClassification', 'areaSystem'],
      readouts: ['features', 'areaWgs84', 'areaClasses'],
      stage: 'measures'
    },
    {
      id: 'flat-map',
      title: 'Flat-map inflation',
      headline: 'The flat map inflates northern area',
      textAlternative:
        'Ratio increases with latitude; deflated polygons show planar-to-WGS84 correction around GPU centroids.',
      body: `The ratio is **planar / WGS84**. Deflate is explanatory geometry: each triangle scales about its GPU centroid by \`1 / sqrt(planar / WGS84)\`. It does not replace Web Mercator. The ratio-versus-latitude scatter includes the Mercator theory curve and equal-true-size reference geometry.`,
      optionsMode: 'fresh',
      options: {dataset: 'us-counties', metric: 'areaDistortion', showDeflate: true},
      highlight: {readout: 'distortion'},
      controls: ['showDeflate'],
      readouts: ['distortion', 'areaPlanar', 'areaWgs84'],
      stage: 'measures'
    },
    {
      id: 'compactness',
      title: 'Compactness',
      headline: 'Roundness needs reference shapes',
      textAlternative:
        'Chicago compactness classes include a long-axis glyph and an area-equivalent circle.',
      body: `Polsby-Popper compares every community area with a circle of equal area. The fixed BuGn classes are calibrated against circle, square and sliver silhouettes; long axis and reference circle make a selected value inspectable. Boundary detail changes perimeter.`,
      camera: {longitude: -87.68, latitude: 41.83, zoom: 9.4, transitionMs: 1600},
      optionsMode: 'fresh',
      options: {
        dataset: 'chicago-community-areas',
        metric: 'compactness',
        showAxes: true,
        showInscribed: true
      },
      highlight: {readout: 'meanCompactness'},
      controls: ['showAxes', 'sliverThreshold', 'metric'],
      readouts: ['meanCompactness', 'slivers', 'selected'],
      stage: 'shape'
    },
    {
      id: 'label-point',
      title: 'Label point',
      headline: 'A label point must stay inside',
      textAlternative:
        'Centroid, interior label point and circle are distinct marks; a named loaded feature shows grid and refinement.',
      body: `Centroid, label point and clearance circle are distinct marks. The O’Hare specimen is resolved from loaded names, then its selected initial grid and refinement lattice are drawn. Grid size and candidate count explain the approximation rather than claiming an exact label.`,
      camera: {zoom: 9.55, longitude: -87.7, latitude: 41.84, transitionMs: 1200},
      optionsMode: 'fresh',
      options: {
        showLabels: true,
        showInscribed: true,
        showCentroids: true,
        showAxes: false,
        metric: 'area',
        showLabelSearch: true
      },
      highlight: {readout: 'centroidOutside'},
      controls: ['initialGridSize', 'refinementCandidates', 'showLabelSearch'],
      readouts: ['centroidOutside', 'labelClearance', 'labelSearch'],
      stage: 'label'
    },
    {
      id: 'validity',
      title: 'Is the layer clean?',
      headline: 'Convention differs from structural failure',
      textAlternative:
        'Neutral, amber and vermilion map classes distinguish valid geometry, convention-only orientation and structural bits.',
      body: `\`GPUGeometryValidity\` writes a per-polygon bitmask. Orientation is a convention; self-intersection, repeated coordinates and other structural bits are not. Colour plus redundant defect outlines separate those meanings. The bar chart is resolved from per-bit counts; overflow is reported separately.`,
      camera: {longitude: -87.68, latitude: 41.83, zoom: 9.9, transitionMs: 1400},
      options: {
        dataset: 'chicago-tracts',
        metric: 'validity',
        orientation: 'counter-clockwise-shell',
        showLabels: false,
        showInscribed: false,
        showCentroids: false
      },
      highlight: {readout: 'validity'},
      controls: ['orientation', 'injectDefects'],
      readouts: ['validity', 'validityConvention', 'validityStructural', 'validityOverflow'],
      optionsMode: 'fresh',
      stage: 'validity'
    },
    {
      id: 'colouring',
      title: 'Neighbour colouring',
      headline: 'Neighbour colours are nominal categories',
      textAlternative:
        'White seams make conflicts visible in a naive FIPS-mod comparison and a greedy nominal assignment.',
      body: `Compare naive FIPS-mod classes with the GPU greedy colouring. The Set3-like palette is nominal, while white seams expose conflicts. Rook and queen change the graph; the rounds chart reports convergence. The greedy count is an upper bound, and generalised boundaries do not model nested-hole connectivity. Next: *Turn points and lines into new shapes around Chicago.*`,
      camera: {longitude: -96, latitude: 38.2, zoom: 3.45, transitionMs: 1600},
      optionsMode: 'fresh',
      options: {
        dataset: 'us-counties',
        metric: 'mapColor',
        coloringMethod: 'algorithm',
        contiguity: 'rook'
      },
      highlight: {readout: 'coloring'},
      controls: ['coloringMethod', 'contiguity'],
      readouts: ['coloring', 'adjacencies'],
      stage: 'colouring'
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
    if (metric === 'area') {
      legends.push(
        getClassTableLegend(POLYGON_AREA_CLASSES, {title: 'WGS84 area', layout: 'list'})
      );
    } else if (metric === 'areaDistortion') {
      legends.push(
        getClassTableLegend(POLYGON_DISTORTION_CLASSES, {
          title: 'Planar / WGS84 area',
          layout: 'list'
        })
      );
    } else if (metric === 'compactness') {
      legends.push(
        getClassTableLegend(POLYGON_COMPACTNESS_CLASSES, {
          title: 'Polsby-Popper compactness',
          layout: 'list'
        })
      );
    } else if (metric === 'sliver') {
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
      legends.push(
        getClassTableLegend(POLYGON_VALIDITY_CLASSES, {
          title: 'Validity status',
          layout: 'list',
          note: 'Amber is convention-only; vermilion is structural.'
        })
      );
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
        kind: 'categories',
        title: METRIC_TITLES[metric],
        entries: [{color: [100, 120, 145, 255], label: 'Continuous GPU value'}]
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
    {id: 'areaClasses', label: 'Area class counts'},
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
    {id: 'distortionScatter', label: 'Ratio / latitude', kind: 'chart'},
    {id: 'meanCompactness', label: 'Mean compactness'},
    {id: 'slivers', label: 'Slivers', help: 'Polygons below the sliver threshold.'},
    {
      id: 'centroidOutside',
      label: 'Centroids outside polygon',
      help: 'Concave shapes whose area-weighted centroid is not inside them.'
    },
    {id: 'labelClearance', label: 'Selected clearance'},
    {id: 'labelSearch', label: 'Grid / candidates', hood: true},
    {
      id: 'validity',
      label: 'Validity bits set',
      help: 'Features with each bit of the validity mask.'
    },
    {id: 'validityConvention', label: 'Convention only'},
    {id: 'validityStructural', label: 'Structural defects'},
    {id: 'validityOverflow', label: 'Validity overflow', hood: true},
    {id: 'validityBits', label: 'Validity bits', kind: 'chart'},
    {id: 'coloring', label: 'Map colouring'},
    {id: 'coloringRounds', label: 'Greedy rounds', kind: 'chart'},
    {
      id: 'adjacencies',
      label: 'Neighbour slots',
      format: 'integer',
      help: 'Directed adjacency entries of the contiguity graph.'
    },
    {id: 'selected', label: 'Selected polygon', help: 'Click a polygon to select it.'}
  ],

  pipeline: [
    {id: 'measures', label: 'Measure', detail: 'Area systems and planar/WGS84 ratio.'},
    {id: 'shape', label: 'Describe', detail: 'Compactness, long axis and reference circle.'},
    {id: 'label', label: 'Place', detail: 'Initial grid and refinement candidates.'},
    {id: 'validity', label: 'Validate', detail: 'Convention and structural bits.'},
    {id: 'colouring', label: 'Colour', detail: 'Contiguity and greedy rounds.'}
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
