// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {CREDITS, joinCredits} from '../../cartography/credits';
import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import {storyFromMarkdown} from '../story-markdown';
import narrative from './wildfire-shapes.md?raw';
import type {WildfireShapesOptions} from './wildfire-shapes.compute';

const YEAR_COLORS = [
  {color: [232, 90, 60, 255], label: '2020'},
  {color: [240, 170, 50, 255], label: '2021'},
  {color: [70, 190, 150, 255], label: '2022'},
  {color: [110, 140, 235, 255], label: '2023'}
] as const;

const METRIC_LEGENDS: Record<
  string,
  {title: string; unit?: string; sqrt?: boolean; labels?: [string, string]}
> = {
  area: {title: 'Fire area', unit: 'acres', sqrt: true},
  perimeter: {title: 'Perimeter', unit: 'km'},
  areaDistortion: {title: 'Planar area / WGS84 area', labels: ['1.0', 'larger']},
  vertices: {title: 'Vertices per fire', unit: 'vertices', sqrt: true},
  polsbyPopper: {title: 'Polsby-Popper compactness', labels: ['stringy', 'round']},
  schwartzberg: {title: 'Schwartzberg compactness', labels: ['stringy', 'round']},
  elongation: {title: 'Elongation', labels: ['round', 'line-like']},
  convexity: {title: 'Convexity (area / hull area)', labels: ['wraps around', 'convex']}
};

export default defineScene<WildfireShapesOptions>({
  id: 'wildfire-shapes',
  title: 'Do big fires get stringier?',
  chapter: 'geometry',
  order: 7,
  summary:
    'Mapped wildfire perimeters compared for compactness, multipart accounting and detail sensitivity with GPU geometry measures and shape descriptors.',
  contributors: ['GPUGeometryMeasures', 'GPUShapeDescriptors'],
  datasets: [{id: 'poopdeck-wildfires', role: 'final fire perimeters, 2020 to 2023'}],
  initialView: {longitude: -116.5, latitude: 40, zoom: 4.2},
  basemap: ground('paperSheet'),
  furniture: {
    title: {
      title: 'Mapped perimeter shape',
      subtitle: 'Compactness is conditional on boundary detail'
    },
    scaleBar: {units: 'metric'},
    credit: joinCredits(CREDITS.usgs, 'NIFC perimeter records (public domain)')
  },

  options: [
    {
      kind: 'select',
      id: 'shapeView',
      label: 'Shape view',
      group: 'Display',
      apply: 'param',
      default: 'map',
      options: [
        {value: 'map', label: 'Mapped perimeters'},
        {value: 'gallery', label: 'Equal-size silhouettes'},
        {value: 'detail', label: 'Prepared detail comparison'}
      ],
      help: 'Equal-size silhouettes preserve outline shape while removing acreage from the display.'
    },
    {
      kind: 'select',
      id: 'metric',
      label: 'Color by',
      group: 'Display',
      apply: 'param',
      default: 'polsbyPopper',
      help: 'Which per-fire GPU column colors the polygons and dots. Switching is a buffer copy; no graph is rebuilt.',
      options: [
        {value: 'year', label: 'Perimeter year'},
        {
          value: 'area',
          label: 'Area',
          help: 'GPUGeometryMeasures areas in the chosen Area system.'
        },
        {
          value: 'perimeter',
          label: 'Perimeter',
          help: 'GPUGeometryMeasures lengths over every ring.'
        },
        {
          value: 'areaDistortion',
          label: 'Planar / WGS84 area',
          help: 'How much Web Mercator inflates each fire.'
        },
        {
          value: 'vertices',
          label: 'Vertices',
          help: 'Number of vertex rows, a proxy for mapping detail.'
        },
        {
          value: 'polsbyPopper',
          label: 'Polsby-Popper compactness',
          help: '4 pi A / P^2, 1 for a circle.'
        },
        {
          value: 'schwartzberg',
          label: 'Schwartzberg compactness',
          help: 'Circle perimeter over fire perimeter, sqrt of Polsby-Popper.'
        },
        {
          value: 'elongation',
          label: 'Elongation',
          help: '0 for a round fire, toward 1 for a line.'
        },
        {value: 'convexity', label: 'Convexity', help: 'Area over convex hull area.'},
        {value: 'sliver', label: 'Sliver flag', help: 'Polsby-Popper below the sliver threshold.'}
      ]
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Fill opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.85,
      help: 'Opacity of the filled polygons.'
    },
    {
      kind: 'toggle',
      id: 'showOutlines',
      label: 'Outlines',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Ring edges of the shown fires (340,000 segments from one buffer).'
    },
    {
      kind: 'toggle',
      id: 'showMarkers',
      label: 'Centroid dots',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'A dot at the WGS84 centroid of every fire, so small fires stay visible when zoomed out. The centroids are a GPUGeometryMeasures output.'
    },
    {
      kind: 'toggle',
      id: 'showAxes',
      label: 'Major axes',
      group: 'Display',
      apply: 'param',
      default: false,
      help: 'The major axis of every fire from GPUShapeDescriptors orientation, longer for more elongated fires. Built from the readback table.'
    },
    {
      kind: 'select',
      id: 'year',
      label: 'Perimeter year',
      group: 'Filter',
      apply: 'param',
      default: 'all',
      help: 'Shows the fires of one year. Hidden fires leave the colors, charts and readouts.',
      options: [
        {value: 'all', label: 'All years'},
        {value: '2020', label: '2020'},
        {value: '2021', label: '2021'},
        {value: '2022', label: '2022'},
        {value: '2023', label: '2023'}
      ]
    },
    {
      kind: 'slider',
      id: 'minAcres',
      label: 'Smallest fire',
      group: 'Filter',
      apply: 'param',
      min: 1000,
      max: 50000,
      step: 500,
      default: 1000,
      unit: 'acres',
      help: 'Hides fires below this size (NIFC acres). All fires in the archive are at least 1,000 acres.'
    },
    {
      kind: 'select',
      id: 'areaSystem',
      label: 'Area system',
      group: 'GPUGeometryMeasures',
      apply: 'param',
      default: 'wgs84',
      disabledWhen: state => state.metric !== 'area' && state.metric !== 'perimeter',
      help: 'Which of the four measure sets feeds Area and Perimeter. All four run in one graph, so this is a column switch, not a rebuild.',
      options: [
        {
          value: 'planar',
          label: 'Planar (Web Mercator meters)',
          help: 'Euclidean on projected coordinates: areas inflate by 1 / cos^2 of latitude.'
        },
        {
          value: 'spherical',
          label: 'Spherical (mean radius)',
          help: 'Haversine and Chamberlain-Duquette on a sphere.'
        },
        {
          value: 'wgs84',
          label: 'WGS84 authalic sphere',
          help: 'Vincenty edges, equal-area latitude: matches the NIFC acres.'
        },
        {
          value: 'geodesic',
          label: 'Geodesic edges',
          help: 'Edges of 20 km or more follow geodesics; the same on fires of this size.'
        }
      ]
    },
    {
      kind: 'select',
      id: 'holeRule',
      label: 'Hole rule',
      group: 'GPUGeometryMeasures',
      apply: 'compile',
      default: 'winding',
      help: 'How the rings of one fire combine. winding sums signed areas (shells counter-clockwise add, holes clockwise subtract). first-ring-exterior takes the first ring as the shell and every other ring as a hole, which is wrong for multi-polygon fires. Rebuilds both graphs.',
      options: [
        {value: 'winding', label: 'Winding (GeoJSON)'},
        {value: 'first-ring-exterior', label: 'First ring is the exterior'}
      ]
    },
    {
      kind: 'select',
      id: 'largeRings',
      label: 'Large-ring path',
      group: 'GPUGeometryMeasures',
      apply: 'compile',
      default: 'cooperative',
      help: 'A ring of more than 512 vertices is measured by a 64-lane workgroup (cooperative); serial turns that off (cooperativeRingRows 0) and one invocation walks it. Same numbers to float precision, different latency. Rebuilds the measures graph.',
      options: [
        {value: 'cooperative', label: 'Cooperative above 512 vertices'},
        {value: 'serial', label: 'Serial'}
      ]
    },
    {
      kind: 'select',
      id: 'convexityMethod',
      label: 'Convexity method',
      group: 'GPUShapeDescriptors',
      apply: 'compile',
      default: 'auto',
      help: 'The hull algorithm behind convexity. Gift wrapping marches a private hull per fire (cost: vertices times hull vertices); the monotone chain sorts (O(n log n), about thirty small passes). auto picks the chain when fires average at least 256 vertices. Rebuilds the shape graph.',
      options: [
        {value: 'auto', label: 'Auto'},
        {value: 'gift-wrapping', label: 'Gift wrapping'},
        {value: 'monotone-chain', label: 'Monotone chain'}
      ]
    },
    {
      kind: 'slider',
      id: 'sliverThreshold',
      label: 'Sliver threshold',
      group: 'GPUShapeDescriptors',
      apply: 'param',
      min: 0.01,
      max: 0.3,
      step: 0.01,
      default: 0.05,
      disabledWhen: state => state.metric !== 'sliver',
      help: 'Polsby-Popper below this flags a fire as a sliver (0.05 is about a 1:60 rectangle). A parameter-buffer write.'
    }
  ],

  story: storyFromMarkdown<WildfireShapesOptions>(narrative, {
    gallery: {
      headline: 'Equal extents isolate shape from acreage',
      textAlternative:
        'Twelve north-up wildfire outlines at the lowest and highest Polsby-Popper values are scaled to equal extents and coloured by fixed compactness classes; this removes acreage scale and retains differences in mapped boundary detail.',
      camera: {longitude: -116.5, latitude: 40, zoom: 4.2, transitionMs: 1400},
      optionsMode: 'fresh',
      options: {metric: 'polsbyPopper', showMarkers: false, shapeView: 'gallery'},
      controls: ['shapeView'],
      readouts: ['fireCount', 'medianCompactness', 'selected']
    },
    multipart: {
      headline: 'One fire may contain many polygons',
      textAlternative:
        'A selected multipart wildfire perimeter shows its rings and WGS84 area alongside agency acres, with fill colour encoding area; treating every later ring as a hole is invalid when separate exterior polygon parts are present.',
      camera: {longitude: -121.6, latitude: 38.6, zoom: 6.6, transitionMs: 1800},
      optionsMode: 'fresh',
      options: {metric: 'area', areaSystem: 'wgs84', shapeView: 'map'},
      controls: ['areaSystem', 'metric'],
      readouts: ['medianRatio', 'gpuAcres', 'nifcAcres']
    },
    compactness: {
      headline: 'Rings need an explicit hole rule',
      textAlternative:
        'Wildfire polygons are coloured by fixed Polsby-Popper compactness classes and a selected outline reports the largest area discrepancy; compactness depends on the chosen ring hole rule and mapped perimeter detail.',
      camera: {longitude: -121.7, latitude: 38.4, zoom: 7.2, transitionMs: 1800},
      optionsMode: 'fresh',
      options: {metric: 'polsbyPopper', areaSystem: 'wgs84', shapeView: 'map'},
      controls: ['holeRule'],
      readouts: ['worstError', 'medianRatio', 'selected']
    },
    trend: {
      headline: 'Compactness describes mapped outlines',
      textAlternative:
        'Wildfire centroids and polygons are coloured by fixed Polsby-Popper classes, with compactness plotted against acreage and summarized by rank correlation; the relationship describes this filtered mapped-perimeter sample, not fire behaviour.',
      camera: {longitude: -116.5, latitude: 40, zoom: 4.2, transitionMs: 1800},
      optionsMode: 'fresh',
      options: {metric: 'polsbyPopper', showMarkers: true, shapeView: 'map'},
      controls: ['year'],
      readouts: ['medianCompactness', 'areaCompactness', 'trendChart', 'classChart']
    },
    detail: {
      headline: 'More vertices increase measured perimeter detail',
      textAlternative:
        'One wildfire ring is shown with its original vertices and coloured by vertex count, while vertices per kilometre and the area-vertex rank correlation quantify detail; no topology-safe simplified comparison is available.',
      camera: {longitude: -121.09, latitude: 40.3, zoom: 8.2, transitionMs: 1800},
      optionsMode: 'fresh',
      options: {metric: 'vertices', showMarkers: false, shapeView: 'detail'},
      controls: ['shapeView', 'largeRings'],
      readouts: ['detailOriginalVertices', 'detailDensity', 'areaVertices', 'metricChart']
    },
    'other-shapes': {
      headline: 'One index cannot describe every form',
      textAlternative:
        'Wildfire polygons are coloured by elongation and a selected feature carries its second-moment major axis, with area-elongation rank correlation reported; elongation, compactness and convexity measure different properties and do not explain fire processes.',
      camera: {longitude: -121.6, latitude: 38.8, zoom: 6.4, transitionMs: 1800},
      optionsMode: 'fresh',
      options: {metric: 'elongation', showAxes: true, shapeView: 'map'},
      controls: ['metric', 'showAxes', 'convexityMethod'],
      readouts: ['areaElongation', 'selected']
    }
  }),

  legends: state => {
    if (state.metric === 'year') {
      return [
        {
          kind: 'categories',
          title: 'Perimeter year',
          entries: YEAR_COLORS,
          note: 'Dots mark the centroid of every fire; zoom in for the polygons.'
        }
      ];
    }
    if (state.metric === 'sliver') {
      return [
        {
          kind: 'categories',
          title: 'Sliver flag',
          entries: [
            {color: [150, 160, 175, 200], label: 'compact enough'},
            {
              color: [226, 96, 80, 255],
              label: `sliver (Polsby-Popper < ${state.sliverThreshold.toFixed(2)})`
            }
          ]
        }
      ];
    }
    if (state.metric === 'polsbyPopper') {
      return [
        {
          kind: 'categories',
          title: 'Polsby-Popper compactness',
          unit: 'PP',
          note: 'Fixed reference classes; strongest colour is the stringiest perimeter.',
          entries: [
            {color: [127, 0, 0, 255], label: '< 0.05 · stringiest'},
            {color: [203, 24, 29, 255], label: '0.05–0.15'},
            {color: [252, 146, 114, 255], label: '0.15–0.30'},
            {color: [254, 224, 210, 255], label: '0.30–0.50'},
            {color: [255, 247, 236, 255], label: '≥ 0.50'}
          ]
        }
      ];
    }
    const spec = METRIC_LEGENDS[state.metric];
    return [
      {
        kind: 'ramp',
        id: 'value',
        title: spec.title,
        ramp: 'ylorrd',
        extent: 'gpu',
        sqrtScale: spec.sqrt,
        unit: spec.unit,
        labels: spec.labels
      }
    ];
  },

  readouts: [
    {
      id: 'fireCount',
      label: 'Fires shown',
      format: 'integer',
      help: 'Fires that pass the year and size filters.'
    },
    {
      id: 'nifcAcres',
      label: 'NIFC acres, shown fires',
      format: 'integer',
      help: 'Sum of the acres attribute of the archive.'
    },
    {
      id: 'gpuAcres',
      label: 'GPU acres, WGS84',
      format: 'integer',
      help: 'Sum of GPUGeometryMeasures WGS84 areas of the shown fires, in acres.'
    },
    {
      id: 'medianRatio',
      label: 'Median GPU / NIFC',
      help: 'Median of GPU WGS84 area over the NIFC acres. The NIFC acres come from the same polygons, so 1.00 validates the measure.'
    },
    {
      id: 'worstError',
      label: 'Largest area error',
      help: 'The shown fire whose GPU WGS84 area differs most from the NIFC acres under the current hole rule.'
    },
    {
      id: 'medianCompactness',
      label: 'Median compactness',
      help: 'Median Polsby-Popper of the shown fires.'
    },
    {
      id: 'areaCompactness',
      label: 'Rank correlation, area vs compactness',
      help: 'Spearman rank correlation of fire area and Polsby-Popper across the shown fires. Negative: bigger fires are less compact.'
    },
    {
      id: 'areaVertices',
      label: 'Rank correlation, area vs vertices',
      help: 'Spearman rank correlation of fire area and vertex count: how much more detail big fires carry.'
    },
    {
      id: 'areaElongation',
      label: 'Rank correlation, area vs elongation',
      help: 'Spearman rank correlation of fire area and elongation.'
    },
    {id: 'metricChart', label: 'Distribution of the colored metric', kind: 'chart'},
    {id: 'trendChart', label: 'Compactness by fire size', kind: 'chart'},
    {id: 'classChart', label: 'Compactness by acreage class', kind: 'chart'},
    {id: 'detailOriginalVertices', label: 'Original ring vertices'},
    {id: 'detailDensity', label: 'Vertices per kilometre'},
    {
      id: 'selected',
      label: 'Selected fire',
      layout: 'block',
      help: 'Click a fire to read its descriptors and outline it in yellow. Click it again to clear.'
    }
  ],

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUShapeDescriptorsParameterValues,
  GPUGeometryMeasures,
  GPUShapeDescriptors
} from '@luma.gl/experimental/gpu-spatial-analysis';

// One feature per fire: featureRingOffsets maps fires to rings, ringOffsets to vertices.
const graph = new GPUCommandGraph(device, {id: 'wildfires'});
graph.add(new GPUGeometryMeasures({spatialContext: {coordinateSpace: ('${state.areaSystem}') === 'planar' ? 'planar' : 'longitude-latitude', metric: ('${state.areaSystem}') === 'planar' ? 'native' : ('${state.areaSystem}') === 'spherical' ? 'great-circle' : 'ellipsoidal', units: ('${state.areaSystem}') === 'planar' ? 'native' : 'meters'},ellipsoidalEdgeModel: ('${state.areaSystem}') === 'wgs84' ? 'coordinate-linear' : undefined,
  positions: lngLat,                    // float32x2 lon/lat (planar: Web Mercator meters)
  ringOffsets, featureRingOffsets,
  geometryType: 'polygons',

  holeRule: '${state.holeRule}',${state.largeRings === 'serial' ? '\n  cooperativeRingRows: 0,' : ''}
  output: {areas, lengths, centroids, vertexCounts}
}));
graph.add(new GPUShapeDescriptors({
  positions: webMercatorMeters,         // planar coordinates
  ringOffsets, featureRingOffsets,
  holeRule: '${state.holeRule}',
  convexityMethod: '${state.convexityMethod}',
  parameters,                           // [sliverThreshold, 0, 0, 0], a per-frame write
  output: {polsbyPopper, elongation, orientation, convexity, sliver}
}));
const compiled = graph.compile();
shapeParameters.write(getGPUShapeDescriptorsParameterValues({sliverThreshold: ${state.sliverThreshold}}));
compiled.encode(commandEncoder, {parameters: undefined});`,

  about: {
    what: '`GPUGeometryMeasures` returns area, perimeter, centroid, bounds and vertex count of every polygon feature in one pass, in planar, spherical, WGS84 or geodesic coordinates. `GPUShapeDescriptors` builds on it: Polsby-Popper and Schwartzberg compactness, convexity, elongation, orientation and a sliver flag.',
    why: 'Shape statistics are how you ask whether a district gerrymanders, a parcel is a sliver or a burn scar is stringy. Doing it on the GPU means the numbers follow a filter or a different hole rule immediately, for every feature at once.',
    howToRead:
      'The fixed compactness classes put the strongest colour on the lowest Polsby-Popper values. Dots mark each fire so small ones stay visible when zoomed out. Perimeter dates are record dates, not ignition. Compactness depends on mapped boundary detail, so its relation to acreage is descriptive rather than causal.'
  },

  create: async ctx => (await import('./wildfire-shapes.compute')).createWildfireShapes(ctx)
});
