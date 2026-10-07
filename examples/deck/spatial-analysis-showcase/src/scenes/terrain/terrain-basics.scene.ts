// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {defineScene} from '../scene';
import type {BasicsOptions} from './terrain-basics.style';
import {getNominalRange} from './terrain-basics.style';

const SEQUENTIAL_RAMPS = [
  {value: 'viridis', label: 'Viridis'},
  {value: 'magma', label: 'Magma'},
  {value: 'inferno', label: 'Inferno'},
  {value: 'cividis', label: 'Cividis (color-blind optimised)'}
] as const;

/** Hue wheel of the aspect product: twin of `getAspectColor` in `b14a-colorize.ts` (kept light here). */
function aspectColor(degrees: number): [number, number, number] {
  const hue = (((degrees % 360) + 360) % 360) / 360;
  const channel = (n: number) => {
    const k = (n + hue * 6) % 6;
    return Math.round((0.95 - 0.95 * 0.62 * Math.max(Math.min(k, 4 - k, 1), 0)) * 255);
  };
  return [channel(5), channel(3), channel(1)];
}

const COMPASS = [
  {degrees: 0, label: 'North'},
  {degrees: 90, label: 'East'},
  {degrees: 180, label: 'South'},
  {degrees: 270, label: 'West'}
] as const;

/**
 * Terrain basics on the Matterhorn tile: decode a Terrarium PNG on the GPU, then slope, aspect,
 * hillshade and three ruggedness measures, with a noisy-tile repair demo.
 */
export default defineScene<BasicsOptions>({
  id: 'terrain-basics',
  title: 'How steep is the Matterhorn?',
  chapter: 'terrain',
  order: 1,
  summary:
    'Decode a Terrarium elevation PNG on the GPU, then compute slope, aspect, hillshade, TPI, TRI, roughness and vector ruggedness for the Matterhorn and Zermatt. Inject noisy spikes and repair them.',
  contributors: [
    'GPUTerrainRGBDecode',
    'GPUTerrainSpikeRepair',
    'GPUTerrainDerivatives',
    'GPUTerrainRuggedness',
    'GPUTerrainVectorRuggedness',
    'GPUReliefShading'
  ],
  datasets: [{id: 'alps-dem', role: 'elevation (Terrarium PNG, Web Mercator)'}],
  initialView: {longitude: 7.742, latitude: 45.985, zoom: 11.9},

  options: [
    {
      kind: 'select',
      id: 'encoding',
      label: 'Decoder',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      default: 'terrarium',
      help: 'How the PNG colors become meters. The file is Terrarium; picking Mapbox shows what a wrong decoder does: every pixel falls outside the valid range and becomes nodata.',
      options: [
        {value: 'terrarium', label: 'Terrarium: R*256 + G + B/256 - 32768'},
        {value: 'mapbox', label: 'Mapbox: -10000 + 0.1*(R*65536 + G*256 + B)'}
      ]
    },
    {
      kind: 'select',
      id: 'validRange',
      label: 'Valid height range',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      default: 'default',
      help: 'Decoded heights outside this range become nodata (NaN plus validity 0). The default removes blank-canvas pixels; the tight range deliberately drops valley floors and the summit.',
      options: [
        {value: 'default', label: 'Default: -11,000 to 9,000 m'},
        {value: 'tight', label: 'Tight: 1,800 to 4,200 m'},
        {value: 'off', label: 'Off: accept everything'}
      ]
    },
    {
      kind: 'toggle',
      id: 'alphaNoData',
      label: 'Alpha 0 means nodata',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      default: true,
      help: 'PNG tiles mark missing data with transparent pixels. Turn this off to ignore alpha and decode the color bytes anyway.'
    },
    {
      kind: 'toggle',
      id: 'clampBathymetry',
      label: 'Clamp below sea level to 0',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      default: false,
      help: 'The mt-image sea clamp flattens every height between -12,000 and 0 m to 0. It changes nothing over the Alps, which is the point: it is an application choice that wrecks the Dead Sea and the Dutch polders.'
    },
    {
      kind: 'toggle',
      id: 'missingTile',
      label: 'Blank a 2.5 km patch (alpha 0)',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'param',
      default: false,
      help: 'Rewrites the input so a square patch east of Riffelalp is transparent, as in a PNG tile with missing data. Nodata flows through every later product as a hole.'
    },
    {
      kind: 'slider',
      id: 'spikeDensity',
      label: 'Inject spikes',
      group: 'Noisy tiles (GPUTerrainSpikeRepair)',
      apply: 'param',
      min: 0,
      max: 0.5,
      step: 0.01,
      default: 0,
      unit: '% of pixels',
      format: value => (value === 0 ? 'none' : `${value.toFixed(2)} %`),
      help: 'Adds +/-256 m errors to the red byte of random pixels, the damage canvas anti-fingerprinting noise does to Terrarium tiles in some browsers.'
    },
    {
      kind: 'toggle',
      id: 'repairSpikes',
      label: 'Repair spikes',
      group: 'Noisy tiles (GPUTerrainSpikeRepair)',
      apply: 'param',
      default: false,
      help: 'Chooses the repaired heights (labelled components shifted by whole 256 m steps) over the raw decode. The repair always runs; this only selects its output.'
    },
    {
      kind: 'select',
      id: 'slopeUnits',
      label: 'Slope units',
      group: 'Slope, aspect, hillshade (GPUTerrainDerivatives)',
      apply: 'compile',
      default: 'degrees',
      disabledWhen: state => !['slope', 'aspect', 'hillshade'].includes(state.product),
      help: 'Degrees from horizontal (0 to 90) or percent grade (rise over run times 100; 100 % is 45 degrees).',
      options: [
        {value: 'degrees', label: 'Degrees'},
        {value: 'percent', label: 'Percent'}
      ]
    },
    {
      kind: 'slider',
      id: 'zFactor',
      label: 'Vertical exaggeration (z factor)',
      group: 'Slope, aspect, hillshade (GPUTerrainDerivatives)',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      unit: 'x',
      help: 'Multiplies heights before slope is taken. 1 is true slope; 2 doubles the rise and makes every face steeper. It also feeds the vector ruggedness.'
    },
    {
      kind: 'slider',
      id: 'sunAzimuth',
      label: 'Sun azimuth',
      group: 'Slope, aspect, hillshade (GPUTerrainDerivatives)',
      apply: 'param',
      min: 0,
      max: 360,
      step: 5,
      default: 315,
      unit: '°',
      help: 'Direction the light comes from, clockwise from north. 315 (north-west) is the cartographic convention because it keeps relief from inverting.'
    },
    {
      kind: 'slider',
      id: 'sunAltitude',
      label: 'Sun altitude',
      group: 'Slope, aspect, hillshade (GPUTerrainDerivatives)',
      apply: 'param',
      min: 5,
      max: 85,
      step: 1,
      default: 45,
      unit: '°',
      help: 'Height of the light above the horizon. Low sun exaggerates texture; high sun flattens it.'
    },
    {
      kind: 'select',
      id: 'borderMode',
      label: 'Border handling',
      group: 'Slope, aspect, hillshade (GPUTerrainDerivatives)',
      apply: 'compile',
      default: 'clamp',
      help: 'How the 3 x 3 Sobel window reads outside the raster or past nodata: repeat the edge value (clamp) or leave the cell nodata.',
      options: [
        {value: 'clamp', label: 'Clamp to the edge'},
        {value: 'nodata', label: 'Edge cells become nodata'}
      ]
    },
    {
      kind: 'select',
      id: 'triAlgorithm',
      label: 'TRI algorithm',
      group: 'Ruggedness (GPUTerrainRuggedness)',
      apply: 'compile',
      default: 'riley',
      disabledWhen: state => state.product !== 'tri',
      help: 'Riley (1999) sums squared differences to the 8 neighbours and takes the square root; Wilson (2007) averages absolute differences. Riley is the GDAL default since 3.3.',
      options: [
        {value: 'riley', label: 'Riley 1999 (GDAL default)'},
        {value: 'wilson', label: 'Wilson 2007'}
      ]
    },
    {
      kind: 'select',
      id: 'edgeMode',
      label: 'Edge mode',
      group: 'Ruggedness (GPUTerrainRuggedness)',
      apply: 'compile',
      default: 'nodata',
      disabledWhen: state => !['tpi', 'tri', 'roughness'].includes(state.product),
      help: 'The 3 x 3 window does not fit on the outermost ring. Nodata leaves it empty (gdaldem default); extrapolate fills it like gdaldem -compute_edges.',
      options: [
        {value: 'nodata', label: 'Edge ring is nodata'},
        {value: 'extrapolate', label: 'Extrapolate edges (-compute_edges)'}
      ]
    },
    {
      kind: 'slider',
      id: 'vrmRadius',
      label: 'VRM window radius',
      group: 'Ruggedness (GPUTerrainVectorRuggedness)',
      apply: 'compile',
      min: 1,
      max: 8,
      step: 1,
      default: 2,
      unit: 'cells',
      disabledWhen: state => state.product !== 'vrm',
      help: 'The window is (2r + 1) cells wide: 5 x 5 at r = 2 is about 33 m on the ground. A bigger window sums more surface normals, so it costs more and reads coarser texture. Changing it rebuilds this one graph.'
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Show',
      group: 'Display',
      apply: 'param',
      default: 'elevation',
      help: 'Which raster to draw. Each product is a small compiled graph built the first time you pick it; switching back and forth afterwards costs nothing.',
      options: [
        {value: 'elevation', label: 'Elevation (decoded heights)'},
        {value: 'slope', label: 'Slope'},
        {value: 'aspect', label: 'Aspect (which way a slope faces)'},
        {value: 'hillshade', label: 'Hillshade (lit from the sun settings)'},
        {value: 'tpi', label: 'TPI: above or below the neighbours'},
        {value: 'tri', label: 'TRI: ruggedness index'},
        {value: 'roughness', label: 'Roughness: largest local height range'},
        {value: 'vrm', label: 'VRM: vector ruggedness'}
      ]
    },
    {
      kind: 'select',
      id: 'ramp',
      label: 'Color ramp',
      group: 'Display',
      apply: 'param',
      default: 'viridis',
      help: 'Ramp for the sequential products (not aspect, hillshade or TPI). All four are perceptually uniform.',
      options: SEQUENTIAL_RAMPS
    },
    {
      kind: 'slider',
      id: 'rangeScale',
      label: 'Color range scale',
      group: 'Display',
      apply: 'param',
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      unit: 'x',
      help: 'Multiplies the value range the ramp spans. Below 1 stretches the low end (subtle features); above 1 keeps high values from saturating.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Layer opacity',
      group: 'Display',
      apply: 'param',
      min: 0.2,
      max: 1,
      step: 0.05,
      default: 0.78,
      help: 'Lower it to let the relief underlay (or the basemap) show through.'
    },
    {
      kind: 'toggle',
      id: 'underlay',
      label: 'Hillshade underlay',
      group: 'Display',
      apply: 'param',
      default: true,
      help: 'Draws a gray multidirectional hillshade (GPUReliefShading) beneath the product so you can read the terrain.'
    },
    {
      kind: 'button',
      id: 'measure',
      label: 'Time this product',
      group: 'Under the hood',
      help: 'Runs the displayed product graph outside the frame and reports GPU time.'
    }
  ],

  readouts: [
    {id: 'grid', label: 'Raster'},
    {
      id: 'validPixels',
      label: 'Valid pixels',
      help: 'Share of pixels the decode accepted. Nodata pixels are NaN with validity 0.'
    },
    {id: 'elevationRange', label: 'Elevation range (GPU decode)'},
    {
      id: 'decodeDifference',
      label: 'GPU vs CPU decode, max error',
      help: "Largest absolute difference between GPUTerrainRGBDecode and the loader's float64 Terrarium decode. Terrarium decoding is exact in float32, so this is 0."
    },
    {id: 'repairJumps', label: 'Spike repair: jumps found'},
    {id: 'repairResult', label: 'Spike repair: result'},
    {id: 'median', label: 'Displayed raster: median'},
    {id: 'p98', label: 'Displayed raster: 98th percentile'},
    {id: 'maximum', label: 'Displayed raster: maximum'},
    {id: 'timing', label: 'Product GPU time'}
  ],

  legends: state => {
    const {low, high} = getNominalRange(state.product, state.slopeUnits);
    const scale = state.rangeScale;
    const titles: Record<BasicsOptions['product'], string> = {
      elevation: 'Elevation',
      slope: `Slope (${state.slopeUnits})`,
      aspect: 'Aspect',
      hillshade: 'Hillshade',
      tpi: 'TPI: height above (+) or below (-) the 8 neighbours',
      tri: 'TRI',
      roughness: 'Roughness',
      vrm: 'Vector ruggedness (0 smooth, 1 chaotic)'
    };
    if (state.product === 'aspect') {
      return [
        {
          kind: 'categories',
          title: 'Aspect: the direction the slope faces',
          entries: [
            ...COMPASS.map(entry => ({color: aspectColor(entry.degrees), label: entry.label})),
            {color: [200, 200, 200], label: 'Flat (no aspect)'}
          ],
          note: 'Hue is the compass bearing of the downhill direction, in degrees clockwise from north.'
        }
      ];
    }
    if (state.product === 'hillshade') {
      return [
        {
          kind: 'ramp',
          title: titles.hillshade,
          ramp: 'grayscale',
          extent: [0, 1],
          labels: ['shaded', 'lit']
        }
      ];
    }
    const unit =
      state.product === 'slope'
        ? state.slopeUnits === 'degrees'
          ? '°'
          : '%'
        : state.product === 'vrm'
          ? ''
          : 'm';
    return [
      {
        kind: 'ramp',
        title: titles[state.product],
        ramp: state.product === 'tpi' ? 'diverging' : state.ramp,
        extent: [low * scale, high * scale],
        unit,
        format: value =>
          state.product === 'vrm' ? value.toFixed(2) : value.toFixed(value % 1 === 0 ? 0 : 1)
      }
    ];
  },

  story: [
    {
      id: 'question',
      controls: ['product'],
      readouts: ['decodeDifference', 'elevationRange'],
      title: 'How steep is the Matterhorn, and where can you walk?',
      body: 'The Matterhorn rises about 2,900 m above the village of Zermatt, and the data for this whole 13.6 km window is one small PNG: a **Terrarium** tile, where every pixel stores a height in its red, green and blue bytes. A map you can query needs numbers, so the first job is turning colors back into meters.\n\n**`GPUTerrainRGBDecode`** does that on the GPU for all 4.2 million pixels. It is exact: the readout *GPU vs CPU decode* compares it with a float64 decode and reports a maximum error of 0 m. The picture is plain elevation, 1,503 m in the Mattertal to about 4,476 m at the summit.',
      options: {product: 'elevation', underlay: true, opacity: 0.8},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1400},
      callout: {coordinate: [7.6586, 45.9766], text: 'Matterhorn 4,478 m'},
      highlight: {readout: 'decodeDifference'}
    },
    {
      id: 'decode',
      controls: ['encoding', 'validRange', 'missingTile'],
      readouts: ['validPixels'],
      title: 'Decode before you filter: pick the wrong decoder',
      body: 'Terrarium and Mapbox "terrain-RGB" are both PNG height encodings, and nothing in the file says which one you have. Set **Decoder** below to Mapbox and the map goes blank: the same bytes now decode to heights near 870 km, which fall outside the **valid range**, so the GPU marks them *nodata* instead of drawing nonsense.\n\nBelow, try **Valid height range** set to *Off* to see why that guard exists, and turn on **Blank a 2.5 km patch (alpha 0)** to see PNG alpha 0 become a hole that every later product inherits. Switch the decoder back to Terrarium to continue.',
      options: {product: 'elevation', encoding: 'mapbox'},
      highlight: {readout: 'validPixels'}
    },
    {
      id: 'slope-aspect',
      controls: ['product', 'slopeUnits'],
      readouts: ['p98', 'maximum'],
      title: 'Slope and aspect: how steep, and which way it faces',
      body: "**`GPUTerrainDerivatives`** applies Horn's method: two 3 x 3 Sobel filters give the rise east-west and north-south, and slope is the angle of that vector, `atan(sqrt(dz/dx^2 + dz/dy^2))`. Because the tile is in Web Mercator, the cell size is set per row (about 6.6 m of ground at 46 degrees north, not the 9.55 m a pixel spans on the projected grid).\n\nRead the legend: dark is flat, bright is steep. The glaciers and the Zermatt valley floor are comparatively gentle; the Matterhorn faces and ridge flanks are the brightest, with the steepest cells above 50 degrees (see the 98th percentile and maximum readouts). Switch **Show** below to *Aspect* to see which way each slope faces, and to *Slope* with **Slope units** set to *Percent* to see the same data as grade.",
      options: {product: 'slope', encoding: 'terrarium', opacity: 0.78},
      camera: {longitude: 7.685, latitude: 45.978, zoom: 13.1, transitionMs: 1600},
      callout: {coordinate: [7.6586, 45.9766], text: 'Matterhorn'},
      highlight: {readout: 'p98'}
    },
    {
      id: 'hillshade',
      controls: ['sunAzimuth', 'sunAltitude'],
      readouts: [],
      title: 'Hillshade: light the terrain from any direction',
      body: 'The same Horn gradients also give a **hillshade**: brightness is the cosine of the angle between the surface normal and the sun, `cos(zenith)*cos(slope) + sin(zenith)*sin(slope)*cos(azimuth - aspect)`. It is the standard way to make elevation readable.\n\nDrag **Sun azimuth** and **Sun altitude** below: they are parameter-buffer writes, so nothing recompiles (the *Rebuilds* counter in *Under the hood* stays put). From the south-east the Matterhorn looks like a pit, the classic relief inversion, which is why maps light from the north-west.',
      options: {product: 'hillshade', sunAzimuth: 315, sunAltitude: 40},
      camera: {longitude: 7.742, latitude: 45.985, zoom: 11.9, transitionMs: 1400}
    },
    {
      id: 'ruggedness',
      controls: ['product', 'vrmRadius'],
      readouts: ['p98'],
      title: 'Rugged or merely steep? TPI, TRI, roughness and VRM',
      body: "Steep is not the same as rough: a smooth steep face is easy to climb, a bouldery field is not. **`GPUTerrainRuggedness`** reproduces the three `gdaldem` indices on 3 x 3 windows: **TPI** (a cell minus the mean of its neighbours, positive on ridges and knolls), **TRI** (Riley's root-sum-squared height differences) and **roughness** (largest minus smallest height). **`GPUTerrainVectorRuggedness`** is different: it measures how much the *directions* of the surface normals scatter, `1 - |sum n| / N`, so a uniform steep slope scores near 0 while crevassed or blocky terrain scores high.\n\nThis step shows TRI. Switch **Show** below to *VRM* (then try **VRM window radius**) and to *Slope* to compare, and see where steepness and ruggedness disagree, for example on the Matterhorn faces versus the broken ice of the glaciers.",
      options: {product: 'tri', opacity: 0.8},
      camera: {longitude: 7.73, latitude: 45.975, zoom: 12.4, transitionMs: 1400},
      highlight: {readout: 'p98'}
    },
    {
      id: 'spikes',
      controls: ['spikeDensity', 'product'],
      readouts: ['repairJumps'],
      title: 'Noisy tiles: what a few bad pixels do to slope',
      body: 'Some browsers add tiny noise to canvas pixels as an anti-fingerprinting measure. In a Terrarium tile that noise lands in the red byte, and one count in red is exactly **256 m**. This step injects +/-256 m errors into 0.12 % of the pixels and shows the slope: every spike becomes a needle of near-vertical slope.\n\nThe damage is hardest to see in elevation (a few bright dots) and obvious in derivatives. Raise **Inject spikes** below, and switch **Show** between *Elevation* and *Slope* to see it.',
      options: {product: 'slope', spikeDensity: 0.12, repairSpikes: false},
      camera: {longitude: 7.76, latitude: 45.99, zoom: 13.0, transitionMs: 1400},
      highlight: {readout: 'repairJumps'}
    },
    {
      id: 'repair',
      controls: ['repairSpikes', 'spikeDensity'],
      readouts: ['repairResult', 'repairJumps'],
      title: 'Repair: shift components by whole 256 m steps',
      body: '**`GPUTerrainSpikeRepair`** labels connected components of the decoded heights across jumps larger than 200 m, then shifts every small component whose borders all agree on a jump of `k * 256 m` back by that amount. The readouts report what it found, what it moved and whether the labelling converged; if it does not converge it fails closed and returns the input.\n\nTurn **Repair spikes** on below and the needles vanish, in a few milliseconds. **Caveat:** the repair cannot tell a bad pixel from a real enclosed butte with 216 to 296 m walls, so use it only on sources you know are noisy, and always before slope, curvature or any filtering. **Try:** raise **Inject spikes** to 0.5 %, then switch the decoder, valid range and edge mode in the *All controls* tab to see each contributor option do its job.',
      options: {product: 'slope', spikeDensity: 0.12, repairSpikes: true},
      highlight: {readout: 'repairResult'}
    }
  ],

  about: {
    what: '`GPUTerrainRGBDecode` converts Terrarium or Mapbox terrain-RGB PNG words to float32 heights plus a validity mask (bit-exact for Terrarium). `GPUTerrainSpikeRepair` fixes +/-256 m red-byte errors. `GPUTerrainDerivatives` gives Horn slope, aspect and hillshade; `GPUTerrainRuggedness` gives gdaldem TPI, TRI and roughness; `GPUTerrainVectorRuggedness` gives the Sappington et al. (2007) VRM.',
    why: 'Slope and ruggedness feed route planning, avalanche and rockfall hazard, habitat models and viewshed or hydrology preparation. Decoding and repairing on the GPU keeps a tile in GPU memory from PNG to product.',
    howToRead:
      'Slope: bright is steep. Aspect: hue is the compass direction a slope faces. TPI: warm above its neighbours (ridge), cool below (hollow). TRI, roughness and VRM: brighter is rougher. Hover any cell for its exact value. Cell sizes are per-row ground meters from the Web Mercator tile geometry; the whole tile is one 2048 x 2048 raster at 6.6 m ground resolution. Reference implementations: gdaldem (slope, aspect, hillshade, TPI, TRI, roughness) and mt-image (Terrarium decode, spike repair). The terrain dataset is bundled, so `?data=synthetic` makes no difference here.'
  },

  create: async ctx => (await import('./terrain-basics.compute')).createTerrainBasics(ctx),

  snippet: state => `import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTerrainRGBDecode, GPUTerrainDerivatives, GPUTerrainRuggedness,
  getGPUTerrainDerivativesParameterValues, GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-terrain';

// \`encoded\` holds the PNG pixels as packed RGBA8 words (getImageData().data viewed as Uint32Array)
const graph = new GPUCommandGraph(device, {id: 'terrain'});
graph.add(new GPUTerrainRGBDecode({
  width: 2048, height: 2048, encoding: '${state.encoding}',
  validRange: [${state.validRange === 'tight' ? '1800, 4200' : '-11000, 9000'}],${state.clampBathymetry ? '\n  clampBathymetry: true,' : ''}
  input: {buffer: encodedView}, values: heightsView, validity: validityView
}));
const elevation = {id: 'dem', format: 'float32', storage: {kind: 'buffer', values: heightsView}, validity: validityView};

graph.add(new GPUTerrainDerivatives({
  width: 2048, height: 2048, elevation, settings: derivativeSettings.importToGraph(graph),
  slope, aspect, hillshade,
  cellSizeMode: 'web-mercator', slopeUnits: '${state.slopeUnits}', borderMode: '${state.borderMode}'
}));
graph.add(new GPUTerrainRuggedness({
  width: 2048, height: 2048, elevation, terrainRuggednessIndex: tri,
  terrainRuggednessAlgorithm: '${state.triAlgorithm}', edgeMode: '${state.edgeMode}'
}));
const compiled = graph.compile(); // once

// per frame: parameter writes only (web-mercator: edges are normalized Mercator y of the tile)
derivativeSettings.write(getGPUTerrainDerivativesParameterValues({
  cellSize: [9.5546, 9.5546], northEdge: 0.3556, southEdge: 0.3112,
  zFactor: ${state.zFactor}, azimuthDegrees: ${state.sunAzimuth}, altitudeDegrees: ${state.sunAltitude}
}));
compiled.encode(commandEncoder, {parameters: undefined});`
});
