// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene, type StoryStep} from '../scene';
import {getTerrainFurniture, terrainCartouche, TERRAIN_FRAMES} from './terrain-furniture';
import {getStepPlaces, STEP_PLACES} from './terrain-basics-labels';
import {
  type BasicsOptions,
  DECISION_SLOPE_DEGREES,
  getBasicsLegends,
  GROUND_CELL_METERS,
  STEEP_FACE_DEGREES
} from './terrain-basics.style';

/** Camera of the establishing shot and of the slope classes: the Matterhorn at the left, the Gornergrat right. */
const FORM_FRAME = {longitude: 7.7, latitude: 45.985, zoom: 12.3, pitch: 0, bearing: 0} as const;
/** The Gorner glacier crevasse fields beside the steep faces of the Breithorn and Gornergrat. */
const RUGGED_FRAME = {
  longitude: 7.765,
  latitude: 45.968,
  zoom: 12.8,
  pitch: 0,
  bearing: 0
} as const;

/** The title cartouche of a step: its question, with a fallback subtitle the scene replaces live. */
const furnitureFor = (question: string, subtitle: string) =>
  getTerrainFurniture({cartouche: terrainCartouche(question, subtitle)});

/** The 3 x 3 Horn window as a diagram: the nine heights, the weights and the two formulas. */
const HORN_DIAGRAM: NonNullable<StoryStep<BasicsOptions>['diagram']> = {
  kind: 'diagram',
  width: 320,
  height: 150,
  description:
    'A three by three window of heights labelled a to i with the centre cell highlighted, and the formulas for the east-west and north-south gradients with weights one, two, one, and for slope as the arctangent of the gradient length.',
  svg: `
<text x="59" y="11" text-anchor="middle" fill="currentColor" font-size="10" class="diagram-muted">north</text>
<rect x="14" y="16" width="90" height="90" fill="none" class="diagram-muted" stroke="currentColor" stroke-opacity="0.45"/>
<rect x="44" y="46" width="30" height="30" class="diagram-signal" fill="currentColor" fill-opacity="0.18" stroke="currentColor" stroke-width="1.6"/>
<path d="M44 16V106M74 16V106M14 46H104M14 76H104" stroke="currentColor" stroke-opacity="0.35" fill="none"/>
<g fill="currentColor" font-size="13" text-anchor="middle" font-family="var(--font-mono, monospace)">
<text x="29" y="36">a</text><text x="59" y="36">b</text><text x="89" y="36">c</text>
<text x="29" y="66">d</text><text x="59" y="66" font-weight="700">e</text><text x="89" y="66">f</text>
<text x="29" y="96">g</text><text x="59" y="96">h</text><text x="89" y="96">i</text>
</g>
<text x="59" y="124" text-anchor="middle" fill="currentColor" font-size="10" class="diagram-muted">weights 1 2 1 across, 1 2 1 down</text>
<g fill="currentColor" font-size="11" font-family="var(--font-mono, monospace)" style="white-space: pre">
<text x="124" y="24">dz/dx = (c + 2f + i)</text>
<text x="124" y="39">      - (a + 2d + g)</text>
<text x="124" y="54">        / (8 s)</text>
<text x="124" y="76">dz/dy = (a + 2b + c)</text>
<text x="124" y="91">      - (g + 2h + i)</text>
<text x="124" y="106">        / (8 s)</text>
<text x="124" y="130" font-weight="700">slope = atan(hypot(dz/dx, dz/dy))</text>
<text x="124" y="144" class="diagram-muted" font-size="10">s = ground metres per cell</text>
</g>`
};

/**
 * Terrain basics on the Matterhorn tile: decode a Terrarium elevation PNG on the GPU, class slope
 * at the thresholds people decide on, measure it in ground metres, and tell steep from rugged.
 */
export default defineScene<BasicsOptions>({
  id: 'terrain-basics',
  title: 'How steep is the Matterhorn?',
  chapter: 'terrain',
  order: 1,
  summary:
    'Decode a Terrarium elevation PNG on the GPU, then class slope at the avalanche thresholds, measure it in ground metres rather than Mercator pixels, colour aspect and tell steep from rugged for the Matterhorn and Zermatt.',
  contributors: [
    'GPUTerrainRGBDecode',
    'GPUTerrainSpikeRepair',
    'GPUTerrainDerivatives',
    'GPUTerrainRuggedness',
    'GPUTerrainVectorRuggedness'
  ],
  datasets: [
    {id: 'alps-dem', role: 'elevation (Terrarium PNG, Web Mercator)'},
    {id: 'alps-context', role: 'glaciers and place names (OpenStreetMap)'}
  ],
  initialView: {...FORM_FRAME},
  basemap: ground('relief'),
  furniture: furnitureFor(
    'How steep is the Matterhorn?',
    'Elevation, metres · pale tint under shaded relief'
  ),

  options: [
    {
      kind: 'select',
      id: 'view',
      label: 'Map shows',
      group: 'Display',
      apply: 'param',
      default: 'analysis',
      display: 'segmented',
      help: 'The shaded relief ground on its own, the heights exactly as the GPU decoded them, or an analysis product over the relief.',
      options: [
        {value: 'relief', label: 'Relief'},
        {value: 'decoded', label: 'Decoded'},
        {value: 'analysis', label: 'Analysis'}
      ]
    },
    {
      kind: 'select',
      id: 'product',
      label: 'Product',
      group: 'Display',
      apply: 'param',
      default: 'slope',
      display: 'chips',
      disabledWhen: state => state.view !== 'analysis',
      help: 'Which analysis to draw. Each product is a small compiled graph built the first time you pick it; going back to one costs nothing.',
      options: [
        {value: 'slope', label: 'Slope'},
        {value: 'aspect', label: 'Aspect'},
        {value: 'tpi', label: 'TPI'},
        {value: 'tri', label: 'TRI'},
        {value: 'vrm', label: 'VRM'}
      ]
    },
    {
      kind: 'select',
      id: 'slopeDisplay',
      label: 'Slope colour',
      group: 'Slope (GPUTerrainDerivatives)',
      apply: 'param',
      default: 'classes',
      display: 'segmented',
      disabledWhen: state => state.view !== 'analysis' || state.product !== 'slope',
      help: 'Classes break slope at the thresholds people act on. Continuous colour paints the same numbers on a ramp and hides where the thresholds are.',
      options: [
        {value: 'classes', label: 'Classes'},
        {value: 'continuous', label: 'Continuous'}
      ]
    },
    {
      kind: 'select',
      id: 'cellModel',
      label: 'Cell size',
      group: 'Slope (GPUTerrainDerivatives)',
      apply: 'param',
      default: 'ground',
      display: 'segmented',
      disabledWhen: state => state.view !== 'analysis' || state.product !== 'slope',
      help: 'Ground metres: the real size of each row, from the Web Mercator edges. Mercator pixels: the projected pixel size taken as metres, which flattens every slope. Both variants are computed together so both readouts stay live; this only chooses which one is drawn.',
      options: [
        {value: 'ground', label: 'Ground metres'},
        {value: 'mercator', label: 'Mercator pixels'}
      ]
    },
    {
      kind: 'select',
      id: 'encoding',
      label: 'Decoder',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      default: 'terrarium',
      display: 'segmented',
      help: 'How the PNG colours become metres. The file is Terrarium; Mapbox is the other common height encoding, and the wrong one puts every pixel out of range.',
      options: [
        {value: 'terrarium', label: 'Terrarium'},
        {value: 'mapbox', label: 'Mapbox'}
      ]
    },
    {
      kind: 'toggle',
      id: 'missingTile',
      label: 'Blank a patch (alpha 0)',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'param',
      default: false,
      help: 'Rewrites the input so a square patch over Riffelberg and Rotenboden is transparent, as in a PNG tile with missing data. No data flows through every later product as a hole.'
    },
    {
      kind: 'select',
      id: 'validRange',
      label: 'Valid height range',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      expert: true,
      default: 'default',
      help: 'Decoded heights outside this range become no data (NaN and validity 0). The default removes blank-canvas pixels; the tight range deliberately drops valley floors and the summit.',
      options: [
        {value: 'default', label: 'Default: -11,000 to 9,000 m'},
        {value: 'tight', label: 'Tight: 1,800 to 4,200 m'},
        {value: 'off', label: 'Off: accept everything'}
      ]
    },
    {
      kind: 'toggle',
      id: 'alphaNoData',
      label: 'Alpha 0 means no data',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      expert: true,
      default: true,
      help: 'PNG tiles mark missing data with transparent pixels. Turn this off to ignore alpha and decode the colour bytes anyway.'
    },
    {
      kind: 'toggle',
      id: 'clampBathymetry',
      label: 'Clamp below sea level to 0',
      group: 'Decode (GPUTerrainRGBDecode)',
      apply: 'compile',
      expert: true,
      default: false,
      help: 'The mt-image sea clamp flattens every height between -12,000 and 0 m to 0. It changes nothing over the Alps, which is the point: it is an application choice that wrecks the Dead Sea and the Dutch polders.'
    },
    {
      kind: 'slider',
      id: 'spikeDensity',
      label: 'Inject spikes',
      group: 'Noisy tiles (GPUTerrainSpikeRepair)',
      apply: 'param',
      expert: true,
      min: 0,
      max: 0.5,
      step: 0.01,
      default: 0,
      unit: '% of pixels',
      format: value => (value === 0 ? 'none' : `${value.toFixed(2)} %`),
      help: 'Adds +/-256 m errors to the red byte of random pixels, the damage canvas anti-fingerprinting noise does to Terrarium tiles in some browsers. Look at Slope: every spike is a needle.'
    },
    {
      kind: 'toggle',
      id: 'repairSpikes',
      label: 'Repair spikes',
      group: 'Noisy tiles (GPUTerrainSpikeRepair)',
      apply: 'param',
      expert: true,
      default: false,
      help: 'Chooses the repaired heights (labelled components shifted by whole 256 m steps) over the raw decode. The repair always runs; this only selects its output. It cannot tell a bad pixel from an enclosed butte with 216 to 296 m walls, so use it only on sources you know are noisy.'
    },
    {
      kind: 'slider',
      id: 'zFactor',
      label: 'Vertical exaggeration (z factor)',
      group: 'Slope (GPUTerrainDerivatives)',
      apply: 'param',
      expert: true,
      min: 0.25,
      max: 3,
      step: 0.05,
      default: 1,
      unit: 'x',
      help: 'Multiplies heights before slope is taken. 1 is true slope; 2 doubles the rise and makes every face steeper. It also feeds the vector ruggedness.'
    },
    {
      kind: 'select',
      id: 'borderMode',
      label: 'Border handling',
      group: 'Slope (GPUTerrainDerivatives)',
      apply: 'compile',
      expert: true,
      default: 'clamp',
      help: 'How the 3 x 3 window reads outside the raster or past no data: repeat the edge value (clamp) or leave the cell no data.',
      options: [
        {value: 'clamp', label: 'Clamp to the edge'},
        {value: 'nodata', label: 'Edge cells become no data'}
      ]
    },
    {
      kind: 'select',
      id: 'triAlgorithm',
      label: 'TRI algorithm',
      group: 'Ruggedness (GPUTerrainRuggedness)',
      apply: 'compile',
      expert: true,
      default: 'riley',
      disabledWhen: state => state.view !== 'analysis' || state.product !== 'tri',
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
      expert: true,
      default: 'nodata',
      disabledWhen: state =>
        state.view !== 'analysis' || (state.product !== 'tpi' && state.product !== 'tri'),
      help: 'The 3 x 3 window does not fit on the outermost ring. No data leaves it empty (gdaldem default); extrapolate fills it like gdaldem -compute_edges.',
      options: [
        {value: 'nodata', label: 'Edge ring is no data'},
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
      disabledWhen: state => state.view !== 'analysis' || state.product !== 'vrm',
      describe: value => {
        const size = 2 * value + 1;
        return `${size} x ${size} cells = ${Math.round(size * GROUND_CELL_METERS)} m, ${size * size} taps per cell`;
      },
      help: 'The window is (2r + 1) cells wide. A bigger window sums more surface normals, so it costs more and reads coarser texture. The radius is a compile option: changing it rebuilds this one graph.'
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
    {
      id: 'relief',
      label: 'Relief, highest minus lowest cell',
      emphasis: 'tile',
      help: 'From the GPU-decoded heights of the tile.'
    },
    {id: 'cells', label: 'Valid cells', emphasis: 'tile', help: 'Cells the decode accepted.'},
    {
      id: 'summitHeight',
      label: 'Highest cell (GPU decode)',
      help: 'The top cell of the decoded tile; the published summit is a little higher because cells are 6.6 m wide and heights are quantised.'
    },
    {
      id: 'decodeDifference',
      label: 'GPU vs CPU decode, max error',
      hood: true,
      help: "Largest absolute difference between GPUTerrainRGBDecode and the loader's float64 Terrarium decode. Terrarium decoding is exact in float32, so this is 0."
    },
    {
      id: 'validPixels',
      label: 'Valid pixels',
      emphasis: 'tile',
      help: 'Share of pixels the decode accepted. No-data pixels are NaN with validity 0 and are hatched.'
    },
    {
      id: 'steeperThan30',
      label: `Steeper than ${DECISION_SLOPE_DEGREES}°`,
      emphasis: 'tile',
      help: 'Share of valid cells in the classes at or above the avalanche decision threshold.'
    },
    {
      id: 'steeperThan45',
      label: `Steeper than ${STEEP_FACE_DEGREES}°`,
      emphasis: 'tile',
      help: 'Share of valid cells in the steepest classes.'
    },
    {
      id: 'slopeHistogram',
      label: 'Slope distribution',
      kind: 'chart',
      help: 'Cells by slope, one bar per degree, coloured by the map classes.'
    },
    {
      id: 'share30Ground',
      label: `Past ${DECISION_SLOPE_DEGREES}°, ground cells`,
      emphasis: 'tile',
      help: 'Share of valid cells at or above the threshold with the true ground cell size of each row.'
    },
    {
      id: 'share30Mercator',
      label: `Past ${DECISION_SLOPE_DEGREES}°, Mercator pixels`,
      emphasis: 'tile',
      help: 'The same share when the Mercator pixel size is taken as metres: smaller, because every slope is flattened.'
    },
    {
      id: 'naive45',
      label: 'A steep face on Mercator pixels',
      emphasis: 'tile',
      help: 'tan(slope) scales with ground cell size over Mercator cell size at the centre of the tile.'
    },
    {
      id: 'maxSlopeGround',
      label: 'Steepest cell, ground cells',
      hood: true,
      help: 'The steepest valid cell of the tile on ground cells.'
    },
    {
      id: 'aspectRose',
      label: 'Which way steep slopes face',
      kind: 'chart',
      help: 'Share of the steep cells in each of sixteen compass sectors, coloured as on the map.'
    },
    {
      id: 'northShare',
      label: 'Steep cells facing NW, N or NE',
      emphasis: 'tile',
      help: 'Share of the steep cells whose aspect lies between north-west and north-east.'
    },
    {
      id: 'p98',
      label: '98th percentile of this product',
      emphasis: 'tile',
      help: 'Value below which 98 % of the valid cells lie (histogram bin accuracy).'
    },
    {id: 'taps', label: 'VRM taps', hood: true, help: 'Heights read per cell: (2r + 1) squared.'},
    {id: 'timing', label: 'Product GPU time', hood: true},
    {
      id: 'rebuilds',
      label: 'Product graphs compiled',
      hood: true,
      help: 'Each product compiles once; a compile option such as the VRM radius compiles its graph again.'
    },
    {id: 'decisionSlope', label: 'Decision slope', hood: true},
    {id: 'groundCell', label: 'Ground cell size', hood: true},
    {id: 'mercatorCell', label: 'Mercator pixel size', hood: true},
    {id: 'aspectFade', label: 'Aspect fades in over', hood: true},
    {id: 'repairJumps', label: 'Spike repair: jumps found', hood: true},
    {id: 'repairResult', label: 'Spike repair: result', hood: true}
  ],

  pipeline: [
    {
      id: 'decode',
      label: 'Decode',
      detail: 'Terrarium bytes to float32 metres, with a validity mask',
      show: {option: 'view', value: 'decoded'}
    },
    {id: 'repair', label: 'Repair', detail: 'Optional: shift +/-256 m red-byte spikes back'},
    {
      id: 'derivatives',
      label: 'Derivatives',
      detail: 'Horn slope and aspect over a window of neighbours, in ground metres'
    },
    {id: 'ruggedness', label: 'Ruggedness', detail: 'TPI and TRI on 3 x 3, VRM on a chosen radius'},
    {
      id: 'colorize',
      label: 'Classes',
      detail: 'Class the float raster into colours at the decision thresholds',
      show: {option: 'view', value: 'analysis'}
    }
  ],

  legends: getBasicsLegends,

  story: [
    {
      id: 'form',
      title: 'One PNG holds the whole mountain',
      headline: 'One image holds the whole mountain',
      textAlternative:
        'Shaded relief of the Matterhorn, Zermatt and the Gornergrat, tinted by height, with glaciers in pale blue and a dashed frame where the data ends west of the Matterhorn.',
      body: 'Every height here is three colour bytes in one PNG, `R*256 + G + B/256 - 32768`: **{{cells}}** of them. `GPUTerrainRGBDecode` turns the bytes into metres on the GPU and agrees with the CPU decode to **{{decodeDifference}}**. The top cell reads **{{summitHeight}}**. The relief, tints and glaciers are drawn from these same heights and OpenStreetMap outlines.\n\n*Decode first: every later number depends on it.*',
      optionsMode: 'fresh',
      options: {view: 'relief'},
      controls: [],
      readouts: ['relief', 'cells', 'summitHeight', 'decodeDifference'],
      camera: {...FORM_FRAME, transitionMs: 1400},
      stage: 'decode',
      furniture: furnitureFor(
        'How steep is the Matterhorn?',
        'Elevation, metres · pale tint under shaded relief'
      ),
      annotations: getStepPlaces(STEP_PLACES.form),
      highlight: {readout: 'decodeDifference'}
    },
    {
      id: 'decode',
      title: 'Decode before you map',
      headline: 'The wrong decoder leaves a blank map',
      textAlternative:
        'The decoded heights painted in height classes on bare paper; with the Mapbox decoder every cell is hatched as no data.',
      body: 'A PNG does not say how its colours encode height. Set **Decoder** to Mapbox and every height lands far outside the valid range, so the GPU marks all of it no data: hatched, never drawn. Switch back to Terrarium, then **Blank a patch (alpha 0)**: the hole survives into every later product. Noisy-tile repair (`GPUTerrainSpikeRepair`) is under Expert in All controls.\n\n*No data is not zero.*',
      optionsMode: 'fresh',
      options: {view: 'decoded', encoding: 'mapbox'},
      controls: ['encoding', 'missingTile'],
      readouts: ['validPixels', 'decodeDifference'],
      camera: {...FORM_FRAME, transitionMs: 1000},
      stage: 'decode',
      furniture: furnitureFor('Which decoder reads this PNG?', 'Heights decoded on the GPU'),
      annotations: getStepPlaces(STEP_PLACES.decode),
      highlight: {readout: 'validPixels'}
    },
    {
      id: 'slope-classes',
      title: 'Slope, classed where decisions change',
      headline: 'Avalanche terrain starts at one slope threshold',
      textAlternative:
        'The same landscape with slope drawn in classes from yellow through red and purple to near black, from the first class upward; ground under the first class is not coloured. A histogram and a Horn window diagram sit in the card.',
      body: '`GPUTerrainDerivatives` fits a plane to each small window of neighbouring heights (diagram), in ground metres. Continuous colour hides the class people act on: most slab avalanches release on slopes of **{{decisionSlope}}** or steeper. Switch **Slope colour** to compare; **{{steeperThan30}}** of the tile is at or past that line.\n\n*Class a continuous surface at the thresholds people act on.*',
      optionsMode: 'fresh',
      options: {view: 'analysis', product: 'slope', slopeDisplay: 'classes'},
      controls: ['slopeDisplay'],
      readouts: ['steeperThan30', 'steeperThan45', 'slopeHistogram'],
      camera: {...FORM_FRAME, transitionMs: 1400},
      stage: 'colorize',
      furniture: furnitureFor(
        'Where is the ground avalanche-steep?',
        'Slope, degrees · Horn window on ground cells'
      ),
      annotations: getStepPlaces(STEP_PLACES.slopeClasses),
      diagram: HORN_DIAGRAM,
      highlight: {readout: 'steeperThan30'}
    },
    {
      id: 'ground-units',
      title: 'Slope in ground units',
      headline: 'Measured on the Mercator grid, faces flatten',
      textAlternative:
        'A close view of the Matterhorn and the Hörnli Hut with slope classes computed on Mercator pixels, with a note at the steepest cell; switching to ground metres makes the steep classes larger.',
      body: 'The PNG is Web Mercator: a pixel is **{{mercatorCell}}** across, but at this latitude it covers only **{{groundCell}}** of ground. Taken as metres, the Mercator pixel flattens every face: **{{naive45}}**. Switch **Cell size** and watch the share past the threshold move from **{{share30Ground}}** on ground cells to **{{share30Mercator}}**.\n\n*Measure in ground units: a projected pixel is not a metre on the mountain.*',
      optionsMode: 'fresh',
      options: {view: 'analysis', product: 'slope', cellModel: 'mercator'},
      controls: ['cellModel'],
      readouts: ['share30Ground', 'share30Mercator', 'naive45', 'maxSlopeGround'],
      camera: {...TERRAIN_FRAMES.matterhorn, transitionMs: 1600},
      stage: 'derivatives',
      furniture: furnitureFor(
        'Is a map pixel a metre of mountain?',
        'Slope, degrees · cell size chosen above'
      ),
      annotations: getStepPlaces(STEP_PLACES.groundUnits),
      highlight: {readout: 'naive45'}
    },
    {
      id: 'aspect',
      title: 'Aspect: which way slopes face',
      headline: 'Which way each slope faces',
      textAlternative:
        'The tile coloured by the direction each steep slope faces on a cyclic ramp; gentle ground is not coloured, and a rose chart shows the share of steep cells in each compass sector.',
      body: "Aspect is the compass direction a slope faces downhill, so its colours wrap from north back to north. Flat ground has no aspect: colour fades in over **{{aspectFade}}** of slope. **{{northShare}}** of the steep cells face north-west, north or north-east; the relief under them is lit from the north-west by gradients of the same heights, the next story's subject.\n\n*A direction needs a cyclic ramp, faded where it means nothing.*",
      optionsMode: 'fresh',
      options: {view: 'analysis', product: 'aspect'},
      controls: [],
      readouts: ['aspectRose', 'northShare'],
      camera: {...TERRAIN_FRAMES.home, transitionMs: 1600},
      stage: 'derivatives',
      furniture: furnitureFor(
        'Which way does each slope face?',
        'Aspect, direction faced downhill'
      ),
      annotations: getStepPlaces(STEP_PLACES.aspect),
      highlight: {readout: 'northShare'}
    },
    {
      id: 'rugged-or-steep',
      title: 'Rugged or merely steep?',
      headline: 'Steep is not the same as rugged',
      textAlternative:
        'The Gorner glacier and the slopes around it coloured by vector ruggedness in classes of purple and red; the lowest class is not drawn.',
      body: 'Steep is one plane tilted; rugged is how much the surface scatters. `GPUTerrainVectorRuggedness` sums unit surface normals over a window: `1 - |sum n| / N`. Pick a **Product**, then widen **VRM window radius**: it is a compile option, so one graph rebuilds (**{{rebuilds}}**), and cost follows the taps (**{{taps}}**).\n\n**Try:** compare VRM with Slope on the Gorner glacier.\n\n*Steepness and ruggedness answer different questions.*',
      optionsMode: 'fresh',
      options: {view: 'analysis', product: 'vrm', vrmRadius: 2},
      controls: ['product', 'vrmRadius', 'measure'],
      readouts: ['p98', 'taps', 'timing', 'rebuilds'],
      camera: {...RUGGED_FRAME, transitionMs: 1600},
      stage: 'ruggedness',
      furniture: furnitureFor('Steep or merely rugged?', 'Vector ruggedness, index 0 to 1'),
      annotations: getStepPlaces(STEP_PLACES.rugged),
      highlight: {readout: 'p98'}
    }
  ],

  about: {
    what: '`GPUTerrainRGBDecode` converts Terrarium or Mapbox terrain-RGB PNG words to float32 heights plus a validity mask (bit-exact for Terrarium). `GPUTerrainSpikeRepair` fixes +/-256 m red-byte errors. `GPUTerrainDerivatives` gives Horn slope and aspect in ground metres; `GPUTerrainRuggedness` gives gdaldem TPI and TRI; `GPUTerrainVectorRuggedness` gives the Sappington et al. (2007) VRM. A small paint pass classes each float raster into colours from a table that the legend shares.',
    why: 'Slope and ruggedness feed route planning, avalanche and rockfall hazard, habitat models and viewshed or hydrology preparation. Decoding and classing on the GPU keeps a tile in GPU memory from PNG to map.',
    howToRead:
      'Relief: pale tint under warm-lit, cool-shaded relief, glaciers in pale blue. Slope: classes at the decision thresholds, nothing drawn under the first class. Aspect: a cyclic hue, faded on flat ground. TPI: orange above its neighbours (ridge), purple below (hollow). TRI and VRM: quantile classes of this tile, the lowest not drawn. Hover any cell for its value, its class and the window of heights behind it. Cell sizes are per-row ground metres from the Web Mercator tile geometry; the tile is one 2048 x 2048 raster. Reference implementations: gdaldem (slope, aspect, TPI, TRI) and mt-image (Terrarium decode, spike repair). The terrain datasets are bundled, so `?data=synthetic` makes no difference here.'
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
  width: 2048, height: 2048, elevation, settings: derivativeSettings.importToGraph(graph), slope, aspect,
  cellSizeMode: '${state.cellModel === 'ground' ? 'web-mercator' : 'uniform'}', borderMode: '${state.borderMode}'
}));
graph.add(new GPUTerrainRuggedness({
  width: 2048, height: 2048, elevation, terrainRuggednessIndex: tri,
  terrainRuggednessAlgorithm: '${state.triAlgorithm}', edgeMode: '${state.edgeMode}'
}));
const compiled = graph.compile(); // once

// per frame: parameter writes only (web-mercator: edges are normalized Mercator y of the tile)
derivativeSettings.write(getGPUTerrainDerivativesParameterValues({
  cellSize: [9.5546, 9.5546], northEdge: 0.3556, southEdge: 0.3112, zFactor: ${state.zFactor}
}));
compiled.encode(commandEncoder, {parameters: undefined});`
});
