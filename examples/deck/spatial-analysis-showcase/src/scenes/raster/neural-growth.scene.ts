// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ground} from '../../cartography/grounds';
import {defineScene} from '../scene';
import type {NeuralGrowthOptions} from './neural-growth.compute';

/** Temporal cloud repair over paired Sentinel-2 observations. */
export default defineScene<NeuralGrowthOptions>({
  id: 'neural-growth',
  title: 'Neural Cloud Repair: two places, two dates',
  chapter: 'raster',
  order: 20,
  summary:
    'Remove real Sentinel-2 clouds over Oʻahu or Venice. The SCL mask chooses missing pixels, an older clear observation supplies detail, and a tiny 8→16→4 cellular MLP adapts the colour residual on WebGPU. Compare it directly with a naive prior-image paste.',
  contributors: ['FrontierNeuralAutomaton'],
  datasets: [
    {
      id: 'sentinel-cloud-repair',
      role: 'two natural-colour cloudy/clear Sentinel-2 pairs with official scene-classification masks'
    }
  ],
  initialView: {longitude: -157.71496, latitude: 21.33493, zoom: 12.55},
  basemap: ground('paperCity'),
  furniture: {
    title: {
      title: 'Neural Cloud Repair',
      subtitle: 'A dated clear image is the prior—not a hallucination'
    },
    scaleBar: {units: 'metric'},
    credit: 'Contains modified Copernicus Sentinel data (2026)',
    caveat: 'The result estimates the cloudy date; changes hidden by cloud cannot be verified.'
  },
  options: [
    {
      kind: 'select',
      id: 'place',
      label: 'Place',
      group: 'Observation pair',
      apply: 'param',
      default: 'oahu',
      options: [
        {
          value: 'oahu',
          label: 'Waimānalo, Oʻahu',
          help: 'Eight-month prior; vegetation and illumination changed.'
        },
        {
          value: 'venice',
          label: 'Venice Lagoon',
          help: 'One-day prior; a clean occlusion-removal control.'
        }
      ],
      help: 'Switches all resident image, prior and mask buffers, then moves the camera.'
    },
    {
      kind: 'select',
      id: 'display',
      label: 'Map shows',
      group: 'Comparison',
      apply: 'param',
      default: 'cloudy',
      options: [
        {value: 'cloudy', label: 'Cloudy observation'},
        {value: 'neural', label: 'Neural residual repair'},
        {value: 'naive', label: 'Naive older-image paste'},
        {value: 'prior', label: 'Older clear reference'}
      ],
      help: 'Compare the observed target, the live GPU repair, a hard paste inside the same mask, and the complete prior.'
    },
    {
      kind: 'toggle',
      id: 'play',
      label: 'Run neural repair',
      group: 'Neural repair',
      apply: 'param',
      default: true,
      help: 'Runs four local MLP updates per frame while the neural result is visible.'
    },
    {
      kind: 'slider',
      id: 'stepScale',
      label: 'Repair rate',
      group: 'Neural repair',
      apply: 'param',
      min: 0.1,
      max: 1,
      step: 0.05,
      default: 0.8,
      help: 'How quickly the target-minus-prior colour residual propagates across masked pixels.'
    },
    {
      kind: 'button',
      id: 'restart',
      label: 'Restart neural repair',
      group: 'Neural repair',
      help: 'Restores the common initialization so the neural and naive results can be compared again.'
    },
    {
      kind: 'slider',
      id: 'opacity',
      label: 'Image opacity',
      group: 'Display',
      apply: 'param',
      min: 0.3,
      max: 1,
      step: 0.05,
      default: 1
    }
  ],
  story: [
    {
      id: 'clouds',
      title: 'Two real cloudy observations',
      headline: 'Clouds hide Oʻahu ridges and Venice docks',
      body: 'Choose **Waimānalo, Oʻahu** or **Venice Lagoon**. Each view is a natural-colour, 512 × 512 Sentinel‑2 Level‑2A crop covering **5.12 km** at **10 metres per pixel**. These are observed clouds—not a painted circular hole. The target Scene Classification Layer marks cloud, cirrus, shadow, uncertainty and defective pixels, then a one-cell halo catches mixed cloud edges.',
      options: {display: 'cloudy'},
      controls: ['place', 'display'],
      readouts: ['image', 'cloud']
    },
    {
      id: 'naive',
      title: 'First establish the naive baseline',
      headline: 'A hard prior paste is already surprisingly strong',
      body: '**Naive older-image paste** copies the clear observation only where the SCL mask is invalid and leaves every clear target pixel untouched. Venice uses the previous day, making it a clean control. Oʻahu uses an eight-month-old image, so seams in water colour, vegetation and illumination reveal the limitation of simple temporal substitution.',
      options: {display: 'naive'},
      controls: ['display', 'place'],
      readouts: ['cloud']
    },
    {
      id: 'neural',
      title: 'Now adapt the old detail to the target',
      headline: 'The GPU propagates observed date-to-date colour change',
      body: 'Switch to **Neural residual repair**. Clear target pixels store their exact target-minus-prior RGB residual and are pinned. Masked pixels start from the scene-wide residual; an 8→16→4 ReLU cellular MLP repeatedly mixes local residual and confidence Laplacians across the boundary. The display remains the sharp older image plus the inferred residual, so the model adjusts tone without blurring streets, roofs or ridges.',
      options: {display: 'neural'},
      controls: ['display', 'play', 'stepScale', 'restart'],
      readouts: ['generation', 'weights']
    },
    {
      id: 'compare',
      title: 'Neural versus naive is the actual test',
      headline: 'Improved seams do not prove hidden change',
      body: 'Toggle between **Neural residual repair** and **Naive older-image paste** after the generation count rises. The neural result should better match the target’s surrounding colour and illumination; the naive result is the unadjusted historical measurement. Neither can recover an object that changed while hidden. Use **Older clear reference** to inspect exactly which real detail both methods borrow.',
      controls: ['display', 'place'],
      readouts: ['generation', 'image']
    }
  ],
  legends: () => [],
  readouts: [
    {id: 'weights', label: 'MLP payload', format: 'text'},
    {id: 'generation', label: 'Generation', format: 'integer'},
    {id: 'cloud', label: 'Masked pixels', format: 'text'},
    {id: 'cells', label: 'Pixels per pass', format: 'integer'},
    {id: 'image', label: 'Observation pair', format: 'text'}
  ],
  snippet: state => `const automaton = new FrontierNeuralAutomaton({
  width, height, state, nextState, weights, parameters, habitat, prior, display
});
parameters.write(getFrontierNeuralParameterValues({
  stepScale: ${state.stepScale}, mutation: 0, damping: 0
}));`,
  about: {
    what: 'A compact neural cellular rule that removes pixels flagged by Sentinel‑2 scene classification and temporally inpaints them from a registered older clear observation.',
    why: 'Cloud removal is a credible browser-sized neural workload with an immediate visual baseline. The naive paste makes the contribution of the GPU model inspectable instead of asking the viewer to trust a “before/after” trick.',
    howToRead:
      'The cloudy image and both dated references are real. White cloud is never fed to the model as valid colour. Neural output is older detail plus an inferred target-date residual; hidden physical changes remain unknowable.'
  },
  create: async ctx => (await import('./neural-growth.compute')).createNeuralGrowth(ctx)
});
