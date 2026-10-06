// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The twelve named recipe builders (`addHotSpotAnalysisRecipe` and eleven more) on real New York
 * data, one recipe at a time. A recipe is a function that adds a whole chain of contributors to
 * one graph, so the panel lists that chain step by step next to the live outputs.
 *
 * Choosing a recipe is a compile-time choice: it builds and compiles that recipe's graph (the
 * footer rebuild counter counts it). Everything a recipe exposes per frame (radius, significance,
 * permutations, class counts, thresholds, drive-time breaks, a moving facility) is a parameter
 * buffer write on the compiled graph and never recompiles it. The controls are a fixed pool of
 * sliders and toggles that each recipe relabels, because a panel section cannot remove controls.
 */

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import type {SpatialAnalysisControlHandle} from '../app-ui';
import type {
  SpatialAnalysisFrame,
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance,
  SpatialAnalysisPointerEvent
} from '../spatial-analysis-mode';
import type {RecipeParameter, RecipeScene, RecipeSceneHost} from './recipes-kit';
import {RECIPE_ENTRIES} from './recipes-scenes';

const SLIDER_COUNT = 4;
const TOGGLE_COUNT = 2;
const CHAIN_LINE_COUNT = 9;
const OUTPUT_LINE_COUNT = 9;
/** Slider positions are integers in `[0, SLIDER_RESOLUTION]` mapped to the parameter's range. */
const SLIDER_RESOLUTION = 1000;
/** Frames Deck may still hold the layers of a replaced scene. */
const RETIRE_DELAY_MILLISECONDS = 400;

/**
 * A panel row the mode relabels and hides: the control builders cannot remove or rename a control,
 * so the pool finds its own rows in the DOM by their initial, unique text and edits them in place.
 */
type PanelRow = {row: HTMLElement | null; name: HTMLElement | null};

function findRowByText(text: string): PanelRow {
  const spans = Array.from(globalThis.document?.querySelectorAll('span') ?? []);
  const name = spans.find(span => span.textContent === text) ?? null;
  return {name, row: name?.parentElement ?? null};
}

function findSliderRow(label: string): PanelRow {
  const input = globalThis.document?.querySelector(`input[aria-label="${label}"]`);
  const row = input?.parentElement ?? null;
  return {row, name: (row?.firstElementChild?.firstElementChild as HTMLElement | null) ?? null};
}

function setRow(panelRow: PanelRow, text: string | null, host: 'block' | 'flex' = 'block'): void {
  if (panelRow.row) panelRow.row.style.display = text === null ? 'none' : host;
  if (panelRow.name && text !== null) panelRow.name.textContent = text;
}

type SliderSlot = {
  panel: PanelRow;
  control: SpatialAnalysisControlHandle<number>;
  spec: Extract<RecipeParameter, {kind: 'slider'}> | null;
};
type ToggleSlot = {
  panel: PanelRow;
  control: SpatialAnalysisControlHandle<boolean>;
  spec: Extract<RecipeParameter, {kind: 'toggle'}> | null;
};

function toPosition(spec: SliderSlot['spec'] & object, value: number): number {
  return Math.round(
    ((value - spec.minimum) / Math.max(spec.maximum - spec.minimum, 1e-12)) * SLIDER_RESOLUTION
  );
}

function toValue(spec: SliderSlot['spec'] & object, position: number): number {
  const raw = spec.minimum + (position / SLIDER_RESOLUTION) * (spec.maximum - spec.minimum);
  const snapped = spec.minimum + Math.round((raw - spec.minimum) / spec.step) * spec.step;
  return Math.min(spec.maximum, Math.max(spec.minimum, snapped));
}

export const recipesMode: SpatialAnalysisModeDefinition = {
  id: 'recipes',
  title: 'Recipes',
  contributors: [
    'addHotSpotAnalysisRecipe',
    'addRateClusterMapRecipe',
    'addPointsInPolygonsChoroplethRecipe',
    'addSpaceTimeHotSpotsRecipe',
    'addClusterAndOutlineRecipe',
    'addSpatialRegressionRecipe',
    'addDriveTimeCatchmentRecipe',
    'addStraightLineCatchmentsRecipe',
    'addChangeOfSupportRecipe',
    'addFleetDwellRecipe',
    'addFleetDwellZoneEventsRecipe',
    'addPeriodComparisonRecipe'
  ],
  description:
    'Each recipe is one function that wires a whole contributor chain; the step list shows what it ' +
    'composes. Pick a recipe (compile-time), then drag its live parameters: they are buffer writes.',
  initialViewState: {longitude: -73.96, latitude: 40.73, zoom: 11.3},

  async create(context) {
    const searchParameters = new URLSearchParams(globalThis.location?.search ?? '');
    const requested = searchParameters.get('recipe');
    let recipeId = RECIPE_ENTRIES.some(entry => entry.id === requested)
      ? (requested as string)
      : RECIPE_ENTRIES[0].id;
    let scene: RecipeScene | null = null;
    let buildToken = 0;
    let destroyed = false;
    const retired: RecipeScene[] = [];

    // --- Controls ------------------------------------------------------------------------------
    const recipeSelect = context.controls.addSelect<string>({
      label: 'Recipe (compile-time: switching builds and compiles that recipe)',
      options: RECIPE_ENTRIES.map(entry => ({value: entry.id, label: entry.label})),
      value: recipeId,
      onChange: value => {
        void switchRecipe(value);
      }
    });
    const summaryNote = context.controls.addNote('');
    const sliderSlots: SliderSlot[] = [];
    for (let index = 0; index < SLIDER_COUNT; index++) {
      const label = `Recipe parameter ${index + 1}`;
      const slot: SliderSlot = {panel: {row: null, name: null}, control: null as never, spec: null};
      slot.control = context.controls.addSlider({
        label,
        min: 0,
        max: SLIDER_RESOLUTION,
        step: 1,
        value: 0,
        format: position => (slot.spec ? slot.spec.format(toValue(slot.spec, position)) : ''),
        onChange: position => {
          if (slot.spec) slot.spec.onChange(toValue(slot.spec, position));
        }
      });
      slot.panel = findSliderRow(label);
      sliderSlots.push(slot);
    }
    const toggleSlots: ToggleSlot[] = [];
    for (let index = 0; index < TOGGLE_COUNT; index++) {
      const label = `Recipe toggle ${index + 1}`;
      const slot: ToggleSlot = {panel: {row: null, name: null}, control: null as never, spec: null};
      slot.control = context.controls.addToggle({
        label,
        value: false,
        onChange: value => slot.spec?.onChange(value)
      });
      slot.panel = findRowByText(label);
      toggleSlots.push(slot);
    }
    const legendNote = context.controls.addNote('');
    const chainReadouts = Array.from({length: CHAIN_LINE_COUNT}, (_, index) => {
      const label = `Step ${index + 1}`;
      const handle = context.controls.addReadout(label);
      return {handle, panel: findRowByText(label)};
    });
    const outputReadouts = Array.from({length: OUTPUT_LINE_COUNT}, (_, index) => {
      const label = `Result ${index + 1}`;
      const handle = context.controls.addReadout(label);
      return {handle, panel: findRowByText(label)};
    });
    const dataReadout = context.controls.addReadout('Data');

    /** Per-scene host: a retired scene can no longer write the shared readouts or layers. */
    const hosts = new Map<RecipeScene, {active: boolean}>();
    const createHost = (): {host: RecipeSceneHost; state: {active: boolean}} => {
      const state = {active: false};
      const host: RecipeSceneHost = {
        context,
        updateLayers: () => {
          if (!destroyed && state.active) context.updateLayers();
        },
        setOutputs: lines => {
          if (!state.active) return;
          outputReadouts.forEach((readout, index) => {
            const line = lines[index];
            setRow(readout.panel, line ? line[0] : null, 'flex');
            if (line) readout.handle.setValue(line[1]);
          });
        }
      };
      return {host, state};
    };

    function applyScene(next: RecipeScene, label: string, summary: string): void {
      summaryNote.setValue(summary);
      sliderSlots.forEach((slot, index) => {
        const spec = next.parameters.filter(parameter => parameter.kind === 'slider')[index] as
          | SliderSlot['spec']
          | undefined;
        slot.spec = spec ?? null;
        setRow(slot.panel, spec ? `${spec.label} (per-frame)` : null);
        slot.control.setDisabled(!spec);
        if (spec) slot.control.setValue(toPosition(spec, spec.value));
      });
      toggleSlots.forEach((slot, index) => {
        const spec = next.parameters.filter(parameter => parameter.kind === 'toggle')[index] as
          | ToggleSlot['spec']
          | undefined;
        slot.spec = spec ?? null;
        setRow(slot.panel, spec ? `${spec.label} (per-frame)` : null, 'flex');
        slot.control.setDisabled(!spec);
        slot.control.setValue(spec?.value ?? false);
      });
      legendNote.setValue(next.legend);
      chainReadouts.forEach((readout, index) => {
        const step = next.chain[index];
        setRow(readout.panel, step ? `Step ${index + 1}` : null, 'flex');
        if (step) readout.handle.setValue(step);
      });
      outputReadouts.forEach(readout => setRow(readout.panel, null));
      dataReadout.setValue(next.dataNote);
      context.setStatus(
        `${label}: ${next.contributorCount} contributors in ${next.chain.length} chain steps`
      );
    }

    async function switchRecipe(id: string): Promise<void> {
      const entry = RECIPE_ENTRIES.find(candidate => candidate.id === id);
      if (!entry) return;
      const token = ++buildToken;
      context.setStatus(`Building ${entry.label}…`);
      recipeSelect.setDisabled(true);
      const created = createHost();
      try {
        const next = await entry.build(created.host);
        if (destroyed || token !== buildToken) {
          created.state.active = false;
          next.destroy();
          return;
        }
        const previous = scene;
        scene = next;
        hosts.set(next, created.state);
        created.state.active = true;
        recipeId = id;
        if (previous) {
          const previousState = hosts.get(previous);
          if (previousState) previousState.active = false;
          retired.push(previous);
          setTimeout(() => {
            const position = retired.indexOf(previous);
            if (position >= 0) retired.splice(position, 1);
            previous.destroy();
          }, RETIRE_DELAY_MILLISECONDS);
        }
        applyScene(next, entry.label, entry.summary);
        context.updateLayers();
      } catch (error) {
        if (!destroyed && !context.signal.aborted) {
          context.setStatus(`Failed to build ${entry.label}: ${String(error)}`);
        }
      } finally {
        if (!destroyed) recipeSelect.setDisabled(false);
      }
    }

    const entry = RECIPE_ENTRIES.find(candidate => candidate.id === recipeId)!;
    const initial = createHost();
    scene = await entry.build(initial.host);
    hosts.set(scene, initial.state);
    initial.state.active = true;
    context.signal.throwIfAborted();
    applyScene(scene, entry.label, entry.summary);

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => (scene ? [scene.compiled] : []),
      encode(commandEncoder: CommandEncoder, frame: SpatialAnalysisFrame) {
        scene?.encode(commandEncoder, frame);
      },
      getLayers: (): Layer[] => scene?.getLayers() ?? [],
      onClick: (event: SpatialAnalysisPointerEvent) => scene?.onClick?.(event) ?? false,
      getTooltip: (event: SpatialAnalysisPointerEvent) => scene?.getTooltip?.(event) ?? null,
      destroy() {
        destroyed = true;
        buildToken++;
        for (const state of hosts.values()) state.active = false;
        scene?.destroy();
        for (const old of retired.splice(0)) old.destroy();
        scene = null;
      }
    };
    return instance;
  }
};
