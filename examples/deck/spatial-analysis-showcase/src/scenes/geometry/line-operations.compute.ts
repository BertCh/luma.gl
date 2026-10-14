// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphSwap} from '../../engine/graph-swap';
import type {SceneContext, SceneInstance} from '../scene';
import {loadRailPaths, loadStreetPaths, type PathSet} from './b3-city-data';
import type {GeometryView} from './b3-common';
import type {LineOperationsOptions} from './b3-line-options';
import {createReshapeView, createSnapView, type CityLineEnvironment} from './b3-line-views-city';
import {createTracksView} from './b3-line-views-far';

export type {LineOperationsOptions};

type Options = LineOperationsOptions;

/**
 * Line operations on three real layers: the L routes (densify, chunk, substring, locate), eligible
 * community places snapped to streets (linear referencing) and AIS ship tracks (Douglas-Peucker and
 * Chaikin). Each tool is created the first time it is shown.
 */
export async function createLineOperations(
  ctx: SceneContext<Options>
): Promise<SceneInstance<Options>> {
  const roads = ctx.datasets.get('chicago-roads');
  const transit = ctx.datasets.get('cta-transit');
  const placesDataset = ctx.datasets.get('chicago-places');
  const vessels = ctx.datasets.get('ais-vessels');

  const origin = roads.defaultOrigin;
  const projection = roads.getProjection(origin);
  const rail = loadRailPaths(transit, projection);
  let streets: PathSet | null = null;
  const cityEnvironment: CityLineEnvironment = {
    projection,
    ctx,
    coordinateOrigin: [origin[0], origin[1], 0],
    rail,
    get streets() {
      streets ??= loadStreetPaths(roads, projection);
      return streets;
    },
    places: {
      local: placesDataset.projectColumn('position', origin),
      categories: placesDataset.column<Uint8Array>('category'),
      count: placesDataset.count
    }
  };

  const factories: Record<Options['view'], () => Promise<GeometryView<Options>>> = {
    reshape: () => createReshapeView(cityEnvironment),
    snap: () => createSnapView(cityEnvironment),
    tracks: () => createTracksView({ctx, vessels})
  };
  const views = new Map<Options['view'], GeometryView<Options>>();
  let destroyed = false;
  const graphSwap = new GraphSwap();
  let active = await factories[ctx.options.view]();
  views.set(ctx.options.view, active);
  let activeName = ctx.options.view;
  // Option changes that a view missed while it was inactive or still being built; replayed on commit.
  const missed = new Map<Options['view'], Set<keyof Options>>();

  const getView = () => active;
  const showView = (name: Options['view']) => {
    // The previous view keeps encoding until the new one has been created.
    void graphSwap.request({
      build: async () => views.get(name) ?? (await factories[name]()),
      commit: view => {
        views.set(name, view);
        active = view;
        activeName = name;
        for (const id of missed.get(name) ?? []) view.setOption(id, ctx.options);
        missed.delete(name);
        ctx.requestLayers();
      },
      discard: view => {
        if (views.get(name) !== view) view.destroy();
      }
    });
  };

  const setActiveOption = (id: keyof Options, options: Options) => {
    getView().setOption(id, options);
    for (const name of Object.keys(factories) as Options['view'][]) {
      if (name === activeName) continue;
      const set = missed.get(name) ?? new Set<keyof Options>();
      set.add(id);
      missed.set(name, set);
    }
  };

  return {
    getCompiledGraphs: () => getView().getCompiledGraphs(),
    setOption(id, _value, options) {
      if (id === 'view') {
        showView(options.view);
      } else {
        setActiveOption(id, options);
      }
      ctx.requestLayers();
    },
    onThemeChange: () => ctx.requestLayers(),
    encode(commandEncoder, frame) {
      if (!destroyed) getView().encode(commandEncoder, frame);
    },
    getLayers: () => getView().getLayers(),
    destroy() {
      destroyed = true;
      graphSwap.destroy();
      for (const view of views.values()) view.destroy();
      views.clear();
    }
  };
}
