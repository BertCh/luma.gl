// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

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

  const factories: Record<Options['view'], () => GeometryView<Options>> = {
    reshape: () => createReshapeView(cityEnvironment),
    snap: () => createSnapView(cityEnvironment),
    tracks: () => createTracksView({ctx, vessels})
  };
  const views = new Map<Options['view'], GeometryView<Options>>();
  let destroyed = false;
  const getView = () => {
    const name = ctx.options.view;
    let view = views.get(name);
    if (!view) {
      view = factories[name]();
      views.set(name, view);
    }
    return view;
  };
  getView();

  return {
    getCompiledGraphs: () => getView().getCompiledGraphs(),
    setOption(id, _value, options) {
      if (id === 'view') {
        getView();
      } else {
        getView().setOption(id, options);
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
      for (const view of views.values()) view.destroy();
      views.clear();
    }
  };
}
