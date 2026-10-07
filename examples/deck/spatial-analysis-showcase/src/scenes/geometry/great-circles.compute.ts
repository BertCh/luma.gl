// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fetchText, getDataFileUrl, parseCsv} from '../../data/loaders';
import type {SceneContext, SceneInstance} from '../scene';
import type {GreatCirclesOptions} from './b3-line-options';
import {createWorldView} from './b3-line-views-far';

export type {GreatCirclesOptions};

/**
 * The OpenFlights network as great-circle arcs, with geodesic distances and bearings from a hub to
 * every airport and a geodesic range ring around it.
 */
export async function createGreatCircles(
  ctx: SceneContext<GreatCirclesOptions>
): Promise<SceneInstance<GreatCirclesOptions>> {
  const flows = ctx.datasets.get('openflights');
  // IATA code to airport row for the hub list.
  const airportRows = new Map<string, number>();
  try {
    const table = parseCsv(
      await fetchText(getDataFileUrl('openflights', 'airports.csv'), ctx.signal)
    );
    const iata = table.header.indexOf('iata');
    const index = table.header.indexOf('index');
    for (const row of table.rows) airportRows.set(row[iata], Number(row[index]));
  } catch (error) {
    if (ctx.signal.aborted) throw error;
  }
  const view = createWorldView({ctx, flows, airportRows});
  return {
    getCompiledGraphs: () => view.getCompiledGraphs(),
    setOption(id, _value, options) {
      view.setOption(id, options);
      ctx.requestLayers();
    },
    onThemeChange: () => ctx.requestLayers(),
    encode: (commandEncoder, frame) => view.encode(commandEncoder, frame),
    getLayers: () => view.getLayers(),
    destroy: () => view.destroy()
  };
}
