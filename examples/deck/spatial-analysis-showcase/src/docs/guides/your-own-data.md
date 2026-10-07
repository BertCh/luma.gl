---
title: Running analysis on your own data
summary: Typed arrays, a local metric projection, importGraphBuffer, and the shape of a scene.
order: 2
---

The stories use shipped datasets, but a contributor only needs GPU buffers. This guide shows the path from your arrays to a drawn result.

## 1. Project to planar meters

Contributors measure distances in meters, so project longitude and latitude to a local planar frame around one origin. `LocalMetricProjection` uses the same distance scales deck.gl applies to `COORDINATE_SYSTEM.METER_OFFSETS`, so layers drawn around that origin line up exactly.

```ts
import {LocalMetricProjection} from './engine/projection';

const origin: [number, number] = [-122.44, 37.76];      // lng, lat
const projection = new LocalMetricProjection(origin);
const meters = new Float32Array(count * 2);
for (let i = 0; i < count; i++) {
  [meters[i * 2], meters[i * 2 + 1]] = projection.project(lng[i], lat[i]);
}
```

In a scene you rarely do this by hand: `dataset.projectColumn('position')` projects a lon/lat column around the dataset's bbox center and memoizes it. Use `dataset.defaultOrigin` as the layer's `coordinateOrigin`.

## 2. Upload once

`SpatialAnalysisResources` creates storage buffers that both compute graphs and the layers can read, and destroys them in reverse order when the scene ends.

```ts
const resources = new SpatialAnalysisResources(device, 'my-scene');
const positions = resources.createBuffer('positions', meters);
const bounds = resources.createParameterBuffer('bounds', 'float32', 4);
```

## 3. Import buffers into a graph

`importGraphBuffer(graph, id, buffer, format, length)` wraps an application-owned buffer as a typed view the contributor accepts. Import a buffer **once** and pass the same view to every contributor that reads it, so the graph tracks hazards on one handle.

```ts
const graph = new GPUCommandGraph<void>(device, {id: 'my-scene'});
graph.add(
  new GPUPointDensity({
    positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', count),
    bounds: bounds.importToGraph(graph),
    gridSize: [160, 100],
    output: {values: importGraphBuffer(graph, 'values', values, 'float32', 16000)}
  })
);
const compiled = resources.track(graph.compile());
```

## 4. Encode every frame, draw the output

```ts
encode(commandEncoder, frame) {
  bounds.write(Float32Array.from(getViewportMetricBounds(frame.viewport, projection)));
  compiled.encode(commandEncoder, {parameters: undefined});
},
getLayers() {
  return [new SpatialAnalysisRasterLayer({values, colormap: 'viridis', /* ... */})];
}
```

Never compile, submit or synchronously read back inside `encode`. For a number you need on the CPU, use `SummaryReader`, which copies a few bytes into a ring ticket and delivers them asynchronously.

## 5. Wrap it as a scene

A scene is a file `src/scenes/<chapter>/<id>.scene.ts` with metadata, options, a story and a `create` function. Datasets are declared by id and arrive loaded in `ctx.datasets`. See the builder guide in the repository (`src/scenes/BUILDER-GUIDE.md`) for the full contract.
