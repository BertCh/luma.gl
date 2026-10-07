---
title: How contributors work
summary: Compile once, rewrite parameter buffers every frame, draw outputs straight from GPU buffers.
order: 1
---

Every analysis on this site follows the same three-beat pattern. Learn it once and every story reads the same way.

<div class="guide-diagram">
<svg viewBox="0 0 760 250" role="img" aria-label="Diagram: data buffers and parameter buffers feed a compiled command graph that writes output buffers, which layers draw directly" font-family="system-ui, sans-serif" font-size="13">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill="currentColor"/></marker>
  </defs>
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="10" y="20" width="150" height="46" rx="8"/><text x="85" y="48" text-anchor="middle" fill="currentColor" stroke="none">Data buffers (once)</text>
    <rect x="10" y="95" width="150" height="46" rx="8" stroke-dasharray="5 4"/><text x="85" y="123" text-anchor="middle" fill="currentColor" stroke="none">Parameter buffers</text>
    <rect x="215" y="35" width="190" height="120" rx="10" stroke="#1f6feb" stroke-width="2"/>
    <text x="310" y="62" text-anchor="middle" fill="#1f6feb" stroke="none" font-weight="600">Compiled graph</text>
    <rect x="235" y="76" width="70" height="26" rx="6"/><text x="270" y="93" text-anchor="middle" fill="currentColor" stroke="none">node</text>
    <rect x="315" y="76" width="70" height="26" rx="6"/><text x="350" y="93" text-anchor="middle" fill="currentColor" stroke="none">node</text>
    <rect x="275" y="114" width="70" height="26" rx="6"/><text x="310" y="131" text-anchor="middle" fill="currentColor" stroke="none">node</text>
    <rect x="460" y="35" width="130" height="46" rx="8"/><text x="525" y="63" text-anchor="middle" fill="currentColor" stroke="none">Output buffers</text>
    <rect x="620" y="35" width="130" height="46" rx="8"/><text x="685" y="63" text-anchor="middle" fill="currentColor" stroke="none">deck.gl layers</text>
    <path d="M160 43H213" marker-end="url(#arrow)"/>
    <path d="M160 118H213" marker-end="url(#arrow)" stroke-dasharray="5 4"/>
    <path d="M405 58H458" marker-end="url(#arrow)"/>
    <path d="M590 58H618" marker-end="url(#arrow)"/>
  </g>
  <g fill="currentColor" font-size="12">
    <text x="10" y="195">1. Compile once: graph.compile()</text>
    <text x="10" y="215">2. Each frame: parameters.write(...) then compiled.encode(commandEncoder)</text>
    <text x="10" y="235">3. Layers read the output buffers on the GPU. No readback in the frame loop.</text>
  </g>
</svg>
</div>

## 1. Compile once

A contributor such as `GPUPointDensity` is not a function you call. It declares *resources* and *compute nodes* into a `GPUCommandGraph`. You add it with `graph.add(...)` and call `graph.compile()` once. Compiling decides the topology: array lengths, grid sizes, which optional outputs exist, and compile-time options such as grid or hexagon binning, or false-discovery-rate correction.

## 2. Rewrite parameter buffers every frame

Everything that should change at interactive speed is read from a small *parameter buffer* (`GPUParameterBuffer`): the viewport bounds, a radius, a threshold, a time, a seed. Each frame you call `parameters.write(...)` and `compiled.encode(commandEncoder, {parameters: undefined})`. Nothing is recompiled.

The options panel marks each control as either a parameter write or a **rebuild**. When you flip a rebuild option, **Under the hood** counts one more rebuild; when you drag a slider it does not move. That counter is the proof of the pattern.

## 3. Draw outputs straight from GPU buffers

Contributors write caller-owned output buffers. The showcase's layers bind those same buffers as read-only storage, so the analysis result never visits the CPU. The compute pass and the draw pass share one command buffer per frame because the effect encodes into deck.gl's own frame encoder.

The only readbacks are tiny summaries: a min and max for the legend, a handful of class counts for the readouts. They go through a small ring of staging buffers and arrive a frame or two later.

## Compile-time or parameter?

Each prop of a contributor says which category it belongs to in its TSDoc and in the [reference](#/reference). As a rule of thumb:

- **Parameter**: viewports, radii, thresholds, seeds, observer positions, budgets, kernel weights, masks.
- **Compile-time**: lengths, capacities, which optional views exist, binning scheme, significance-testing variant.

When a scene exposes a compile-time option, it usually compiles every variant up front and switches between them, so toggling stays instant.
