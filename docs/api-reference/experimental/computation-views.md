---
title: Computation views authoring guide
description: Embed recorded, explorable views of a computation in the docs with the lens directive.
---

# Computation views authoring guide

:::caution Experimental
`@luma.gl/experimental/computation-views` and `@luma.gl/experimental/computation-lenses` are
experimental entry points. The directive syntax, record files and views may change between releases.
:::

A **lens** is a view of one real computation at one level of abstraction: the formula an operator
computes, one datum's receipt, the iterations a loop actually ran, the kernel dispatch, or the command
graph. Every lens is drawn from a recorded run of the real GPU computation, never from a CPU replay. The
same Markdown renders on this site and in the Reference section of the spatial analysis showcase
(`examples/deck/spatial-analysis-showcase`).

This page is for people writing the operator reference. The record format is documented in
[GPU computation records](/docs/api-reference/experimental/gpu-core/computation-record).

## The lens directive

A lens is a Markdown leaf directive on a line of its own:

```md
::lens{scenario=gpu-spatial-analysis.local-moran rung=datum bind=page datum=17 caption="Local Moran I of tract 17"}
```

| Attribute | Required | Meaning |
| --- | --- | --- |
| `scenario` | yes | Registered lens scenario, `<entry>.<element>[.<variant>]` in kebab case, for example `gpu-core.scan` |
| `rung` | yes | Level of abstraction to show; see [Rungs](#rungs) |
| `bind` | no | Page coordinator the lens links through; lenses with the same name share selections. Defaults to `page` |
| `datum` | no | Non-negative integer: the entity (row, cell, node) the view opens bound to |
| `iteration` | no | Non-negative integer: the loop iteration the view opens at |
| `caption` | no | Quoted caption. Without it the generated description is the caption |
| `view` | no | Id of a registered view to use for the rung instead of the preferred one, for example `neighbourhood-view` |

Unknown attributes, an invalid scenario id or an unknown rung fail the website build with the file and
line. The showcase prints the same error in place instead of failing.

## Rungs

| Rung | View | Use it for |
| --- | --- | --- |
| `formula` | FormulaCard | The statement the operator computes, with symbols bound to buffers and kernels |
| `datum` | DatumReceipt | One entity's worked example: inputs, neighbourhood and result, in analysis-stage order |
| `iterations` | SkeletonView and RealizedView | The declared step structure and what this run actually did, iteration by iteration |
| `kernel` | DispatchView | Real dispatch geometry, workgroups and lanes, with the kernel source |
| `graph` | CommandGraphView | The command graph at node or contributor grain, folded by degree of interest |
| `timing` | TimingView | Per-node GPU time with the observer effects of timing stated |
| `compare` | CompareView | Two runs or two references as an explicit delta |
| `scope` | ScopeDiagram | Which inputs an output reads, generated from the operator's `scope × reduction` |
| `table` | Table | The typed table view every record has, for anything without a custom view |

A rung the record cannot satisfy is shown disabled with its reason and the table view, never faked.

## Page templates

Each operator section in the reference ends with a **How it computes** block. Its order depends on the
audience of the entry:

- **Analyst entries** (spatial analysis, network, raster, terrain, dataframe analysis) go concrete first:
  `datum` with a substituted formula, then `formula`, then `iterations` if the operator is iterative,
  then `kernel`, then `graph`.
- **Engineer entries** (GPU Core, gpgpu) open at `kernel` or `graph`, with the other rungs one level up.

Each entry's Concepts page has a **GPU execution model** section drawn from the real `describe()` output
of a representative graph: the notional machine made explicit.

### Lens slots

The operator pages carry one marker per exported contributor at the end of the section that documents
it:

```md
<!-- lens:GPULocalMoran -->
```

To fill a slot, keep the marker and add the heading and directives around it:

```md
#### How it computes

<!-- lens:GPULocalMoran -->

::lens{scenario=gpu-spatial-analysis.local-moran rung=datum datum=17 caption="Tract 17, worked example"}

::lens{scenario=gpu-spatial-analysis.local-moran rung=formula}

::lens{scenario=gpu-spatial-analysis.local-moran rung=kernel}
```

The coverage contract finds a contributor's section by its marker. The Concepts pages use the marker
`<!-- lens:execution-model -->`.

## Static twins

Every lens has a static twin printed at build time: a `<figure>` with the recorded view, a caption, an
inline key number, the receipt as a table, the formula as LaTeX with MathML, and the adapter, date and
observer effects of the recorded run. Search engines, print, the llms text export and readers without
WebGPU read the twin. A twin needs no JavaScript.

Twins are produced in three steps:

1. A scenario in `@luma.gl/experimental/computation-lenses` runs the real computation on docs-sized open
   data (`modules/experimental/src/computation-lenses/scenarios/<entry>/<element>.lens.ts`).
2. A headless capture run on a real GPU writes `docs/lens-records/<scenario>.json`, labelled with
   adapter, variant and date. Only maintainers run it; it is not part of the docs build. Records are
   compacted losslessly on disk (repeated captures and word runs, WGSL source lines, interned graph
   strings, object shapes and repeated arrays); read them with `fromRecordedLensFile`. Each record has a
   512 KB budget (`record-budget.node.spec.ts`): a scenario shrinks by capturing key frames rather than
   every iteration, only the captures its views read, or a smaller docs input, and says so in the
   capture's meaning.
3. A node spec turns each record into one twin per directive key, using the same description generator
   that feeds alt text and the live region, and checks them in:

   ```sh
   UPDATE_LENS_TWINS=1 npx vitest run --project node modules/experimental/test/computation-docs/lens-twins.node.spec.ts
   ```

   The files land in `docs/lens-records/twins/<scenario>~<rung>[~d<datum>][~i<iteration>][~v<view>].json`. Without
   the variable the spec fails when a twin is stale, and it fails when a twin has no caption or no key
   number.

The website and the showcase only read twin files, so a docs build never imports view or GPU code.

A twin also prints a verdict chip for each guard that applies to the element: a domain pack returns
them from `guardsFor({record, element})`, and the same guards run in the live lens. A chip says
*pass*, *fail* with the reason, or *unknown* with the missing metadata.

### Record pending

A directive whose scenario has no recorded run yet prints a hatched **Record pending** twin with its
caption and a pending key number. It is never an error, so a page can be written before its records
exist.

## Live mode

On a browser with WebGPU each twin offers **Run live on this GPU**. Nothing runs during render or
hydration. On the reader's click the page loads `computation-views` and `computation-lenses`, mounts the
view into the twin's `<ladder-lens>` element, and can re-run the same scenario on the reader's adapter.
On failure the twin comes back with the error stated.

- One device per page, using the `webgpu-analysis` profile; a page scheduler serializes runs.
- Lenses far offscreen revert to their twin, and views pause when the page is hidden.
- Leaving the page disposes every live lens.

## Linking lenses

Lenses on one page link by default through the coordinator named by `bind`: selecting a datum in one
lens highlights it in every lens bound to the same name, through declared key-space relations rather than
geometry. Give unrelated lenses different `bind` names to keep them independent.

## Coverage contract

`modules/experimental/test/computation-docs/lens-coverage.node.spec.ts` checks the docs, packs and
scenarios together. For each entry listed in its `COVERED_ENTRIES`, every exported contributor needs:

1. an operator descriptor `scope × reduction` in the entry's domain pack, plus `variableKinds` for spatial
   analysis;
2. a formula reference, or an explicit `{formula: 'none', reason}` with an `iterations` or `graph` lens;
3. at least one registered lens scenario whose `element` is the exported name and whose id is
   `<entry>.<element-kebab>[.<variant>]`;
4. a `::lens` directive for that scenario in its section;
5. a recorded run, which may be missing until the capture run (set
   `LENS_COVERAGE_REQUIRE_RECORDS=1` to require it).

Independent of `COVERED_ENTRIES`, the spec checks that every lens slot marker is present, that every
directive is well formed, and that every record file names its scenario.

## Writing good lenses

- Draw from a real run. Folded state recovered by a GPU replay is labelled as a replay; a value is never
  re-simulated on the CPU for display.
- State observer effects. A timing lens says that timestamps disabled pass coalescing.
- Choose captions that say what the reader should see, and make the key number the one fact a reader
  should take away.
- Keep data small and real: each scenario dataset is at most 150 KB, with its source and licence listed in
  `modules/experimental/src/computation-lenses/data/README.md`.

## See also

- [GPU computation records](/docs/api-reference/experimental/gpu-core/computation-record)
- [GPU Spatial Analysis concepts](/docs/api-reference/experimental/gpu-spatial-analysis/concepts)
- [GPU Network concepts](/docs/api-reference/experimental/gpu-network/concepts)
- [GPU Terrain concepts](/docs/api-reference/experimental/gpu-terrain/concepts)
