import {GPUCoreDocsTabs} from '@site/src/components/docs/gpu-core-docs-tabs';
import {GPUOperationContract} from '@site/src/components/docs/gpu-operation-contract';

# GPUFlagOffsets

<GPUCoreDocsTabs active="flag-offsets" />

## Overview

Converts binary flags into exclusive offsets and a total selected count.

::lens{scenario=gpu-core.flag-offsets rung=result caption="Flags mark rainfall of at least 100 mm; offsets assign dense destinations."}

`GPUFlagOffsets` turns one packed binary flag stream into exclusive dense indices and a
GPU-resident count. It is the smallest useful materialization step after a classifier.

<GPUOperationContract operation="gpu-flag-offsets" />

#### How it computes

{/* lens:GPUFlagOffsets */}

::lens{scenario=gpu-core.flag-offsets rung=kernel caption="Inspect 256 lanes in the nested scan and the scalar count publication."}

::lens{scenario=gpu-core.flag-offsets rung=graph caption="Trace the nested scan into 1 final count publication."}

::lens{scenario=gpu-core.flag-offsets rung=datum caption="Follow station 9 from its wet flag to its dense offset."}

::lens{scenario=gpu-core.flag-offsets rung=formula caption="Compare exclusive offsets with the 1 final selected count."}

## When to use it

Use it when later GPU work needs a stable destination for every accepted slot, or when an indirect
consumer needs the number of accepted slots. Typical uses include nullable-value layout, stable
scatter, variable-length value offsets, and the logical-element stream at one nesting depth.

Use `GPUScan` directly if no total count is needed. Use `GPUCompaction` if the operation should also
move `uint32` values. Use `GPUSegmentedLayout` if value, element, and group layout are all needed at
one depth. For nested layouts, compose one shared `GPUFlagOffsets` for leaf validity with one
element `GPUFlagOffsets` and `GPUSegmentOffsets` per depth.

## Contract

| View | Length | Meaning |
| --- | ---: | --- |
| `flags` | slot count | Packed values that must be exactly zero or one |
| `offsets` | at least slot count | Exclusive prefix sum; the dense destination for each slot |
| `count` | at least 1 | Total set flags in element zero |

`flags` and `offsets` may independently be atomic views or `GraphVectorView`s with different
partitions. The scan carries across chunks without repacking their buffers, and `count` covers
the complete flag sequence. Extra offset capacity is left untouched. An empty input writes
`count[0] = 0`. No offset element exists for an empty input. Counts and offsets use `uint32`
arithmetic, so callers must retain page or batch boundaries before overflow.

## Usage

```ts
import {GPUCommandGraph, GPUFlagOffsets} from '@luma.gl/gpgpu/gpu-core';

const graph = new GPUCommandGraph(device, {id: 'nullable-column'});

graph.add(new GPUFlagOffsets({
  id: 'present-values',
  flags: validity,
  offsets: valueOffsets,
  count: nonNullValueCount
}));
```

The class contributes graph nodes only. It does not allocate public outputs, compile the graph,
submit work, or map the count. A compiled graph can be reused when its view sizes and topology stay
fixed; imported buffers may be rebound for each encoding.

## Costs and mistakes

The complete flag stream is scanned even when few values are present. Keep flags GPU-resident when
several consumers reuse the result; a CPU round trip solely to obtain the count defeats the main
benefit. Values greater than one are summed, not clamped, so normalize arbitrary predicates before
calling the operation.
