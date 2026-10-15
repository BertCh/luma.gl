---
title: GPU Dataframe grouping and aggregation
description: Group categorical values and compute reductions and histograms on GPU-resident tables.
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Dataframe grouping and aggregation

<ExperimentalDocsTabs active="gpu-dataframe-aggregation" />

## Group dense categorical values

### `GPUDataFrameGroupedAggregationQuery`

Group rows by a dense `uint32` key and aggregate each group, planned by `GPUDataFrameGroupByQuery` and compiled to a GPU table.

::lens{scenario=gpu-dataframe.data-frame-grouped-aggregation-query rung=result caption="Category 0 holds 505 of 1,500 observations; category 1 has 370 research grade of 384 (96%)."}

Group keys must use `uint32` GPU storage. Dictionary-backed keys infer their dense group count from
the adapter-owned labels; raw `uint32` keys require an explicit `groupCount`. The following example
assumes the dataframe also contains a dictionary-backed `category` column.

```ts
const grouped = dataframe
  .filter(column('fare').greaterThan(parameter('minimumFare', 10)))
  .groupBy('category')
  .aggregate({
    rides: 'count',
    totalFare: {sum: 'fare'},
    minimumFare: {min: 'fare'},
    maximumFare: {max: 'fare'},
    averageFare: {mean: 'fare'}
  });

const explicitGroups = dataframe.groupBy('category', {groupCount: 4});
```

Grouping preserves the category dictionary and publishes one row for every dense group, including
empty groups. Nullable keys are excluded. Count results are `uint32`; summed, minimum, maximum,
and mean values currently require `float32` input. Null, NaN, and infinite metric values do not
contribute. Empty numeric groups have an explicit invalid output mask; their sum payload is zero
and minimum, maximum, and mean payloads are NaN.

Cross-batch grouping accumulates contributions from every original source batch without repacking
the source table. `CompiledGPUDataFrameGroupedAggregation.groupCount` exposes the dense domain.

#### How it computes

{/* lens:GPUDataFrameGroupedAggregationQuery */}
{/* lens:GPUDataFrameGroupByQuery */}

`GPUDataFrameGroupByQuery` is the plan that `groupBy()` returns; it shares this card because `aggregate()` is what compiles it.

::lens{scenario=gpu-dataframe.data-frame-grouped-aggregation-query rung=datum bind=page datum=1 caption="Group 1: 384 observations, 370 research grade, mean 0.9635."}
::lens{scenario=gpu-dataframe.data-frame-grouped-aggregation-query rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.data-frame-grouped-aggregation-query rung=kernel caption="Look at atomicAddFloat: each accepted row adds its value into its group's slot."}
::lens{scenario=gpu-dataframe.data-frame-grouped-aggregation-query rung=graph caption="Clear and initialise the outputs, accumulate count, sum and mean per group key, then finalize validity and means."}

## Compute global reductions and explicit histograms

### `GPUDataFrameAggregationQuery`

Reduce selected rows to one row of count, sum, minimum, maximum and mean statistics in a single GPU command graph.

::lens{scenario=gpu-dataframe.data-frame-aggregation-query rung=result caption="Population sums to 2,716,047 across 791 tracts; the largest tract has 12,213 residents."}

Global reductions support packed `float32`, `sint32`, and `uint32` metric columns:

```ts
const totals = dataframe.aggregate({
  rows: 'count',
  totalFare: {sum: 'fare'},
  minimumFare: {min: 'fare'},
  maximumFare: {max: 'fare'},
  averageFare: {mean: 'fare'}
});
```

`count` counts selected source rows and produces `uint32`. A metric's sum, minimum, and maximum
retain its input format; its mean is `float32`. Metric nulls and nonfinite floating-point values
are excluded independently, and each potentially empty metric has an explicit one-row validity
mask. Native integer sums wrap to their 32-bit representation, floating-point reductions retain
`float32` precision, and oversized row counts are rejected instead of silently overflowing.

#### How it computes

{/* lens:GPUDataFrameAggregationQuery */}

::lens{scenario=gpu-dataframe.data-frame-aggregation-query rung=datum bind=page datum=0 caption="Statistics row: 791 rows, sum 2,716,047, minimum 0, maximum 12,213, mean 3,433.69."}
::lens{scenario=gpu-dataframe.data-frame-aggregation-query rung=formula bind=page datum=0}
::lens{scenario=gpu-dataframe.data-frame-aggregation-query rung=kernel caption="Look at the reduce-level and finalize kernels: masked rows contribute nothing to any statistic."}
::lens{scenario=gpu-dataframe.data-frame-aggregation-query rung=graph caption="A multi-level reduction per statistic over the shared selection mask, plus count-valid and validity nodes."}

### `GPUDataFrameHistogramQuery`

Count selected numeric values into equal-width or explicit-edge bins, publishing dense `bin` and `count` columns.

::lens{scenario=gpu-dataframe.data-frame-histogram-query rung=result caption="Income bin 1, $32,000 to $64,000, holds the most tracts: 261 of 772 with an estimate."}

```ts
const equalWidth = dataframe.histogram('fare', {
  bins: 8,
  domain: [0, 80]
});

const customEdges = dataframe.histogram('fare', {
  edges: [0, 10, 25, 50, 100]
});
```

Histograms publish a dense GPU table of `uint32` `bin` identifiers and `count` values. Supply either
an explicit equal-width domain or 2–257 strictly ascending literal edges; automatic domains are not
supported because masked or nullable source values must not influence an inferred extent. Existing
filters, null masks, derived columns, and repeated query parameters apply before binning.

#### How it computes

{/* lens:GPUDataFrameHistogramQuery */}

::lens{scenario=gpu-dataframe.data-frame-histogram-query rung=datum bind=page datum=1 caption="Bin 1 covers $32,000 up to $64,000 and counts 261 tracts, the most of any bin."}
::lens{scenario=gpu-dataframe.data-frame-histogram-query rung=formula bind=page datum=1}
::lens{scenario=gpu-dataframe.data-frame-histogram-query rung=kernel caption="Look at binIndex: equal-width bins use the float32 ratio, and the maximum joins the last bin."}
::lens{scenario=gpu-dataframe.data-frame-histogram-query rung=graph caption="Initialise and clear the bins, combine the validity mask, then accumulate each accepted row into its bin."}

## Related pages

- [GPU Dataframe overview](/docs/api-reference/experimental/gpu-dataframe)
- [GPU Dataframe operations index](/docs/api-reference/experimental/gpu-dataframe-operations)
- [GPU tables](/docs/api-reference/experimental/gpu-tables)
