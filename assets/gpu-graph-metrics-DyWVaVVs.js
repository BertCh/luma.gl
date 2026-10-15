var e=`---
title: GPU Graph metrics and ranking
description: Core numbers, clustering coefficients, and PageRank for GPU-resident graphs.
---

import {ExperimentalDocsTabs} from '@site/src/components/docs/experimental-docs-tabs';

# GPU Graph metrics and ranking

<ExperimentalDocsTabs active="gpu-graph-metrics" />

## Find durable network backbones with GPUGraphCoreNumber

Refine distinct-neighbor degrees by H-index in synchronized rounds, publishing each vertex's k-core number.

::lens{scenario=gpu-graph.graph-core-number rung=result caption="Compare core numbers across 485 stations; 18 stations reach core three."}

**Question: Which vertices stay connected to a mutually supportive group after peripheral
relationships disappear?**

\`GPUGraphCoreNumber\` separates a genuinely cohesive network backbone from a vertex that merely has
many fragile spokes. A vertex belongs to the **k-core** when it remains in a largest subgraph where
every remaining vertex has at least \`k\` other remaining neighbors; its core number is the highest
such \`k\`. An isolated vertex has core number zero.

Consider a popular account with 100 followers who do not follow one another. Its degree is 100,
but the account and every follower have core number one: remove the leaves and no mutually
supporting dense group remains. A four-person clique instead has core number three for each
member. Use core decomposition to find resilient social backbones, tightly sustained fraud rings,
interdependent service groups, or the stable center of a citation network. Choose vertex degree
for immediate popularity, local clustering for connections among one vertex's neighbors, community
labels for a proposed group assignment, and core numbers for repeated structural support.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| \`id\` | \`string\` | \`'gpu-graph-core-number'\` | Prefix for generated command-graph nodes and imported resources. |
| \`topology\` | \`GPUGraphTopology\` or \`GPUGraphTopologyView\` | Required | Existing compressed adjacency; directed weak neighborhoods require reverse CSR. |
| \`output\` | \`GPUVector<'uint32'>\` or \`GraphDataView<'uint32'>\` | Required | Caller-owned unsigned core-number row per vertex. |
| \`iterations\` | \`number\` | \`32\` | Maximum synchronized refinement rounds, from zero through 1,024. |
| \`converged\` | \`GPUVector<'uint32'>\` or \`GraphDataView<'uint32'>\` | Omitted | Optional scalar reporting whether the core numbers reached a fixed point. |
| \`degeneracy\` | \`GPUVector<'uint32'>\` or \`GraphDataView<'uint32'>\` | Omitted | Optional scalar receiving the maximum currently published core number. |

\`output\` is a caller-owned, packed \`GPUVector<'uint32'>\` with exactly one row per vertex. The
optional one-row \`converged\` output becomes one only when the final synchronized refinement proves
that no core estimate changed. The optional one-row \`degeneracy\` output receives the maximum
published core number, describing the deepest available graph core. When convergence is zero,
both the per-vertex values and the maximum are **upper bounds**, not proven exact answers.

The operation explicitly projects all relationships into a **simple undirected weak graph**. A
distinct incoming or outgoing neighbor counts once; reciprocal directed edges and parallel edges
do not increase support, self-loops do not make a vertex support itself, and edge weights do not
change structural core membership. This directed convention is not interchangeable with an
in-degree-plus-out-degree convention that counts reciprocal incidences separately. Directed
graphs require complete forward and reverse CSR; undirected graphs reuse symmetric forward CSR.

Each vertex starts at its distinct weak-neighbor degree. Every bounded round simultaneously
replaces that upper estimate with the H-index of its neighbors' preceding estimates: the largest
\`k\` for which at least \`k\` distinct neighbors still have estimates of at least \`k\`. The default is
\`32\` rounds; \`iterations\` may be any integer from \`0\` through \`1024\`. Zero rounds publish the
initial unique-neighbor degrees and conservatively leave convergence unproven unless the graph has
no edges. An empty graph has no core rows, convergence one, and optional degeneracy zero. There
is no automatic early termination, CPU synchronization, or implicit result readback.

If either required CSR neighbor allocation overflows, every core number and the optional
degeneracy become \`0xffffffff\`, and optional convergence becomes zero. For \`K\` configured rounds
and distinct weak degree \`d\`, unsorted CSR deduplication and H-index selection require worst-case
\`O(K × sum(d² × log(d + 1)))\` work. The implementation needs \`O(V)\` graph-owned scratch, plus an
optional bounded reduction workspace when reporting degeneracy; large hubs or insufficient round
budgets require explicit measurement rather than assumed convergence.

#### How it computes

{/* lens:GPUGraphCoreNumber */}

::lens{scenario=gpu-graph.graph-core-number rung=kernel caption="Inspect the initialization, H-index refinement, publish and convergence kernels run each round."}
::lens{scenario=gpu-graph.graph-core-number rung=graph caption="Follow topology construction into 12 rounds of refine, publish and convergence passes."}
::lens{scenario=gpu-graph.graph-core-number rung=datum datum=409 caption="Inspect Utrecht Centraal, station 409, with its 15 neighbors and core number three."}
::lens{scenario=gpu-graph.graph-core-number rung=formula caption="Follow the H-index refinement to the WGSL buffers that hold it."}
::lens{scenario=gpu-graph.graph-core-number rung=iterations caption="Compare six recorded core-number frames through the final round; no intermediate states are interpolated."}

## Measure neighborhood density with GPUGraphLocalClusteringCoefficient

Count closed neighbor pairs per vertex, publishing Graphalytics clustering coefficients and optional triangle counts.

::lens{scenario=gpu-graph.graph-local-clustering-coefficient rung=result caption="Compare clustering across 485 stations; 115 stations have at least one closed neighbor pair."}

**Question: Do this vertex's neighbors actually connect to one another, or do they only share the
same central contact?**

\`GPUGraphLocalClusteringCoefficient\` distinguishes a tightly connected friend circle from a loose
hub. A person with ten friends does not necessarily belong to a close community: if those friends
also know one another, local clustering is high; if none of them connect, it is zero. Use the
coefficient to identify closed friendship circles, mutually connected transaction accounts,
strongly interlinked services, or citation neighborhoods with many shared relationships.

Choose degree when you only need the number of direct relationships, local clustering when the
relationships *among those neighbors* matter, and label propagation when you need one community
identifier describing a larger connected region. The coefficient describes each vertex's immediate
surroundings; it does not assign community membership or optimize a global clustering objective.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| \`id\` | \`string\` | \`'gpu-graph-local-clustering-coefficient'\` | Prefix for generated command-graph node and imported-resource identifiers. |
| \`topology\` | \`GPUGraphTopology\` | Required | Existing GPU adjacency; directed weak neighborhoods require reverse CSR. |
| \`output\` | \`GPUVector<'float32'>\` | Required | Caller-owned floating-point clustering coefficient per vertex. |
| \`triangles\` | \`GPUVector<'uint32'>\` | Omitted | Optional per-vertex triangle count or directed neighbor-edge closure count. |
| \`algorithm\` | \`'auto'\`, \`'direct'\` or \`'canonical'\` | \`'auto'\` | Direct suits sparse low-degree graphs; canonical suits dense or duplicate-heavy graphs. |

\`output\` is a caller-owned, packed \`GPUVector<'float32'>\` with exactly one row per vertex. The
optional \`triangles\` result is a separate, caller-owned, packed \`GPUVector<'uint32'>\` with one
row per vertex. Let \`d\` be the number of distinct incoming-or-outgoing neighbors. For a directed
graph, count every distinct **directed** edge between those neighbors as \`C\`; the coefficient is
\`C / (d × (d - 1))\`, and the optional \`triangles\` result contains that directed closure count \`C\`.
Reciprocal neighbor relationships count as two directed closures, while repeated copies of the
same directed edge count once. For an undirected graph, let \`T\` be the number of unique incident
triangles: its coefficient is \`2 × T / (d × (d - 1))\`, and the optional \`triangles\` result contains
\`T\`. A vertex with fewer than two distinct neighbors has coefficient and closure count zero.

The neighbor set is an undirected, or **weak**, neighborhood: either an incoming or outgoing edge
introduces the same neighbor. Its closure numerator still preserves direction for directed graphs;
the two directions of a reciprocal pair are not silently collapsed. Directed graphs therefore
require both forward and reverse CSR. Undirected graphs reuse their symmetric forward adjacency.
Self-loops never make a vertex its own neighbor, and duplicate edges never duplicate neighbors,
directed closures, or possible neighbor pairs. Edge weights do not change this structural
coefficient.

The contributor writes all results on the GPU without sorting adjacency, repacking edge columns,
allocating a graph-owned scratch buffer, submitting commands, or reading results back. If either
required adjacency overflows, or a closure count cannot fit in \`uint32\`, every affected
coefficient fails closed to zero; optional triangle counts become \`0xffffffff\`. Empty graphs have
no output rows. Because source adjacency is intentionally unsorted, exact neighbor deduplication
and membership scans can require \`O(sum(degree³))\` work across all vertices. Dense regions and
high-degree hubs can therefore be substantially more expensive than degree or traversal; this is
not a claim of constant-time triangle counting, weighted clustering, or global community
detection.

#### How it computes

{/* lens:GPUGraphLocalClusteringCoefficient */}

::lens{scenario=gpu-graph.graph-local-clustering-coefficient rung=kernel caption="Inspect the direct kernel that checks every neighbor pair of each vertex for a closing edge."}
::lens{scenario=gpu-graph.graph-local-clustering-coefficient rung=graph caption="Follow topology construction into the single clustering pass over its CSR."}
::lens{scenario=gpu-graph.graph-local-clustering-coefficient rung=datum datum=26 caption="Inspect Amsterdam Centraal, station 26, with 13 neighbors and a coefficient of 0.115."}
::lens{scenario=gpu-graph.graph-local-clustering-coefficient rung=formula caption="Follow the closure count and its normalization to the WGSL buffers that hold them."}

## Rank incoming influence with GPUGraphPageRank

Rank graph vertices by incoming influence, redistributing dangling mass and normalizing scores after each bounded iteration.

::lens{scenario=gpu-graph.graph-page-rank rung=result caption="Compare incoming influence across 485 stations; the highlighted station has the highest final rank."}

**Question: Which vertices receive influence from other important vertices?**

\`GPUGraphPageRank\` estimates vertex importance from the importance flowing through incoming
relationships. A citation from an influential paper or a dependency from an important package can
matter more than many links from otherwise disconnected vertices.

Use PageRank when raw degree is not enough: prioritize influential accounts, rank connected
documents, identify widely depended-on services, or choose salient vertices for application-owned
visualization. The metric is unweighted even when the source topology retains edge-weight columns.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| \`id\` | \`string\` | \`'gpu-graph-page-rank'\` | Prefix for generated command-graph nodes and imported resources. |
| \`topology\` | \`GPUGraphTopology\` or \`GPUGraphTopologyView\` | Required | Existing GPU topology; directed graphs require reverse adjacency. |
| \`output\` | \`GPUVector<'float32'>\` or \`GraphDataView<'float32'>\` | Required | Caller-owned packed score column, one row per vertex. |
| \`damping\` | \`number\` | \`0.85\` | Probability of following an outgoing edge rather than teleporting, between zero and one. |
| \`iterations\` | \`number\` | \`40\` | Bounded number of compiled, normalized iterations; integer from 1 through 1024. |
| \`residual\` | \`GPUVector<'float32'>\` or \`GraphDataView<'float32'>\` | Omitted | Optional caller-owned scalar receiving the final iteration's absolute rank change. |

In a directed graph, \`paper A -> paper B\` contributes influence from A to B. A paper cited by one
highly influential source can outrank a paper cited by several obscure sources. Degree would count
those citations without asking how influential their sources are; PageRank propagates that
additional context through the surrounding network.

Directed graphs require reverse CSR so each vertex can gather incoming influence; undirected graphs
reuse their symmetric forward adjacency. Every fixed iteration redistributes probability from
dangling vertices with no outgoing edges, applies teleportation, and normalizes the published
\`float32\` scores so their total is approximately one.

The default damping is \`0.85\`: each iteration models an 85% chance of following an outgoing link
and a 15% chance of jumping to a uniformly chosen vertex. This prevents disconnected or cyclic
regions from permanently trapping all influence. A dangling vertex has no outgoing link to follow;
its influence is redistributed uniformly instead of disappearing from the probability vector.
The default bounded iteration count is \`40\`.

The optional one-row \`float32\` \`residual\` reports the final iteration's L1 score change: the sum of
absolute differences between the last two normalized score vectors. It is an observable error
signal, not an automatic convergence threshold, early-termination mechanism, or promise that a
fixed budget reached the stationary distribution. Reductions use subgroup additions when the
max-feature device exposes both required WebGPU capabilities, with a portable workgroup fallback.
Both paths use ordinary \`float32\` arithmetic, not floating-point atomics or native GPU \`float64\`.

Overflow in either required adjacency publishes zero scores and zero residual. Invalid normalization
totals also fail closed to zero. Parallel edges retain their multiplicity in this unweighted update.

#### How it computes

{/* lens:GPUGraphPageRank */}

::lens{scenario=gpu-graph.graph-page-rank rung=kernel caption="Inspect the incoming gather, dangling-mass reduction, and normalization kernels."}
::lens{scenario=gpu-graph.graph-page-rank rung=graph caption="Follow topology construction into 40 repeated gather, reduction, and normalization rounds."}
::lens{scenario=gpu-graph.graph-page-rank rung=datum datum=409 caption="Inspect Utrecht Centraal, station 409, alongside its CSR neighborhood and final score."}
::lens{scenario=gpu-graph.graph-page-rank rung=formula caption="Follow dangling-mass redistribution and the normalization denominator to their WGSL bindings."}
::lens{scenario=gpu-graph.graph-page-rank rung=iterations caption="Compare eight recorded score frames through the final round; no intermediate states are interpolated."}

## Related pages

- [GPU Graph overview](/docs/api-reference/experimental/gpu-graph)
- [GPU Graph operations index](/docs/api-reference/experimental/gpu-graph-operations)
- [GPU Core](/docs/api-reference/experimental/gpu-core)
`;export{e as default};