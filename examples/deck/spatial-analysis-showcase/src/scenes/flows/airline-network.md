## Which airports are the backbone of the world's airline network? {#backbone}

Every airport pair in OpenFlights is an edge of an undirected graph: 3,257 vertices, 18,930 edges. Connections per airport (`GPUGraphDegree`) ranks the big European hubs first, but "most connections" is not the same as "most central". `GPUGraphPageRank` asks a different question: which airports are connected to other airports that are themselves well connected, with a damping of 0.85 that models a traveller who occasionally jumps anywhere.

Discs are sized by the square root of PageRank, and coloured by continent for now. Read the bar chart below: bars drawn in the accent colour are airports in the PageRank top 20 that are **not** among the 20 most connected airports. Try **Disc size by** below and flip between PageRank, connections and core number (`GPUGraphCoreNumber`, the deepest "everyone has at least k partners" shell the airport belongs to). The degree histogram shows why hubs matter: the median airport has 3 connections, the busiest 248, and almost a quarter of airports are dead-end spokes with a single route.

## Do the communities follow the continents? {#communities}

`GPUGraphLabelPropagation` gives every airport a label and lets each one repeatedly adopt the most common label among its neighbours, until groups stabilise. It knows nothing about maps or countries: it only sees which airports are connected. The colours are the communities it found, each named for its highest-PageRank airport.

Compare them with continents. **Continent purity** is the share of airports sitting on the dominant continent of their community, and `GPUGraphModularity` scores both partitions with the same yardstick (Newman modularity: how much more route-internal weight a grouping keeps than a degree-matched random network would). Flip **Colour groups by** below between *Community* and *Continent* and watch where the colours disagree: communities mostly follow continents, but some cross borders: the hub communities around Istanbul and Dubai are well below 100% on one continent (see the community list). Remember the legend: red-orange routes are the ones that cross between groups.

## Resolution: how fine should a community be? {#resolution}

Label propagation is a heuristic with no knob. `GPUGraphModularityOptimization` starts from its result and moves one airport at a time into the neighbouring community that raises modularity most. Its **Resolution** gamma sets the baseline: low values favour a few large communities, high values many small ones. The line chart sweeps eight resolutions, one compiled graph each, and scores three partitions at every one: the optimized communities, plain label propagation, and the six continents.

Slide **Resolution** below (it rebuilds one graph) and read the dot on the curve. The surprise is how competitive the continents are: at coarse resolutions the six continents out-score plain label propagation, and at gamma 1 the two are within a hundredth of each other. See where the optimized curve separates from the continents curve. Geography is a strong prior for who flies where. **Rounds** is the budget of moves (one per round) and **Minimum gain** the improvement a move must beat; the status readout says whether the optimizer reached a local optimum or just ran out of rounds.

## Edge bundling, coloured by community {#bundles}

Eighteen thousand great-circle arcs are unreadable. `GPUEdgeBundling` pulls routes that run side by side into shared corridors (kernel-density bundling, as in the flight-bundling scene), and here each bundle is coloured by the community of its endpoints. Routes within a community keep that community's colour; routes between two communities are red-orange, so the inter-community links stand out against the regional fabric.

Switch **Routes shown** below to *Between groups only* to keep just the bridges: these are the long-haul and gateway routes that tie the continental communities together, and they funnel through a short list of airports. Raise **Bundle iterations** or **Kernel radius** to merge them harder. The bundles are straight lines in longitude and latitude, not great circles, so treat their curvature as grouping, not as flight paths.

## Geography versus topology: morph to the force layout {#morph}

`GPUGraphForceLayout` ignores coordinates entirely. Every airport repels every other, every route is a spring, and a gentle gravity keeps the cloud together; the simulation runs on the GPU and writes a positions buffer that is also a vertex buffer. The scene morphs each airport from its real position to its layout position and back (this view draws straight lines between the current positions; routes across the Pacific fade in as their ends leave the map).

Press **Animate morph** below, or drag **Morph** to scrub. Geography is a poor guide to the network's structure: expect hubs to gather towards the middle, spokes to spread out around them, and regions to pull apart. **Edge length correlation** is the Pearson correlation between how long each route is on the map and in the layout; a low value means geography explains little of what the layout considers close. **Layout steps** counts the simulation steps so far; the layout keeps settling while it runs. The force constants rebuild the layout graph and keep its current state; **Reset layout** below starts again.

## Honest limits, and things to try {#limits}

**Caveats.** OpenFlights is community-maintained and largely frozen around June 2014: no post-2014 routes or carriers, and a route pair exists or it does not. The graph is unweighted by frequency, so a once-weekly regional service counts the same as a trunk route. Airline identity is not in this dataset (only the number of airlines per pair), so the question of whether communities follow airline alliances cannot be tested here. Modularity optimization is single-level local moving with one move per round, not Louvain or Leiden, and float32 atomic accumulation means near-ties can resolve differently on another GPU. Label propagation and core numbers report whether they converged; the layout is a picture, not a measurement, and the spatial approximation freezes if airports leave its bounds.

**Try it.** Raise **Minimum route records** to keep only pairs with several airline-route records. Switch the layout to *Spatial approximation* and compare its status readout with the exact one. Move **PageRank damping** towards 1 and see the ranking sharpen towards the dense core, or towards 0 and see it collapse to the degree ordering.
