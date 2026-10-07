## Which routes do the world's airlines really share? {#hairball}

Here are 18,930 airline route pairs between 3,257 airports, each drawn as a straight line. It is the classic hairball: you can see that Europe, the US and East Asia are dense, but you cannot see *where* the traffic travels, because thousands of lines cross the same ground.

The analyst's question is about **corridors**: which stretches of the world does a lot of traffic share (the North Atlantic, the Europe to Asia belt), and which airports are the junctions? `GPUEdgeBundling` answers it by pulling nearby edges together. At this step **Iterations** (below) is set to 0, so you are looking at the raw input. **Colour edges by** is set to route distance: short regional hops are purple, long-haul routes yellow. Orange dots are the 24 best connected airports.

## Bundle them: kernel-density edge bundling {#bundle}

`GPUEdgeBundling` implements kernel-density edge bundling (KDEEB). Every edge is a polyline of 16 control points with its two endpoints pinned. Each **iteration** splats the density of all control points onto a grid, moves every interior point a step *uphill* along the density gradient (towards where other edges already are), resamples the line to even spacing, and smooths it with a Laplacian pass. The search radius shrinks a little every iteration, so lines first merge into broad bundles and then tighten.

Set to 15 iterations, the hairball resolves into corridors: the North Atlantic, Europe to the Middle East and South Asia, and the dense lattice over China and Japan. Press **Play** to sweep the iteration count from 0 to 32 and watch the corridors condense out of the hairball. The **Iterations** slider below is a parameter-buffer write; the compiled graph is the same one that drew the straight lines. Pairs that span the antimeridian are split in two so each half leaves the map on its own side.

## Kernel radius: how far apart can two edges merge? {#kernel}

The **Kernel radius** is the distance, as a fraction of the map width, within which control points attract each other. A small radius merges only edges that already run side by side; a larger one pulls in distant edges and gives thicker, fewer bundles.

This step raises it to 0.04 (drag **Kernel radius** below to compare, and **Radius decay** for how fast it shrinks): the North Atlantic becomes a single trunk and the intra-Asia network fuses into a few long arteries. The histogram of **path length over straight length** shows the price in distance: most routes grow by only a few percent, and a tail is dragged into a distant trunk. Detail is the price too. A bundle hides which individual edges are inside it, so use bundling for the *structure* of the network, not for reading single routes. Compare the readouts **Mean path stretch** (how much longer bundled lines are than straight ones) and **Map cells: bundled**, how many half-degree map cells the lines cross; **Map coverage removed** is the saving against straight lines.

## Zoom to one region {#europe}

Filters are an **edge mask**: a per-edge on/off buffer that the contributor honours without recompiling, and masked edges leave the density and the work box too. Here **Region** (below) is set to Europe, so only edges with both endpoints in Europe are live.

The bundling is recomputed on those edges alone, so the corridors inside Europe come out sharper than they were in the world picture: the Frankfurt, Amsterdam, Paris and Istanbul hubs each anchor a fan, and the Iberia to Scandinavia axis shows as a single band. Hover an airport to see how many airports it serves; in this 2014 snapshot AMS reaches 248, FRA 244, CDG 240 and IST 235.

## The long-haul backbone {#long-haul}

Edges are also filtered by distance. With **Route distance** (below) at 3,000 km and above, only intercontinental routes stay live: the short routes that make up 22% of all pairs (under 500 km) are masked, and what remains is the global backbone. The route-length histogram shows the cut: the short, tall bars on the left are gone.

Notice how few airports serve as junctions: the lines funnel through the Gulf, Europe's big hubs and the US coasts. Routes that cross the antimeridian leave one edge of the map and re-enter at the other. Bundling the long-haul layer alone makes the flows between continents far easier to follow than the combined hairball.

## US flights in July 2023, coloured by delay {#us-delay}

Now a second dataset on the same graph: **real US schedule data**, 601,582 flights in July 2023 between 335 airports, merged to undirected pairs. Colour is mean departure delay (flights-weighted over both directions), and the filter is set to the lower 48 states.

Delay is not spread evenly: the worst pairs run through Florida, the Northeast corridor and the big connecting hubs (the dataset's worst pair with at least 100 flights is LAS to BOS at 98 minutes of mean delay). Mean delay overall is 21.3 minutes and 2.4% of flights were cancelled. Switch **Colour edges by** to *Cancellation rate* for the other side of reliability.

## Limits and things to try {#limits}

**Caveats.** Bundling is a visual transform: bundled lines are not routes and their curvature carries no meaning, only their grouping does. Edges follow straight lines in longitude and latitude, not great circles. OpenFlights is community-maintained and largely frozen around 2014, so treat the world picture as network structure, not current traffic; the US data is real July 2023 scheduled flights from the BTS On-Time file for the largest carriers, not passengers. Float32 chaos means the bundles are visually stable but not bit-identical between GPUs.

**Try it.** Raise **Iterations** to 32 and **Stiffness (smoothing)** to 1 for smooth, tight arcs, or drop it to 0 for angular ones. The scene is back on the world routes. Set **Density grid** to 512 for crisper bundles (this one rebuilds the graph), set **Colour edges by** to *Traffic* to see which corridors carry the most route records, and turn on **Straight edges (ghost)** to compare with the original.
