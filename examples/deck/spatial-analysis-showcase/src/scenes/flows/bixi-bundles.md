## What does a month of Montreal bike trips look like without the hairball? {#the-question}

Each line is a pair of BIXI stations that riders connected in August 2024, drawn straight between the two docks. The 30,000 busiest pairs carry 77% of all rides between stations, and drawn like this they are an unreadable web: the lines pile up downtown and hide every pattern. **Iterations** below is at zero, so you are looking at the straight edges.

The question is how to show *where the traffic concentrates* without the clutter. `GPUEdgeBundling` pulls lines that run the same way toward each other until they share a corridor, the way a good transit map does.

## Pull compatible lines together {#bundle}

`GPUEdgeBundling` is kernel-density edge bundling. Every edge is a polyline of control points. Each iteration splats the points onto a density grid, moves every point uphill along that density, resamples the polyline and smooths it. Lines that start and end near each other are carried to the same ridge and merge; lines going elsewhere stay apart. Drag **Iterations** from 0 to 16 below: the corridors form.

**Kernel radius** is how far apart two lines can be and still attract (a fraction of the map width): small radii keep the neighbourhood structure, large ones merge whole districts. **Edges used** picks how many of the busiest pairs go in. The readouts count the cost and the benefit: the ink saved (map cells the lines no longer touch) and how much longer a bundled pair is than its straight line, with the histogram showing the spread.

## Tune the bundles {#tune}

**Radius decay** shrinks the kernel every iteration, so the first rounds make coarse corridors and later ones tighten them; **Stiffness** is the smoothing after each step, and a low value leaves kinks. **Control points per edge** and **Density grid** are compile-time: a finer grid resolves smaller gaps between bundles, more points follow tighter bends, and each change rebuilds the graph.

**Colour edges by** decides what the ramp shows: rides on the pair (square-root scale), the length of the pair, or whether the two stations are in different boroughs. The last is revealing: the corridors that cross borough lines are the commuter arteries between the Plateau, Mile End and downtown. **Minimum rides on a pair** thins the weak edges out of both the density and the picture.

## The morning on the corridors {#rides}

Turn on the rides: 8,365 routed BIXI rides on 15 August 2024 between 07:30 and 10:00, running over the bundles. A `GPUTrajectoryPlayhead` graph interpolates every ride at the clock and a `GPUTimeWindowFilter` graph selects the trail segments of the last few minutes. Each bike is coloured by its speed, from dark (slow, climbing or waiting) to yellow (fast).

Press **Play** and watch the **rides in progress** chart climb to about 1,300 at 09:00. **Speed** is minutes of the morning per real second. The bikes follow the corridors the bundling found: they were routed over the street network, so they run along the same avenues and the Rue Rachel and Maisonneuve cycle tracks. **Trail length** and **Trail fade** set how far behind each bike the trail runs.

## Limits and things to try {#limits}

**Caveats.** The routes are derived, not GPS: the archive routes each ride on the bike network with OSRM between its start and end station and spreads the time along the route, so speeds are modelled and a ride that took a detour is drawn on the shortest path. Bundling shows the pairs, not the street network, and its result depends on the kernel settings and is chaotic in 32-bit floats. Rebalancing trucks are not in the data. Rides still underway at 07:30 begin at the first frame. The corridors use August pairs from the whole month; the animation is one day.

**Try it.** Set **Colour edges by** to borough crossing and turn **Straight edges** on to see what bundling removed. Raise **Kernel radius** to 0.06 and watch the districts merge. Hide the stations and the bundles, and let only the rides run: the street network appears from the bikes alone.
