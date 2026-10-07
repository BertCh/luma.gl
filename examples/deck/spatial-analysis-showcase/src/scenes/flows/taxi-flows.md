## Where do Chicago's taxis carry people? {#the-question}

Every curved arc is one **flow**: all the 2023 taxi trips that began in one community area and ended in another. Orange is the pickup end, cyan the drop-off end, and the arc bends to the right of travel so a flow and its return trip do not overlap. The shaded areas show how many trips start in each area. **Flow dataset**, **Zone shading** and **Flows drawn** below change what you see.

The question is the one a transport planner or a dispatcher asks first: *which pairs of neighbourhoods carry the most traffic, and does that change through the day?* With 119,777 origin, destination, hour and day-type records the answer has to be computed, not eyeballed. `GPUFlowAggregation` does it on the GPU: it counts every pair, ranks them, and writes the heaviest as a list the arc layer reads directly.

## The top flows carry most of the story {#top-flows}

`GPUFlowAggregation` groups records by *(origin zone, destination zone)*, sums their weight, sorts the pairs by that sum and keeps the **top 512**. Here the weight is trips, the zones are Chicago's 77 community areas, and trips that start and end in the same area are excluded so the arcs show movement between places.

Slide **Flows drawn** below down to 25 and read **Share carried by drawn arcs**: a handful of pairs around the Loop, the Near North Side and O'Hare carry a large part of all inter-area trips. Width follows the square root of the flow, so the heaviest arc is wide without hiding the rest. The curve below is the same thing for every rank: the **cumulative share** of all flow carried by the largest flows, and the rule on it is how many arcs are drawn. If **Top-512 list truncated** says yes, more pairs exist than the list holds; the zone totals are still exact.

## Slide the time window {#time-window}

Each record carries a pickup hour, and `GPUFlowAggregation` accepts a **time window** that gates records on the GPU: the window is four numbers in a parameter buffer, so changing it re-runs the same compiled graph with no rebuild. This step shows weekday 17:00 to 20:00 (**Window start**, **Window length** and **Day type** below).

The shading now switches to **net balance** (arrivals minus departures, from the per-zone totals the contributor writes). On the diverging ramp a cool blue is an area that sends out more trips than it receives (O'Hare in this window), a warm red receives more than it sends (the Loop and the Near North Side), and the neutral midpoint is balance. The chart shows the day itself: the share of each day type's trips in every pickup hour, with rules on the window. Weekdays have a commuter peak at 17:00 to 19:00; weekends keep going through the night. Slide **Window start** to 6 with **Window length** 3 for the morning rush, or press **Play** to sweep the window across the day: the slider follows the clock, the shading and arcs re-rank as it moves, and **Loop** repeats it. Switch **Day type** to weekends and watch the downtown pull weaken.

## The same trips, different zones {#zone-size}

Zone choice changes the picture: this is the modifiable areal unit problem. Instead of community areas, trips are now assigned to a **hexagon lattice** built from each area's centroid, and the **Lattice size** slider is the hexagon radius. Both the lattice size and the radius are per-frame parameters under a compile-time capacity, so dragging the slider never recompiles.

At small radii every area keeps its own hexagon and the flows match the native map. At 4 to 6 km neighbouring areas merge, flows between them become same-zone trips and drop out, and the arcs consolidate into a few long corridors. Choose **Square grid** in **Zones** to see the other lattice; hexagons have six equidistant neighbours, which avoids the grid-aligned look squares can impose.

## Commuters: 40,000 tract-to-tract flows {#commute}

Now a different dataset on the same contributor: US Census **LODES** home-to-work flows between Chicago's 791 census tracts, the 40,000 largest by jobs. Zones are tracts, weight is jobs, and there is no timestamp, so the time window is off (the graph is compiled without one). The shading is **arrivals**: where the jobs are.

The Loop dominates: its busiest tract receives 169,415 of the 534,765 commuters in these flows, and the single largest flow (2,033 workers) ends there. Almost every arc converges on one point, yet even 150 arcs carry only about 15% of the flow: commuting is far more dispersed than taxi traffic. Raise **Flows drawn** to reveal the secondary centres: Near North, the Medical District, O'Hare and Midway. The concentration curve rises far more slowly than for taxis, which is the same fact as a chart.

## Who commutes where: low versus high earners {#earnings}

LODES splits jobs by monthly earnings, so **Weight (commute)** can be switched in place: **low** is up to $1,250 a month, **high** is over $3,333. This is a weight-buffer write, not a recompile. Compare the two: high earners' flows converge on the Loop and Near North, while low-earning jobs are spread across the whole city and its outer centres.

In the kept flows, 66% of jobs are high-earning and 11% low-earning. The readout **Share carried by drawn arcs** tells you how concentrated each group is: the same 150 arcs carry about 15% of all jobs but only about 6% of low-earning ones, so low-earning commutes are less funnelled into one place.

## Limits and things to try {#limits}

**Caveats.** Taxi trips are a sample of urban movement, not all of it: they skew to downtown, airports and night hours, and the city suppresses pickup or drop-off areas for privacy on about 11% of trips, which are missing here. Pairs of areas are compared by centroid, so arcs show *which* areas are linked, not routes. LODES blocks are noise-infused for privacy, and only the top 40,000 tract pairs are kept (80% of inter-tract jobs). Mean fare excludes tips and extras.

**Try it.** This step returns to the taxi data. Switch **Weight (taxi)** to *Fare revenue* to see where the money moves rather than the trips. Turn **Exclude same-zone flows** off and note how the busiest arc becomes a loop inside the Near North Side. Press **Compare summation orders**; the **Summation order** option picks between them: the default `sorted` order gives bitwise-identical sums on every run, while `atomic` can differ in the last digits.
