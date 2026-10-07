## Where does Manhattan fill up and empty? {#the-question}

Every taxi trip moves a passenger from one place to another, so each hexagon on the map gains passengers when trips end in it and loses them when trips start there. **Net balance** is arrivals minus departures in the time window: red hexagons took in more riders than they sent out, blue ones sent out more, and white is balance. The question is the one a dispatcher or a transport planner asks: *where does the city's population shift during the day?*

This is Friday 2 January 2015, 08:00 to 09:00. `GPUFlowAggregation` assigns each of 440,000 trips to a **hexagon lattice** by its pickup and dropoff point, gates them by a **time window**, and sums trips per hexagon. The arcs are the 60 biggest flows of the window. Slide **Window start** below to move the window.

## The Friday morning pull {#morning-rush}

In this hour Midtown gains riders: the hexagon over Midtown West (around Columbus Circle and 57th Street) takes in a net 611 trips and the one over Midtown East (Grand Central to the United Nations) a net 465, out of roughly 8,000 trips citywide. The biggest loser is the Upper East Side, which sends out a net 436. The arcs, drawn from origin (orange) to destination (cyan), show the heaviest flows of the hour.

Widen **Window length** to three hours (07:00 to 10:00) and the pattern holds: Columbus Circle gains 1,256. Change **Weight** to fare or passengers to see where the *money* and the *riders* go rather than the trip count; it is a buffer write, so the graph does not recompile.

## Press play: a day in net arrivals {#play-the-day}

The playback clock moves the window start across the 38 hours and writes the slider back; each step changes only the eight numbers of a window buffer, and both graphs run again. Watch Midtown turn red on Friday morning and slowly blue in the evening: the chart plots the hourly net of a Midtown box, and the marker follows the clock.

In the Midtown box the net rises to +453 and +546 trips at 07:00 and 08:00 on Friday, then falls back. On Thursday, New Year's Day, the same box loses riders all evening: -340 at 18:00, -385 at 19:00 and -667 at 20:00. Holiday traffic goes *out* of the core, not into it. **Playback speed** below sets how many hours of data pass each second.

## Arrivals and departures are separate clocks {#two-clocks}

A trip departs at its pickup time and arrives at its dropoff time, minutes later. This scene therefore runs **two** aggregations over the same trip pairs: departures are gated by pickup time and arrivals by dropoff time, and the net is arrivals minus departures per hexagon. Switch **Show** to departures or arrivals to see each side alone: the sequential ramp shows volume instead of balance.

Departures and arrivals in a window never match exactly, because trips are in flight across the window edge. The readouts show both totals. The dropoff time is the pickup time plus the routed duration (a derived value), so arrivals are only as good as the routing.

## Hexagon size changes the answer {#hexagon-size}

Drag **Hexagon radius** between 400 m and 3 km. The lattice capacity is fixed when the graph is built (sized for 400 m), but the radius and the active lattice size are per-frame parameters, so dragging never recompiles. At 3 km the hexagons average the Midtown peaks together; at 400 m single blocks stand out.

The choice of zones changes what you conclude: this is the modifiable areal unit problem. **Zones** also offers a square grid, a compile-time choice with a rebuild badge. Hexagons have six equidistant neighbours, so adjacent cells compare fairly; squares do not.

## Airports are the evening sources {#airports}

Move to Thursday 18:00 to 21:00. Midtown's balance is now small (no hexagon gains more than 250), and the strongest signal is at the airports: LaGuardia sends out a net 799 trips and JFK about 450 per hexagon, because returning holiday travellers take taxis away from the terminals and few taxis arrive. The arcs fan out from the airports into the boroughs.

Try **Show** with departures or arrivals, or change **Arcs** to hide the flows. Because the lattice is fine, a single terminal is one hexagon.

## Limits and what to trust {#limits}

**Caveats.** Yellow taxis only, in 2015, before ride-hail dominated; they are a sample of movement (440,000 of the day's trips) and skew to Manhattan. The archive begins at midnight, so trips that began on 31 December are missing and the arrivals of the first hour are incomplete (the slider starts at 01:00, and the chart leaves the first and last hours blank). Dropoff times come from routes, not meters.

**Exclude same-zone trips** below keeps trips inside a hexagon out of the arcs and the totals; for the net it makes no difference, because such trips cancel. **Summation order** chooses between a deterministic sorted sum and an atomic sum whose rounding depends on GPU scheduling; both are compile-time options.
