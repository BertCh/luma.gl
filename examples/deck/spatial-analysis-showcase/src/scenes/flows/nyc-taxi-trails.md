## A rush hour, one cab at a time {#eight-oclock}

Each white dot is a yellow cab on the road right now, and the faint line behind it is the last stretch of its route. **{{active}}** cabs are moving at **{{clock}}**, drawn from a sample of **{{trips}}** trips. Where do they all go, and where do they gather? Press **Play** below, or change **Playback speed**.

## Longer trails turn dots into streets {#trail-is-memory}

A trail is the window of the last few minutes of each route. Press play next to **Trail length** and it sweeps from a short memory to a long one: dots become corridors, and **{{segments}}** route segments are drawn. The line fades with age, as the curve below shows, so the cab stays the brightest thing.

*A trail is the map's memory.*

## These lines are routes, not GPS {#routes-not-traces}

Every faint line is a modelled route: a routing engine joins each pickup to its drop-off, so no cab ever waits at a light. The dashed chords are the straight lines between the ends of the routes. The median route is **{{circuity}}**, and **{{detourShare}}** of trips make a clear detour. Scrub the **Clock**, or hide the **Chords**.

*Modelled routes never wait at a light.*

## Avenues move one way, in pairs {#heading}

Each route segment takes the colour of the direction it points, clockwise from north, on a cyclic ramp that ends where it starts. The grid stands out as opposite pairs: **{{dominantHeading}}**, and the rose counts the cabs by heading. Switch **Colour by** to bring back the single yellow.

*Direction is a cycle, so its ramp must be one.*

## Match the ramp to the ground {#fare-colour}

Fare classes, each holding an equal share of the trips. Left of the divider the ramp runs against the dark ground and the dearest trips sink into it; right, brightest means dearest. Drag the divider, or turn off **Compare ramps** and flip **Ramp** by hand.

*A sequential ramp must run toward contrast with its ground.*

## Every frame is a binary search per cab {#how-the-gpu-does-it}

One thread per cab searches its own timestamps for the clock, then interpolates between the two vertices around it: for this cab, **{{searchSteps}}**. A second pass keeps the segments inside the trail, **{{segments}}** now, and writes the draw count on the GPU. Press **Pick another long taxi**, change **Trail length**, or try **Colour by**.
