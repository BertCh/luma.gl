## Where do taxi rides begin? {#the-question}

Each of the **440,000 trips** has a pickup and a dropoff. `GPUPointDensity` counts the pickups that fall in each cell of a grid laid over the screen, then blurs the counts slightly (**Smoothing** below) so the picture reads as a surface instead of speckle. Brighter cells have more pickups; the scale is a square root so quiet neighbourhoods stay visible next to the hot spot.

The hottest cell is around Times Square and Bryant Park: at a 1 km hexagon size it holds about 62,000 pickups, 14% of every trip in the data. Two more cells, the Village and the Columbus Circle area, hold about 40,000 each. Switch **Show** to dropoffs and the picture is almost the same, which is the next question: where is it *not* the same?

## The grid follows the screen {#follow-the-camera}

Zoom into Midtown. The grid is not a fixed raster: every frame the four-number **bounds** parameter buffer is rewritten from the camera, so the 110 × 70 cells always span what you see and the **Cell size** readout shrinks as you zoom. The **Resolution** select below sets how many cells there are. That is a compile-time `gridSize`, so each resolution is a separately compiled graph that the scene keeps.

At this zoom a cell is a few blocks. Hexagons (**Cell shape**) avoid the grid-aligned look of squares and are not smoothed; the square grid can be blurred with a Gaussian kernel whose weights are a parameter buffer, so **Smoothing radius** never recompiles.

## Pickups against dropoffs: the balance map {#balance}

**Balance** gives each pickup a weight of -1 and each dropoff +1 in the same 880,000-point buffer and sums them per cell, so a cell is positive where more rides end than start. Warm cells are *dropoff surplus*, cool cells *pickup surplus*, and each sign has its own ramp and legend.

Over both days, the Upper Manhattan hexagons (East Harlem, Morningside Heights and the northern Upper East Side, about +1,300 to +2,000 per 1 km hexagon) are net destinations. The nightlife neighbourhoods are net origins, the Village at about -8,600, the Lower East Side -6,100 and Hell's Kitchen -5,100, and so is JFK at about -3,100. That is the shape of a holiday night out: rides start where the bars and the airport are and end in residential neighbourhoods.

## Where a ride costs the most {#fare}

Change **Measure** to fare and **Statistic** to average: each cell now shows the *mean fare* of the trips that start there, the weights buffer divided by the count. The two JFK cells stand out at about $45 to $46 a ride, LaGuardia is about $28, and almost everywhere in Manhattan the average is $10 to $13.

Pickup density says where demand is; mean fare says how far it travels. Midtown has by far the most rides, and some of the shortest. **Total** returns to sums, so choosing fare shows where the revenue is collected (still in Midtown, by volume) rather than the average.

## Ask "when?" with a mask {#clock}

Pick an **Hour of day** window and the GPU skips every point outside it: the mask buffer is rewritten, the graph is not rebuilt. Here the window is the first two hours of 2015. The hot spot is still Midtown (about 5,500 pickups in a 1 km cell) but the Village (about 4,700) and the Lower East Side (about 3,300) are close behind, while on Friday from 07:00 to 10:00 Midtown holds about 4,000 and nothing else comes close.

The line chart shows pickups and dropoffs for every hour; the markers show your window. **Outside that window** keeps the hours outside the range instead, which is how you ask for an evening or a night.

## What to trust {#limits}

**Caveats.** These are yellow taxis only, in 2015, before ride-hail dominated; they under-serve the outer boroughs, so quiet cells there mean "few yellow taxis". The 440,000 trips are the first 38 hours of 2015, which include New Year's night, so the nightlife signal is stronger than on an ordinary day. Dropoffs are the end of an OSRM route and snap to the road network.

Numbers quoted in the text are for a 1 km hexagon lattice over the whole timeline; this scene's lattice follows the screen, so the numbers in the readouts differ a little with zoom and resolution.
