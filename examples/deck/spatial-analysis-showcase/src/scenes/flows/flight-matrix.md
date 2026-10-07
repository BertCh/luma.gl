## Where is the structure? {#question}

**{{airports}}** airports joined by **{{routes}}** route pairs, drawn faintly on a night ground: routes inside a continent in its own hue, routes between continents in off-white. A few hubs, named from the data, carry the picture, but a map cannot say how much of the network stays inside a region. Switch **Routes** off to see the airports alone.

*Is the world's airline network organised by continent?*

## Unordered, the matrix is noise {#matrix}

The same graph with no geography: row *i*, column *j* is marked when airport *i* flies to airport *j*. `GPUAdjacencyMatrix` bins the pairs into **{{cellScale}}**; only **{{occupied}}** cells are occupied, at most **{{fillBound}}** could be. Airports are in alphabetical order, which has nothing to do with the world, so the marks scatter. A matrix is not geography, so it gets a paper sheet. Peek ahead with **Order airports by**.

## Sort by continent and blocks appear {#order}

`GPUAdjacencyMatrixOrder` turns a label per airport into a position with two stable GPU sorts, so **Order airports by** re-sorts the same cells and they slide to their new places. Routes inside a block now: **{{blockShare}}**. Try *Shuffle (null)*: the same block sizes with random labels spread the routes evenly across the blocks.

*The ordering is the analysis.*

## Far more routes stay home than chance predicts {#chance}

A big block may just be a big group. Keep every airport's number of routes, pair the route ends at random, and only **{{expectedShare}}** of routes would stay inside a continent, against **{{blockShare}}** observed: modularity **{{modularity}}**. The six-by-six chart divides observed by expected for each pair of continents; **Tint blocks** paints the same colours on the matrix. Compare it with *Shuffle (null)* under **Order airports by**.

*Normalise, don't count.*

## Inside Europe, countries form blocks {#one-block}

Focus on one continent and the window holds **{{blockAirports}}** airports, each cell **{{cellScale}}**. Raise **Resolution** past the window and the cells stop shrinking: a cell never holds less than one airport, so the resolution is clamped instead of striping. **Focus block** moves to another continent.

*Resolution is an aggregation unit.*

## Collapse it: continents and countries {#coarsen}

`GPUNetworkCoarsening` collapses the network to one supernode per group, on the group's busiest airport, and one superedge per pair of groups. By continent **{{shareContinent}}** of routes stay inside; by country only **{{shareCountry}}**. Switch **Coarsen by**; the busiest pair is **{{largestPair}}**.

*What did the aggregation hide? Pick the grouping and you pick how modular it looks.*
