## Is the world's airline network organised by continent? {#question}

The map shows 3,257 airports coloured by continent and about 19,000 airline route pairs drawn faintly. You can see that routes are dense inside Europe, North America and East Asia, but a map cannot answer the structural question: *how much of the network stays inside a region, and which airports tie regions together?*

That is a job for a **matrix view** of the graph and for **coarsening**; **Show** below switches between the map and the matrix. This scene uses three contributors on one compiled graph: `GPUAdjacencyMatrixOrder` arranges the airports, `GPUAdjacencyMatrix` draws who-connects-to-whom as an image, and `GPUNetworkCoarsening` collapses the network into one node per continent or country. The data is OpenFlights, frozen around 2014, so read it as structure, not current traffic. The histogram below counts airports by number of routes: most have a handful, a few hubs have hundreds, and those hubs are what the matrix will light up.

## The same graph as a matrix {#matrix-input}

In an adjacency matrix, row *i* and column *j* hold a mark when airport *i* has a route to airport *j*. `GPUAdjacencyMatrix` bins the pairs into a 512 by 512 image on the GPU; each cell counts the pairs that fall in it, and brighter means more. Because the graph is undirected the picture is symmetric about the diagonal.

Here airports are in input order (**Order airports by**), alphabetical by IATA code, which has nothing to do with geography. The result is a speckle: the connections are there, but no pattern is readable. A matrix is only as informative as its ordering. The chart of cell fill explains the dark: nearly every occupied cell holds only one or two pairs.

## Order by continent {#order-continent}

`GPUAdjacencyMatrixOrder` computes a permutation: it sorts airports by a **group** label (here the continent) and returns each airport's matrix position. Two stable GPU sorts do the work, and the result is bit-identical to the CPU helper `computeAdjacencyMatrixOrder`; the **Order check** readout below compares them on every change.

Now blocks appear. The coloured strips along the top and left are the continents. Bright squares on the diagonal are routes that stay inside a continent; the sparse rectangles off the diagonal are routes between continents. The ordering is a parameter buffer, so switching it never recompiles.

## Country blocks, hubs first {#order-country}

**Order airports by** is now **country within continent, hubs first**: a **tie key** of descending degree is added, inside each block the best-connected airports come first. The tie key is the second input of `GPUAdjacencyMatrixOrder`; it is sorted first, then the group, so groups stay contiguous and ties break by degree.

The effect is a bright, dense corner in every block: hubs are connected to nearly everything in their country, while the long tail of small airports is connected to few. Thin lines mark country boundaries inside the continent blocks. Hover a cell for the two airports and the number of pairs the cell holds.

## Zoom into one block {#zoom-europe}

The matrix has a **window**: four numbers, `[rowStart, rowEnd, colStart, colEnd)`, in matrix positions. Zooming rewrites them and the next encoding re-bins the visible part, so a 512-cell image can show 600 airports at several cells each, without recompiling.

Focused on Europe, the country structure inside the block is readable: the large German, French, Spanish and Italian networks, dense with themselves and with each other, against thin links to small countries. Pick another continent in **Focus block**, or switch it to *Manual zoom and pan* and use **Zoom**, **Pan columns** and **Pan rows** (disabled while a block is focused), or raise the **Resolution** (a compile-time option) to 2048 for finer cells.

## Collapse the network: coarsening {#coarsen}

`GPUNetworkCoarsening` summarises the graph by a label: one **supernode** per group (airport count, mean position, summed degree) and one **superedge** per pair of groups (route count and weight). Discs are sized by airports, arcs by routes between the two groups.

The readout **Routes inside their group** is the share of the diagonal blocks: for continents, most routes stay inside the continent they start in, and the thin arcs between continents are the long-haul backbone. Switch **Coarsen by** below to country for about two hundred supernodes. Sums are accumulated in fixed point, so the numbers are identical on every run, and the **Dropped (label overflow)** readout would show any label outside the compile-time capacity.

## Limits and things to try {#limits}

**Caveats.** A matrix cell at 512 bins holds several airports each, so a bright cell means a dense neighbourhood, not one route. Block order depends on the grouping you choose: country within continent is a convention, and Russia and Turkey are placed in Europe here. Supernode positions are mean longitude and latitude, so countries with distant territories (the US with Hawaii and Guam, Russia) are pulled toward their middle. OpenFlights is a community dataset frozen near 2014.

**Try it.** Set **Order airports by** to *Degree only (hubs first)* for a hub-first matrix with no groups, set **Cell value** to *Weight sum*, then switch **Edge weight** to *Airlines serving the pair* or *Distance*, and compare **Cells occupied**. Turn **Show** to *Map and matrix* to see both at once: the matrix card sits over the Atlantic, centred on longitude 0, latitude 0.
