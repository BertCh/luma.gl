## Does Montreal's bike network split into neighbourhoods of riding? {#the-question}

Every dot is a BIXI station and every line is a pair of stations that riders connect: the 1.93 million rides of August 2024, between 905 stations, summed over both directions. Colours are **communities**: groups of stations that exchange more rides with each other than with the rest. Lines between two stations of the same community take that community's colour; lines that cross a boundary are faint grey.

The question is the one a network planner asks before drawing service areas or rebalancing routes: *do the administrative boroughs match how people actually ride?* `GPUGraphLabelPropagation` and `GPUGraphModularityOptimization` answer it on the GPU, from the same graph the lines come from. **Partition shown** and **Strongest links drawn** below switch the view.

## Label propagation proposes groups {#propagation}

`GPUGraphLabelPropagation` starts with every station in its own group. Each round, a station adopts the label most common among its neighbours, ties going to the lowest label, and after a few rounds the labels settle into groups. It ignores edge weights and does not optimise anything, so it is cheap and a good first guess.

A graph with 88,000 station pairs is too dense for that vote, because every busy station neighbours every other. Set **Strongest links per station** below to 60 and one group swallows more than half of all stations; with every pair it would swallow nearly all of them. Keep only each station's strongest links (the default is 8: an edge stays when it is among the strongest links of either end) and about 25 distinct groups appear. **Label propagation rounds** is a compile-time constant; **Minimum rides on a pair** drops weak pairs in a buffer write. The readout says whether the vote converged inside the budget.

## Modularity optimization refines them {#refine}

`GPUGraphModularityOptimization` takes the propagation as a starting partition and improves it against an objective: weighted **modularity**, the share of rides that stay inside groups minus the share a random graph with the same station degrees would keep inside. Each round accepts the single best move of one station to a neighbouring community, and only if it strictly raises the score. This is single-level local moving, not full Louvain, and one move per round means the **Refinement rounds** budget matters: look at the convergence readout and the quality chart below.

**Resolution** scales the penalty: below 1 it favours a few large communities, above 1 many small ones. **Edge weighting** decides whether a pair counts by its rides, their square root, or equally. Both are checked against the borough partition at the same resolution by `GPUGraphModularity`.

## Communities against boroughs {#vs-boroughs}

Switch **Partition shown** to boroughs and back. The borough partition scores lower modularity than the refined communities on the same graph, and slightly fewer of the rides it draws stay inside its groups: the readouts give both shares and a normalised mutual information between the two partitions (1 means identical, 0 means unrelated).

The two agree on the big picture and disagree at the seams. In a reference run the largest community is the Plateau with most of Rosemont's edge stations; another joins half of Ville-Marie (the Old Port and Griffintown side) with the Sud-Ouest along the canal; Verdun and LaSalle form a pocket together; Villeray joins Ahuntsic across the rail line; Côte-des-Neiges pairs with Outremont; and Ville-Marie itself splits into several groups. Click a station to see how much of its riding stays inside its community.

## Limits and things to try {#limits}

**Caveats.** The graph is built from stations that rides start and end at, so it describes where BIXI is docked, not where people want to go. Rebalancing trucks are not in the trip data, so a station that is always refilled looks busier than it is. Only trips of at least a minute count. The communities are a heuristic: label propagation is order-free but not optimal, the refinement stops at a local optimum, and the weighted sums use atomic float additions, so the last digits can differ between GPUs.

**Try it.** Raise **Resolution** to 2 and watch communities split along streets. Set **Edge weighting** to equal and see the optimised score fall. Lower **Refinement rounds** to 64 and read the convergence readout: the refinement has not finished. Turn **Between communities** off to leave only the inside of each group.
