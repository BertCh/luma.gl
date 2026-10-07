## Do riders follow the borough map? {#the-question}

Each dot is a BIXI station, coloured by the riding group the GPU found: stations whose riders mostly ride to each other. There are **{{communityCount}}** groups; the largest is {{largestCommunity}}, and modularity is **{{modularityRefined}}**. Lines inside a group take its hue. Thin the links with **Links drawn**. Do Montreal's riders follow the borough map?

## Each station copies its neighbours {#neighbours-vote}

Label propagation starts with every station as its own group. Each round, all stations at once adopt the label most common among themselves and their neighbours, ties to the lowest. Play **Voting round** and watch grey stations merge into hues. This is a CPU replay of the vote; its last round matches the GPU: {{replayMatch}}.

*The vote ignores ride counts and can oscillate.*

## Moving single stations raises modularity {#climb}

Refinement moves one station at a time to the neighbouring group that most raises modularity: the rides kept inside groups minus what a random network with the same degrees would keep. Zero is chance. Q rises from **{{modularityPropagation}}** to **{{modularityRefined}}**. Only one move is accepted per round, so **Refinement rounds** is a budget.

## Boroughs and riding groups disagree at the seams {#seams}

Drag the divider: boroughs left, riding groups right, each borough in the hue of the group it overlaps most. Rings mark **{{seamStations}}** stations whose group is mostly in another borough. Links stay inside a group {{withinShare}} of the time and inside a borough {{withinBoroughShare}}, but bigger groups keep more by size alone: modularity subtracts that.

*Zones drawn for administration need not match how people move.*

## Resolution decides how many groups exist {#resolution}

Resolution gamma scales what chance expects. Low gamma merges groups, high gamma splits them into many small ones, and only the seven largest keep a hue: now **{{communityCount}}** groups. The chart holds eight analyses, each compiled and run once. Drag **Resolution** or click the chart; hues follow a group when it splits.

*The number of groups is a parameter, not a fact.*

## Distance alone would make clusters too {#read-with-care}

Half the rides on the kept links span less than **{{medianLink}}**, so a network of nearby stations clusters by place before it clusters by habit (Austwick et al. 2013). Change **Edge weighting** or **Day type** and the groups shift; agreement with the boroughs is **{{agreement}}**. Now explore: read communities with care.
