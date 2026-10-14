# Analysis contributors review, 2026-10-13: stream reports

These are the evidence reports behind the
[analysis contributors quality roadmap](../../roadmaps/analysis-contributors-quality-roadmap.md).
It was a read-only review of the live `map-contributors` tree (HEAD 731e2a066 plus the uncommitted correctness and
projection rounds). Five Opus coordinators ran it, each with Sonnet sub-reviewers, and every kept finding was re-read
in code by its coordinator. Nothing ran on a GPU. The evidence is code reads plus CPU replays of the kernel arithmetic
against reference packages (shapely/GEOS 3.14, scipy, h3 4.5, networkx, esda) or exact rationals.

| Report | Scope |
|---|---|
| [A](./a-api-consistency-ergonomics.md) | API consistency, ergonomics, docs and tests across every entry, plus the canonical vocabulary proposal |
| [B](./b-network-dataframe-geospatial.md) | Correctness of `gpu-network`, the `gpu-dataframe` subdirs, `geospatial` and `src/utils` |
| [C](./c-terrain-raster.md) | Correctness of `gpu-terrain` and the `gpu-raster` subdirs |
| [D](./d-geometry-cells-movement.md) | Correctness of the geometry, topology, cells/DGGS and movement half of `gpu-spatial-analysis` |
| [E](./e-performance-lifecycle.md) | Performance engineering and resource lifecycle across every entry |

The statistics methodology was out of scope because the 2026-10-13 correctness round already covered it.

Line numbers are from a tree that two peer sessions were editing at the time, so they drift; search for the quoted
code instead. The reports mention `work/<stream>/...` notes and oracle scripts that lived in a session scratchpad.
They were not kept. Regenerate an oracle from the reference package named in the finding.
