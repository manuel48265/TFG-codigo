# Real_Data

Scope: the real data sets and reference graphs of Experiment 3 (falsifying graphs with the
independence statements of Kook, on the three cases of Eulig et al.). This folder only
loads the data; the CI tests and the analysis live in `Experimental_cases/`.

`datasets.py` exposes `load_sachs()`, `load_auto_mpg()` and `load_apm()`. Each one returns a
`RealDataset` (data, nodes and edges). All the files in `data/` are CSV files with a header:

- `<name>_data.csv`: one column per variable.
- `<name>_edges.csv`: one row `parent,child` per directed edge. Every node of the graph is
  in at least one edge, so the node list is taken from the edges.

## Sources

The Sachs and APM files and the Auto MPG graph come from the code repository of

> E. Eulig, A. A. Mastakouri, P. Blöbaum, M. Hardt and D. Janzing, "Toward Falsifying Causal
> Graphs Using a Permutation-Based Test", Proceedings of the AAAI Conference on Artificial
> Intelligence, vol. 39, no. 25, pp. 26778-26786, 2025.

(<https://github.com/eeulig/dag-falsification>, Apache 2.0 license, commit `227fde2`,
directory `falsifydags/`).

## How each file was obtained

| File | Process |
|---|---|
| `sachs_data.csv` | `data/sachs_data.xls` of Eulig et al., read with `pandas.read_excel` and written with `to_csv(index=False)`; no value changed. The original data are from Sachs et al. (2005), *Science* 308(5721), 523-529. |
| `sachs_edges.csv` | `data/sachs_g_true.gml` of Eulig et al., read with `networkx.read_gml`; the edges are written in the order of `graph.edges`. |
| `auto_mpg_data.csv` | `auto-mpg.data-original` of the UCI Machine Learning Repository (<http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original>, the same URL as `load_auto_data` in `falsifydags/datasets.py`). Read as whitespace-separated with the nine columns named as in that function; rows with missing values (`dropna()`) and the `car name` column are removed, as Eulig et al. do. |
| `auto_mpg_edges.csv` | The nine edges written by hand in `load_auto_data` of Eulig et al. |
| `apm_data.csv` | `data/APM_data.csv` of Eulig et al., copied unchanged (only renamed). Application performance monitoring data of Amazon microservices. |
| `apm_edges.csv` | `data/APM_g_true.gml` of Eulig et al. (the inverse of the call graph), converted like `sachs_edges.csv`. |

The `networkx` and `xlrd` packages were only used once to convert the original GML and XLS
files; they are not needed to load the data.

## Cleaning

- **Auto MPG**: done when the file was produced (see above), so the CSV has no missing values.
- **APM**: done on loading. Every column with a missing value is dropped from the data, as in
  `experiments/run_with_config.py` of Eulig et al. (line 208; Appendix A.5 of the paper: for
  some nodes no data was recorded or the data had missing values). The graph keeps all its
  nodes: 13 of its 39 nodes have no data (4 are not in the file and 9 have missing values).
