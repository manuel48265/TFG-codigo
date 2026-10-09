# Real topologies: bnlearn Bayesian Network Repository

`.bif` files downloaded from [bnlearn.com/bnrepository](https://www.bnlearn.com/bnrepository/)
(© Marco Scutari), licensed under
[Creative Commons Attribution-Share Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

Only the DAG **structure** is used (`probability(child | parents)` in each
`.bif`, extracted by `real_dag_topologies.load_bif_topology`); the original
CPDs and each variable's type/cardinality are discarded — data is simulated
with our own pipeline (`Node_Generation/`), not the real network's
parameters. See [[dato-opcional-no-flag-de-metodo]] for the same criterion of
decoupling structure from mechanism.

## Networks

The 4 chosen are the ones cited as "standard benchmark networks" 

- **asia.bif** — ASIA (8 nodes, 8 edges), lung disease diagnosis.
  Lauritzen, S. L. (1988). *Local computations with probabilities on
  graphical structures and their application to expert systems (with
  discussion)*. J. Roy. Statist. Soc. B, 50(2), 251–263.
- **alarm.bif** — ALARM (37 nodes, 46 edges), ICU patient monitoring.
  Beinlich, I. A., Suermondt, H. J., Chavez, R. M., & Cooper, G. F. (1989).
  *The ALARM monitoring system: A case study with two probabilistic
  inference techniques for belief networks*. AIME 89.
- **insurance.bif** — INSURANCE (27 nodes, 52 edges), car insurance risk
  assessment, mix of discrete cardinalities (2/3/4/5).
  Binder, J., Koller, D., Russell, S., & Kanazawa, K. (1997). *Adaptive
  probabilistic networks with hidden variables*. Machine Learning, 29(2),
  213–244.
- **sachs.bif** — SACHS (11 nodes, 17 edges), protein signaling, all
  variables discretized to 3 levels.
  Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P.
  (2005). *Causal protein-signaling networks derived from multiparameter
  single-cell data*. Science, 308(5721), 523–529.
