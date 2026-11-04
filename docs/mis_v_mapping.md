# MIS-V traceability

23MAT303, Mathematics for Intelligent Systems 5; Semester 5; L-T-P-C 2-0-2-3.

| Topic | Equation/operation | Source module | Notebook | Assessment | Outcomes |
|---|---|---|---|---|---|
| Unit 1: PGM / probabilistic AI | P(V)=product CPDs | pgm/bayesian_quality.py | 11 | Why is a posterior conditional on modeling assumptions? | CO1, CO2 |
| Unit 1: Bayesian networks | P(child|parents) | pgm/bayesian_quality.py | 11 | What must every CPD row sum to? | CO2, CO3 |
| Unit 1: representation learning | EM responsibilities | pgm/latent_quality.py | 11 | Why can latent state labels permute? | CO1, CO2 |
| Unit 1: inference | sum-product variable elimination | pgm/bayesian_quality.py | 11 | Compare enumeration and elimination. | CO2 |
| Unit 2: Markov networks | P proportional to product potentials | pgm/markov_network.py | 12 | How is Z computed on a tiny graph? | CO1, CO3 |
| Unit 2: independencies | graph separation | pgm/markov_network.py | 12 | Does conditioning on b separate a-b-c? | CO3 |
| Unit 2: HMM | forward / backward recursions | pgm/hmm.py | 13 | Which inference requires future frames? | CO1, CO4 |
| Unit 3: MRF | unary + beta Potts disagreement | pgm/mrf.py | 12 | How can smoothing erase a small region? | CO3, CO4 |
| Unit 3: decision networks | argmin expected loss | pgm/decisions.py | 14 | Compute losses at reliability 0.8. | CO1, CO4 |
| Unit 3: BN to Markov | moralization + CPD clique factors | pgm/moralization.py | 14 | Why is moralization not triangulation? | CO2, CO3 |

Every listed notebook contains an executable demonstration and worked question. PGM evidence uses educational assumed tables or synthetic observations, not clinical calibration. CO1 models the quality system; CO2 derives and tests Bayesian inference; CO3 compares directed and undirected models; CO4 connects posterior uncertainty to spatial quality and research acquisition decisions.
