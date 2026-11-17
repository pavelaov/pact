# PACT Artifact — Code

Anonymous review artifact for **PACT** (publicly auditable key generation for distributed point functions). It contains the reference implementation and the twelve experiment drivers used for the evaluation. No author names, repository URL, or machine-specific absolute paths are included.

## Directory layout

```
.
├─ requirements.txt          # pinned dependencies
└─ code/
   ├─ run_all.py             # suite entry point (self-test + exp1–exp12)
   ├─ pact/                  # protocol implementation
   │  ├─ primitives.py       # AES-128 / MMO hash, deterministic RNG
   │  ├─ dpf.py              # distributed point function keys and evaluation
   │  ├─ protocol.py         # chips, hosts, GEN/BIND key generation
   │  ├─ audit.py            # sacrificial audit, certificates, judgement
   │  ├─ adversary.py        # covert/backdoor chip adversaries
   │  ├─ placements.py       # key-generation placement counterfactuals
   │  └─ stats.py            # Wilson / exact confidence intervals
   └─ experiments/
      ├─ common.py           # shared harness, pinned seeds, output writer
      └─ exp1_correctness.py … exp12_singlechip.py
```

## Requirements

- Python 3.10 or newer (tested with Python 3.13.9 on x86-64; works on Linux, macOS and Windows).
- Dependencies (pinned in `requirements.txt`): `cryptography`, `numpy`, `scipy`, `matplotlib`.

```bash
python -m pip install -r requirements.txt
```

## How to run

Run every command from this directory (the folder that contains `requirements.txt` and `code/`).

```bash
# built-in self-test only (fast sanity check, no experiment runs)
python code/run_all.py --check

# quick smoke run: all experiments with reduced repetitions
python code/run_all.py --fast

# full experiment suite with the paper's repetition counts
python code/run_all.py

# run a single experiment through the suite runner
python code/run_all.py --only exp1_correctness

# or run one driver directly
python code/experiments/exp1_correctness.py
```

## Outputs

Each experiment writes files at the artifact root (created automatically):

- `out/<name>.csv` — result rows;
- `out/<name>.json` — seed, parameters, wall-clock time, environment metadata and summary;
- `logs/<name>.log` — console transcript.

All paths recorded in the logs are relative to this directory. The code performs no network access.

## Experiments

| Script | Content                                                      |
|---|---|
| `exp1_correctness` | Protocol and audit correctness over the full domain |
| `exp2_detection` | Sacrificial-audit deterrent factor versus the closed form |
| `exp3_yield` | Sacrifice calibration, yield cost and provisioning |
| `exp4_adaptive` | Adaptive chip: blind, coin-grinding and oracle arms |
| `exp5_defamation` | Nine forgery strategies against certificate verification |
| `exp6_consistency` | Input-consistency arms and ablation |
| `exp7_frontier` | Placement–auditability/privacy frontier |
| `exp8_auditcost` | Publicly verifiable assurance cost versus analytic baselines |
| `exp9_backdoor` | Implemented chip backdoor and its leakage budget |
| `exp10_costmicro` | Object/certificate sizes, AES counts, exposure window, audit latency |
| `exp11_distinguisher` | Whether the backdoored target stream is distinguishable |
| `exp12_singlechip` | Deterministic single-chip detection versus colluding chips |

## Notes

- The master seed of each experiment is pinned in `SEEDS` in `code/experiments/common.py`; changing a seed or repetition count changes the reported numbers.
- `--fast` is only a functional check.
