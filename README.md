# AegisFlow: temporal hypergraph state-space learning with Mondrian conformal alerting

Reference implementation for the manuscript "AegisFlow: Temporal Hypergraph State-Space Learning with
Mondrian Conformal Alerting for Open-World Network Intrusion Detection under Concept Drift".
All files sit in one flat folder.

## Data
* UNSW-NB15 full records: `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv` and `NUSW-NB15_features.csv`
  (UNSW Canberra). Place them in `data/unsw/` (paths are set at the top of `prep_unsw.py` and `aegis_data.py`).
* ToN-IoT subset (`ToN-IoT.parquet`, 93,199 de-duplicated flows) for `exp_toniot.py`.

## Pipeline
| step | script | output |
|---|---|---|
| clean and sort records | `prep_unsw.py` | `unsw_all.pkl` |
| split statistics, label-twin analysis | `data_stats.py` | `data_stats.json` |
| closed-set benchmark (neural) | `exp_nn.py VARIANT SEED closed` | `runs/*.json`, `pred_B_*.npz` |
| tree baselines | `exp_trees.py`, `exp_lgbm.py` | `res_trees_B.json` |
| leave-one-family-out | `exp_nn.py VARIANT SEED loafo:FAMILY` | `runs/*_loafo-*.json` |
| ablations / sensitivity | `exp_nn.py A-noSSL 0 closed`, `exp_nn.py AegisFlow 0 closed L=4 tag=_L4` | `runs/` |
| cross-session stream | `exp_stream.py METHOD 0` | `stream/*.json` |
| context spill-over check | `spill_check.py` | `res_spill.json` |
| ToN-IoT external check | `exp_toniot.py` | `res_toniot.json` |
| latency | `bench_latency.py` | `res_latency.json` |
| aggregation and figures | `analyze_closed.py`, `analyze_open.py`, `analyze_rest.py`, `fig_arch.py` | `res_*.json`, figures |

`run_all.sh` lists the exact commands used. Variants: `AegisFlow`, `MLP`, `E-GraphSAGE`, `HGNN`, `GRU-Seq`,
`Transformer-Seq`, `SSM-Seq`, `A-noSSL`, `A-attrSSL`, `A-noHyper`, `A-noTemporal`, `A-Transformer`, `A-GRU`, `A-softmax`.
Stream methods: `Static`, `SelfTrain`, `GatedST`, `GatedST+ER`, `GatedST+EWC`, `AegisFlow-CA`, `Label-1%`.

## Modules
* `aegis_data.py` featurisation (log-scaling, vocabularies fitted on training rows), typed temporal hyperedges, host trajectories
* `aegis_model.py` flow encoder, type-attentive hyperconvolution, closed-form selective SSM, prototype head, baselines
* `aegis_train.py` masked attribute + incidence-recovery pretraining, supervised training, Mondrian conformal p-values, decision rule with resolvable classes
* `aegis_metrics.py` macro-F1, MCC, ECE, Brier, detection and false-alarm rates

## Protocol summary
Session B (18 Feb 2015): train 0-6 h, calibration 6-7.5 h, test 7.5-12 h. Blocks of 4096 consecutive flows, built separately
per partition. Excluded attributes: sttl, dttl, ct_state_ttl (testbed artefacts), stcpb, dtcpb, addresses, source port, times.
LOAFO removes the held-out family from training and calibration (including pretraining data) before blocks are built.
Cross-session: train on session A (first 4 h, blocks split 80/20 into train/calibration), stream session B hourly.

All runs used CPU only (two threads); default budget 2 pretraining + 10 supervised epochs.
