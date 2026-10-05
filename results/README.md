here is the results for all combination of the parameters, the experiments are running on polimi VM
### Switching scenarios

| Scenario | K | Start | Rule tested | Observed transitions |
|---|---|---|---|---|
| S1 | 3 | SL | Latency: communication spike | r2: SL→SFL |
| S2 | 10 | FL | Latency control (FL is not left by latency) | none |
| S3 | 3 | SL | Time: round time above threshold | r2: SL→FL |
| S4 | 10 | SFL | Time + return | r1: SFL→FL, r2: FL→SFL |
| S5 | 3 | SL | Accuracy watch (switch only on a real drop) | none |
| S6 | 3 | FL | Static FL reference | none |
| S7 | 3 | FL | Forced switch (FL ×3 → SFL ×2) | r3: FL→SFL |

**Unstable results observed (ResNet-18)**

- **SFL, K=10, non-IID oscillates strongly.** Repeated runs with the same data partition differ a lot. MNIST α=0.9 reaches 96.03 / 97.74 / 55.13% after 10 rounds. CIFAR-10 α=0.9 reaches 24.95 / 39.59 / 53.91 / 55.32%. 
- **Accuracy collapses when entering FL.** FL starts at ~10% after round 1. In the switching runs, every switch into FL drops accuracy to 11–13% for one round (SL→FL 52.86→11.72%, SFL→FL 32.43→12.83%). The CNN (no BatchNorm) shows neither effect. Suspected cause: FedAvg over BatchNorm statistics on non-IID data (not verified yet).
- **2 CPUs per client are slower than 1.** CIFAR-10, K=3, IID: median round time goes FL 147→190 s, SL 248→261 s, SFL 245→261 s, with unchanged accuracy. Cause not confirmed yet.

##TO DO
repeate the experiments

