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
