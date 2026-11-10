# Runtime Adaptation Across FL, SL and SFL

Code for the thesis *Adaptative Runtime Switch of Distributed Learning Paradigms*.
Federated Learning (FL), Split Learning (SL) and SplitFed Learning (SFL) are
simulated with Flower and Ray, share one model definition and one checkpoint
format, and can switch paradigm at the end of any round.

## Files

**Training programs**

| File | Purpose |
|---|---|
| `fl_mnist_minimal.py` | FL training (FedAvg on the full model) |
| `sl_mnist_minimal.py` | SL training (cut model, client-side models not aggregated) |
| `fsl_mnist_minimal.py` | SFL training (cut model, client-side models aggregated with FedAvg) |
| `flower_split_message.py` | Message-based training engine shared by SL and SFL |
| `split_learning_utils.py` | Models (ResNet, CNN), cut layers, data loading, Dirichlet partition, seed, shared checkpoint |
| `mnist_evaluation.py` | Test-set evaluation |
| `noniid_jsd_switch.py` | Jensen–Shannon divergence of the client label distributions |
| `main.py` | Single entry point: selects the initial paradigm and optionally switches on communication time |
| `local_resnet_baseline.py` | Centralized reference (one client, whole training set) |

**Experiment scripts**

| File | Purpose |
|---|---|
| `run_baseline.py` | Static baseline grid (FL / SL / SFL × clients × data distribution), RQ1 |
| `run_switch.py` | Runtime switching scenarios S1–S7, RQ2 |
| `run_switch_seeds.sh` | Repeats the switching scenarios with seeds 1235–1238 |
| `run_cutlayer.py` | SFL with the cut after each ResNet-18 stage and leakage of each cut, RQ3 |
| `run_cutswitch.py` | Privacy-driven cut-layer switching, RQ3 |
| `measure_all_leakage.py`, `measure_cutlayer_leakage.py` | Distance correlation and reconstruction attack on the smashed data |
| `dcor_privacy.py` | Distance correlation and reconstruction-attack helpers |
| `run_resnet18_paradigm_comparison.py` | Earlier comparison script for ResNet-18 runs and leakage |

**Data and results**

| Path | Content |
|---|---|
| `switch_baselines/` | Baseline round times used to calibrate the switching thresholds |
| `results/` | Archived results of the experiments |
| `data/` | Datasets (downloaded on the first run) |

## How to run

Install the dependencies (Python 3.10+):

```bash
pip install -r requirements.txt
```

The random seed is set with the environment variable `THESIS_SEED`
(default `1234`).

**Single run**

```bash
# FL, SL or SFL on MNIST with 3 clients
python main.py --method fl  --num-clients 3 --num-rounds 3
python main.py --method sfl --num-clients 3 --num-rounds 3 --dataset cifar10 --noniid-alpha 0.5

# initial paradigm chosen automatically, switching on communication time
python main.py --method auto --num-clients 3 --num-rounds 5 \
    --adaptive-communication-switch --communication-delay 0,8,5
```

**Baseline grid (RQ1)**

```bash
python run_baseline.py plan                                   # show the grid
python run_baseline.py --model resnet18 --dataset cifar10 run # run it (resumable)
python run_baseline.py --repeats 1-4 run                      # repetitions, seeds 1235-1238
python run_baseline.py summarize                              # build the CSV tables
```

**Runtime switching (RQ2)**

```bash
python run_switch.py run --setup resnet_cifar10 --baseline switch_baselines/resnet_cifar10_rounds.csv
python run_switch.py run --setup cnn_mnist      --baseline switch_baselines/cnn_mnist_rounds.csv --only S1,S3
python run_switch.py report --setup resnet_cifar10 --baseline switch_baselines/resnet_cifar10_rounds.csv
bash run_switch_seeds.sh resnet_cifar10                       # seeds 1235-1238
```

**Cut layer and privacy (RQ3)**

```bash
python run_cutlayer.py --dataset cifar10 --rounds 20
python run_cutswitch.py --name rho04 --rho-min 0.4
python run_cutswitch.py --name rho07 --rho-min 0.7
python run_cutswitch.py --name static --static
```
