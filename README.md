# Runtime Adaption switch Between FL,SL,SFL



1. Core Module Description
   - `main.py`: Unified scheduling entry point
   - `fl_mnist_minimal.py`: Federated Learning
   - `sl_mnist_minimal.py`: Split Learning
   - `fsl_mnist_minimal.py`: Federated Split Learning
   - `.\checkpoints`: Store the previous model parameters
2. Automatic Method Selection Strategy
   When `--method auto` is used, the program selects a training method based on
   the number of clients and the CPU resources available on each client.

   The current selection rules are:
   - If the number of clients is fewer than 5, the system selects SL.
   - If the number of clients is greater than 5 and every client has at least
     2 CPUs, the system selects FL.
   - In other middle cases, the system selects SFL.

   Users can also manually override the automatic selection by specifying:

   ```bash
   python .\main.py --num-clients 5 --method fl
   ```

3. Communication Time Measurement and Simulation

   We measure the communication-related runtime for each training
   executionn by taking timestamp.

   A simulated delay can be added with `--communication-delay`. This makes it
   possible to test how different methods behave under bad network condition,

   Example:
   ```bash
   python .\main.py --num-clients 5 --num-rounds 3 --communication-delay 0.1,0.3,0.6
   ```

4. Adaptive Communication-Based Switching Mechanism

   In adaptive mode, the program runs one training round at a time. After each
   round, it compares the current communication time with the previous round.
   If the communication time increases beyond the configured threshold , the
   method is switched for the next round.

   The default switching threshold is 20 percent and can be changed with
   `--switch-threshold`.

   The switching order is:

   ```text
   SL -> SFL -> FL
   ```

5. Checkpoint Saving and Recovery

   Checkpoints are used to save model parameters during adaptive training.
   This allows the project to preserve the state of each method across rounds
   instead of restarting from scratch every time.

   The shared data partition configuration is stored in the checkpoint. If a
   checkpoint was created with a different `--num-clients`, `--noniid-alpha`,
   or seed, recovery stops with an error instead of silently mixing
   incompatible client data splits.

   By default, checkpoints are stored in:

   ```text
   .\checkpoints
   ```

6. Non-IID Data Splitting

   Training data can be split by a shared Dirichlet distribution with
   `--noniid-alpha` in the range `[0, 1]`. Smaller values create stronger
   label skew across clients. The default value is `1.0`, which keeps the
   original IID split.

   Example:
   ```bash
   python .\main.py --method fl --num-clients 3 --num-rounds 3 --noniid-alpha 0.2
   ```

7. Boundary-Scalar Non-IID Detection and Pattern Toggling

   The training path does not send raw per-class counts or full label
   distribution vectors to the server. Each Flower client computes one local
   boundary scalar:

   ```text
   JSD(local_label_distribution || uniform_reference)
   ```

   The scalar uses log base 2, so the value is in `[0, 1]`. The server-side
   Pattern Toggling Manager consumes only these scalar scores. If the mean score
   is at or above `--strong-noniid-jsd-threshold`, SL toggles to the SFL pattern
   by enabling client-side FedAvg.

   Inspect the current split:

   ```bash
   python .\noniid_jsd_switch.py --num-clients 3 --noniid-alpha 0.2
   ```

   Enable method switching before training:

   ```bash
   python .\main.py --method auto --num-clients 3 --num-rounds 3 --noniid-alpha 0.2 --adaptive-noniid-switch
   ```

   The offline `noniid_jsd_switch.py` report still prints full counts and a JSD
   matrix for debugging, but that path is not used by the privacy-preserving
   Flower message switch.

   

## Quick Start

```bash
python .\main.py --method sl --num-clients 2 --client-cpus 1,1 --num-rounds 3 --adaptive-communication-switch --communication-delay 0,8,5 
```
