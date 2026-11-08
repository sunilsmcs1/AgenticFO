
Code for the paper "An Agentic Framework for Adaptive Feature Optimization in Intrusion Detection Systems"

## Setup
pip install -r requirements.txt
Download NSL-KDD (KDDTrain+.txt, KDDTest+.txt) into data/ and UNSW-NB15 (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv) into data_unsw/.

## Reproduce
python afo2.py --dataset nsl  --seeds 0 1 2 3 4 --lams 0.03 0.1 0.3
python afo2.py --dataset unsw --seeds 0 1 2 3 4 --lams 0.03 0.1 0.3

## Options added after the first submission (on by default)
- `--crossfit K` (default 3): the policy's reward uses out-of-fold predictions from K classifiers, so no classifier scores rows it was trained on. Without this the classifier looks over-confident on the policy's training rows and the policy learns to stop too early. `--crossfit 0` = original behaviour.
- `--thr {youden,f1,<number>}` (default `youden`): the attack cut-off is chosen on the validation split for every method (AFO, masked classifier, static baselines). Youden (max TPR - FPR) does not depend on the attack ratio, which differs between the UNSW-NB15 training and testing sets. `--thr 0.5` = original behaviour. Metrics at the fixed 0.5 cut-off are also reported (`acc_t05`, `f1_t05`, `fpr_t05`).
- `python afo2.py --dataset unsw --crossfit 0 --thr 0.5 ...` reproduces the original numbers.

## Notes
Results are the mean over 5 seeds; lambda = 1.0 was run for 2 seeds. Costs are simulated. Tested on Google Colab.