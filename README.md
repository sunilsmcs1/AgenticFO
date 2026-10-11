
Code for the paper "An Agentic Framework for Adaptive Feature Optimization in Intrusion Detection Systems"

## Setup
pip install -r requirements.txt
Download NSL-KDD (KDDTrain+.txt, KDDTest+.txt) into data/ and UNSW-NB15 (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv) into data_unsw/.

## Reproduce
python afo2.py --dataset nsl  --seeds 0 1 2 3 4 --lams 0.03 0.1 0.3
python afo2.py --dataset unsw --seeds 0 1 2 3 4 --lams 0.03 0.1 0.3

## Notes
Results are the mean over 5 seeds; lambda = 1.0 was run for 2 seeds. Costs are simulated. Tested on Google Colab.