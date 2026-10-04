# Intrusion Detection on CIC-IDS2017: 1D CNN vs Random Forest

Binary network intrusion detection (benign vs attack) on CIC-IDS2017 flow features, comparing a 1D convolutional neural network with a random forest under two evaluation protocols:

- **Random split:** both capture days combined, balanced, and split 80/20 at random. Train and test share attack families.
- **Day-based split:** train on Tuesday (FTP-Patator, SSH-Patator), test on Friday afternoon (DDoS). No attack family is shared, so this measures generalization to an unseen attack type.

The project started as coursework for AIM 602 (Artificial Neural Network & Deep Learning, MS in Information Technology, Westcliff University). It adapts the CNN-based approach of Sudyana et al. (2024), who study generalization of ML-based IDS to traffic outside the training data. This is an adaptation on flow features, not a replication of that paper's raw-packet, cross-dataset experiments.

## Results

All numbers come from `results/<split>/<model>/results.json` (seed 42, one run each).

| Protocol | Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| Random split | 1D CNN | 0.9852 | 0.9911 | 0.9792 | 0.9851 | 0.9995 |
| Random split | Random forest | 0.9998 | 0.9999 | 0.9997 | 0.9998 | 1.0000 |
| Day-based split | 1D CNN | 0.4098 | 0.0000 | 0.0000 | 0.0000 | 0.3291 |
| Day-based split | Random forest | 0.4999 | 0.0000 | 0.0000 | 0.0000 | 0.9444 |

Test sets are balanced: 56,745 flows for the random split, 195,436 flows (97,718 per class) for the day-based split.

What the table shows:

1. **On a random split both models look excellent, and the forest is better.** The forest makes 10 errors on 56,745 test flows, the CNN makes 840.
2. **On the day-based split both models detect none of the DDoS flows** at the default 0.5 threshold (0 of 97,718 for each). The CNN also flags 17,619 benign flows as attacks. The forest flags 12.
3. **The two failures are different.** The forest's ROC AUC of 0.944 means DDoS flows still receive higher attack scores than benign flows, so the ranking carries signal and the decision threshold is what breaks under the shift. The CNN's AUC of 0.329 is below chance: it ranks the unseen DDoS flows as more benign than actual benign traffic. A high AUC alone does not make a usable detector, since choosing a better threshold would need labeled data from the new distribution.
4. **The two protocols reward different cues.** The forest's most important features in the random split are forward packet length statistics. In the day-based split they are TCP initial window sizes and minimum segment size. I have not investigated why.

The random-split scores are what most papers on this dataset report, and the day-based result shows how little they say about unseen attack types.

## Data

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity, University of New Brunswick). Two of the flow CSV files are used, fetched automatically from the Hugging Face mirror `c01dsnap/CIC-IDS2017`:

| File | Flows | Content |
|---|---|---|
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 | 432,074 benign, 13,835 FTP-Patator and SSH-Patator |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 | 97,718 benign, 128,027 DDoS |

The dataset is not included in this repository. Check the original terms of use before redistributing it.

## Pipeline

1. Normalize column names, convert labels to binary (`BENIGN` = 0, everything else = 1).
2. Drop identifiers and metadata (flow ID, IP addresses, ports, timestamps, label), leaving 77 numeric features. This reduces the chance of the model keying on the capture environment.
3. Treat infinities and non-numeric values as missing. Balance the classes by undersampling.
4. Split (random or day-based), drop all-missing columns, and median-impute. Imputation statistics and the scaler are fitted on the training split only.
5. **CNN:** standardize, reshape to (samples, 77, 1), train with early stopping on a 20% validation hold-out of the training split. **Random forest:** 100 trees, default hyperparameters, trained on the full training split.
6. Evaluate once on the test split. The test split is never used for fitting, validation or model selection.

CNN architecture (83,841 parameters): three Conv1D blocks (64, 128, 128 filters, kernel size 3) with batch normalization, two max-pooling layers and dropout 0.25, then global average pooling, a 64-unit dense layer with dropout 0.30, and a sigmoid output. Adam (learning rate 1e-3), binary cross-entropy, up to 15 epochs, batch size 256, learning-rate reduction on plateau.

## Reproduce

Tested with Python 3.12 on CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/ids_cnn.py --split random
python src/ids_cnn.py --split day
python src/ids_rf.py  --split random
python src/ids_rf.py  --split day
```

Each run downloads the data on first use (about 200 MB into `data/hf_cache`), then writes `results.json` and figures to `results/<split>/<model>/`. The CNN runs also save the trained model and scaler there; these are git-ignored. The CNN random-split run reproduced the same metrics and confusion matrix across two separate runs.

## Limitations

- Two of the eight CIC-IDS2017 files, and only three attack types (FTP-Patator, SSH-Patator, DDoS).
- One seed and one train/test pairing per protocol. No confidence intervals or repeated runs.
- A random flow-level split can place near-duplicate flows from the same attack burst on both sides of the split, so random-split scores are likely optimistic.
- Balanced test sets do not reflect real traffic, where attacks are rare. Precision at realistic base rates would be lower.
- The CNN's validation loss was unstable in the day-based run, which has only about 22,000 training samples (see `results/day/cnn/fig2_training_curves.png`). Read that CNN number as indicative.
- The forest is untuned and sees the full training split, while the CNN holds out 20% for early stopping.
- Flow statistics from CICFlowMeter are used, not raw packets as in the original paper.

## Repository layout

```
src/common.py      data loading, splits, preprocessing, evaluation, shared figures
src/ids_cnn.py     1D CNN training and evaluation
src/ids_rf.py      random forest baseline
results/           metrics (results.json) and figures for each protocol and model
docs/              project report (PDF)
```

The report in `docs/` was written for the random-split CNN experiment and matches `results/random/cnn`. Its Appendix A refers to the original notebook and script names, which are now consolidated into `src/`. The day-based split and the random forest were added afterwards and are not in the report.

## References

- Sudyana, D., Lin, Y.-D., Verkerken, M., Hwang, R.-H., Lai, Y.-C., D'Hooge, L., Wauters, T., Volckaert, B., & De Turck, F. (2024). Improving generalization of ML-based IDS with lifecycle-based dataset, auto-learning features, and deep learning. *IEEE Transactions on Machine Learning in Communications and Networking, 2*, 645-662. https://doi.org/10.1109/TMLCN.2024.3402158
- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP 2018*.

## Author

Gaurav Basyal
