# SAE

## 1️⃣ Train the SAE model
```bash
python dna_training.py
```

## 2️⃣ Downstream evaluations with Ridge regression
*(Edit the three variables in each script)*

### A. First-token latent embedding
Edit the following variables in `dna_sequence_analysis_firsttoken.py`:
```python
csv_path            = "<PATH_TO_YOUR_DATASET_CSV>"
sae_checkpoint_path = "<PATH_TO_YOUR_SAE_CHECKPOINT>"
output_path         = "<PATH_WHERE_YOU_WANT_FIRSTTOKEN_WEIGHTS_CSV>"
```

Then run:
```bash
python dna_sequence_analysis_firsttoken.py
```

### B. Mean-pooled latent embedding
Edit the following variables in `dna_sequence_analysis.py`:
```python
csv_path            = "<PATH_TO_YOUR_DATASET_CSV>"
sae_checkpoint_path = "<PATH_TO_YOUR_SAE_CHECKPOINT>"
output_path         = "<PATH_WHERE_YOU_WANT_MEANPOOL_WEIGHTS_CSV>"
```

Then run:
```bash
python dna_sequence_analysis.py
```

```
