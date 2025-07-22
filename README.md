# sae

# 1️⃣ Train the SAE model
python dna_training.py

# 2️⃣ Down‑stream evaluations with Ridge regression
#    (edit the three variables in each script)

## A. First‑token latent embedding
# dna_sequence_analysis_firsttoken.py
csv_path            = "<PATH_TO_YOUR_DATASET_CSV>"
sae_checkpoint_path = "<PATH_TO_YOUR_SAE_CHECKPOINT>"
output_path         = "<PATH_WHERE_YOU_WANT_FIRSTTOKEN_WEIGHTS_CSV>"
python dna_sequence_analysis_firsttoken.py

## B. Mean‑pooled latent embedding
# dna_sequence_analysis.py
csv_path            = "<PATH_TO_YOUR_DATASET_CSV>"
sae_checkpoint_path = "<PATH_TO_YOUR_SAE_CHECKPOINT>"
output_path         = "<PATH_WHERE_YOU_WANT_MEANPOOL_WEIGHTS_CSV>"
python dna_sequence_analysis.py
