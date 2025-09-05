import config
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from scipy.optimize import linear_sum_assignment
import numpy as np

# 1. Load the DataFrame from the parquet file
df = pd.read_parquet(config.FINAL_DATA_PATH)

# 2. Deduplicate entries to ensure each (id, session_id, cluster_label) triple is unique
df_unique = df.drop_duplicates(subset=["id", "session_id", "cluster_label"])

# 3. Ensure that each id corresponds to exactly one ground-truth and one predicted cluster
id_mapping = df_unique.drop_duplicates(subset=["id"])

if id_mapping["id"].duplicated().any():
    raise ValueError("Conflict: some 'id' values map to multiple 'session_id' or 'cluster_label'")

# 4. Extract labels
y_true = id_mapping["session_id"].astype(str).values
y_pred = id_mapping["cluster_label"].astype(str).values

# 5. Compute Adjusted Rand Index (ARI)
ari = adjusted_rand_score(y_true, y_pred)
print(f"Adjusted Rand Index (ARI): {ari:.4f} → 1.0 means identical partitions, 0 means random agreement, <0 worse than random.")

# 6. Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
print(f"Normalized Mutual Information (NMI): {nmi:.4f} → 1.0 means identical, 0.0 means independent.")
