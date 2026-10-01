import os
import gc
import torch
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

TEST_PATH = "/Users/shaswatshrey/Development/AmazonML/final_text_pairs_test.tsv"
MODEL_PATH = "/Users/shaswatshrey/Development/AmazonML/s_bert_entity_matching/sbert_entity_matching_v1"
PREDICTION_OUTPUT = "/Users/shaswatshrey/Development/AmazonMLtest_predictions.tsv"
MATCHING_OUTPUT = "/Users/shaswatshrey/Development/AmazonML/matching_results.tsv"

THRESHOLD = 0.965
CHUNK_SIZE = 1_000_000
BATCH_SIZE = 512

device = "mps"
print(torch.backends.mps.is_available())

MODEL_FILE = os.path.join(
    MODEL_PATH,
    "model.safetensors"
)
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model weights not found:\n{MODEL_FILE}"
    )
print("Model weights found:")
print(MODEL_FILE)

model = SentenceTransformer(
    MODEL_PATH,
    device=device
)

print("\nFine-tuned SBERT loaded successfully!")

sample = pd.read_csv(
    TEST_PATH,
    sep="\t",
    nrows=5
)

print("\nTest columns:")
print(sample.columns.tolist())

print("\nSample:")
print(sample.head())

REQUIRED_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "test_s1",
    "test_cand"
]

missing = [
    col
    for col in REQUIRED_COLUMNS
    if col not in sample.columns
]
if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

first_prediction_chunk = True

total_processed = 0
total_matches = 0

for chunk in pd.read_csv(
    TEST_PATH,
    sep="\t",
    chunksize=CHUNK_SIZE,
    usecols=[
        "source1_entity_id",
        "candidate_entity_id",
        "test_s1",
        "test_cand"
    ],
    dtype=str
):
    chunk = chunk.dropna(
        subset=[
            "test_s1",
            "test_cand"
        ]
    ).reset_index(drop=True)

    if len(chunk) == 0:
        continue

    total_processed += len(chunk)
    print(f"\nProcessing {total_processed:,} pairs...")

    emb_s1 = model.encode(
        chunk["test_s1"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )
    emb_cand = model.encode(
        chunk["test_cand"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    similarities = np.sum(
        emb_s1 * emb_cand,
        axis=1
    )
    y_pred = (
        similarities >= THRESHOLD
    ).astype(int)

    chunk["similarity"] = similarities
    chunk["y_pred"] = y_pred

    matches = int(
        y_pred.sum()
    )
    total_matches += matches


    print(f"Pairs: {len(chunk):,}")
    print(f"Predicted 1 : {matches:,}")
    print(f"Total match : {total_matches:,}")

    chunk.to_csv(
        PREDICTION_OUTPUT,
        sep="\t",
        index=False,
        mode="w" if first_prediction_chunk else "a",
        header=first_prediction_chunk
    )
    first_prediction_chunk = False

    del emb_s1
    del emb_cand
    del similarities
    del y_pred
    del chunk

    gc.collect()


print("\n========================================")
print("INFERENCE COMPLETE")
print("========================================")

print(f"Total pairs processed : {total_processed:,}")
print(f"Total predicted match : {total_matches:,}")
print(f"Predictions saved to  : {PREDICTION_OUTPUT}")

# ============================================================
# CREATE FINAL MATCHING RESULTS
# ============================================================

print("\nCreating matching_results.tsv...")
matched_parts = []

for chunk in pd.read_csv(
    PREDICTION_OUTPUT,
    sep="\t",
    usecols=[
        "source1_entity_id",
        "candidate_entity_id",
        "y_pred"
    ],
    chunksize=100_000
):
    matched = chunk[
        chunk["y_pred"] == 1
    ][
        [
            "source1_entity_id",
            "candidate_entity_id"
        ]
    ]

    if len(matched) > 0:
        matched_parts.append(matched)

# Combine only MATCHED rows
matched_df = pd.concat(
    matched_parts,
    ignore_index=True
)

matched_df = matched_df.drop_duplicates()
matching_results = (
    matched_df
    .groupby("source1_entity_id")["candidate_entity_id"]
    .agg(",".join)
    .reset_index()
)

matching_results.columns = [
    "source1_entity_id",
    "candidate_entity_ids"
]

matching_results.to_csv(
    MATCHING_OUTPUT,
    sep="\t",
    index=False
)

print("\n========================================")
print("FINAL MATCHING RESULTS")
print("========================================")

print(matching_results.head(10))
print(
    f"\nTotal source entities with matches: "
    f"{len(matching_results):,}"
)
print(f"\nSaved to:\n{MATCHING_OUTPUT}")

