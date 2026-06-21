import pandas as pd
import numpy as np
import json


if __name__ == "__main__":

    metadata_embed = pd.read_csv("data/processed/metadata_embeddings_pca_0.8_01.csv", index_col=0)
    metadata = pd.read_csv("data/input/metadata.csv")
    metadata = metadata.drop(columns=['developer_name', 'content_rating', 'price', 'num_reviews', 'description'])
    # Load used items
    print(metadata.shape)
    with open("data/input/sampled_data_01.json", "r") as f:
        sampled_data = json.load(f)
    used_items = set(sampled_data["items"])

    metadata = metadata[metadata["app_package"].isin(used_items)]
    metadata = metadata.reset_index(drop=True)
    print(metadata.shape)

    metadata = pd.merge(metadata, metadata_embed, on="app_package", how="left")

    for col in metadata.columns:
        if col.startswith("emb_"):
            metadata[col] = metadata[col].astype(np.float16)

    print(metadata.head())
    metadata.to_csv("data/output/metadata_artifact.csv", index=False)
    metadata.to_parquet("data/output/metadata_artifact")

    print(metadata.columns)
