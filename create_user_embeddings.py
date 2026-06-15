import pandas as pd
import numpy as np


def split_historic_and_recommendation(interactions: pd.DataFrame, historic_ratio: float = 0.75) -> pd.DataFrame:
    interactions = interactions.sort_values(by=["uid", "formated_date"], ascending=True)
    interactions.reset_index(drop=True, inplace=True)

    # Sequential index
    interactions["i"] = interactions.groupby("uid").cumcount() + 1

    # total interactions per user
    interactions["total_interactions"] = interactions.groupby("uid")["uid"].transform("count")

    # percentage of the interaction
    interactions["pct"] = interactions["i"] / interactions["total_interactions"]

    interactions["split_type"] = np.select(
        [
            interactions["pct"] < historic_ratio,
            interactions["pct"] >= historic_ratio
        ],
        [
            "historic",
            "recommendation"
        ],
        default="error"
    )

    interactions = interactions.drop(columns=["i", "total_interactions", "pct"])
    interactions.reset_index(drop=True, inplace=True)
    return interactions

def create_historical_dataset(interactions: pd.DataFrame) -> pd.DataFrame:

    interactions = interactions.sort_values(by=["uid", "formated_date"], ascending=True)
    interactions.reset_index(drop=True, inplace=True)

    historical_data = interactions.loc[interactions["split_type"] == "historic"].groupby("uid").agg({"rating": "median", "formated_date": "max"})
    historical_data = historical_data.rename(columns={"rating": "ref_rating", "formated_date": "ref_date"}).reset_index()

    return historical_data

if __name__ == "__main__":

    interactions = pd.read_csv("data/input/sampled_interactions.csv")
    metadata = pd.read_csv("data/processed/metadata_embeddings_pca_0.8.csv", index_col=0)

    print(interactions.head())
    print(metadata.head())

    # Pre process
    interactions = interactions.drop(columns=['review', 'votes', 'date', 'unix_timestamp', 'app_category'])
    interactions["app_package"] = interactions["app_package"].astype("category")
    interactions["uid"] = interactions["uid"].astype("category")
    interactions["formated_date"] = pd.to_datetime(interactions["formated_date"])
    interactions["rating"] = interactions["rating"].astype("int8")

    print(f"Unique users: {len(interactions['uid'].unique())}")

    # Split historic data and future access
    interactions = split_historic_and_recommendation(interactions)

    # Creating historical references
    historical_dataset = create_historical_dataset(interactions)

    interactions = pd.merge(interactions, historical_dataset, on="uid", how="left")

    # Create features based on historical behavior
    interactions["centered_rating"] = interactions["rating"] - interactions["ref_rating"]

    ## Delta T = Last historical interaction - interaction (result in months)
    ## For unknown items it is set to 0
    interactions['delta_t'] = (
        interactions['ref_date'].dt.year - interactions['formated_date'].dt.year
    ) * 12 + (
        interactions['ref_date'].dt.month - interactions['formated_date'].dt.month
    )

    interactions.loc[interactions["split_type"] == "recommendation", "delta_t"] = 0

    # Relevancy factor e^(alpha * cr) * e^(-lambda * delta_t)
    ALPHA = 0.1
    LAMBDA = 0.01
    interactions["relevancy"] = np.exp(
        ALPHA * interactions["centered_rating"]
        - LAMBDA * interactions["delta_t"]
    )

    # Selecting only historical data and merging with metadata
    interactions_historic = interactions.loc[interactions["split_type"] == "historic"]

    embedding_cols = [col for col in metadata.columns if col.startswith("emb_")]
    interactions_metadata = pd.merge(interactions_historic, metadata, on="app_package", how="left")

    aggregation_methods = {col: "sum" for col in embedding_cols}
    aggregation_methods["relevancy"] = "sum"
    interactions_metadata[embedding_cols] = interactions_metadata[embedding_cols].multiply(interactions_metadata["relevancy"], axis=0)
    interactions_metadata_group = interactions_metadata.groupby("uid").agg(aggregation_methods)
    interactions_metadata_group = interactions_metadata_group[embedding_cols].div(
        interactions_metadata_group["relevancy"], axis=0
    ).reset_index(drop=False)

    interactions_metadata_group.to_csv("data/processed/historic_embeddings.csv")
    print(interactions_metadata_group.head())
    print(interactions_metadata_group.tail(20))
    print(interactions_metadata_group.shape)

    # Creating dataset
    print("Creating dataset")

    # interactions
    print(f"Interactions:\n{interactions.head()}\n{interactions.columns}")    
    # metadata
    print(f"Metadata:\n{metadata.head()}\n{metadata.columns}")    
    # historic_embeddings
    historic_embeddings = interactions_metadata_group.copy()
    print(f"Historic embeddings:\n{historic_embeddings.head()}\n{historic_embeddings.columns}")    

    # Only after historical period
    interactions_rec = interactions.loc[interactions["split_type"] == "recommendation"]
    metadata.columns = [f"item_{col}" if "emb_" in col else col for col in metadata.columns]
    historic_embeddings.columns = [f"user_{col}" if "emb_" in col else col for col in historic_embeddings.columns]

    model_dataset = pd.merge(
        interactions_rec, metadata, on="app_package", how="left"
    )
    model_dataset = pd.merge(
        model_dataset, historic_embeddings, on="uid", how="left"
    )
    print("Final dataset")
    print(model_dataset.shape)
    print(model_dataset.head(10))
    print(model_dataset.tail(10))
    for c in model_dataset.columns:
        print(c, end=" | ")

    model_dataset.to_csv("data/output/model_data.csv")

