import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_illegal_char(column: pd.Series):
    # ASCII characters
    column = column.str.encode('ascii', 'ignore').str.decode("utf-8")

    # Remove illegal XML characters that cause issues with to_excel
    # Define a regex for illegal XML 1.0 characters (excluding valid control characters like tab, newline, carriage return)
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    column = column.apply(lambda x: ILLEGAL_CHARACTERS_RE.sub('', x) if isinstance(x, str) else x)
    return column

def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").lower().strip()

def generate_embeddings(model, texts: list):
    return model.encode(texts, show_progress_bar=True)

if __name__ == "__main__":

    MODEL_NAME = "all-MiniLM-L6-v2"

    # Load metadata
    metadata = pd.read_csv("data/input/metadata.csv")
    metadata = metadata.drop(columns=['app_name', 'developer_name', 'content_rating', 'price', 'num_reviews' , 'avg_rating', 'app_category'])
    metadata["app_package"] = metadata["app_package"].astype("category")

    # Load used items
    with open("data/input/sampled_data_01.json", "r") as f:
        sampled_data = json.load(f)
    used_items = set(sampled_data["items"])

    metadata = metadata[metadata["app_package"].isin(used_items)]
    metadata = metadata.reset_index(drop=True)

    metadata["description"] = remove_illegal_char(metadata["description"])
    metadata["description"] = metadata["description"].apply(clean_text)

    print("Preprocess finished")

    # GENERATE EMBEDDINGS
    print("Generating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = generate_embeddings(model, metadata["description"].tolist())

    embeddings_df = pd.DataFrame(embeddings)
    cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    embeddings_df.columns = cols
    embeddings_df.insert(0, "app_package", metadata["app_package"])

    embeddings_df.to_csv("data/processed/metadata_embeddings_01.csv")
    print("Embeddings generated...")
    print(embeddings_df.head())

    # RUN PCA - OPTIONAL
    embeddings_df = pd.read_csv("data/processed/metadata_embeddings_01.csv", index_col=0)
    MIN_EXPLAINABILITY = 0.8

    scaler = StandardScaler()

    emb_cols = [col for col in embeddings_df.columns if col.startswith("emb_")]
    embeddings_df[emb_cols] = scaler.fit_transform(embeddings_df[emb_cols])
    embedding_values = embeddings_df[emb_cols].to_numpy()

    PCA_DIM = embedding_values.shape[1]
    
    print(f"Finding best number of components to explain {MIN_EXPLAINABILITY * 100}%")
    for i in range(1, embedding_values.shape[1]-1):
        pca_desc = PCA(n_components=i)
        pca_desc.fit(embedding_values)
        evr = sum(pca_desc.explained_variance_ratio_)
        if evr >= MIN_EXPLAINABILITY:
            PCA_DIM = i
            break
    print(f"Selected {PCA_DIM} components to explain {round(evr*100, 2)} %")

    pca_model = PCA(n_components=PCA_DIM)
    emb_reduced = pca_model.fit_transform(embedding_values)
    pca_emb_names = [f"emb_{i}" for i in range(PCA_DIM)]

    pca_df = pd.DataFrame(emb_reduced)
    pca_df.columns = pca_emb_names
    pca_df.insert(0, "app_package", embeddings_df["app_package"])

    pca_df.to_csv(f"data/processed/metadata_embeddings_pca_{MIN_EXPLAINABILITY}_01.csv")

    print("Embeddings reduced...")
    print(pca_df.head())
    print(pca_df.tail(20))