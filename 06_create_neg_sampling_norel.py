import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_NEG_SAMPLES = 100


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)

    # model_data: catálogo completo de itens + histórico de consumo de todos os usuários
    # test_data:  usuários de teste e suas features já calculadas
    model_data = pd.read_csv("data/output/model_data_01_pca_norel.csv")
    test_data  = pd.read_csv("data/output/test_data_01_pca_norel.csv", index_col=0)
    neg_with_rel = pd.read_csv("data/output/neg_sample_test_01_pca_norel.csv")

    # ── Identificação de grupos de colunas ───────────────────────────────────
    item_emb_cols  = [c for c in test_data.columns if c.startswith("item_emb")]
    item_cat_cols  = [c for c in test_data.columns if c.startswith("item_cat_")]
    user_emb_cols  = [c for c in test_data.columns if c.startswith("user_emb")]
    cat_hist_cols  = [c for c in test_data.columns if c.startswith("cat_")]

    item_feature_cols = ["app_package"] + item_emb_cols + item_cat_cols
    user_level_cols   = ["ref_rating", "ref_date"] + cat_hist_cols + user_emb_cols

    print(f"Item feature cols : {len(item_feature_cols)}")
    print(f"User level cols   : {len(user_level_cols)}")

    # ── Catálogo de itens com features (uma linha por item) ──────────────────
    item_catalog = (
        model_data[item_feature_cols]
        .drop_duplicates(subset="app_package")
        .set_index("app_package")
    )
    all_packages = set(item_catalog.index)
    print(f"Items no catálogo : {len(all_packages)}")

    # ── Itens consumidos por cada usuário (base completa, não só test) ───────
    user_consumed = model_data.groupby("uid")["app_package"].apply(set).to_dict()

    # ── Features de nível de usuário (invariantes por item) ──────────────────
    # Pega a primeira linha de cada usuário no test_data (os valores são iguais em todas as linhas)
    user_features = (
        test_data.drop_duplicates(subset="uid")
        .set_index("uid")[user_level_cols]
    )

    test_users = test_data["uid"].unique()
    print(f"Usuários de teste : {len(test_users)}")

    # ── Negative sampling ────────────────────────────────────────────────────
    segments = []
    for uid in test_users:
        # consumed   = user_consumed.get(uid, set())
        # candidates = list(all_packages - consumed)

        # n      = min(N_NEG_SAMPLES, len(candidates))
        # chosen = np.random.choice(candidates, size=n, replace=False)

        chosen = neg_with_rel.loc[neg_with_rel["uid"] == uid, "app_package"].values.tolist()

        # Features dos itens amostrados
        items_df = item_catalog.loc[chosen].reset_index()  # app_package + item cols

        # Features do usuário (broadcast para todas as linhas do bloco)
        u_feats = user_features.loc[uid]
        for col in user_level_cols:
            items_df[col] = u_feats[col]
        items_df["uid"] = uid

        segments.append(items_df)

    neg_df = pd.concat(segments, ignore_index=True)

    # ── Adiciona colunas de interação ausentes (NaN para amostras negativas) ─
    interaction_cols = [c for c in test_data.columns if c not in neg_df.columns]
    for col in interaction_cols:
        neg_df[col] = np.nan
    neg_df["split_type"] = "negative_sample"

    # ── Reordena para coincidir exatamente com test_data ────────────────────
    neg_df = neg_df[test_data.columns]

    print(f"\nShape final : {neg_df.shape}")
    print(f"Amostras por usuário (esperado={N_NEG_SAMPLES}):")
    print(neg_df.groupby("uid").size().describe())
    print(neg_df.head(3))

    neg_df.to_csv("data/output/neg_sample_test_01_pca_norel2.csv")
    print("\nSalvo em data/output/neg_sample_test_01_pca_norel2.csv")
