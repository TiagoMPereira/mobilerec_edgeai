import pandas as pd
import numpy as np
import json


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    interactions = pd.read_csv("data/input/interactions.csv", usecols=["uid", "app_package"])
    interactions["app_package"] = interactions["app_package"].astype("category")
    interactions["uid"] = interactions["uid"].astype("category")

    # Remove itens with less than 100 interactions
    app_counts = interactions["app_package"].value_counts()
    interactions = interactions[interactions["app_package"].isin(app_counts[app_counts >= 100].index)]

    # Get amount of interactions per user
    user_counts = pd.DataFrame({'uid': interactions['uid'].value_counts().index, 'count': interactions['uid'].value_counts().values})
    user_counts = user_counts.loc[user_counts["count"] >= 5]

    # Calculate quartiles from original data
    quartiles = user_counts['count'].quantile([0.25, 0.5, 0.75])

    q1 = quartiles[0.25]
    q2 = quartiles[0.5]
    q3 = quartiles[0.75]

    print("Original Quartiles:")
    print(f"Q1 (25%): {q1:.2f}")
    print(f"Q2 (50%): {q2:.2f}")
    print(f"Q3 (75%): {q3:.2f}")

    user_counts["strata"] = "Outlier"
    user_counts.loc[user_counts["count"] <= q1, "strata"] = "Q1"
    user_counts.loc[(user_counts["count"] > q1) & (user_counts["count"] <= q2), "strata"] = "Q2"
    user_counts.loc[(user_counts["count"] > q2) & (user_counts["count"] <= q3), "strata"] = "Q3"
    user_counts.loc[user_counts["count"] > q3, "strata"] = "Q4"

    total_users = len(user_counts)
    sample_ratio = 0.05
    sample_size = int(sample_ratio * total_users)

    quartiles_proportions = user_counts[["strata"]].value_counts(normalize=True).reset_index()
    quartiles_proportions["reduced_samples"] = (quartiles_proportions["proportion"] * sample_size).astype(int)
    print(quartiles_proportions)

    np.random.seed(42)
    selected_users = []
    for _, row in quartiles_proportions.iterrows():
        strata_users = user_counts.loc[user_counts["strata"] == row["strata"], "uid"].tolist()
        selected_users.extend(np.random.choice(strata_users, row["reduced_samples"], replace=False))

    
    # Filter interactions to only include sampled users
    sampled_interactions = interactions[interactions['uid'].isin(selected_users)]
    sampled_items = sampled_interactions['app_package'].unique().tolist()

    # Save the sampled interactions and items to new CSV files
    sampled_interactions.to_csv("data/input/sampled_interactions.csv", index=False)
    with open("data/input/sampled_data.json", "w") as f:
        json.dump({"users": selected_users, "items": sampled_items}, f)
