import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data = pd.read_csv("data/output/model_data.csv", index_col=0)

    # Get amount of interactions per user
    user_counts = pd.DataFrame({'uid': data['uid'].value_counts().index, 'count': data['uid'].value_counts().values})

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

    # Stratified sampling
    train_users, temp_users = train_test_split(
        user_counts,
        test_size=0.3,
        stratify=user_counts['strata'],
        random_state=42
    )

    val_users, test_users = train_test_split(
        temp_users,
        test_size=0.5,   # 15% final
        stratify=temp_users['strata'],
        random_state=42
    )

    train_df = data[data['uid'].isin(train_users['uid'])]
    val_df = data[data['uid'].isin(val_users['uid'])]
    test_df = data[data['uid'].isin(test_users['uid'])]

    train_df.to_csv("data/output/train_data.csv")
    val_df.to_csv("data/output/val_data.csv")
    test_df.to_csv("data/output/test_data.csv")