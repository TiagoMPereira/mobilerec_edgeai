import pandas as pd
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    
    # Importing data
    train = pd.read_csv("data/output/train_data.csv", index_col=0)
    validation = pd.read_csv("data/output/val_data.csv", index_col=0)
    test = pd.read_csv("data/output/test_data.csv", index_col=0)

    # Importing data
    print(train[["uid", "app_package", "rating", "ref_rating", "centered_rating"]].head(20))
    print(validation[["uid", "app_package", "rating", "ref_rating", "centered_rating"]].head(20))
    print(test[["uid", "app_package", "rating", "ref_rating", "centered_rating"]].head(20))

    # Creating target
    print("Creating target")
    train["target"] = train["centered_rating"]/5
    validation["target"] = validation["centered_rating"]/5
    test["target"] = test["centered_rating"]/5

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    validation = validation.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Splitting columns
    print("Spliting columns")
    target_col = "target"
    user_cols = [col for col in train.columns if "_user" in col]
    item_cols = [col for col in train.columns if "_item" in col]

    X_user_train = train[user_cols].values.astype('float32')
    X_item_train = train[item_cols].values.astype('float32')
    y_train = train[target_col].values.astype('float32')

    X_user_validation = validation[user_cols].values.astype('float32')
    X_item_validation = validation[item_cols].values.astype('float32')
    y_validation = validation[target_col].values.astype('float32')

    X_user_test = test[user_cols].values.astype('float32')
    X_item_test = test[item_cols].values.astype('float32')
    y_test = test[target_col].values.astype('float32')

    # Creating user tower
    print("User tower")
    user_input = tf.keras.Input(
        shape=(X_user_train.shape[1],),
        name="user_embedding"
    )

    user_model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Dense(128, kernel_initializer="he_normal", name="user_hidden_1"),
            # tf.keras.layers.LeakyReLU(alpha=0.1),
            # tf.keras.layers.Dropout(0.1),
            # tf.keras.layers.Dense(64, name="user_hidden_2"),
            # tf.keras.layers.Dropout(0.1),
            # tf.keras.layers.Dense(32, name="user_tower_output", kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            tf.keras.layers.Dense(32, kernel_initializer="he_normal", name="user_tower_output")
        ]
    )

    user_vector = user_model(user_input)

    # Creating item tower
    print("Item tower")
    item_input = tf.keras.Input(
        shape=(X_item_train.shape[1],),
        name="item_embedding"
    )

    item_model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Dense(128, kernel_initializer="he_normal", name="item_hidden_1"),
            # tf.keras.layers.LeakyReLU(alpha=0.1),
            # tf.keras.layers.Dropout(0.1),
            # tf.keras.layers.Dense(64, activation='relu', name="item_hidden_2"),
            # tf.keras.layers.Dropout(0.1),
            # tf.keras.layers.Dense(32, name="item_tower_output", kernel_regularizer=tf.keras.regularizers.l2(1e-5))
            tf.keras.layers.Dense(32, kernel_initializer="he_normal", name="item_tower_output")
        ]
    )

    item_vector = item_model(item_input)

    # Creating combined layer
    print("Combined layer")
    combined_tensor = tf.keras.layers.Concatenate(name="combined_layer")(
        [
            user_vector,
            item_vector
        ]
    )

    combined_layer = tf.keras.layers.Dense(
        32, activation='relu', name="combined_hidden"
        # 32, activation='relu', name="combined_hidden", kernel_regularizer=tf.keras.regularizers.l2(1e-5)
    )(combined_tensor)

    model_output = tf.keras.layers.Dense(1, activation="linear", name="combined_output")(combined_layer)

    # Creating model
    print("Creating final model")
    model = tf.keras.Model(
        inputs=[user_input, item_input],
        outputs=model_output
    )

    opt = tf.keras.optimizers.Adam(
        learning_rate=1e-2
    )

    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )

    print(f"Model summary:\n{model.summary()}")

    # Fitting
    print("================\nFITTING\n================")
    history = model.fit(
        x={
            'user_embedding': X_user_train,
            'item_embedding': X_item_train
        },
        y=y_train,
        validation_data=(
            {
                'user_embedding': X_user_validation,
                'item_embedding': X_item_validation
            },
            y_validation
        ),
        epochs=50,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=50,
                restore_best_weights=True
            )
        ]
    )

    # Predict
    print("================\nPREDICTING\n================")
    predictions = model.predict({
        'user_embedding': X_user_test,
        'item_embedding': X_item_test
    }).flatten()

    results = test.copy()[["app_package", "uid", "rating", "ref_rating", "target"]]
    results["prediction"] = predictions
    print(results.sample(20))
    results.to_csv("data/output/predictions.csv", index=False)

    extractor = tf.keras.Model(
        inputs=[user_input, item_input],
        outputs=[user_vector, item_vector, combined_layer, model_output]
    )
    batch_u = X_user_validation[:256]
    batch_i = X_item_validation[:256]
    user_emb, item_emb, combined_emb, output = extractor.predict(
        {
            'user_embedding': batch_u,
            'item_embedding': batch_i
        }
    )

    print("User vector - std entre amostras:", np.std(user_emb, axis=0).mean())   # média dos desvios por dimensão  
    print("Item vector - std entre amostras:", np.std(item_emb, axis=0).mean())
    print("Combined layer - std entre amostras:", np.std(combined_emb, axis=0).mean())

    # Distância entre duas amostras diferentes
    print("Distância user 0 vs 1:", np.linalg.norm(user_emb[0] - user_emb[-1]))
    print("Distância item 0 vs 1:", np.linalg.norm(item_emb[0] - item_emb[-1]))
    print("Predições min/max/mean:", output.min(), output.max(), output.mean())

