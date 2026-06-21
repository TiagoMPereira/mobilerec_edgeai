import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.integration import TFKerasPruningCallback

# No seu script, após carregar os dados (antes da normalização)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


class RatingModelOptimizer:
    def __init__(self, train_path, val_path, test_path, target_col='centered_rating', 
                 user_cols_suffix='user_emb', item_cols_suffix='item_emb', random_state=42):
        """
        Carrega os dados e prepara as matrizes de treino, validação e teste.
        """
        self.random_state = random_state
        self.target_col = target_col
        self.user_cols_suffix = user_cols_suffix
        self.item_cols_suffix = item_cols_suffix
        
        # Carrega CSVs
        self.train = pd.read_csv(train_path, index_col=0)
        self.val   = pd.read_csv(val_path, index_col=0)
        self.test  = pd.read_csv(test_path, index_col=0)

        
        # Embaralha
        self.train = self.train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.val   = self.val.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.test  = self.test.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Prepara target
        self.train["target"] = self.train[self.target_col]
        self.val["target"] = self.val[self.target_col]
        self.test["target"] = self.test[self.target_col]

        self.target_col = "target"

        self.train[self.target_col] = (self.train[self.target_col] > 0).astype('float32')
        self.val[self.target_col] = (self.val[self.target_col] > 0).astype('float32')
        self.test[self.target_col] = (self.test[self.target_col] > 0).astype('float32')

        print(self.train)

        # Separa colunas
        self.user_cols = [c for c in self.train.columns if user_cols_suffix in c]
        self.item_cols = [c for c in self.train.columns if item_cols_suffix in c]
        self.cat_historic_cols = [c for c in self.train.columns if c.startswith("cat_")]
        self.cat_item_cols = [c for c in self.train.columns if c.startswith("item_cat_")]

        print(self.user_cols)
        print(self.item_cols)
        
        # Converte para float32
        self.X_user_train = self.train[self.user_cols].values.astype('float32')
        self.X_item_train = self.train[self.item_cols].values.astype('float32')
        self.hist_cat_train = self.train[self.cat_historic_cols].values.astype('float32')
        self.item_cat_train = self.train[self.cat_item_cols].values.astype('float32')
        self.y_train = self.train[self.target_col].values.astype('float32')
        
        self.X_user_val = self.val[self.user_cols].values.astype('float32')
        self.X_item_val = self.val[self.item_cols].values.astype('float32')
        self.hist_cat_val = self.val[self.cat_historic_cols].values.astype('float32')
        self.item_cat_val = self.val[self.cat_item_cols].values.astype('float32')
        self.y_val = self.val[self.target_col].values.astype('float32')
        
        self.X_user_test = self.test[self.user_cols].values.astype('float32')
        self.X_item_test = self.test[self.item_cols].values.astype('float32')
        self.hist_cat_test = self.test[self.cat_historic_cols].values.astype('float32')
        self.item_cat_test = self.test[self.cat_item_cols].values.astype('float32')
        self.y_test = self.test[self.target_col].values.astype('float32')
        
        # Dimensões
        self.user_input_dim = self.X_user_train.shape[1]
        self.item_input_dim = self.X_item_train.shape[1]
        self.output_bias = np.mean(self.y_train)
        print(f"Dados carregados: treino={self.X_user_train.shape}, "
              f"val={self.X_user_val.shape}, teste={self.X_user_test.shape}")

    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1-score como métrica (threshold 0.5)."""
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def _build_tower(self, input_shape, units, activation, dropout, l2_reg, name_prefix):
        inp = tf.keras.Input(shape=(input_shape,))
        x = inp
        for i, u in enumerate(units[:-1]):
            x = tf.keras.layers.Dense(u, kernel_initializer='he_normal',
                                      kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                      name=f'{name_prefix}_hidden_{i}')(x)
            if activation == 'relu':
                x = tf.keras.layers.ReLU()(x)
            elif activation == 'leaky_relu':
                x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
            elif activation == 'elu':
                x = tf.keras.layers.ELU()(x)
            x = tf.keras.layers.Dropout(dropout)(x)
        out = tf.keras.layers.Dense(units[-1] if units else 64,   # fallback
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                    name=f'{name_prefix}_output')(x)
        return tf.keras.Model(inputs=inp, outputs=out, name=f'{name_prefix}_tower')
    
    def _build_model(self, trial):
        # Limpa sessão para evitar conflitos entre trials
        tf.keras.backend.clear_session()
        
        # ---- Hiperparâmetros ----
        n_layers = trial.suggest_int('n_layers', 2, 3)
        units = [trial.suggest_int(f'units_L{i}', 16, 128, log=True) for i in range(n_layers)]
        emb_units = trial.suggest_int('emb_units', 16, 64, log=True)
        combined_units = trial.suggest_int('combined_units', 16, 64, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        l2_reg = trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True)
        learning_rate = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        combined_type = trial.suggest_categorical('combined_type', ['dot_product', 'concat_dense'])
        category_emb_units = trial.suggest_int('category_emb_units', 2, 8, log=True)

        units.append(emb_units)  # Última camada da torre é a de embedding
        
        # Torres (compartilhando a mesma arquitetura para user e item)
        user_tower = self._build_tower(self.user_input_dim, units, activation, dropout, l2_reg, 'user')
        item_tower = self._build_tower(self.item_input_dim, units, activation, dropout, l2_reg, 'item')
        
        user_input = tf.keras.Input(shape=(self.user_input_dim,), name='user_embedding')
        item_input = tf.keras.Input(shape=(self.item_input_dim,), name='item_embedding')

        historic_cat_input = tf.keras.Input(shape=(len(self.cat_historic_cols),), name='historic_cat')
        item_cat_input = tf.keras.Input(shape=(len(self.cat_item_cols),), name='item_cat')
        
        user_vec = user_tower(user_input)
        item_vec = item_tower(item_input)

        emb_hist_cat = tf.keras.layers.Dense(category_emb_units, use_bias=False, kernel_initializer='glorot_uniform')(historic_cat_input)
        emb_hist_cat = tf.keras.layers.Dropout(dropout)(emb_hist_cat)
        emb_item_cat = tf.keras.layers.Dense(category_emb_units, use_bias=False, kernel_initializer='glorot_uniform')(item_cat_input)
        emb_item_cat = tf.keras.layers.Dropout(dropout)(emb_item_cat)
        
        # Camada combinada
        if combined_type == 'dot_product':
            user_norm = tf.keras.layers.LayerNormalization(axis=1)(user_vec)
            item_norm = tf.keras.layers.LayerNormalization(axis=1)(item_vec)
            hist_cat_norm = tf.keras.layers.LayerNormalization(axis=1)(emb_hist_cat)
            item_cat_norm = tf.keras.layers.LayerNormalization(axis=1)(emb_item_cat)
            combined1 = tf.keras.layers.Dot(axes=1, normalize=False)([user_norm, item_norm])
            combined2 = tf.keras.layers.Dot(axes=1, normalize=False)([hist_cat_norm, item_cat_norm])
            combined = tf.keras.layers.Concatenate()([combined1, combined2])

        else:  # concat_dense
            combined = tf.keras.layers.Concatenate()([user_vec, item_vec, emb_hist_cat, emb_item_cat])
            combined = tf.keras.layers.Dense(combined_units, kernel_initializer='he_normal')(combined)

        combined = tf.keras.layers.ReLU()(combined)
        combined = tf.keras.layers.Dropout(dropout)(combined)
        
        output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
        model = tf.keras.Model(inputs=[user_input, item_input, historic_cat_input, item_cat_input], outputs=output)
        
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            self.f1_score   # métrica customizada
        ]
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=metrics
        )
        return model
    
    def objective(self, trial):
        model = self._build_model(trial)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]
        history = model.fit(
            x={'user_embedding': self.X_user_train, 'item_embedding': self.X_item_train, 'historic_cat': self.hist_cat_train, 'item_cat': self.item_cat_train},
            y=self.y_train,
            validation_data=(
                {'user_embedding': self.X_user_val, 'item_embedding': self.X_item_val, 'historic_cat': self.hist_cat_val, 'item_cat': self.item_cat_val},
                self.y_val
            ),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=1
        )
        return max(history.history['val_auc'])
    
    def optimize(self, n_trials=50, study_name='rating_model_opt', storage=None, save_path='study.pkl', initial_params={}):
        """
        Executa a otimização Optuna e salva o estudo em disco.
        """
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            storage=storage  # Se quiser SQLite, ex: 'sqlite:///optuna.db'
        )

        if initial_params:
            if isinstance(initial_params, list):
                for params in initial_params:
                    study.enqueue_trial(params)
            else:
                study.enqueue_trial(initial_params)

        study.optimize(self.objective, n_trials=n_trials, callbacks=[])
        
        # Salva com pickle
        with open(save_path, 'wb') as f:
            pickle.dump(study, f)
        print(f"Estudo salvo em {save_path}")
        
        # Melhor resultado
        print("Melhor MAE:", study.best_value)
        print("Melhores parâmetros:", study.best_params)
        self.best_study = study
        return study
    
    def train_final_model(self, study=None, model_save_path='final_model.h5', history_save_path='history.pkl'):
        """
        Treina o modelo final com os melhores hiperparâmetros encontrados,
        usando treino+validação (sem validação separada) e avalia no teste.
        """
        if study is None:
            if hasattr(self, 'best_study'):
                study = self.best_study
            else:
                raise ValueError("Nenhum estudo fornecido. Rode optimize() primeiro.")
        
        # Ativa o melhor trial para usar seus parâmetros
        best_trial = study.best_trial
        params = best_trial.params
        
        # Constrói modelo com os melhores parâmetros (sem validação)
        tf.keras.backend.clear_session()
        # Recria o modelo manualmente utilizando os parâmetros fixos
        # (evitando chamar objective, pois não queremos trial)
        # Vamos copiar a lógica de _build_model, mas passando os valores diretamente.
        units = [params[f'units_L{i}'] for i in range(params['n_layers'])]
        emb_units = params['emb_units']
        units.append(emb_units)  # Última camada da torre é a de embedding
        user_tower = self._build_tower(self.user_input_dim, units, params['activation'],
                                       params['dropout'], params['l2_reg'], 'user')
        item_tower = self._build_tower(self.item_input_dim, units, params['activation'],
                                       params['dropout'], params['l2_reg'], 'item')
        
        user_input = tf.keras.Input(shape=(self.user_input_dim,), name='user_embedding')
        item_input = tf.keras.Input(shape=(self.item_input_dim,), name='item_embedding')
        user_vec = user_tower(user_input)
        item_vec = item_tower(item_input)

        historic_cat_input = tf.keras.Input(shape=(len(self.cat_historic_cols),), name='historic_cat')
        item_cat_input = tf.keras.Input(shape=(len(self.cat_item_cols),), name='item_cat')

        emb_hist_cat = tf.keras.layers.Dense(params['category_emb_units'], use_bias=False, kernel_initializer='glorot_uniform')(historic_cat_input)
        emb_hist_cat = tf.keras.layers.Dropout(params['dropout'])(emb_hist_cat)
        emb_item_cat = tf.keras.layers.Dense(params['category_emb_units'], use_bias=False, kernel_initializer='glorot_uniform')(item_cat_input)
        emb_item_cat = tf.keras.layers.Dropout(params['dropout'])(emb_item_cat)
        
        # Camada combinada
        if params['combined_type'] == 'dot_product':
            user_norm = tf.keras.layers.LayerNormalization(axis=1)(user_vec)
            item_norm = tf.keras.layers.LayerNormalization(axis=1)(item_vec)
            hist_cat_norm = tf.keras.layers.LayerNormalization(axis=1)(emb_hist_cat)
            item_cat_norm = tf.keras.layers.LayerNormalization(axis=1)(emb_item_cat)
            combined1 = tf.keras.layers.Dot(axes=1, normalize=False)([user_norm, item_norm])
            combined2 = tf.keras.layers.Dot(axes=1, normalize=False)([hist_cat_norm, item_cat_norm])
            combined = tf.keras.layers.Concatenate()([combined1, combined2])

        else:  # concat_dense
            combined = tf.keras.layers.Concatenate()([user_vec, item_vec, emb_hist_cat, emb_item_cat])
            combined = tf.keras.layers.Dense(params['combined_units'], kernel_initializer='he_normal')(combined)

        combined = tf.keras.layers.ReLU()(combined)
        combined = tf.keras.layers.Dropout(params['dropout'])(combined)
    
        output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)
        model = tf.keras.Model(inputs=[user_input, item_input, historic_cat_input, item_cat_input], outputs=output)

        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            self.f1_score
        ]
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss="binary_crossentropy",
            metrics=metrics
        )
        
        # Combina treino e validação
        X_user_all = np.concatenate([self.X_user_train, self.X_user_val])
        X_item_all = np.concatenate([self.X_item_train, self.X_item_val])
        hist_cat_all = np.concatenate([self.hist_cat_train, self.hist_cat_val])
        item_cat_all = np.concatenate([self.item_cat_train, self.item_cat_val])
        y_all = np.concatenate([self.y_train, self.y_val])
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
        history = model.fit(
            x={'user_embedding': self.X_user_train, 'item_embedding': self.X_item_train, 'historic_cat': self.hist_cat_train, 'item_cat': self.item_cat_train},
            y=self.y_train,
            validation_data=(
                {'user_embedding': self.X_user_val, 'item_embedding': self.X_item_val, 'historic_cat': self.hist_cat_val, 'item_cat': self.item_cat_val},
                self.y_val
            ),
            epochs=200,
            batch_size=256,
            callbacks=[early_stop],
            verbose=1
        )

        model.save(model_save_path)
        with open(history_save_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"Modelo salvo em {model_save_path}")
        print(f"Histórico salvo em {history_save_path}")
        
        # Avalia no teste
        test_loss, test_acc, test_auc, test_prec, test_rec, test_f1 = model.evaluate(
            x={'user_embedding': self.X_user_test, 'item_embedding': self.X_item_test, 'historic_cat': self.hist_cat_test, 'item_cat': self.item_cat_test},
            y=self.y_test,
            verbose=0
        )
        print(f"Teste final - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, AUC: {test_auc:.4f}, "
              f"Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}")
        
        # Predições
        preds = model.predict(
            {'user_embedding': self.X_user_test, 'item_embedding': self.X_item_test, 'historic_cat': self.hist_cat_test, 'item_cat': self.item_cat_test}
        ).flatten()
        classes = (preds > 0.5).astype(int)

        results = self.test.copy()[["app_package", "uid", "rating", "ref_rating", "target"]]
        results["probability"] = preds
        results["predicted_class"] = classes
        results["true_class"] = self.y_test

        return model, results


if __name__ == "__main__":
    
    # Importing data

    optimizer = RatingModelOptimizer(
        train_path='data/output/train_data_01_pca.csv',
        val_path='data/output/val_data_01_pca.csv',
        test_path='data/output/test_data_01_pca.csv',
        target_col='centered_rating'
    )

    initial_params = [
        {
            'n_layers': 2,
            'units_L0': 128,
            'units_L1': 128,
            'emb_units': 64,
            'combined_units': 64,
            'activation': 'leaky_relu',
            'dropout': 0.2,
            'l2_reg': 1e-2,
            'lr': 1e-3,
            'combined_type': 'concat_dense',
            'category_emb_units': 4
        },
        {
            'n_layers': 2,
            'units_L0': 128,
            'units_L1': 64,
            'emb_units': 64,
            'combined_units': 64,
            'activation': 'leaky_relu',
            'dropout': 0.2,
            'l2_reg': 1e-3,
            'lr': 5e-4,
            'combined_type': 'concat_dense',
            'category_emb_units': 6
        },
        {
            'n_layers': 2,
            'units_L0': 64,
            'units_L1': 32,
            'emb_units': 32,
            'combined_units': 16,
            'activation': 'relu',
            'dropout': 0.25,
            'l2_reg': 1e-2,
            'lr': 1e-3,
            'combined_type': 'dot_product',
            'category_emb_units': 4
        },
        {
            'n_layers': 3,
            'units_L0': 64,
            'units_L1': 32,
            'units_L2': 16,
            'emb_units': 32,
            'combined_units': 16,
            'activation': 'elu',
            'dropout': 0.4,
            'l2_reg': 1e-2,
            'lr': 5e-4,
            'combined_type': 'concat_dense',
            'category_emb_units': 3
        },
        {
            'n_layers': 2,
            'units_L0': 128,
            'units_L1': 64,
            'emb_units': 64,
            'combined_units': 64,
            'activation': 'relu',
            'dropout': 0.1,
            'l2_reg': 1e-3,
            'lr': 1e-3,
            'combined_type': 'concat_dense',
            'category_emb_units': 8
        }
]

    # Otimização
    study = optimizer.optimize(
        n_trials=100,
        save_path='best_study_01_bin_pca.pkl',
        initial_params=initial_params
    )

    # Modelo final
    model, predictions = optimizer.train_final_model(
        study,
        model_save_path='data/output/final_model_01_bin_pca.h5',
        history_save_path='data/output/history_01_bin_pca.pkl'
    )

    # Predict
    print("================\nPREDICTING\n================")
    # predictions = model.predict({
    #     'user_embedding': X_user_test,
    #     'item_embedding': X_item_test
    # }).flatten()

    # results = test.copy()[["app_package", "uid", "rating", "ref_rating", "target"]]
    # results["prediction"] = predictions
    # results["prediction_restored"] = results["prediction"] * 5 + results["ref_rating"]
    print(predictions.sample(20))
    predictions.to_csv("data/output/predictions_01_bin_pca.csv", index=False)
    print("Binary ==")
