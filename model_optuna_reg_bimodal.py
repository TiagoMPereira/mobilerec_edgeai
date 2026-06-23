import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna


class RatingModelOptimizer:
    def __init__(self, train_path, val_path, test_path, target_col='centered_rating',
                 user_cols_suffix='user_emb', item_cols_suffix='item_emb', random_state=42):
        self.random_state = random_state
        self.target_col = target_col
        self.user_cols_suffix = user_cols_suffix
        self.item_cols_suffix = item_cols_suffix
        self.target_scale = 5.0

        # Carrega CSVs
        self.train = pd.read_csv(train_path, index_col=0)
        self.val   = pd.read_csv(val_path, index_col=0)
        self.test  = pd.read_csv(test_path, index_col=0)

        # Embaralha
        self.train = self.train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.val   = self.val.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.test  = self.test.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Normaliza target para [-1, 1] (centered_rating ∈ [-5, 5])
        self.train["target"] = (self.train[self.target_col] / self.target_scale).astype('float32')
        self.val["target"]   = (self.val[self.target_col]   / self.target_scale).astype('float32')
        self.test["target"]  = (self.test[self.target_col]  / self.target_scale).astype('float32')
        self.target_col = "target"

        print(self.train[self.target_col].describe())

        # Separa colunas
        self.user_cols = [c for c in self.train.columns if user_cols_suffix in c]
        self.item_cols = [c for c in self.train.columns if item_cols_suffix in c]
        self.cat_historic_cols = [c for c in self.train.columns if c.startswith("cat_")]
        self.cat_item_cols = [c for c in self.train.columns if c.startswith("item_cat_")]

        print(f"user_cols: {len(self.user_cols)}  item_cols: {len(self.item_cols)}")

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
        self.output_bias = float(np.mean(self.y_train))
        print(f"Dados carregados: treino={self.X_user_train.shape}, "
              f"val={self.X_user_val.shape}, teste={self.X_user_test.shape}")
        print(f"Target bias (mean normalizado): {self.output_bias:.4f}")

    @staticmethod
    def r2_score(y_true, y_pred):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())

    @staticmethod
    def _make_loss(loss_type, huber_delta, aux_weight):
        """
        Retorna uma função de perda por-amostra compatível com sample_weight.

        Combina a loss de regressão principal com uma loss auxiliar de classificação
        binária (sinal da nota), que força separação explícita entre avaliações
        positivas e negativas — crítico para a distribuição bimodal dos dados.

        Retorno por-amostra (shape batch,) permite que o Keras aplique
        sample_weight corretamente antes de reduzir.
        """
        def loss_fn(y_true, y_pred):
            y_true = tf.squeeze(y_true, axis=-1) if y_true.shape.rank > 1 else y_true
            y_pred = tf.squeeze(y_pred, axis=-1) if y_pred.shape.rank > 1 else y_pred

            diff     = y_true - y_pred
            abs_diff = tf.abs(diff)

            if loss_type == 'mse':
                main = tf.square(diff)
            elif loss_type == 'mae':
                main = abs_diff
            else:  # huber
                main = tf.where(
                    abs_diff <= huber_delta,
                    0.5 * tf.square(diff),
                    huber_delta * (abs_diff - 0.5 * huber_delta)
                )

            if aux_weight > 0.0:
                # BCE auxiliar: sinal da nota (positivo/negativo)
                # sigmoid(y_pred * 3) mapeia [-1,1] para [0.05, 0.95] c/ boa inclinação
                y_sign    = tf.cast(y_true > 0, tf.float32)
                y_prob    = tf.sigmoid(y_pred * 3.0)
                eps       = 1e-7
                bce       = -(y_sign * tf.math.log(y_prob + eps) +
                              (1.0 - y_sign) * tf.math.log(1.0 - y_prob + eps))
                return main + aux_weight * bce

            return main

        return loss_fn

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
        out = tf.keras.layers.Dense(units[-1] if units else 64,
                                    kernel_initializer='he_normal',
                                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                    name=f'{name_prefix}_output')(x)
        return tf.keras.Model(inputs=inp, outputs=out, name=f'{name_prefix}_tower')

    def _build_model(self, trial):
        tf.keras.backend.clear_session()

        # ---- Hiperparâmetros ----
        n_layers       = trial.suggest_int('n_layers', 2, 3)
        units          = [trial.suggest_int(f'units_L{i}', 16, 128, log=True) for i in range(n_layers)]
        emb_units      = trial.suggest_int('emb_units', 16, 64, log=True)
        combined_units = trial.suggest_int('combined_units', 16, 64, log=True)
        activation     = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
        dropout        = trial.suggest_float('dropout', 0.05, 0.3)
        l2_reg         = trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True)
        learning_rate  = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        combined_type  = trial.suggest_categorical('combined_type', ['dot_product', 'concat_dense'])
        cat_emb_units  = trial.suggest_int('category_emb_units', 2, 8, log=True)
        loss_type      = trial.suggest_categorical('loss_type', ['mse', 'huber', 'mae'])
        huber_delta    = trial.suggest_float('huber_delta', 0.3, 2.0)
        # Peso amostral: extremos recebem mais atenção (combate à distribuição bimodal)
        weight_power   = trial.suggest_categorical('weight_power', [0.5, 1.0, 1.5, 2.0])
        # Peso da loss auxiliar BCE (sinal da nota — positivo vs negativo)
        aux_weight     = trial.suggest_categorical('aux_weight', [0.0, 0.1, 0.3, 0.5])

        units.append(emb_units)

        user_tower = self._build_tower(self.user_input_dim, units, activation, dropout, l2_reg, 'user')
        item_tower = self._build_tower(self.item_input_dim, units, activation, dropout, l2_reg, 'item')

        user_input         = tf.keras.Input(shape=(self.user_input_dim,), name='user_embedding')
        item_input         = tf.keras.Input(shape=(self.item_input_dim,), name='item_embedding')
        historic_cat_input = tf.keras.Input(shape=(len(self.cat_historic_cols),), name='historic_cat')
        item_cat_input     = tf.keras.Input(shape=(len(self.cat_item_cols),), name='item_cat')

        user_vec = user_tower(user_input)
        item_vec = item_tower(item_input)

        emb_hist_cat = tf.keras.layers.Dense(cat_emb_units, use_bias=False, kernel_initializer='glorot_uniform')(historic_cat_input)
        emb_hist_cat = tf.keras.layers.Dropout(dropout)(emb_hist_cat)
        emb_item_cat = tf.keras.layers.Dense(cat_emb_units, use_bias=False, kernel_initializer='glorot_uniform')(item_cat_input)
        emb_item_cat = tf.keras.layers.Dropout(dropout)(emb_item_cat)

        if combined_type == 'dot_product':
            user_norm      = tf.keras.layers.LayerNormalization(axis=1)(user_vec)
            item_norm      = tf.keras.layers.LayerNormalization(axis=1)(item_vec)
            hist_cat_norm  = tf.keras.layers.LayerNormalization(axis=1)(emb_hist_cat)
            item_cat_norm  = tf.keras.layers.LayerNormalization(axis=1)(emb_item_cat)
            combined1      = tf.keras.layers.Dot(axes=1, normalize=False)([user_norm, item_norm])
            combined2      = tf.keras.layers.Dot(axes=1, normalize=False)([hist_cat_norm, item_cat_norm])
            combined       = tf.keras.layers.Concatenate()([combined1, combined2])
            combined       = tf.keras.layers.Dense(combined_units, kernel_initializer='he_normal')(combined)
        else:  # concat_dense
            combined = tf.keras.layers.Concatenate()([user_vec, item_vec, emb_hist_cat, emb_item_cat])
            combined = tf.keras.layers.Dense(combined_units, kernel_initializer='he_normal')(combined)

        combined = tf.keras.layers.ReLU()(combined)
        combined = tf.keras.layers.Dropout(dropout)(combined)

        # Saída linear — sem tanh para evitar saturação de gradiente nos extremos
        output_bias_init = tf.keras.initializers.Constant(self.output_bias)
        output = tf.keras.layers.Dense(1, bias_initializer=output_bias_init)(combined)
        model  = tf.keras.Model(inputs=[user_input, item_input, historic_cat_input, item_cat_input], outputs=output)

        loss_fn = self._make_loss(loss_type, huber_delta, float(aux_weight))
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            self.r2_score,
        ]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=metrics
        )
        return model

    def _compute_sample_weights(self, y, weight_power):
        """Peso proporcional à magnitude da nota — extremos recebem mais atenção."""
        w = 1.0 + np.abs(y) ** float(weight_power)
        return (w / w.mean()).astype('float32')   # normaliza para média 1

    def objective(self, trial):
        model        = self._build_model(trial)
        weight_power = trial.params['weight_power']
        sw_train     = self._compute_sample_weights(self.y_train, weight_power)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]
        history = model.fit(
            x={'user_embedding': self.X_user_train, 'item_embedding': self.X_item_train,
               'historic_cat': self.hist_cat_train, 'item_cat': self.item_cat_train},
            y=self.y_train,
            sample_weight=sw_train,
            validation_data=(
                {'user_embedding': self.X_user_val, 'item_embedding': self.X_item_val,
                 'historic_cat': self.hist_cat_val, 'item_cat': self.item_cat_val},
                self.y_val
            ),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=0
        )
        return min(history.history['val_mae'])

    def optimize(self, n_trials=50, study_name='rating_model_reg', storage=None, save_path='study.pkl', initial_params={}):
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            storage=storage
        )

        if initial_params:
            if isinstance(initial_params, list):
                for params in initial_params:
                    study.enqueue_trial(params)
            else:
                study.enqueue_trial(initial_params)

        study.optimize(self.objective, n_trials=n_trials, callbacks=[])

        with open(save_path, 'wb') as f:
            pickle.dump(study, f)
        print(f"Estudo salvo em {save_path}")

        print("Melhor MAE (normalizado):", study.best_value)
        print(f"Melhor MAE (escala original ×{self.target_scale}): {study.best_value * self.target_scale:.4f}")
        print("Melhores parâmetros:", study.best_params)
        self.best_study = study
        return study

    def train_final_model(self, study=None, model_save_path='final_model.h5', history_save_path='history.pkl'):
        if study is None:
            if hasattr(self, 'best_study'):
                study = self.best_study
            else:
                raise ValueError("Nenhum estudo fornecido. Rode optimize() primeiro.")

        best_trial = study.best_trial
        params     = best_trial.params

        tf.keras.backend.clear_session()

        units = [params[f'units_L{i}'] for i in range(params['n_layers'])]
        units.append(params['emb_units'])

        user_tower = self._build_tower(self.user_input_dim, units, params['activation'],
                                       params['dropout'], params['l2_reg'], 'user')
        item_tower = self._build_tower(self.item_input_dim, units, params['activation'],
                                       params['dropout'], params['l2_reg'], 'item')

        user_input         = tf.keras.Input(shape=(self.user_input_dim,), name='user_embedding')
        item_input         = tf.keras.Input(shape=(self.item_input_dim,), name='item_embedding')
        historic_cat_input = tf.keras.Input(shape=(len(self.cat_historic_cols),), name='historic_cat')
        item_cat_input     = tf.keras.Input(shape=(len(self.cat_item_cols),), name='item_cat')

        user_vec = user_tower(user_input)
        item_vec = item_tower(item_input)

        emb_hist_cat = tf.keras.layers.Dense(params['category_emb_units'], use_bias=False, kernel_initializer='glorot_uniform')(historic_cat_input)
        emb_hist_cat = tf.keras.layers.Dropout(params['dropout'])(emb_hist_cat)
        emb_item_cat = tf.keras.layers.Dense(params['category_emb_units'], use_bias=False, kernel_initializer='glorot_uniform')(item_cat_input)
        emb_item_cat = tf.keras.layers.Dropout(params['dropout'])(emb_item_cat)

        if params['combined_type'] == 'dot_product':
            user_norm     = tf.keras.layers.LayerNormalization(axis=1)(user_vec)
            item_norm     = tf.keras.layers.LayerNormalization(axis=1)(item_vec)
            hist_cat_norm = tf.keras.layers.LayerNormalization(axis=1)(emb_hist_cat)
            item_cat_norm = tf.keras.layers.LayerNormalization(axis=1)(emb_item_cat)
            combined1     = tf.keras.layers.Dot(axes=1, normalize=False)([user_norm, item_norm])
            combined2     = tf.keras.layers.Dot(axes=1, normalize=False)([hist_cat_norm, item_cat_norm])
            combined      = tf.keras.layers.Concatenate()([combined1, combined2])
            combined      = tf.keras.layers.Dense(params['combined_units'], kernel_initializer='he_normal')(combined)
        else:
            combined = tf.keras.layers.Concatenate()([user_vec, item_vec, emb_hist_cat, emb_item_cat])
            combined = tf.keras.layers.Dense(params['combined_units'], kernel_initializer='he_normal')(combined)

        combined = tf.keras.layers.ReLU()(combined)
        combined = tf.keras.layers.Dropout(params['dropout'])(combined)

        output_bias_init = tf.keras.initializers.Constant(self.output_bias)
        output = tf.keras.layers.Dense(1, bias_initializer=output_bias_init)(combined)
        model  = tf.keras.Model(inputs=[user_input, item_input, historic_cat_input, item_cat_input], outputs=output)

        loss_fn = self._make_loss(params['loss_type'], params['huber_delta'], float(params['aux_weight']))
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name='mae'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            self.r2_score,
        ]
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss=loss_fn,
            metrics=metrics
        )

        sw_train  = self._compute_sample_weights(self.y_train, params['weight_power'])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        history = model.fit(
            x={'user_embedding': self.X_user_train, 'item_embedding': self.X_item_train,
               'historic_cat': self.hist_cat_train, 'item_cat': self.item_cat_train},
            y=self.y_train,
            sample_weight=sw_train,
            validation_data=(
                {'user_embedding': self.X_user_val, 'item_embedding': self.X_item_val,
                 'historic_cat': self.hist_cat_val, 'item_cat': self.item_cat_val},
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

        test_loss, test_mae, test_rmse, test_r2 = model.evaluate(
            x={'user_embedding': self.X_user_test, 'item_embedding': self.X_item_test,
               'historic_cat': self.hist_cat_test, 'item_cat': self.item_cat_test},
            y=self.y_test,
            verbose=0
        )
        scale = self.target_scale
        print(f"Teste final | "
              f"MAE: {test_mae:.4f} ({test_mae * scale:.4f} original) | "
              f"RMSE: {test_rmse:.4f} ({test_rmse * scale:.4f} original) | "
              f"R²: {test_r2:.4f}")

        preds_norm = model.predict(
            {'user_embedding': self.X_user_test, 'item_embedding': self.X_item_test,
             'historic_cat': self.hist_cat_test, 'item_cat': self.item_cat_test}
        ).flatten()

        results = self.test.copy()[["app_package", "uid", "rating", "ref_rating", "target"]]
        results["predicted_rating_norm"] = preds_norm
        results["predicted_rating"]      = preds_norm * scale
        results["true_rating_norm"]      = self.y_test
        results["true_centered_rating"]  = self.y_test * scale

        return model, results


if __name__ == "__main__":

    optimizer = RatingModelOptimizer(
        train_path='data/output/train_data_01_pca.csv',
        val_path='data/output/val_data_01_pca.csv',
        test_path='data/output/test_data_01_pca.csv',
        target_col='centered_rating'
    )

    initial_params = [
        # MSE + peso alto nos extremos + aux BCE forte → combate à bimodalidade
        {
            'n_layers': 2, 'units_L0': 128, 'units_L1': 128,
            'emb_units': 64, 'combined_units': 64,
            'activation': 'leaky_relu', 'dropout': 0.15, 'l2_reg': 1e-4,
            'lr': 1e-3, 'combined_type': 'concat_dense', 'category_emb_units': 4,
            'loss_type': 'mse', 'huber_delta': 1.0, 'weight_power': 1.5, 'aux_weight': 0.3
        },
        # MSE + peso muito alto + aux BCE moderado
        {
            'n_layers': 2, 'units_L0': 128, 'units_L1': 64,
            'emb_units': 64, 'combined_units': 64,
            'activation': 'leaky_relu', 'dropout': 0.1, 'l2_reg': 1e-4,
            'lr': 5e-4, 'combined_type': 'concat_dense', 'category_emb_units': 6,
            'loss_type': 'mse', 'huber_delta': 1.0, 'weight_power': 2.0, 'aux_weight': 0.1
        },
        # Huber + peso médio + aux BCE desligado (baseline)
        {
            'n_layers': 2, 'units_L0': 64, 'units_L1': 32,
            'emb_units': 32, 'combined_units': 32,
            'activation': 'relu', 'dropout': 0.2, 'l2_reg': 5e-5,
            'lr': 1e-3, 'combined_type': 'dot_product', 'category_emb_units': 4,
            'loss_type': 'huber', 'huber_delta': 1.0, 'weight_power': 1.0, 'aux_weight': 0.0
        },
        # MAE + peso alto + aux BCE forte
        {
            'n_layers': 3, 'units_L0': 64, 'units_L1': 32, 'units_L2': 16,
            'emb_units': 32, 'combined_units': 32,
            'activation': 'elu', 'dropout': 0.25, 'l2_reg': 1e-4,
            'lr': 5e-4, 'combined_type': 'concat_dense', 'category_emb_units': 3,
            'loss_type': 'mae', 'huber_delta': 1.5, 'weight_power': 1.5, 'aux_weight': 0.5
        },
        # MSE + peso máximo + aux BCE moderado + LR alto
        {
            'n_layers': 2, 'units_L0': 128, 'units_L1': 64,
            'emb_units': 64, 'combined_units': 64,
            'activation': 'relu', 'dropout': 0.1, 'l2_reg': 1e-4,
            'lr': 2e-3, 'combined_type': 'concat_dense', 'category_emb_units': 8,
            'loss_type': 'mse', 'huber_delta': 0.7, 'weight_power': 2.0, 'aux_weight': 0.3
        }
    ]

    study = optimizer.optimize(
        n_trials=100,
        save_path='best_study_01_reg_bim_pca.pkl',
        initial_params=initial_params
    )

    model, predictions = optimizer.train_final_model(
        study,
        model_save_path='data/output/final_model_01_reg_bim_pca.h5',
        history_save_path='data/output/history_01_reg_bim_pca.pkl'
    )

    print("================\nPREDICTING\n================")
    print(predictions.sample(20))
    predictions.to_csv("data/output/predictions_01_reg_bim_pca.csv", index=False)
    print("Regression ==")
