import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.integration import TFKerasPruningCallback

class RatingModelOptimizer:
    def __init__(self, train_path, val_path, test_path, target_col='centered_rating', 
                 user_cols_suffix='user', item_cols_suffix='item', random_state=42):
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

        self.train[self.target_col] = self.train[self.target_col] / 5
        self.val[self.target_col] = self.val[self.target_col] / 5
        self.test[self.target_col] = self.test[self.target_col] / 5

        print(self.train)

        # Separa colunas
        self.user_cols = [c for c in self.train.columns if user_cols_suffix in c]
        self.item_cols = [c for c in self.train.columns if item_cols_suffix in c]

        print(self.user_cols)
        print(self.item_cols)

        # Normalização das features usando apenas estatísticas do treino
        self.user_mean = self.train[self.user_cols].mean()
        self.user_std = self.train[self.user_cols].std(ddof=0).replace(0, 1)
        self.item_mean = self.train[self.item_cols].mean()
        self.item_std = self.train[self.item_cols].std(ddof=0).replace(0, 1)

        self.train[self.user_cols] = (self.train[self.user_cols].fillna(0.0) - self.user_mean) / self.user_std
        self.val[self.user_cols] = (self.val[self.user_cols].fillna(0.0) - self.user_mean) / self.user_std
        self.test[self.user_cols] = (self.test[self.user_cols].fillna(0.0) - self.user_mean) / self.user_std

        self.train[self.item_cols] = (self.train[self.item_cols].fillna(0.0) - self.item_mean) / self.item_std
        self.val[self.item_cols] = (self.val[self.item_cols].fillna(0.0) - self.item_mean) / self.item_std
        self.test[self.item_cols] = (self.test[self.item_cols].fillna(0.0) - self.item_mean) / self.item_std
        
        # Converte para float32
        self.X_user_train = self.train[self.user_cols].values.astype('float32')
        self.X_item_train = self.train[self.item_cols].values.astype('float32')
        self.y_train = self.train[self.target_col].values.astype('float32')
        
        self.X_user_val = self.val[self.user_cols].values.astype('float32')
        self.X_item_val = self.val[self.item_cols].values.astype('float32')
        self.y_val = self.val[self.target_col].values.astype('float32')
        
        self.X_user_test = self.test[self.user_cols].values.astype('float32')
        self.X_item_test = self.test[self.item_cols].values.astype('float32')
        self.y_test = self.test[self.target_col].values.astype('float32')
        
        # Dimensões
        self.user_input_dim = self.X_user_train.shape[1]
        self.item_input_dim = self.X_item_train.shape[1]
        self.output_bias = np.mean(self.y_train)
        print(f"Dados carregados: treino={self.X_user_train.shape}, "
              f"val={self.X_user_val.shape}, teste={self.X_user_test.shape}")
    
    @staticmethod
    def mse_std_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        std_true = tf.math.reduce_std(y_true)
        std_pred = tf.math.reduce_std(y_pred)
        penalty = tf.where(
            tf.logical_and(std_true > 1e-6, std_pred > 1e-6),
            tf.abs(std_true - std_pred),
            0.0
        )
        return mse + 0.1 * penalty

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
        n_layers = trial.suggest_int('n_layers', 2, 4)
        units = [trial.suggest_int(f'units_L{i}', 32, 512, log=True) for i in range(n_layers)]
        emb_units = trial.suggest_int('emb_units', 16, 256, log=True)
        combined_units = trial.suggest_int('combined_units', 16, 256, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        l2_reg = trial.suggest_float('l2_reg', 1e-7, 1e-2, log=True)
        learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        combined_type = trial.suggest_categorical('combined_type', ['dot_product', 'concat_dense'])

        units.append(emb_units)  # Última camada da torre é a de embedding
        
        # Torres (compartilhando a mesma arquitetura para user e item)
        user_tower = self._build_tower(self.user_input_dim, units, activation, dropout, l2_reg, 'user')
        item_tower = self._build_tower(self.item_input_dim, units, activation, dropout, l2_reg, 'item')
        
        user_input = tf.keras.Input(shape=(self.user_input_dim,), name='user_embedding')
        item_input = tf.keras.Input(shape=(self.item_input_dim,), name='item_embedding')
        user_vec = user_tower(user_input)
        item_vec = item_tower(item_input)
        
        # Camada combinada
        if combined_type == 'dot_product':
            user_norm = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(user_vec)
            item_norm = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(item_vec)
            combined = tf.keras.layers.Dot(axes=1, normalize=False)([user_norm, item_norm])

        else:  # concat_dense
            combined = tf.keras.layers.Concatenate()([user_vec, item_vec])

        combined = tf.keras.layers.Dense(combined_units, kernel_initializer='he_normal')(combined)
        combined = tf.keras.layers.ReLU()(combined)
        combined = tf.keras.layers.Dropout(dropout)(combined)
        
        output = tf.keras.layers.Dense(1, activation='linear')(combined)
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self.mse_std_loss,
            # loss=tf.keras.losses.Huber(),
            metrics=[
                tf.keras.metrics.MeanSquaredError(name='mse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse')
            ]
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
            x={'user_embedding': self.X_user_train, 'item_embedding': self.X_item_train},
            y=self.y_train,
            validation_data=(
                {'user_embedding': self.X_user_val, 'item_embedding': self.X_item_val},
                self.y_val
            ),
            epochs=100,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        return min(history.history['val_loss'])
    
    def optimize(self, n_trials=50, study_name='rating_model_opt', storage=None, save_path='study.pkl', initial_params={}):
        """
        Executa a otimização Optuna e salva o estudo em disco.
        """
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
            storage=storage  # Se quiser SQLite, ex: 'sqlite:///optuna.db'
        )

        if initial_params:
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
        
        if params['combined_type'] == 'dot_product':
            user_norm = tf.keras.layers.LayerNormalization(axis=1)(user_vec)
            item_norm = tf.keras.layers.LayerNormalization(axis=1)(item_vec)
            combined = tf.keras.layers.Dot(axes=1, normalize=False)([user_norm, item_norm])
        else:
            combined = tf.keras.layers.Concatenate()([user_vec, item_vec])
        combined = tf.keras.layers.Dense(params['combined_units'], kernel_initializer='he_normal')(combined)
        combined = tf.keras.layers.ReLU()(combined)
        combined = tf.keras.layers.Dropout(params['dropout'])(combined)
    
        output = tf.keras.layers.Dense(1, activation='linear')(combined)
        model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss=self.mse_std_loss,
            # loss=tf.keras.losses.Huber(),
            metrics=[
                tf.keras.metrics.MeanSquaredError(name='mse'),
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        
        # Combina treino e validação
        X_user_all = np.concatenate([self.X_user_train, self.X_user_val])
        X_item_all = np.concatenate([self.X_item_train, self.X_item_val])
        y_all = np.concatenate([self.y_train, self.y_val])
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
        history = model.fit(
            x={'user_embedding': X_user_all, 'item_embedding': X_item_all},
            y=y_all,
            epochs=200,
            batch_size=128,
            callbacks=[early_stop],
            verbose=1
        )

        model.save(model_save_path)
        with open(history_save_path, 'wb') as f:
            pickle.dump(history.history, f)
        print(f"Modelo salvo em {model_save_path}")
        print(f"Histórico salvo em {history_save_path}")
        
        # Avalia no teste
        test_loss, test_mse, test_mae, test_rmse = model.evaluate(
            x={'user_embedding': self.X_user_test, 'item_embedding': self.X_item_test},
            y=self.y_test,
            verbose=0
        )
        print(f"Teste final - Loss: {test_loss:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        # Predições
        preds = model.predict(
            {'user_embedding': self.X_user_test, 'item_embedding': self.X_item_test}
        ).flatten()

        results = self.test.copy()[["app_package", "uid", "rating", "ref_rating", "target"]]
        results["prediction"] = preds
        results["prediction_restored"] = results["prediction"] * 5 + results["ref_rating"]

        return model, results


if __name__ == "__main__":
    
    # Importing data

    optimizer = RatingModelOptimizer(
        train_path='data/output/train_data_01.csv',
        val_path='data/output/val_data_01.csv',
        test_path='data/output/test_data_01.csv',
        target_col='centered_rating'
    )

    initial_params = {
        'n_layers': 2,
        'units_L0': 256,
        'units_L1': 128,
        'emb_units': 64,
        'combined_units': 32,
        'activation': 'leaky_relu',
        'dropout': 0.1,
        'l2_reg': 1e-5,
        'lr': 1e-4,
        'combined_type': 'dot_product'
    }

    # Otimização
    study = optimizer.optimize(
        n_trials=50,
        save_path='best_study_01_msestd.pkl',
        initial_params=initial_params
    )

    # Modelo final
    model, predictions = optimizer.train_final_model(
        study,
        model_save_path='data/output/final_model_01_msestd.h5',
        history_save_path='data/output/history_01_msestd.pkl'
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
    predictions.to_csv("data/output/predictions_01_msestd.csv", index=False)
    print("MSE == FINALIZADO")
