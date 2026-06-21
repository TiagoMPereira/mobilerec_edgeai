import tensorflow as tf
import numpy as np
import pandas as pd

import tensorflow as tf
import numpy as np
import pandas as pd

# 1. Carregar o modelo TFLite
interpreter = tf.lite.Interpreter(model_path='data/output/final_model_015.tflite')
interpreter.allocate_tensors()

# 2. Obter detalhes das entradas/saídas
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Número de entradas: {len(input_details)}")
for i, detail in enumerate(input_details):
    print(f"  Entrada {i}: shape={detail['shape']}")

# 3. Ler o dataset CSV
df = pd.read_csv('data/output/test_data_015.csv')  # Substitua pelo caminho do seu CSV

user_cols = [col for col in df.columns if col.startswith('user')]
item_cols = [col for col in df.columns if col.startswith('item')]

X_user = df[user_cols].values.astype(np.float32)
X_item = df[item_cols].values.astype(np.float32)


# print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")

# 5. Fazer predict para cada amostra individualmente
predictions = []

for i in range(X_user.shape[0]):
    
    # Separar user e item
    X_user_current = X_user[i:i+1, :]
    X_item_current = X_item[i:i+1, :]
    
    # Definir os tensores de entrada

    interpreter.set_tensor(input_details[0]['index'], X_user_current)
    interpreter.set_tensor(input_details[1]['index'], X_item_current)
    
    # Executar a inferência
    interpreter.invoke()
    
    # Obter o resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions.append(output_data[0][0])

# 6. Adicionar predições ao DataFrame
df['prediction'] = predictions

# 7. Salvar resultados
df.to_csv('predicoes.csv', index=False)
print(f"\n✅ Predições salvas em 'predicoes.csv'")
print(f"Primeiras 10 predições:")
for i in range(min(10, len(predictions))):
    print(f"  Amostra {i}: {predictions[i]:.4f}")





# # Carregar modelo
# interpreter = tf.lite.Interpreter(model_path='data/output/final_model_015.tflite')
# interpreter.allocate_tensors()
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Ler CSV
# df = pd.read_csv('data/output/test_data_015.csv')

# # Separar user e item features (ajuste os nomes das colunas)
# # Exemplo: colunas que começam com 'user_' são do usuário
# user_cols = [col for col in df.columns if col.startswith('user')]
# item_cols = [col for col in df.columns if col.startswith('item')]

# X_user = df[user_cols].values.astype(np.float32)
# X_item = df[item_cols].values.astype(np.float32)

# # Fazer predict
# interpreter.set_tensor(input_details[0]['index'], X_user)
# interpreter.set_tensor(input_details[1]['index'], X_item)
# interpreter.invoke()

# predictions = interpreter.get_tensor(output_details[0]['index']).flatten()

# # Salvar
# df['prediction'] = predictions
# df.to_csv('predicoes.csv', index=False)
# print("✅ Predições salvas em 'predicoes.csv'")