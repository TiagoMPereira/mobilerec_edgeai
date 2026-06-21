import tensorflow as tf

model = tf.keras.models.load_model(
    "data/output/final_model_01_bin_pca.h5",
    compile=False
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS
]

tflite_model = converter.convert()

with open("data/output/final_model_01_bin_pca.tflite", "wb") as f:
    f.write(tflite_model)