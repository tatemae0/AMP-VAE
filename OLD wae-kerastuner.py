


import tensorflow as tf
import kerastuner as kt
import seaborn as sns
import numpy as np

def build_model(hp):
  # Define input layers for the two strings
  input_aa = tf.keras.Input(shape=(50,), name='aa')
  input_species = tf.keras.Input(shape=(None,), name='species')

  # Vectorize the first string using TextVectorization
  vectorizer_aa = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=20)
  vectorized_aa = vectorizer_aa(input_aa)

  # Vectorize the second string using TextVectorization
  vectorizer_species = tf.keras.layers.experimental.preprocessing.TextVectorization()
  vectorized_species = vectorizer_species(input_species)

  # Concatenate the two vectorized strings
  inputs = tf.keras.layers.concatenate([vectorized_aa, vectorized_species])

  # Encoder
  encoder = tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True)(inputs)
    # encoder = tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32))(encoder)

  # Latent space
  latent = tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=256, step=32))(encoder)

  # Decoder
  decoder = tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=256, step=32))(latent)
  decoder = tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True)(decoder)
    # decoder = tf.keras.layers.LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32))(decoder)

  # Output layers
  output_aa = tf.keras.layers.Dense(20, activation='softmax', name='aa_output')(decoder)
  output_species = tf.keras.layers.Dense(vectorizer_species.vocab_size, activation='softmax', name='species_output')(decoder)

  # Wasserstein loss function
  def wasserstein_loss(y_true, y_pred):
    return tf.keras.backend.mean(y_true * y_pred)

  # Compile the model
  model = tf.keras.Model(inputs=[input_aa, input_species], outputs=[output_aa, output_species])
  model.compile(optimizer='adam', loss=[wasserstein_loss, wasserstein_loss])

    return model


# Generate 100 random samples from the model
num_samples = 100
random_aa = np.random.randint(0, 20, size=(num_samples, 50))
random_species = [np.random.choice(vectorizer_species.get_vocabulary()) for _ in range(num_samples)]
random_outputs = model.predict([random_aa, random_species])

# Pass the random samples through the encoder to get their latent vectors
latent_vectors = model.encoder.predict([random_aa, random_species])

# Extract the second feature from the random samples
second_feature = [sample[1] for sample in random_species]

# Plot the latent vectors using seaborn
sns.scatterplot(latent_vectors[:, 0], latent_vectors[:, 1], hue=second_feature)
