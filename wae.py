# VAE ChatGPT Edited

import keras
from keras import layers
from keras import backend as K

import numpy as np
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Input data is a list of strings with two elements: an amino acid sequence and a species name
data = [("MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLV", "Homo sapiens"),
        ("HLTPEAEFTPAVHASLDKFLASVSTVLTSKYR", "Pan troglodytes"),
        ("KVKAHGKKVLGAFSDGLAHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR", "Gorilla gorilla"),
        ("KSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPLASVSTVLTSKYR", "Pongo abelii")]

# Vectorize the amino acid sequences using the TextVectorization function in Keras
max_aa_length = 50
aa_vocab_size = 20
aa_vectorizer = keras.preprocessing.text.Tokenizer(num_words=aa_vocab_size, char_level=True)
aa_vectorizer.fit_on_texts([d[0] for d in data])
aa_sequences = aa_vectorizer.texts_to_sequences([d[0] for d in data])
aa_sequences = keras.preprocessing.sequence.pad_sequences(aa_sequences, maxlen=max_aa_length, padding='post')

# Vectorize the species names using the TextVectorization function in Keras
species_vocab_size = 4
species_vectorizer = keras.preprocessing.text.Tokenizer(num_words=species_vocab_size)
species_vectorizer.fit_on_texts([d[1] for d in data])
species_sequences = species_vectorizer.texts_to_sequences([d[1] for d in data])

# Concatenate the amino acid sequences and species names into a single input tensor
inputs = np.concatenate([aa_sequences, species_sequences], axis=-1)

# Replace LSTM & Dense arguments with hyper parameters to train using keras tuner
# Build the encoder model
#    hp_lstm_dim = hp.Int("n_hidden", min_value=1, max_value=20)
#    hp_dense_dim = hp.Int("n_hidden", min_value=1, max_value=20)

# def build_model(hp):
latent_dim = 2                   #  Chosen to be 2 for visualization.
inputs_shape = inputs.shape[1:]
encoder_inputs = keras.Input(shape=inputs_shape)
x = layers.LSTM(32, return_sequences=True)(encoder_inputs)
x = layers.Dense(16, activation='relu')(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Define the sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var) * epsilon

# Define the encoder model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var], name='encoder')

# Define the latent inputs for the decoder
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(16, activation='relu')(latent_inputs)
x = layers.RepeatVector(max_aa_length)(x)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.TimeDistributed(layers.Dense(aa_vocab_size, activation='softmax'))(x)
decoder_outputs = x

# Define the decoder model
decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

# Define the VAE model
outputs = decoder(encoder(encoder_inputs)[0])
vae = keras.Model(encoder_inputs, outputs, name='vae')

# Define the loss function as the Wasserstein metric
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

hp_lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4, 1e-5])
tf.keras.optimizers.adam(learning_rate=hp_lr)

# Compile the model
vae.compile(optimizer='adam', loss=wasserstein_loss)

# Train the VAE model
# vae.fit(inputs, inputs, epochs=10, batch_size=32)

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy', # CHECK THIS
    max_epochs=30,
    hyperband_iterations=2)
# To sample 100 outputs from the model, pass them into the model encoder, and plot their latent vectors in seaborn, you can use the following code:

# Sample 100 outputs from the VAE model
num_samples = 100
outputs = vae.predict(inputs[:num_samples])

# Pass the outputs through the encoder to obtain the latent vectors
latent_vectors = encoder.predict(outputs)[0]

# Color the latent vectors according to their second feature
colors = np.array(species_sequences)[:num_samples,0]

# Plot the latent vectors using seaborn
sns.scatterplot(x=latent_vectors[:,0], y=latent_vectors[:,1], hue=colors, palette=sns.color_palette('hls', species_vocab_size))
