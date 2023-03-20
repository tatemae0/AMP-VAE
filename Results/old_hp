
# def build_model(hp):
#     # Define input layers for the two strings
#     input_aa = tf.keras.Input(shape=(50,aa_vocab_size), name='aa')
#     input_species = tf.keras.Input(shape=(species_vocab_size,), name='species')

#     vectorized_aa = input_aa
#     vectorized_species = input_species

#     # hyperparams
#     aa_LSTM_units = hp.Int('aa_LSTM_units', min_value=5, max_value=200, step=5)
#     species_Dense_units = hp.Int('species_Dense_units', min_value=1, max_value=30, step=4)
#     latent_dim = hp.Int('latent_dim', min_value=1, max_value=12, step=1) # Must be < 20 for undercompleteness. There are ~10 descriptors
#     # latent_dim=2
#     output_activation = hp.Choice('output_activation', values=['softmax','linear'])

#     # aa encoder
#     encoder_aa = tf.keras.layers.LSTM(units=aa_LSTM_units, return_sequences=True)(vectorized_aa)

#     # species encoder
#     encoder_species = tf.keras.layers.Dense(units=species_Dense_units)(vectorized_species)
#     encoder_species = tf.keras.layers.Dense(units=aa_LSTM_units)(encoder_species) # for concatenation
#     encoder_species = keras.layers.Reshape((1,encoder_species.shape[1]))(encoder_species)
#     encoder = tf.keras.layers.Concatenate(axis=1)([encoder_aa, encoder_species]) # GEt SHAPES RIGHT so this works.
#     # encoder= [encoder_aa, encoder_species]


#     # Latent space parameters

#     latent_mean = tf.keras.layers.Dense(units=latent_dim)(encoder) # fully encoded
#     latent_log_var = tf.keras.layers.Dense(units=latent_dim)(encoder) # fully encoded


#     # Sample from the latent space
#     def sampling(args):
#         mean, log_var = args
#         epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean))
#         return mean + tf.exp(0.5 * log_var) * epsilon

#     latent = tf.keras.layers.Lambda(sampling)([latent_mean, latent_log_var])

#     # Decoder
#     decoder_aa = tf.keras.layers.LSTM(units=aa_LSTM_units, return_sequences=True)(latent)
#     decoder_species = tf.keras.layers.Dense(units=species_Dense_units)(latent)

#     # Output layers
#     output_aa = tf.keras.layers.Dense(aa_vocab_size, activation=output_activation, name='aa_output')(decoder_aa[:,:max_aa_length]) # What activation?

#     output_species = tf.keras.layers.Dense(species_vocab_size, activation=output_activation, name='species_output')(decoder_species[:,max_aa_length:])

#     # Wasserstein loss function
#     # def wasserstein_loss(y_true, y_pred):
#     #     product = tf.keras.layers.Multiply()([y_true, y_pred])
#     #     return tf.keras.layers.Average()([product])
#     # Compute the KL divergence loss
# #     def kl_divergence_loss(y_true, y_pred):
# #         kl_loss = 1 + latent_log_var - tf.square(latent_mean) - tf.exp(latent_log_var)
# #         kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(kl_loss, axis=-1))
# #         return kl_loss

#     # Compile the model
#     mse = tf.keras.losses.MeanSquaredError()
#     model = tf.keras.Model(inputs=[input_aa, input_species], outputs=[output_aa, output_species])
#     model.compile(optimizer='adam', loss=[mse, mse],
#                   loss_weights=[1, .05], metrics=[tf.keras.metrics.KLDivergence()])
#     return model
