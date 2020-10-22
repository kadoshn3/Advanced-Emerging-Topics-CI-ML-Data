from nmt import load_dataset, Encoder, BahdanauAttention, Decoder, train_step, translate
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import time

# Declare path and filename
path = 'c:\\Users\\Neeve Kadosh\\Desktop\\College\\AI_ML\\datasets\\LanguageTranslation\\'
filename = path + 'spa.txt'

# Set to None to use full dataset
num_samples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(filename, num_samples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split
test_size = 0.2
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = \
                    train_test_split(input_tensor, target_tensor, test_size=test_size)

# Hyper-parameters
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

# Create tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

# Instantiate encoder model
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
# Instantiate attention model
attention_layer = BahdanauAttention(10)
# Instantiate decoder model
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

# Sample encoder input states
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
# Sample attention layers
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)
# Sample decoder output
sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

# Loss function and optimizers
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
# Create training checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

# Declare number of epochs
EPOCHS = 10

# Train model
for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden, encoder, targ_lang, 
                                decoder, BATCH_SIZE, optimizer, loss_object)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

sentence = u'hace mucho frio aqui.'
translate(sentence, max_length_targ, max_length_inp, inp_lang, units, 
              encoder, targ_lang, decoder)