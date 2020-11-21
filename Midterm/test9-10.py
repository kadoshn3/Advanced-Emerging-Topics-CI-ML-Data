from keras.models import Model
from keras.layers import Dense, Input, GRU
from keras.layers.embeddings import Embedding
import re

# Load and clean a text file
def fClean_Load(filename):
    file = open(filename, encoding="utf8", errors='ignore') 
    #file = open(filename, 'rb')
    text = file.read()
    file.close()
    # Clean text
    words = re.findall(r'[a-z\.]+', text.lower())
    return ' '.join(words)

# load text / Complete novel "A Tale of Two Cities"
raw_text = fClean_Load('AToTC.txt').split(' ')

word_dim = 50
num_tokens = 15000

# Define the layers
word_vec_input = Input(shape=(word_dim,))
decoder_inputs = Input(shape=(None,))
decoder_embed = Embedding(input_dim=num_tokens, output_dim=word_dim, mask_zero=True)
decoder_gru_1 = GRU(word_dim, return_sequences=True, return_state=False)
decoder_gru_2 = GRU(word_dim, return_sequences=True, return_state=True)
decoder_dense = Dense(num_tokens, activation='softmax')

# Connect the layers
embedded = decoder_embed(decoder_inputs)
gru_1_output = decoder_gru_1(embedded, initial_state=word_vec_input)
gru_2_output, state_h = decoder_gru_2(gru_1_output)
decoder_outputs = decoder_dense(gru_2_output)

# Define the model that will be used for training
training_model = Model([word_vec_input, decoder_inputs], decoder_outputs)

# Also create a model for inference (this returns the GRU state)
decoder_model = Model([word_vec_input, decoder_inputs], [decoder_outputs, state_h])


training_model.fit(raw_txt)