import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# --------------------------
# 1. Eğitim verisi
# --------------------------
data = [
    ("I love you", "Seni seviyorum"),
    ("Good morning", "Günaydın"),
    ("Thank you", "Teşekkür ederim"),
    ("See you", "Görüşürüz"),
    ("Hello world", "Merhaba dünya"),
    ("How are you?", "Nasılsın"),
    ("Nice to meet you", "Seni tanımak güzel"),
    ("Good night", "İyi geceler"),
    ("I'm happy", "Mutluyum"),
    ("Let's go", "Hadi gidelim"),
] * 10  # Veriyi çoğaltıyoruz

# --------------------------
# 2. Karakter setlerini hazırla
# --------------------------
input_texts = [pair[0] for pair in data]
target_texts = [pair[1] for pair in data]

input_chars = sorted(set("".join(input_texts)))
target_chars = sorted(set("".join(target_texts)))

num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)

max_encoder_seq_length = max(len(txt) for txt in input_texts)
max_decoder_seq_length = max(len(txt) for txt in target_texts)

input_token_index = {char: i for i, char in enumerate(input_chars)}
target_token_index = {char: i for i, char in enumerate(target_chars)}
reverse_target_char_index = {i: char for char, i in target_token_index.items()}

# --------------------------
# 3. One-hot encode et
# --------------------------
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        decoder_target_data[i, t, target_token_index[char]] = 1.0

# --------------------------
# 4. Encoder modeli
# --------------------------
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(512, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# --------------------------
# 5. Decoder modeli
# --------------------------
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(512, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# --------------------------
# 6. Modeli oluştur ve derle
# --------------------------
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------
# 7. Eğit
# --------------------------
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=16,
    epochs=685,
    validation_split=0.2
)

# --------------------------
# 8. Tahmin için encoder modeli
# --------------------------
encoder_model = Model(encoder_inputs, encoder_states)

# --------------------------
# 9. Tahmin için decoder modeli
# --------------------------
decoder_state_input_h = Input(shape=(512,))
decoder_state_input_c = Input(shape=(512,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# --------------------------
# 10. Çeviri yapan fonksiyon
# --------------------------
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    decoded_sentence = ''
    stop_condition = False

    #  Eklenen kısım: aynı karakterin çok tekrarını engelle
    repeat_limit = 2  # maksimum ardışık tekrar sayısı
    last_char_count = 0
    last_sampled_char = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Eğer aynı karakter tekrar tekrar geliyorsa say
        if sampled_char == last_sampled_char:
            last_char_count += 1
        else:
            last_char_count = 1
            last_sampled_char = sampled_char

        decoded_sentence += sampled_char

        #  Durdurma koşulları (boşluk, çok uzun cümle veya tekrar limiti)
        if (sampled_char == ' ' and len(decoded_sentence) > 5) or \
           len(decoded_sentence) > max_decoder_seq_length or \
           last_char_count > repeat_limit:
            stop_condition = True

        #  Sonraki karakteri hazırlamak için girişi güncelle
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence.strip()


# --------------------------
# 11. Test et
# --------------------------
test_text = "I love you"
test_input = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
for t, char in enumerate(test_text):
    if char in input_token_index:
        test_input[0, t, input_token_index[char]] = 1.0

translated = decode_sequence(test_input)
print(f"İngilizce: {test_text} -----> Türkçe: {translated}")
