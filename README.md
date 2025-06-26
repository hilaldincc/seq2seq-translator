# seq2seq-translator

Bu proje, Python ile geliştirilmiş temel bir Sequence-to-Sequence (seq2seq) çeviri modelidir. Model, bir dili başka bir dile çevirebilmek için Encoder-Decoder yapısını ve LSTM katmanlarını kullanır.

## 🔍 Özellikler

- Encoder–Decoder mimarisi
- LSTM tabanlı çeviri modeli
- Basit metin ön işleme
- Eğitim sonrası tahmin yapılabilir

## 🚀 Nasıl Çalıştırılır?

1. Reposu klonla:
   ```bash
   git clone https://github.com/hilaldincc/seq2seq-translator.git
   cd seq2seq-translator

pip install -r requirements.txt
python main.py

Gerekli Kütüphaneler
Python 3.8+
TensorFlow veya PyTorch (kullandığın framework'e göre)
NumPy
scikit-learn
tqdm
