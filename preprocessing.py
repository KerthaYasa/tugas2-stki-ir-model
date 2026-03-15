import os
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(teks):
    teks = teks.lower()
    
    teks = re.sub(r'[^a-z\s]', '', teks)
    
    tokens = teks.split()
    
    tokens_stem = [stemmer.stem(token) for token in tokens]
    
    return tokens_stem

def load_corpus(folder_path):
    corpus = {} 
    
    for nama_file in sorted(os.listdir(folder_path)):
        if nama_file.endswith('.txt'):
            path = os.path.join(folder_path, nama_file)
            
            with open(path, 'r', encoding='utf-8') as f:
                isi = f.read()
            
            hasil_preprocessing = preprocess(isi)
            corpus[nama_file] = hasil_preprocessing
            
    return corpus

if __name__ == "__main__":
    folder_corpus = "corpus"
    
    corpus = load_corpus(folder_corpus)
    
    for nama_doc, tokens in corpus.items():
        print(f"\n{'='*50}")
        print(f"Dokumen: {nama_doc}")
        print(f"Jumlah token: {len(tokens)}")
        print(f"Token hasil preprocessing:")
        print(tokens)

