import pandas as pd
from preprocessing import load_corpus

def build_vocabulary(corpus):
    vocabulary = set()
    for tokens in corpus.values():
        for token in tokens:
            vocabulary.add(token)
    return sorted(vocabulary)

def build_incidence_matrix(corpus, vocabulary):
    nama_dokumen = list(corpus.keys())
    matrix = {}
    for term in vocabulary:
        baris = []
        for doc in nama_dokumen:
            if term in corpus[doc]:
                baris.append(1)
            else:
                baris.append(0)
        matrix[term] = baris
    df = pd.DataFrame(matrix, index=nama_dokumen).T
    return df

def build_inverted_index_full(corpus):
    inverted_index = {}
    for nama_doc, tokens in corpus.items():
        for posisi, token in enumerate(tokens):
            if token not in inverted_index:
                inverted_index[token] = {}
            if nama_doc not in inverted_index[token]:
                inverted_index[token][nama_doc] = {
                    'frekuensi': 0,
                    'posisi': []
                }
            inverted_index[token][nama_doc]['frekuensi'] += 1
            inverted_index[token][nama_doc]['posisi'].append(posisi)
    return inverted_index

def format_inverted_index_table(corpus):
    inv_idx = build_inverted_index_full(corpus)
    
    print("\n=== INVERTED INDEX ===")
    print(f"{'Term':<20} {'Inverted List'}")
    print("-" * 80)
    
    for term in sorted(inv_idx.keys()):
        entries = inv_idx[term]
        formatted = []
        for doc, info in entries.items():
            nama = doc.replace('.txt', '')
            frek = info['frekuensi']
            pos  = info['posisi']
            formatted.append(f"<{nama},{frek},{pos}>")
        inverted_list = ", ".join(formatted)
        print(f"{term:<20} {inverted_list}")

if __name__ == "__main__":
    folder_corpus = "corpus"
    corpus = load_corpus(folder_corpus)
    
    vocab = build_vocabulary(corpus)
    print(f"Total kata unik (vocabulary): {len(vocab)}")
    print(f"Contoh 10 kata pertama: {vocab[:10]}")
    
    incidence_matrix = build_incidence_matrix(corpus, vocab)
    print(f"\nUkuran matrix: {incidence_matrix.shape[0]} term x {incidence_matrix.shape[1]} dokumen")
    print("\n=== INCIDENCE MATRIX ===")
    print(incidence_matrix)
    
    format_inverted_index_table(corpus)