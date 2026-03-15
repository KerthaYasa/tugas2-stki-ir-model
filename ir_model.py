import math
from preprocessing import load_corpus, preprocess
from indexing import build_vocabulary, build_incidence_matrix, build_inverted_index_full


def compute_tf_normalized(corpus):
    tf_norm = {}
    for doc, tokens in corpus.items():
        total = len(tokens)
        tf_norm[doc] = {}
        for token in tokens:
            tf_norm[doc][token] = tf_norm[doc].get(token, 0) + 1
        for token in tf_norm[doc]:
            tf_norm[doc][token] = tf_norm[doc][token] / total
    return tf_norm

def get_postings(term, corpus):
    stemmed = preprocess(term)
    if not stemmed:
        return {}
    t = stemmed[0]
    return {doc: (1 if t in tokens else 0) for doc, tokens in corpus.items()}

def boolean_and(p1, p2):
    return {doc: p1.get(doc, 0) & p2.get(doc, 0) for doc in p1}

def boolean_or(p1, p2):
    return {doc: p1.get(doc, 0) | p2.get(doc, 0) for doc in p1}

def boolean_not(p1):
    return {doc: 1 - v for doc, v in p1.items()}

def tokenize_query(query):
    query = query.replace('(', ' ( ').replace(')', ' ) ')
    return [t for t in query.split() if t]

def parse_query(tokens, corpus, pos=0):
    result, pos = parse_or(tokens, corpus, pos)
    return result, pos

def parse_or(tokens, corpus, pos):
    left, pos = parse_and(tokens, corpus, pos)
    while pos < len(tokens) and tokens[pos].upper() == 'OR':
        pos += 1
        right, pos = parse_and(tokens, corpus, pos)
        left = boolean_or(left, right)
    return left, pos

def parse_and(tokens, corpus, pos):
    left, pos = parse_not(tokens, corpus, pos)
    while pos < len(tokens) and tokens[pos].upper() == 'AND' \
          and (pos + 1 >= len(tokens) or tokens[pos + 1].upper() != 'NOT'):
        pos += 1
        right, pos = parse_not(tokens, corpus, pos)
        left = boolean_and(left, right)
    return left, pos

def parse_not(tokens, corpus, pos):
    if pos < len(tokens) and tokens[pos].upper() == 'NOT':
        pos += 1
        operand, pos = parse_primary(tokens, corpus, pos)
        return boolean_not(operand), pos
    result, pos = parse_primary(tokens, corpus, pos)
    while pos + 1 < len(tokens) and tokens[pos].upper() == 'AND' \
          and tokens[pos + 1].upper() == 'NOT':
        pos += 2
        right, pos = parse_primary(tokens, corpus, pos)
        result = boolean_and(result, boolean_not(right))
    return result, pos

def parse_primary(tokens, corpus, pos):
    if pos >= len(tokens):
        return {doc: 0 for doc in corpus}, pos
    token = tokens[pos]
    if token == '(':
        pos += 1
        result, pos = parse_or(tokens, corpus, pos)
        if pos < len(tokens) and tokens[pos] == ')':
            pos += 1
        return result, pos
    elif token.upper() not in ('AND', 'OR', 'NOT', '(', ')'):
        pos += 1
        return get_postings(token, corpus), pos
    else:
        return {doc: 0 for doc in corpus}, pos

def extended_boolean_score(terms, ops, tf_norm, doc):
    """
    AND : skor = sqrt( (tf1² + tf2² + ... + tfn²) / n )
    OR  : skor = 1 - sqrt( ((1-tf1)² + (1-tf2)² + ... + (1-tfn)²) / n )
    NOT : tf term yang di-NOT → 1 - tf_norm sebelum dihitung
    """
    if not terms:
        return 0

    values = []
    for i, term in enumerate(terms):
        tf = tf_norm[doc].get(term, 0)
        op = ops[i] if i < len(ops) else None
        if op == 'NOT':
            tf = 1 - tf
        values.append(tf)

    n          = len(values)
    op_dominan = 'OR' if 'OR' in [o for o in ops if o] else 'AND'

    if op_dominan == 'AND':
        skor = math.sqrt(sum(v**2 for v in values) / n)
    else:
        skor = 1 - math.sqrt(sum((1 - v)**2 for v in values) / n)

    return round(skor, 4)

def extract_terms_and_ops(query):
    """Ambil list term dan operator NOT dari query untuk scoring."""
    tokens    = query.upper().split()
    terms, ops = [], []
    i = 0
    while i < len(tokens):
        t = tokens[i].strip('()')
        if t == 'NOT' and i + 1 < len(tokens):
            next_term = tokens[i + 1].strip('()').lower()
            stemmed   = preprocess(next_term)
            if stemmed:
                terms.append(stemmed[0])
                ops.append('NOT')
            i += 2
        elif t == 'OR' and i + 1 < len(tokens):
            # Tandai term berikutnya pakai operator OR
            ops.append('OR') if ops else None
            i += 1
        elif t not in ('AND', 'OR', 'NOT', '', '(', ')'):
            stemmed = preprocess(t.lower())
            if stemmed:
                terms.append(stemmed[0])
                ops.append(None)
            i += 1
        else:
            i += 1
    return terms, ops

def search(query_input, corpus, tf_norm):
    if not query_input.strip():
        return None

    # Step 1 — Boolean: saring dokumen relevan
    tokens_query = tokenize_query(query_input)
    try:
        postings, _ = parse_query(tokens_query, corpus)
    except Exception:
        return None

    dokumen_relevan = [doc for doc, val in postings.items() if val == 1]
    terms, ops      = extract_terms_and_ops(query_input)

    if not dokumen_relevan:
        return terms, ops, []

    hasil = []
    for doc in dokumen_relevan:
        skor = extended_boolean_score(terms, ops, tf_norm, doc)
        hasil.append((doc, skor))

    hasil = sorted(hasil, key=lambda x: x[1], reverse=True)
    return terms, ops, hasil

if __name__ == "__main__":
    corpus  = load_corpus("corpus")
    tf_norm = compute_tf_normalized(corpus)

    queries = [
        "bukti AND kasus",
        "bukti AND kasus AND NOT korban",
        "(bukti AND kasus) AND NOT korban",
        "saksi OR korban",
        "ardan AND NOT bunuh",
        "curi AND NOT (bakar OR bunuh)",
    ]

    for q in queries:
        hasil = search(q, corpus, tf_norm)
        if hasil:
            terms, ops, docs = hasil
            print(f"\nQuery : {q}")
            print(f"Terms : {terms}  |  Ops: {ops}")
            print(f"Hasil :")
            for doc, skor in docs:
                print(f"  {doc} → {skor}")
        else:
            print(f"\nQuery : {q} → tidak valid")