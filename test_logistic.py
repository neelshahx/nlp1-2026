from nli import data, tokenizers, representations

snli = data.load_snli()

train = snli['train']

train

# train tokenizer on entire corpus
tok = tokenizers.BPETokenizer()
tok.train(list(train['premise']) + list(train['hypothesis']))

rep = representations.NgramRepresentation(tok, n=1, pairing='concat')

rep.fit_transform(train['premise'][:10])
