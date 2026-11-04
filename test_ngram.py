# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from nli import data, representations, tokenizers
representations.REPRESENTATIONS

# %%
toy_data = data.load_toy()
train, dev = toy_data['train'], toy_data['dev']

# %%
tok = tokenizers.BPETokenizer(vocab_size=1000, min_frequency=2)
tok.train(list(train['premise']) + list(train['hypothesis']))

# %%
rep = representations.NgramRepresentation(tok, n=2)
sentence = train['premise'][0]
tok.tokenize(sentence), rep.n_gram_finder(sentence)

# %%
rep.fit_texts(list(train['premise']) + list(train['hypothesis']))
rep.side_dim, list(rep._vectorizer.vocabulary_.items())[:10]

# %%
X = rep.transform_texts([sentence, 'Zebras juggle quantum xylophones.'])
X.shape, X.nnz

# %%
names = rep._vectorizer.get_feature_names_out()
row = X[0]
dict(zip(names[row.indices], row.data, strict=True))

# %%
# pairing decides how many side blocks get stacked
concat = representations.build_representation({'kind': 'ngram', 'n': 2, 'pairing': 'concat'}, tok).fit(train)
hyp_only = representations.build_representation({'kind': 'ngram', 'n': 2, 'pairing': 'hypothesis_only'}, tok).fit(train)
assert concat.dim == 2 * concat.side_dim
assert hyp_only.dim == hyp_only.side_dim
concat.transform(dev), hyp_only.transform(dev)

# %%
# binary=True gives presence instead of counts
counts = representations.NgramRepresentation(tok, n=1).fit(train)
present = representations.NgramRepresentation(tok, n=1, binary=True).fit(train)
repeated = 'A man and a man.'
counts.transform_texts([repeated]).max(), present.transform_texts([repeated]).max()
