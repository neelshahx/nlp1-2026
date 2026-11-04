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
from nli import data
from nli import tokenizers 

# %%
toy_data = data.load_toy()

# %%
tok = tokenizers.BPETokenizer(vocab_size=50, min_frequency=2)
tok.train(toy_data['train']['premise'] + ' ' + toy_data['train']['hypothesis'])
sentence = toy_data['train']['premise'][0] + ' ' + toy_data['train']['hypothesis'][0]
before = tok.tokenize(sentence)
tok.save('tmp')
del tok
tok = tokenizers.BPETokenizer()
tok = tok.load('tmp')
after = tok.tokenize(sentence)
assert before == after

# %%
tok = tokenizers.build_tokenizer({'kind': 'bpe', 'min_frequency':2, 'vocab_size': 20})
tok.train(toy_data['train']['premise'])
tok.tokenize_many(toy_data['train']['premise'])

# %%

# %%
