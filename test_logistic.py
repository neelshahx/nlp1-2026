from nli import data, tokenizers, representations
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression

# +
snli = data.load_snli()
train = snli['train']

# train tokenizer on entire corpus
tok = tokenizers.BPETokenizer()
tok.train(list(train['premise']) + list(train['hypothesis']))

rep = representations.NgramRepresentation(tok, n=6, pairing='concat')
rep = rep.fit(train)

X_train = rep.transform(train).payload
y_train = train['label'].to_numpy()

logistic_model = LogisticRegression(solver='saga', max_iter=100)
logistic_model.fit(X_train, y_train)

X_dev = rep.transform(snli['dev']).payload
y_dev = snli['dev']['label'].to_numpy()
# -

y_dev_pred = logistic_model.predict(X_dev)

accuracy_score(y_dev, y_dev_pred)

f1_score(y_dev, y_dev_pred ,average='macro')

confusion_matrix(y_dev,y_dev_pred, normalize='true')


