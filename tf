import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham':0, 'spam':1})
x = df['message']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(stop_words='english')
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy is:", accuracy_score(y_test, y_pred))
