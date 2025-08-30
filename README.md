🐦 Twitter Sentiment Analysis using Python
📌 Overview

This project implements Twitter Sentiment Analysis using Python.
By analyzing tweets, we classify them into Positive (1) or Negative (0) sentiments.

We use the Sentiment140 dataset (1.6M tweets) to train models with NLP and Machine Learning techniques.
This helps businesses, researchers, and developers to track public mood, brand reputation, and opinions in real-time.

⚙️ Libraries & Tools Used

Python 3.x

Pandas → data handling

Scikit-learn → ML models and vectorization

TfidfVectorizer → text to numeric features

Bernoulli Naive Bayes, SVM, Logistic Regression → classifiers

Matplotlib (optional) → visualization

📂 Dataset

Dataset: Sentiment140

Source: Kaggle

Format: training.1600000.processed.noemoticon.csv.zip

We keep:

Column 0 → Polarity (0 = Negative, 2 = Neutral, 4 = Positive)

Column 5 → Text (tweet)

🚀 Step-by-Step Implementation
Step 1: Install Dependencies
pip install pandas scikit-learn

Step 2: Load Dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv.zip', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']

Step 3: Keep Only Positive & Negative
df = df[df.polarity != 2]
df['polarity'] = df['polarity'].map({0: 0, 4: 1})

Step 4: Clean the Tweets
def clean_text(text):
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text)

Step 5: Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['polarity'], test_size=0.2, random_state=42
)

Step 6: TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

Step 7: Train Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

bnb = BernoulliNB()
bnb.fit(X_train_tfidf, y_train)
bnb_pred = bnb.predict(X_test_tfidf)

print("Bernoulli Naive Bayes Accuracy:", accuracy_score(y_test, bnb_pred))

Step 8: Train Support Vector Machine (SVM)
from sklearn.svm import LinearSVC

svm = LinearSVC(max_iter=1000)
svm.fit(X_train_tfidf, y_train)
svm_pred = svm.predict(X_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

Step 9: Train Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train_tfidf, y_train)
logreg_pred = logreg.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))

Step 10: Predictions on Sample Tweets
sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
sample_vec = vectorizer.transform(sample_tweets)

print("BernoulliNB:", bnb.predict(sample_vec))
print("SVM:", svm.predict(sample_vec))
print("Logistic Regression:", logreg.predict(sample_vec))

📊 Results

Models used: Bernoulli Naive Bayes, SVM, Logistic Regression

Each classifier predicts tweet sentiment as:

1 → Positive

0 → Negative

Sample Predictions:

Tweet: "I love this!" → Positive
Tweet: "I hate that!" → Negative
Tweet: "It was okay, not great." → Negative

🔮 Future Work

Improve preprocessing (stopwords removal, stemming, lemmatization).

Try deep learning (LSTM, BERT) for better accuracy.

Deploy as a web app with Flask/Streamlit.

Integrate with Twitter API for real-time sentiment analysis.

🙌 Acknowledgements

Dataset: Sentiment140

Libraries: Pandas, Scikit-learn
