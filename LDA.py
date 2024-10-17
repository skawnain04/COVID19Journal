import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from IPython.display import display


# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load the CSV file containing your tweets
df = pd.read_csv('data/data_sampled.csv')

# Assuming the tweets are in a column called 'polished_text'
tweets = df['polished_text'].dropna().tolist()
# print(tweets)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocess the text: tokenization, stopword removal, and lemmatization
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenization and lowercase
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return tokens

# Apply preprocessing to all tweets
processed_tweets = [preprocess_text(tweet) for tweet in tweets]

# Create a dictionary from the processed text
dictionary = corpora.Dictionary(processed_tweets)

# Filter out extreme cases (optional, e.g., very rare or very frequent words)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)

# Create a bag-of-words representation of the corpus
corpus = [dictionary.doc2bow(text) for text in processed_tweets]

# Number of topics you want to extract
num_topics = 5

# Train the LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)

# Print the topics and their top words
for idx, topic in lda_model.print_topics(num_words=10):
    print(f"Topic {idx}: {topic}")

# Create the visualization
lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)

# Display the visualization
pyLDAvis.display(lda_display)