import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Load in Article-Bias-Prediction.csv from https://huggingface.co/datasets/cjziems/Article-Bias-Prediction
data = pd.read_csv('Article-Bias-Prediction.csv')

# Drop unnecessary columns
data = data.drop(['Unnamed: 0', 'split', 'content', 'date', 'source_url',
                  'authors', 'ID', 'url', 'bias_text'], axis=1)

# List of unimportant words
removed_words = [' a ', ' of ', ' but ', ' after ', ' the ', ' and ', ' on ',
                 ' this ', ' in ', ' very ', ' as ', ' with ', ' is ', ' in ', ' her '
                                                                               ' or ', ' am ', ' i ', '?', '\n', ',',
                 '!', ':', ' his ', ' hers '
                                    ';', '\"', '\'', '-', '(', ')', ' theirs ', ' he ', ' him ', ' she ']


# Function to remove words
def remove_words(text, words_to_remove):
    text = text.lower()
    for word in words_to_remove:
        text = text.replace(word, ' ')
    return text

# Remove the unnecessary words from the 'content_original' and 'title' columns
data['content_original'] = data['content_original'].apply(remove_words, args=([removed_words]))
data['title'] = data['title'].apply(remove_words, args=([removed_words]))

# Create lists of unique topics and sources
topics = data.topic.unique()
sources = data.source.unique()
topic_map = {}
source_map = {}

# Assign values for each topic
for i in range(len(topics)):
    topic_map[topics[i]] = i

# Assign values for each source
for i in range(len(sources)):
    source_map[sources[i]] = i

# Replace topic and source columns with new values
data['topic'].replace(topic_map, inplace=True)
data['source'].replace(source_map, inplace=True)

# Create new DataFrame with content and title in one column
data['title_content'] = data['content_original'] + ' ' + data['title']

# Split content into list of words and find the row with the least amount of words
data['content_list'] = data['title_content'].str.split()
data['content_length'] = data['content_list'].apply(len)
print('Least amount of Words:', data.content_length.min())

# DOC2VEC

# Download NLTK resources (if not already downloaded)
# nltk.download('punkt')

# Tokenize paragraphs
documents = [TaggedDocument(words=word_tokenize(paragraph.lower()),
                            tags=[str(idx)]) for idx, paragraph in enumerate(data['title_content'])]

# Train Doc2Vec model
model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4, epochs=20)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)


def embed_paragraph(paragraph, model):
    return model.infer_vector(word_tokenize(paragraph.lower()))


# Create a new column with the vector representations
data['paragraph_vectors'] = data['title_content'].apply(lambda x: embed_paragraph(x, model))

# Export to csv file
data.to_csv('doc2vec_bias.csv')

# Standard scaler
scaler = StandardScaler()

# Read doc2vec_bias file and drop unnecessary columns
data = pd.read_csv('doc2vec_bias.csv')
data = data.drop(columns='Unnamed: 0', axis=1)


def str_to_array(str):
    start = str.index('[')
    end = str.index(']')
    p_nums = str[start + 1: end].replace('\n', '')
    p_vectors = list(map(lambda y: float(y), filter(lambda x: x != '', p_nums.split(' '))))
    return p_vectors


# Apply str_to_array to the paragraph vectors
data['paragraph_vectors'] = data['paragraph_vectors'].apply(str_to_array)

# Create extra column names 0 to 100
column_names = [str(i) for i in range(100)]

# Add new columns and drop paragraph_vectors
expanded_df = data.join(pd.DataFrame(data['paragraph_vectors'].tolist(), columns=column_names))
expanded_df.drop('paragraph_vectors', axis=1, inplace=True)

# Scale data
data_unscaled = expanded_df[
    ['topic', 'source', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
     '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35',
     '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54',
     '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73',
     '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92',
     '93', '94', '95', '96', '97', '98', '99']]
data_scaled = scaler.fit_transform(data_unscaled.to_numpy())
data_scaled_pd = pd.DataFrame(data_scaled)

# Create index for topic and source
expanded_df['topic'] = data_scaled_pd[0]
expanded_df['source'] = data_scaled_pd[1]

# Create columns for paragraph vector data
for i in range(100):
    expanded_df[str(i)] = data_scaled_pd[i + 2]

# Convert to csv
expanded_df.to_csv('normalizedDataWithCenter.csv')

# Use data with center for bias
center = pd.read_csv('normalizedDataWithCenter.csv')

# Create a DataFrame with only left and right bias values
df_filtered = center[center['bias_score'] != 1]
df_filtered['bias_score'] = df_filtered['bias_score'].replace(2, 1)

# Convert to csv
df_filtered.to_csv('normalizedDataNoCenter.csv')

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''FAILED DATA ATTEMPTS'''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''''TF-IDF'''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Drop bias from DataFrame and set X and y
no_bias = data.drop(['bias'], axis=1)
X = tfidf.fit_transform(no_bias['title_content'])
feature_names = tfidf.get_feature_names_out()
y = data['bias']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model and calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Create DataFrame of results
tfidf_df = pd.DataFrame(X.toarray(), columns=feature_names)

# Create DataFrame with topic, source, title, and bias
topic_source = data[['topic', 'source', 'title', 'bias']]

# Combine DataFrames
with_bias = pd.concat([topic_source, tfidf_df], axis=1)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''''''''''''''''''WORD2VEC'''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Assuming df['word_list_column'] contains lists of words for each entry
sentences = data['content_list'].tolist()

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4)


def embed_words(word_list, model):
    # Filter out words not in the model's vocabulary
    tokens = [word for word in word_list if word in model.wv.key_to_index]

    # Calculate the mean vector for the words
    if len(tokens) > 0:
        vector = sum(model.wv[word] for word in tokens) / len(tokens)
    else:
        vector = None

    return vector


# Create a new column with the vector representations
data['w2_vectors'] = data['content_list'].apply(lambda x: embed_words(x, model))
