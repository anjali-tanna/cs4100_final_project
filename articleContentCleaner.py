import pandas as pd

# clean the data



def remove_words(text, words_to_remove):
    text = text.lower()
    for word in words_to_remove:
        text = text.replace(word, ' ')
    return text

data = pd.read_csv('articlebias.csv')

data = data.drop(columns=['Unnamed: 0', 'split', 'content', 'source_url', 'date', 'authors', 'ID', 'bias_text', 'url'])

topics = data.topic.unique()
sources = data.source.unique()
topic_map = {}
source_map = {}

for i in range(len(topics)):
    topic_map[topics[i]] = i

for i in range(len(sources)):
    source_map[sources[i]] = i

data['topic'].replace(topic_map, inplace=True)
data['source'].replace(source_map, inplace=True)

removed_words = [' a ', ' of ', ' but ', ' after ', ' the ', ' and ', ' on ', 
                 ' this ', ' in ', ' very ', ' as ', ' with ', ' is ', ' in ', ' her '
                 ' or ', ' am ', ' i ', '?', '\n', ',', '!', ':', ' his ', ' hers '
                 ';', '\"', '\'', '-', '(', ')', ' theirs ', ' he ', ' him ', ' she ']
data['content_original'] = data['content_original'].apply(remove_words, args=([removed_words]))

print(data.head(10))


# train the model

# report the accuracy