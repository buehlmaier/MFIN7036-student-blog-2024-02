---
Title: Topic modeling using LSA and LDA approach (by Group "Word Wizards")
Date: 2024-03-17 17:12
Category: Progress Report
Tags: Group Word Wizards
---

After we obtained the news that can move the market, we can analyse and extract 
the common topics among them. This blog introduce how to conduct the 
Latent Semantic Analysis (LSA) and the Latent Dirichlet Allocation (LDA). 
We leveraged on the sklearn and genism library.
 
## Text preprocessing
Firstly, we did text preprocessing. For example, removing special characters, 
short words and stop words, etc.

```python

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
import re
import nltk
import matplotlib.colors as mcolors
# Fold warnings
warnings.filterwarnings('ignore')
os.chdir('/Users/wanjing_dai/Desktop/HKU/NLP_7036/Group project/nlp_group')
# Read news that is important
stock_news = pd.read_parquet('stock_news_and_mark.parquet')
mark_stock_news = stock_news[stock_news['news_mark'] == 1].reset_index(drop=True)
mark_stock_news = mark_stock_news[['PERMNO', 'date_news', 'headline']]
# Preprocessing text
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from collections import Counter
from nltk.probability import FreqDist
# Remove punctuations and special characters
mark_stock_news['clean_1'] = mark_stock_news['headline'].apply(lambda x: re.sub('[,\!?$:;.()%\']', '', x))
#Convert to lower case
mark_stock_news['clean_1'] = mark_stock_news['clean_1'].apply(lambda x: x.lower())
# Remove short words
mark_stock_news['clean_1'] = mark_stock_news['clean_1'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
# Stemming
wnl = WordNetLemmatizer()
mark_stock_news['clean_1'] = mark_stock_news['clean_1'].apply(lambda x: ' '.join([wnl.lemmatize(y) for y in x.split()]))
# Remove stopwords
stop_words = set(stopwords.words('english'))
mark_stock_news['clean_1'] = mark_stock_news['clean_1'].apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))
# Remove spaces
mark_stock_news['clean_1'] = mark_stock_news['clean_1'].apply(lambda x: x.strip())
# Tokenization
mark_stock_news['token'] = mark_stock_news['clean_1'].apply(lambda x: [w for w in word_tokenize(x.lower())])

```
## Create a word cloud

Then, let's create a word cloud to have a taste on the news.

```python

# Create a new column containing the length each headline text
mark_stock_news["headline_len"] = mark_stock_news["headline"].apply(lambda x : len(x.split()))
# Plot the news length distribution
sns.displot(mark_stock_news.headline_len, kde=False)
print("The longest headline has: {} words".format(mark_stock_news.headline_len.max()))
# Frequency distribution of words
tokens = mark_stock_news['token'].tolist()
flat_tokens = [item for sublist in tokens for item in sublist]
fd = FreqDist(flat_tokens)
# Plot top 15 words
fd.plot(15)
# Wordcloud
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(width=800,height=800,background_color='white',stopwords=STOPWORDS)\
.generate(' '.join(flat_tokens))
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation='Bilinear') 
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

```

## LSA and LDA analysis using sklearn

sklearn offered us packages to conduct LSA and LDA analysis, but not so powerful as genism.

```python
# LSA analysis using sklearn
from sklearn.decomposition import TruncatedSVD
def get_keys(topic_matrix):
    '''
    returns an integer list of predicted topic 
    categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

def get_top_n_words(n, keys, document_term_matrix, tfidf_vectorizer):
    '''
    returns a list of n_topic strings, where each string contains the n most common 
    words in a predicted category, in order
    '''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = tfidf_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words
# Construct a CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer(stop_words='english') 
# Construct a bag of words
bow = v.fit_transform(mark_stock_news['clean_1'])
# Set number of topics and train the data by SVD model
n_topics = 10
lsa_model = TruncatedSVD(n_components=n_topics)
lsa_topic_matrix = lsa_model.fit_transform(bow)
# Use the function we defined before to get the top 10 topics 
lsa_keys = get_keys(lsa_topic_matrix)
lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
top_n_words_lsa = get_top_n_words(5, lsa_keys, bow, v)
for i in range(len(top_n_words_lsa)):
    print("Topic {}: ".format(i+1), top_n_words_lsa[i])
    
# Plot the number of headlines containing topics generated by LSA
labels = ['Topic {}: \n'.format(i) + top_n_words_lsa[i] for i in lsa_categories]
fig, ax = plt.subplots(figsize=(40,20))
ax.bar(lsa_categories, lsa_counts);
ax.set_xticks(lsa_categories);
ax.set_xticklabels(labels);
ax.set_ylabel('Number of headlines');
ax.set_title('LSA topic counts');
plt.show()

# LDA analysis (Using Sklearn)
from sklearn.decomposition import LatentDirichletAllocation
# Train the data using LDA model
lda_model_s = LatentDirichletAllocation(n_components=n_topics, learning_method='online', 
                                          random_state=0, verbose=0)
lda_topic_matrix = lda_model_s.fit_transform(bow)
# Use the function we defined before to get the top 10 topics generated by LDA
lda_keys = get_keys(lda_topic_matrix)
lda_categories, lda_counts = keys_to_counts(lda_keys)
top_n_words_lda = get_top_n_words(5, lda_keys, bow, v)
for i in range(len(top_n_words_lda)):
    print("Topic {}: ".format(i+1), top_n_words_lda[i])
    
# Plot the number of headlines containing topics generated by LDA 
labels = ['Topic {}: \n'.format(i) + top_n_words_lda[i] for i in lda_categories]
fig, ax = plt.subplots(figsize=(16,8))
ax.bar(lda_categories, lda_counts);
ax.set_xticks(lda_categories);
ax.set_xticklabels(labels);
ax.set_title('LDA topic counts');
ax.set_ylabel('Number of headlines');
plt.show()

```



## LDA approach using genism

Genism offered us more handy functions to do further analysis. As a result, 
our analysis focused on genism from now on.

```python
# LDA analysis (Using genism, more powerful features)
import gensim, logging, warnings
import gensim.corpora as corpora
import matplotlib.pyplot as plt
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
tokens = mark_stock_news['token'].tolist()
# Construct a dictionary and corpus for the tokens
d = corpora.Dictionary(tokens)
cp = [d.doc2bow(text) for text in tokens]
# Get the top 10 topics generated by LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus=cp,
                                           id2word=d,
                                           num_topics=4)
print(lda_model.print_topics(num_words=5))

# Find the Dominant topic and its percentage contribution in each document
def format_topics_sentences(ldamodel=lda_model, corpus=cp, texts=tokens):
    # Init output
    sent_topics_df = pd.DataFrame()
    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                temp = pd.DataFrame()
                temp['Dominant_Topic']=[int(topic_num)]
                temp['Topic_Perc_Contrib']=[round(prop_topic,4)]
                temp['Topic_Keywords']=[topic_keywords]
                sent_topics_df = pd.concat([sent_topics_df, temp], ignore_index=True)
            else:
                break
    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=cp, texts=tokens)
# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Topic_Perc_Contrib'], ascending=False).head(1)], 
                                            axis=0)
# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
# Show
sent_topics_sorteddf_mallet.head(10)

```

## Visualize the results by graphs

As before, we would like to check the most popular topics among the news. 
To go one step further, we also investigate the importance of each keyword in the topics.  
Last but not least, we can create an interactive graph. Each bubble represents a topic. 
The larger the bubble, the more news is about that topic. The further the bubbles are away 
from each other, the more different they are. Blue bars represent the overall frequency of 
each word. Red bars is the times that the term shown in a given topic. 


```python
# Distribution of Document Word Counts by Dominant Topic
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):    
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 30, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 30), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i])
    ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))
fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,30,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
plt.show()

# Word Counts of Topic Keywords
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in tokens for w in w_list]
counter = Counter(data_flat)
out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])
df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count']) 
       
# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0,0.15); ax.set_ylim(0, 95000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()

# Create an interactive graph.
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, cp, dictionary=lda_model.id2word)
vis
```



