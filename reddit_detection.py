#1: data exploration

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

reddit_df=pd.read_csv('reddits.csv')

print(reddit_df)


# In[5]:


print(reddit_df.info())


# In[6]:


print(reddit_df.describe())


# In[7]:


print(reddit_df['clean_comment'])


# In[8]:


sns.heatmap(reddit_df.isnull(), yticklabels=False, cbar=False, cmap="Blues")


# In[9]:


reddit_df.hist(bins=30, figsize=(13,5), color="blue")


# In[10]:


reddit_df['length'] = reddit_df['clean_comment'].apply(lambda x: len(x) if isinstance(x, str) else 0)
print(reddit_df)


# In[11]:


reddit_df.describe()


# In[12]:


reddit_df[reddit_df['length']==3]['clean_comment']


# In[13]:


reddit_df['length'].plot(bins=100, kind='hist')


#2: wordcloud

# In[14]:


positive=reddit_df[reddit_df['category']==1]
print(positive)


# In[15]:


negative=reddit_df[reddit_df['category']==-1]
print(negative)


# In[16]:


neutral=reddit_df[reddit_df['category']==0]
print(neutral)


# In[17]:


sentences=reddit_df['clean_comment'].tolist()
all_sentences = " ".join([str(sentence) for sentence in sentences if isinstance(sentence, str)])
print(all_sentences)


# In[18]:


from wordcloud import WordCloud

plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(all_sentences))


# In[19]:


negative_sent = negative['clean_comment'].tolist()
all_neg_sent = " ".join([str(sentence) for sentence in negative_sent if isinstance(sentence, str)])
print(all_neg_sent)


# In[20]:


plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(all_neg_sent))


#3: data cleaning

# In[21]:


import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


def cleaning(message):
    if not isinstance(message, str):
        return []
    
    stop_words = set(stopwords.words('english'))
    
    cleaned_words = [word.lower() for word in message if word not in string.punctuation]
    cleaned_message = "".join(cleaned_words).split()
    
    filtered_words = [word for word in cleaned_message if word not in stop_words]
    
    return filtered_words


# In[22]:


reddit_df['clean_comment'] = reddit_df['clean_comment'].fillna('')
reddit_df_clean = reddit_df['clean_comment'].apply(cleaning)


# In[23]:


print(reddit_df_clean[5])


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(analyzer=cleaning, dtype=np.uint8, max_features=20000)
reddit_countvec=vectorizer.fit_transform(reddit_df['clean_comment'])


# In[34]:


print(reddit_countvec.toarray())


# In[35]:


reddit_countvec.shape


# In[36]:


x=pd.DataFrame(reddit_countvec.toarray())
y=reddit_df['category']
print(pd.DataFrame(reddit_countvec.toarray()))


#4: training and evaluating naive bayes classifier model

# In[37]:


print(x.shape)
print(y.shape)


# In[38]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)


# In[39]:


from sklearn.naive_bayes import MultinomialNB
nb_class=MultinomialNB()


# In[40]:


nb_class.fit(x_train, y_train)


# In[41]:


from sklearn.metrics import classification_report, confusion_matrix

y_pred_test=nb_class.predict(x_test)
cm=confusion_matrix(y_test,y_pred_test)
sns.heatmap(cm, annot=True)


# In[42]:


print(classification_report(y_test, y_pred_test))
