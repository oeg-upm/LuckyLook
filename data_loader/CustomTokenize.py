from nltk.corpus import stopwords
import string
import re
import ast

# keyword cleaning
def keyword_transf(text):
    lst = ast.literal_eval(text)
    return ', '.join(lst)

# Turn a doc into clean tokens punctuation, not alphabetic, stopwords and short tokens
def clean_doc(doc):
    # split into tokens by white space
    tokens = doc.split()
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

# Given doc, clean and return line of tokens
def doc_to_line(doc):
    tokens = clean_doc(doc)
    # filter by 
    return ' '.join(tokens)

# Load all samples in a data set with custom tokenizer
def process_samples(data):
    lines = list()
    # walk through all files in the folder
    for index, row in data.iterrows():
        # input = title + abstract + keywords
        input = row['AKE_pubmed_title'] + '\n' + row['AKE_abstract'] + '\n' + keyword_transf(row['AKE_keywords'])

        line = doc_to_line(input)
        lines.append(line)
    return lines

# Load all samples in a data set
def process_samples2(data):
    lines = list()
    # walk through all files in the folder
    for index, row in data.iterrows():
        # input = title + abstract + keywords
        input = row['AKE_pubmed_title'] + '\n' + row['AKE_abstract'] + '\n' + keyword_transf(row['AKE_keywords'])

        lines.append(input)
    return lines

# load all samples in a data set with different contents.
def process_samples3(data, content):
    lines = list()
    # walk through all files in the folder
    if content == 'title':
        for index, row in data.iterrows():
            # input = title + abstract + keywords
            input = row['AKE_pubmed_title']

            lines.append(input)
    elif content == 'abstract':
        for index, row in data.iterrows():
            # input = title + abstract + keywords
            input = row['AKE_abstract']

            lines.append(input)
    elif content == 'keywords':
        for index, row in data.iterrows():
            # input = title + abstract + keywords
            input = keyword_transf(row['AKE_keywords'])

            lines.append(input)
    elif content == 'title+abstract':
        for index, row in data.iterrows():
            # input = title + abstract + keywords
            input = row['AKE_pubmed_title'] + '\n' + row['AKE_abstract']

            lines.append(input)
    elif content == 'title+keywords':
        for index, row in data.iterrows():
            # input = title + abstract + keywords
            input = row['AKE_pubmed_title'] + '\n' + keyword_transf(row['AKE_keywords'])

            lines.append(input)
    elif content == 'abstract+keywords':
        for index, row in data.iterrows():
            # input = title + abstract + keywords
            input = row['AKE_abstract'] + '\n' + keyword_transf(row['AKE_keywords'])

            lines.append(input)
            
    return lines