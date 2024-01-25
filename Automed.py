"""
This python script will extract all the information in pubmed using the pymed API of each article 
of the database PubMedAKE.
Download PubMedAKE data set from: https://zenodo.org/records/6330817 train.json, validate.json and test.json

For each 10000 articles save it inside a csv as:
data_pubmed_index1-index2.csv
index1 = index of first article
index2 = index of last article

Example of use:
    python Script PubMedAKE index1
    python Automed.py train 0
    python Automed.py validate 0
    python Automed.py test 0

In case of error:
    The index of the article not retrieved is printed:
        i.e. Index of article not retrieved: 9
    
    Execute script starting from this index.
        i.e. python Automed.py train 9
"""

import argparse
import os
import sys
import time

import pandas as pd
from pymed import PubMed
import pubmedake

def extract_info(articleList):
    """
    Extracts relevant information from articleList and returns it as a pandas DataFrame.
       
    Parameters:
        - articleList (list): List of articles obtained from PubMed search.
        ['pubmed_id', 'title', 'abstract', 'keywords', 'journal', 'publication_date', 'authors', 'methods', 'conclusions', 'results', 'copyrights', 'doi', 'xml', 'pcmid']
    
    Returns:
        - pandas.DataFrame: DataFrame with relevant information extracted from articleList.
    """

    ### EXTRACT THE INFORMATION NEEDED FROM EACH ARTICLE IN PUBMED.

    articleInfo = []

    for article in articleList:
        pubmedId = article['pubmed_id'].partition('\n')[0]

        articleInfo.append({u'pubmed_id':pubmedId,
                            u'title':article['title'],
                            u'keywords':article['keywords'],
                            u'journal':article['journal'],
                            u'abstract':article['abstract'],
                            u'conclusions':article['conclusions'],
                            u'methods':article['methods'],
                            u'results': article['results'],
                            u'copyrights':article['copyrights'],
                            u'doi':article['doi'],
                            u'publication_date':article['publication_date'], 
                            u'authors':article['authors'],
                            u'AKE_pubmed_id':article['pcmid_AKE'],
                            u'AKE_pubmed_title':article['title_AKE'],
                            u'AKE_abstract':article['abstract_AKE'],
                            u'AKE_keywords':article['keywords_AKE']})

    articlesPD = pd.DataFrame.from_dict(articleInfo)   
    return articlesPD


def not_found(articleList, items, ind1, end):
    """
    Saves a list of PubMed IDs (PCMID) that are not found in a given articleList to a CSV file.

    Args:
        - articleList (list): A list of dictionaries representing articles with PubMed IDs (PMIDs) and other information.
        - items (list): A list of tuples representing PMIDs and article bodies.

    Returns:
        None.

    Saves a CSV file to './Dataset/NotFound/data_pubmed_not_found.csv'.
    """

    article_not_found = []
    for pmid, body in items:
        for article in articleList:
            if article['pcmid_AKE'] == pmid:
                break
        else:
            article_not_found.append({u'pcmid_AKE':pmid})

    article_not_found = pd.DataFrame.from_dict(article_not_found)  
    article_not_found.to_csv(f'./Dataset/NotFound/{sys.argv[1]}/data_pubmed_{ind1}-{end}_not_found.csv', index = False)
    

def find_article(data, ind1, ind2):
    """
    Searches for PubMed articles using the provided PMIDs in the 'data' dictionary 
    and returns the corresponding article information in a list.
    
    Parameters:
        - data (dict): A dictionary containing the PMIDs as keys and the article information as values.
        - ind1 (int): The starting index for the range of PMIDs to search.
        - ind2 (int): The ending index for the range of PMIDs to search.
    
    Returns:
        articleList (list): A list containing dictionaries of article information for the searched PMIDs.
    """

    # GENERATE 'pubmed' OBJECT TO RUN THE SEARCH LATER.
    pubmed = PubMed(tool = 'PubMedSearch')


    # CREATE AN EMPTY LIST TO STORE THE RESULTS
    articleList = []

    # GET A LIST OF THE PMIDS TO SEARCH
    items = list(data.items())[ind1:ind2]

    # LOOP OVER THE PMIDS AND SEARCH FOR EACH ONE
    for i, (pmid, body) in enumerate(items):
        abstract    = body['abstract']
        title       = body['title']
        kwds_in     = body['keywords_in']
        kwds_not_in = body['keywords_not_in']
        
        # DEFINE THE SEARCH QUERY
        query = f"{pmid}"
        print(pmid)

        # RUN THE SEARCH AND SPECIFY THE MAXIMUM NUMBER OF RESULTS YOU WANT
        # RETRY A MAXIMUM OF 10 TIMES
        MAX_RETRIES = 10
        num_retries = 0

        while num_retries < MAX_RETRIES:
            try:
                results = pubmed.query(query, max_results=1000)
                
                # LOOP OVER THE RETRIEVED ARTICLES AND SAVE IT IN 'articleList'
                # ALSO STORE THE DATA OF PUBMEDAKE
                for article in results:
                    articleDict = article.toDict()
                    articleDict['pcmid_AKE'] = pmid
                    articleDict['title_AKE'] = title
                    articleDict['abstract_AKE'] = abstract
                    articleDict['keywords_AKE'] = kwds_in+kwds_not_in
                    articleList.append(articleDict)
                
                # IF THERE WERE NO EXCEPTIONS, BREAK THE LOOP
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                num_retries += 1
                if num_retries == 9:
                    print(f"Too many retries")
                    print(f"Index of article not retrieved: {ind1+i}")
                    not_found(articleList, items, ind1, ind1+i-1)
                    return articleList, ind1+i-1
                # WAIT BEFORE RETRYING
                time.sleep(1)

    not_found(articleList, items, ind1, ind2-1)
    return articleList, ind2-1
    


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify the index you want to Start(start from 0)")
        exit(1)

    # READ PubMedAKE data set
    # PubMedAKE data set: pmid, abstract, title, keywords_in and keywords_not_in
    # Download PubMedAKE data set from: https://zenodo.org/records/6330817 train.json, validate.json and test.json

    data = pubmedake.read_pubmedake("./PubMedAKE/" + sys.argv[1] + ".json")

    # Create the directory if it doesn't exist
    if not os.path.exists(f'./Dataset/Found/{sys.argv[1]}'):
        print("Creating directory")
        os.makedirs(f'./Dataset/Found/{sys.argv[1]}')
        os.makedirs(f'./Dataset/NotFound/{sys.argv[1]}')


    # INDEX USED FOR EXTRACTING 

    ind1 = int(sys.argv[2])

    ind2 = len(data)

    pos = ind1

    # SEARCH FOR 10.000 ARTICLES STARTING FROM IND1 AND STORE IT INSIDE A CSV UNTIL REACHING IND2.
    # AFTER STORING IT IN THE CSV THE LIST IS DELETED IN ORDER TO NOT SATURATE THE MEMORY
    while pos < ind2:

        if pos < (ind2-10000):
            pos = pos + 10000
        else:
            pos = ind2
        
        articleList, end = find_article(data, ind1, pos)

        articlesPD = extract_info(articleList)
        articlesPD.to_csv(f'./Dataset/Found/{sys.argv[1]}/data_pubmed_{ind1}-{end}.csv', index = False)
        del articleList

        if end != pos-1:
            print("Extraction not succesful")
            exit(1)
        
        print(f"Index of last article: {end}")
        ind1 = pos

    print("Extraction succesful")    