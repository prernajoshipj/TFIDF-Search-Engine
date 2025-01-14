import os
import math
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def processDocuments():
    documents = {}
    corpusroot = './US_Inaugural_Addresses'
    stopwordsList = stopwords.words('english')
    stopwordsSet = set(stopwordsList)
    stemmer = PorterStemmer()

    # Document Frequencies for each term
    documentFrequencies = {}
    # Documents with their term frequencies performed after stemming
    stemmedTokens = {}

    for filename in os.listdir(corpusroot):
        if filename.startswith(('0', '1', '2', '3', '4')):
            with open(os.path.join(corpusroot, filename), "r", encoding='windows-1252') as file:
                doc = file.read().lower()
                documents[filename] = doc
            
            # Tokenizing
            tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
            tokens = tokenizer.tokenize(doc)
            temp = {}

            for word in tokens:
                if word not in stopwordsSet:
                    stemmedVal = stemmer.stem(word)
                    if stemmedVal in temp:
                        temp[stemmedVal] += 1
                    else:
                        temp[stemmedVal] = 1
                        # Document frequencies for each term
                        if stemmedVal in documentFrequencies:
                            documentFrequencies[stemmedVal] += 1
                        else:
                            documentFrequencies[stemmedVal] = 1

            stemmedTokens[filename] = temp

    return documents, stemmedTokens, documentFrequencies, stopwordsSet

def computeTfIdf(tokens, doc_freqs):
    num_docs = len(tokens)  # Total number of documents
    tfidf_vector = {}

    # Precompute IDF values
    idf_values = {term: math.log10(num_docs / df) for term, df in doc_freqs.items()}

    for doc, freqs in tokens.items():
        doc_tfidf = {}
        sum_sq = 0

        for term, freq in freqs.items():
            # Calculate TF-IDF
            tfidf_weight = (1 + math.log10(freq)) * idf_values.get(term, 0)
            doc_tfidf[term] = tfidf_weight
            sum_sq += tfidf_weight ** 2  # Accumulate squared weights

        # Normalize TF-IDF weights
        if sum_sq > 0:
            for term in doc_tfidf:
                doc_tfidf[term] /= math.sqrt(sum_sq)

        tfidf_vector[doc] = doc_tfidf

    return tfidf_vector

def getidf(token):
    n = 40 # Total number of documents

    # Normalize and stem the token
    stemmed_token = PorterStemmer().stem(token.lower())

    # Check if the stemmed token is in the document frequencies
    df = documentFrequencies.get(stemmed_token, 0)  # Default to 0 if not found

    # Calculate and return the IDF score
    if df > 0:
        return math.log10(n / df)
    else:
        return -1  # Return -1 if token doesn't exist in the corpus

def getweight(filename, token):
    # Normalize and stem the token
    stemmed_token = PorterStemmer().stem(token.lower())
    
    # Check if the stemmed token is a stop word
    if stemmed_token in set(stopwords.words('english')):
        return 0
    
    # Return the normalized TF-IDF weight or 0 if it doesn't exist in the document
    return tfIdfVector.get(filename, {}).get(stemmed_token, 0)

def preprocessQuery(query):
    lower_case_query = query.lower()
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokenized_query = tokenizer.tokenize(lower_case_query)
    query_tf = {}
    query_doc_set = set()
    stemmer = PorterStemmer()

    # Initial posting list structure
    posting_lists = {}
    for token in tokenized_query:
        if token not in stopwords.words('english'):
            stemmed = stemmer.stem(token)
            posting_lists[stemmed] = {"heap": [], "dict": {}}
            query_tf[stemmed] = query_tf.get(stemmed, 0) + 1

    return posting_lists, query_tf, query_doc_set

def updatePostingLists(posting_lists, query_tf, query_doc_set):
    for token, info in posting_lists.items():
        for filename, tfidf in tfIdfVector.items():
            if token in tfidf:
                query_doc_set.add(filename)
                info["dict"][filename] = tfidf[token]

        # Sort the dict by weights to mimic the heap behavior
        sorted_docs = sorted(info["dict"].items(), key=lambda item: item[1], reverse=True)

        # Keep top 10 only
        info["dict"] = dict(sorted_docs[:10])

    return posting_lists, query_tf, query_doc_set


def buildPostingLists(query):
    posting_lists, query_tf, query_doc_set = preprocessQuery(query)
    return updatePostingLists(posting_lists, query_tf, query_doc_set)


def calculateQueryWeights(query_tf):
    # Calculate and normalize query weights
    query_weight = {token: 1 + math.log10(freq) for token, freq in query_tf.items()}
    total_weight = sum(weight ** 2 for weight in query_weight.values())
    
    normalized_query_weight = {token: weight / math.sqrt(total_weight) for token, weight in query_weight.items()}

    return normalized_query_weight

def getQueryPostingList(query):
    posting_lists, query_tf, query_doc_set = buildPostingLists(query)
    normalized_query_weight = calculateQueryWeights(query_tf)

    return normalized_query_weight, posting_lists, query_doc_set

def calculateDocumentSimilarities(query_input):
    class QueryNode:
        def __init__(self, document=None, weight=0, has_false_value=False):
            self.document = document
            self.weight = weight
            self.has_false_value = has_false_value

    document_similarity = {}

    query_weight, posting_lists, query_document_set = getQueryPostingList(query_input)
    
    if query_weight is None:
        return None

    for document in query_document_set:
        node = QueryNode(document)
        
        for token, docs in posting_lists.items():
            if document in docs["dict"]:
                # Document contains this token
                node.weight += docs["dict"][document] * query_weight[token]
            else:
                # Fallback: if no document contains the token, handle gracefully
                # Check if the heap is non-empty, otherwise assign a small default value
                if len(docs["heap"]) > 0:
                    node.weight += docs["heap"][0].weight * query_weight[token]
                else:
                    # Default weight for tokens not in the heap
                    node.weight += 0
                node.has_false_value = True

        document_similarity[document] = node

    # Sort the documents by their weights in descending order
    sorted_similarities = sorted(document_similarity.items(), key=lambda item: item[1].weight, reverse=True)

    return sorted_similarities

def retrieveMostSimilarDocument(document_similarity):
    if not document_similarity:
        return ("None", 0)

    # The first item in the sorted list is the most similar document
    most_similar = document_similarity[0][1]
    
    # Return "fetch more" if the document had incomplete matches
    return ("fetch more", 0) if most_similar.has_false_value else (most_similar.document, most_similar.weight)

def query(query_input):
    document_similarity = calculateDocumentSimilarities(query_input)
    return retrieveMostSimilarDocument(document_similarity)

if __name__ ==  "__main__":
    documents, stemmedTokens, documentFrequencies, stopwordsSet = processDocuments()
    tfIdfVector = computeTfIdf(stemmedTokens, documentFrequencies)

    print("%.12f" % getidf('democracy'))
    print("%.12f" % getidf('foreign'))
    print("%.12f" % getidf('states'))
    print("%.12f" % getidf('honor'))
    print("%.12f" % getidf('great'))
    print("--------------")
    print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
    print("%.12f" % getweight('23_hayes_1877.txt','public'))
    print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
    print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
    print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
    print("--------------")
    print("(%s, %.12f)" % query("states laws"))
    print("(%s, %.12f)" % query("war offenses"))
    print("(%s, %.12f)" % query("british war"))
    print("(%s, %.12f)" % query("texas government"))
    print("(%s, %.12f)" % query("world civilization"))
