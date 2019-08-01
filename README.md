# JobDeepSearch
Job Search Engine based on semantics and documents embedding Word2Vec


Notebook available on [Google Colab](https://colab.research.google.com/drive/1TbudtYymDFseL9JOs2dYlMmwLdSs9xmn).

## Job Search and Ranking function based on Semantics and Word Embeddings

This notebook presents a mockup of a search and ranking function based on semantics of 20000 job descriptions (dataset extracted from Monster.com jobs).
This methodology using word embeddings captures the context and the semantics of the analysed text, compared to a classic search function based on words counts per documents (Vector Space Model and Term Frequency-Inverse Document Frequency).

A word embedding Work2Vec model is build from these descriptions to capture the semantics and the context.
This model is then enriched with a generic Word2Vec model based on a Google News corpus, the job descriptions being not sufficient to build a full language model.

The resulted ranking of a search is based on the cosine similarity between the query and the different job descriptions scored with the word embedding model (300 dim vector).

The TSNE dimension reduction method allows to visualise the job descritions in a 3D space. 

![SegmentLocal](tensorboard.gif "segment")

Possible improvement: TF-IDF weighting for job descriptions scoring

### Dependencies
- Numpy
- Gensim for text processing and word2vec model
- nltk
- tensorflow for T-SNE visualisation
