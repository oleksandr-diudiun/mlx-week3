# Notes on Week 3 RNN


## Data Structures

### Static Data - These data will only need to be calculated once

SentencePiece trained on both queries and answers (stored as .vocab file)

Token embeddings (stored as pkl)

### Embeddings

SentencePiece with GloVe and Word2Vec (Pretrain and finetune))

BERT (Pretrained and finetuned)

### Dataset for quick iterative training


### Negative Sampling

Random

Consider taking from high in distribution


### Postive Sampling

Take all from the bing relevant


### Data

Create triples of (q_i p_ij n_ij)



### Precompute Embeddings

Use mapping functions 



DatasetDict({
    validation: Dataset({
        features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],
        num_rows: 10047
    })
    train: Dataset({
        features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],
        num_rows: 82326
    })
    test: Dataset({
        features: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers'],
        num_rows: 9650
    })
})

Share random state


Share code from last week (training loop)
