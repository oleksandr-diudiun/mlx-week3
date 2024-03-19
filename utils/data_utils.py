import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from tqdm import tqdm
from collections import namedtuple


def create_lookups(dataset):
    print(f"dataset shape: {dataset.shape}")
    all_urls = dataset["passages"].apply(lambda x: x["url"]).tolist()

    unique_urls = set([item for sublist in all_urls for item in sublist])
    print(f"Total number of urls: {sum(len(i) for i in all_urls)}")
    print(f"Total number of unique urls: {len(unique_urls)}")

    # Use an md5 hash for the urls for deterministic mapping
    def generate_md5_hash(s):
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    ids_to_urls = {generate_md5_hash(url): url for url in unique_urls}

    urls_to_ids = {url: i for i, url in ids_to_urls.items()}
    print(f"Total number of hashed urls: {len(ids_to_urls)}")

    url_to_doc_mapping = {}
    for row in dataset[["passages"]].iterrows():
        # assert len(url_list) == len(set(url_list))
        passages = row[1]["passages"]
        for url, passage_text in zip(passages["url"], passages["passage_text"]):
            url_to_doc_mapping[urls_to_ids[url]] = passage_text

    query_ids = dataset["query_id"].tolist()
    assert len(query_ids) == len(set(query_ids))
    print(f"Total number of queries: {len(query_ids)}")
    return (ids_to_urls, urls_to_ids, url_to_doc_mapping)


def add_hashed_urls(dataset, urls_to_ids):
    dataset.loc[:, "hashed_urls"] = dataset["passages"].progress_apply(
        lambda x: np.array(list(set([urls_to_ids[url] for url in x["url"]])))
    )


# Function to be applied to each row, modified to accept master_url_id_set
def process_row(x, master_url_id_set, is_deterministic=True):
    """
    Parallelized function thanks to chatGPT for creating
    negative samples for each row really fast.
    """
    if is_deterministic:
        np.random.seed(seed=999999)
    return x.apply(
        lambda y: np.random.choice(
            np.setdiff1d(master_url_id_set, y, assume_unique=True),
            size=len(y),
            replace=False,
        )
    )


# Function to apply process_row in parallel, passing master_url_id_set to each task
def apply_in_parallel(
    series, func, master_url_id_set, num_partitions=10, is_deterministic=True
):
    # Splitting the series into chunks
    chunks = np.array_split(series, num_partitions)

    # Creating a partial function to include master_url_id_set as a static argument
    func_with_args = partial(
        func, master_url_id_set=master_url_id_set, is_deterministic=True
    )

    # Use a list to collect the results

    with ProcessPoolExecutor() as executor:
        # Processing chunks in parallel
        futures = [executor.submit(func_with_args, chunk) for chunk in chunks]
        ordered_results = [None] * len(chunks)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Batches"
        ):
            index = futures.index(
                future
            )  # Get the index of the future based on the original submission order
            ordered_results[index] = (
                future.result()
            )  # Store the result at the correct index

    return pd.concat(ordered_results, ignore_index=True)


def _add_negative_samples_wrapper(dataset, ids_to_urls, is_deterministic=True):
    """
    {query_hash_id: (relevant_urls, irrelevant_urls)}
    Query ID is from the query_id column in the parquet dataset.
    """
    if is_deterministic:
        np.random.seed(seed=999999)
    master_url_id_set = np.array(list(set(ids_to_urls.keys())))
    # This is faster to do there than in the loop

    return apply_in_parallel(
        dataset["hashed_urls"],
        process_row,
        master_url_id_set,
        num_partitions=10,
        is_deterministic=is_deterministic,
    )


def add_negative_samples(dataset, ids_to_urls, is_deterministic=False):
    results = _add_negative_samples_wrapper(
        dataset, ids_to_urls, is_deterministic=is_deterministic
    )
    dataset.reset_index(drop=True, inplace=True)
    dataset["negative_sample_urls"] = results


Triple = namedtuple(
    "triple",
    [
        "query_id",
        "query_embedding",
        "relevant_doc_embedding",
        "irrelevant_doc_embedding",
    ],
)


def triples_to_embeddings(
    triples_df,
    url_to_doctext_mapping,
    sentence_piece_model,
    w2v_model,
    embedding_size=128,
):
    # triples_df = dataset[triple_columns]
    triples = []

    for row in tqdm(triples_df.iterrows(), total=triples_df.shape[0]):
        # print(len(row[1]["hashed_urls"].tolist()))
        embedding_hashed_urls = []
        embedding_negative_sample_urls = []
        for url_id, neg_url_id in zip(
            row[1]["hashed_urls"].tolist(), row[1]["negative_sample_urls"].tolist()
        ):

            embedding_hashed_urls.append(
                to_embedding(
                    sentence_piece_model,
                    url_to_doctext_mapping[url_id],
                    embedding_size,
                    w2v_model,
                )
            )
            embedding_negative_sample_urls.append(
                to_embedding(
                    sentence_piece_model,
                    url_to_doctext_mapping[neg_url_id],
                    embedding_size,
                    w2v_model,
                )
            )
        query_embedding = to_embedding(
            sentence_piece_model, row[1]["query"], embedding_size, w2v_model
        )
        triples.append(
            Triple(
                row[1]["query_id"],
                query_embedding,
                embedding_hashed_urls,
                embedding_negative_sample_urls,
            )
        )
    return triples


def to_embedding(sp, text, vector_size, w2v_model):
    tokens = sp.encode_as_pieces(text)

    embeddings = []
    for token in tokens:
        if token in w2v_model.wv:
            embeddings.append(w2v_model.wv[token])

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(vector_size)
