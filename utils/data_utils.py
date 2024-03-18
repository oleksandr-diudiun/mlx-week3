import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from tqdm import tqdm

def create_lookups(full_dataset):
    """
    full_dataset: pandas dataframe from the parquet files.
    Returns: urls_to_ids, ids_to_urls dictionaries for
    fast negative sampling.
    """
    print(f"dataset shape: {full_dataset.shape}")
    all_urls = full_dataset["passages"].apply(lambda x: x["url"]).tolist()
    unique_urls = set([item for sublist in all_urls for item in sublist])
    print(f"Total number of urls: {sum(len(i) for i in all_urls)}")
    print(f"Total number of unique urls: {len(unique_urls)}")

    # Use an md5 hash for the urls for deterministic mapping
    def generate_md5_hash(s):
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    ids_to_urls = {generate_md5_hash(url): url for url in unique_urls}

    urls_to_ids = {url: i for i, url in ids_to_urls.items()}
    print(f"Total number of hashed urls: {len(ids_to_urls)}")

    query_ids = full_dataset["query_id"].tolist()
    assert len(query_ids) == len(set(query_ids))
    print(f"Total number of queries: {len(query_ids)}")
    return ids_to_urls, urls_to_ids


# def create_triples(dataset, ids_to_urls, urls_to_ids, create_small=False):
#     """
#     Input: dataset: pandas dataframe from the parquet files
#     ids_to_urls: dictionary of hashed urls to urls from create_lookups
#     urls_to_ids: dictionary of urls to hashed urls from create_lookups

#     returns: Triples for the torch Dataset
#         [(query_hash_id, relevant_urls, irrelevant_urls))]

#     The relevant_urls and irrelevant_urls are hashed urls.

#     Query ID is from the query_id column in the parquet dataset.
#     """
#     triples = []
#     master_url_id_set = np.array(list(set(ids_to_urls.keys())))
#     for row in dataset.iterrows() if not create_small else dataset.head(10).iterrows():
#         urls = row[1]["passages"]["url"]
#         query_id = row[1]["query_id"]
#         relevant_url_ids = np.array(list(set([urls_to_ids[url] for url in urls])))
#         irrelevant_url_ids = np.setdiff1d(
#             master_url_id_set, relevant_url_ids, assume_unique=True
#         )
#         sampled_ids = np.random.choice(
#             irrelevant_url_ids, size=len(relevant_url_ids), replace=False
#         )

#         triple = (query_id, list(relevant_url_ids), sampled_ids)
#         triples.append(triple)
#     return triples
    # NL = '\n'
    # TAB = '\t'
    # print(f"Query ID, Query: {query_id}, {row[1]['query']}")
    # print(f"R {NL.join([ids_to_urls[i]+NL+TAB+str(i) for i in relevant_url_ids])}")
    # print(f"IR: {NL.join([ids_to_urls[i]+NL+TAB+str(i) for i in irrelevant_url_ids])}")

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
