import unittest
import pandas as pd
import numpy as np
from utils.data_utils import add_negative_samples

# Assuming the provided functions are defined here or imported correctly


class TestAddNegativeSamples(unittest.TestCase):
    def test_add_negative_samples(self):
        # Mock dataset after add_hashed_urls has been applied
        data = {
            "query_id": [1, 2],
            "passages": [None, None],  # Placeholder, not used in this test
            "hashed_urls": [
                np.array(["hash1", "hash2"]),
                np.array(["hash3"]),
            ],
        }
        train_dataset = pd.DataFrame(data)

        # Mock ids_to_urls assuming some hashed URL values
        ids_to_urls = {
            "hash1": "url1",
            "hash2": "url2",
            "hash3": "url3",
            "hash4": "url4",
            "hash5": "url5",
        }

        # Apply the function under test
        add_negative_samples(train_dataset, ids_to_urls, is_deterministic=True)

        # Check if 'negative_sample_urls' column was added
        self.assertIn("negative_sample_urls", train_dataset.columns)

        # Check if the negative samples do not include any hashed URL present in 'hashed_urls' for each row
        self.assertTrue(train_dataset.shape[0] > 0)
        for index, row in train_dataset.iterrows():
            hashed_urls = set(row["hashed_urls"])
            negative_samples = set(row["negative_sample_urls"])
            self.assertTrue(
                hashed_urls.isdisjoint(negative_samples),
                msg=f"Row {index} contains overlapping hashed URLs in 'hashed_urls' and 'negative_sample_urls'",
            )


# python -m tests.negative_sampling_test
if __name__ == "__main__":
    unittest.main()
