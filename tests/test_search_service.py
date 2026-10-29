
import unittest

import faiss
import numpy as np

from services.search_service import search_text_loaded


class SearchTextLoadedTests(unittest.TestCase):

    def setUp(self):
        self.index = faiss.IndexFlatIP(2)

        self.index.add(
            np.asarray(
                [
                    [0.99, 0.0],
                    [0.24, 0.0],
                    [0.40, 0.0],
                ],
                dtype=np.float32,
            )
        )

        self.paths = [
            "dataset/flamingo/image.jpg",
            "dataset/dog/dog.jpg",
            "dataset/bonsai/image.jpg",
        ]

        self.query_embedding = np.asarray(
            [[1.0, 0.0]],
            dtype=np.float32,
        )

    def test_results_are_sorted_by_ranking_score(self):
        results = search_text_loaded(
            query="A dog running on grass",
            index=self.index,
            paths=self.paths,
            query_embedding=self.query_embedding,
            top_k=3,
        )

        self.assertTrue(results)

        self.assertEqual(
            [result["rank"] for result in results],
            list(range(1, len(results) + 1)),
        )

        # Results should be ordered by their final ranking score.
        # ranking_score is internal and removed before results return,
        # so validate the ordering using the returned score and
        # category bonus for this test's no-attribute-embedding case.
        expected_scores = [
            result["score"]
            + (0.04 if result["category_match"] else 0.0)
            for result in results
        ]

        self.assertEqual(
            expected_scores,
            sorted(expected_scores, reverse=True),
        )

    def test_category_match_is_reported(self):
        results = search_text_loaded(
            query="A dog running on grass",
            index=self.index,
            paths=self.paths,
            query_embedding=self.query_embedding,
            top_k=3,
        )

        dog_results = [
            result
            for result in results
            if "/dog/" in result["image_path"].replace("\\", "/")
        ]

        self.assertTrue(dog_results)

        for result in dog_results:
            self.assertTrue(result["category_match"])

    def test_no_category_matches_preserves_similarity_order(self):
        results = search_text_loaded(
            query="A green flower",
            index=self.index,
            paths=self.paths,
            query_embedding=self.query_embedding,
            top_k=2,
        )

        self.assertEqual(
            [result["image_path"] for result in results],
            [
                "dataset/flamingo/image.jpg",
                "dataset/bonsai/image.jpg",
            ],
        )

        self.assertFalse(
            any(result["category_match"] for result in results)
        )

        self.assertGreaterEqual(
            results[0]["score"],
            results[1]["score"],
        )

    def test_top_k_limits_number_of_results(self):
        results = search_text_loaded(
            query="A dog running on grass",
            index=self.index,
            paths=self.paths,
            query_embedding=self.query_embedding,
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["rank"], 1)

    def test_empty_index_returns_no_results(self):
        empty_index = faiss.IndexFlatIP(2)

        results = search_text_loaded(
            query="A dog running on grass",
            index=empty_index,
            paths=[],
            query_embedding=self.query_embedding,
            top_k=3,
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()