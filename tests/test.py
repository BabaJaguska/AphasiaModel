# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:49:52 2023

@author: mbelic
"""
import unittest
from languageModel import LanguageProcessingModel

class TestLanguageProcessingModel(unittest.TestCase):

    def setUp(self):
        # Initialize the model before each test
        self.model = LanguageProcessingModel()

    def test_comprehension(self):
        # Test the comprehension method
        input_text = "Hello, world!"
        embeddings, gen_sentence = self.model.comprehension(input_text)

        # Check if embeddings and generated sentence are not None
        self.assertIsNotNone(embeddings)
        self.assertIsNotNone(gen_sentence)
        self.assertTrue(isinstance(gen_sentence, str))


if __name__ == '__main__':
    unittest.main()
