import unittest
import os
from cbert.tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

class TestTokenizers(unittest.TestCase):

    def setUp(self):
        self.sample_c_code = 'int main() { printf("hello world"); return 0; }'
        self.temp_dir = "tests/temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))
        os.rmdir(self.temp_dir)

    def test_char_tokenizer(self):
        tokenizer = CharTokenizer()
        tokens = tokenizer.tokenize(self.sample_c_code)
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens[0], 'i')

    def test_keychar_tokenizer(self):
        tokenizer = KeyCharTokenizer()
        # This will fail until the vocab is built correctly
        tokens = tokenizer.tokenize(self.sample_c_code)
        self.assertIsInstance(tokens, list)
        # 'int' should be a single token
        self.assertIn('int', tokens)

    def test_sentencepiece_tokenizer_train_and_tokenize(self):
        # Create a dummy corpus file
        corpus_path = os.path.join(self.temp_dir, "corpus.txt")
        with open(corpus_path, "w") as f:
            f.write(self.sample_c_code + "\n")
            f.write("for (int i = 0; i < 10; ++i) {}\n")

        model_prefix = os.path.join(self.temp_dir, "spm_model")
        
        # This will fail as train is not implemented
        SentencePieceTokenizer.train(corpus_path, model_prefix, vocab_size=100)

        # Check if model files were created
        self.assertTrue(os.path.exists(model_prefix + ".model"))
        self.assertTrue(os.path.exists(model_prefix + ".vocab"))

        tokenizer = SentencePieceTokenizer(model_prefix + ".model")
        tokens = tokenizer.tokenize(self.sample_c_code)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

if __name__ == '__main__':
    unittest.main()
