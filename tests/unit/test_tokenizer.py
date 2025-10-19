import unittest
import os
from cbert.tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

class TestTokenizers(unittest.TestCase):

    def test_char_tokenizer(self):
        tokenizer = CharTokenizer()
        text = "hello world"
        self.assertEqual(tokenizer.tokenize(text), ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd'])

    def test_keychar_tokenizer(self):
        tokenizer = KeyCharTokenizer()
        text = "int main() { return 0; }"
        self.assertEqual(tokenizer.tokenize(text), ['int', ' ', 'm', 'a', 'i', 'n', '(', ')', ' ', '{', ' ', 'return', ' ', '0', ';', ' ', '}'])

    def test_sentencepiece_tokenizer_train(self):
        # Create a dummy corpus file
        corpus_path = "test_corpus.txt"
        with open(corpus_path, "w") as f:
            f.write("hello world\n")
            f.write("this is a test\n")

        # Train a SentencePiece model
        model_prefix = "test_spm"
        vocab_size = 20
        SentencePieceTokenizer.train(corpus_path, model_prefix, vocab_size)

        # Check if the model and vocab files were created
        model_path = f"{model_prefix}.model"
        vocab_path = f"{model_prefix}.vocab"
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(vocab_path))

        # Test loading and tokenizing with the trained model
        tokenizer = SentencePieceTokenizer(model_path)
        tokens = tokenizer.tokenize("hello test")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)

        # Clean up the created files
        os.remove(corpus_path)
        os.remove(model_path)
        os.remove(vocab_path)

if __name__ == '__main__':
    unittest.main()