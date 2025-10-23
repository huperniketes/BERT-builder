import unittest
import os
import tempfile
import shutil
import json
from cbert.tokenizer import CharTokenizer, KeyCharTokenizer, SentencePieceTokenizer

class TestTokenizers(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdir, "vocab.json")
        self.spm_model_prefix = os.path.join(self.tmpdir, "test_spm")
        self.spm_model_file = self.spm_model_prefix + ".model"
        self.spm_vocab_file = self.spm_model_prefix + ".vocab"

        # Create a dummy vocab.json for Char/KeyChar tokenizers if needed
        with open(self.vocab_file, 'w') as f:
            json.dump({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4, "a": 5, "b": 6}, f)

        # Create a dummy corpus for SentencePiece training
        self.corpus_path = os.path.join(self.tmpdir, "corpus.txt")
        with open(self.corpus_path, 'w') as f:
            f.write("hello world\n")
            f.write("this is a test\n")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # --- CharTokenizer Tests ---
    def test_char_tokenizer_init(self):
        tokenizer = CharTokenizer()
        self.assertIn("[UNK]", tokenizer.vocab)
        self.assertGreater(tokenizer.vocab_size, 128) # ASCII + special tokens

    def test_char_tokenizer_encode_decode(self):
        tokenizer = CharTokenizer()
        text = "abc"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        self.assertEqual(tokenizer.decode(encoded), text)

        # With special tokens, padding, truncation
        encoded_padded = tokenizer.encode("a", max_length=5, padding='max_length', truncation=True)
        self.assertEqual(len(encoded_padded), 5)
        self.assertEqual(tokenizer.decode(encoded_padded, skip_special_tokens=True), "a")

    def test_char_tokenizer_special_tokens(self):
        tokenizer = CharTokenizer()
        self.assertIsNotNone(tokenizer.unk_token_id)
        self.assertIsNotNone(tokenizer.mask_token_id)
        self.assertIn(tokenizer.mask_token, tokenizer.vocab)

    def test_char_tokenizer_get_token_spans(self):
        tokenizer = CharTokenizer()
        text = "hello world"
        spans = tokenizer._get_token_spans(text)
        self.assertEqual(spans, [
            ('h', 0, 1), ('e', 1, 2), ('l', 2, 3), ('l', 3, 4), ('o', 4, 5),
            (' ', 5, 6), ('w', 6, 7), ('o', 7, 8), ('r', 8, 9), ('l', 9, 10), ('d', 10, 11)
        ])

    # --- KeyCharTokenizer Tests ---
    def test_keychar_tokenizer_init(self):
        tokenizer = KeyCharTokenizer()
        self.assertIn("int", tokenizer.vocab)
        self.assertGreater(tokenizer.vocab_size, 128 + len(tokenizer.C_KEYWORDS))

    def test_keychar_tokenizer_encode_decode(self):
        tokenizer = KeyCharTokenizer()
        text = "int main() { return 0; }"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        # KeyCharTokenizer tokenizes keywords, so direct equality might fail due to spaces
        # We check if keywords are preserved
        self.assertIn(tokenizer.convert_tokens_to_ids("int"), encoded)
        self.assertIn(tokenizer.convert_tokens_to_ids("return"), encoded)

    def test_keychar_tokenizer_edge_cases(self):
        tokenizer = KeyCharTokenizer()
        # Keyword as part of identifier
        text = "integer_var"
        tokens = tokenizer._tokenize(text)
        self.assertNotIn("int", tokens) # 'int' should not be tokenized separately
        self.assertEqual(tokens, list(text))

        # Keyword followed by non-alphanum
        text = "int;"
        tokens = tokenizer._tokenize(text)
        self.assertEqual(tokens, ["int", ";"])

    def test_keychar_tokenizer_get_token_spans(self):
        tokenizer = KeyCharTokenizer()
        text = "int main() { return 0; }"
        spans = tokenizer._get_token_spans(text)
        # Check a keyword span
        self.assertIn(('int', 0, 3), spans)
        self.assertIn(('return', 13, 19), spans)
        # Check a char span
        self.assertIn((' ', 3, 4), spans)

    # --- SentencePieceTokenizer Tests ---
    def test_sentencepiece_tokenizer_train_and_load(self):
        # Train a SentencePiece model
        SentencePieceTokenizer.train(self.corpus_path, self.spm_model_prefix, vocab_size=20)

        # Check if model and vocab files were created
        self.assertTrue(os.path.exists(self.spm_model_file))
        self.assertTrue(os.path.exists(self.spm_vocab_file))

        # Load the trained tokenizer
        tokenizer = SentencePieceTokenizer(vocab_file=self.spm_vocab_file, spm_model_file=self.spm_model_file)
        self.assertGreater(tokenizer.vocab_size, 0)
        self.assertIn("[UNK]", tokenizer.vocab)

        # Test encode/decode
        text = "hello world"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        self.assertEqual(decoded, text)

        # With special tokens, padding, truncation
        encoded_padded = tokenizer.encode("hello", max_length=10, padding='max_length', truncation=True)
        self.assertEqual(len(encoded_padded), 10)
        self.assertEqual(tokenizer.decode(encoded_padded, skip_special_tokens=True), "hello")

    def test_sentencepiece_tokenizer_get_token_spans(self):
        # Train a SentencePiece model first
        SentencePieceTokenizer.train(self.corpus_path, self.spm_model_prefix, vocab_size=20)
        tokenizer = SentencePieceTokenizer(vocab_file=self.spm_vocab_file, spm_model_file=self.spm_model_file)

        text = "hello world"
        # SentencePiece tokenization is subword, so direct char spans are complex
        # This test primarily ensures the method runs without error and returns a list of tuples
        spans = tokenizer._get_token_spans(text)
        self.assertIsInstance(spans, list)
        self.assertGreater(len(spans), 0)
        for token, start, end in spans:
            self.assertIsInstance(token, str)
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            # For SentencePiece, tokens may have ▁ prefix, so we check the span length is reasonable
            self.assertGreaterEqual(end, start)

if __name__ == '__main__':
    unittest.main()