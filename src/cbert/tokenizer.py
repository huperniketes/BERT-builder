import sentencepiece as spm
import os

class CharTokenizer:
    """Tokenizes a string into a list of characters."""
    def tokenize(self, text):
        return list(text)

class KeyCharTokenizer:
    """Tokenizes a string, treating C keywords as single tokens."""
    C_KEYWORDS = {
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "int", "long", "register", "return", "short", "signed", "sizeof", "static",
        "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while"
    }

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            # Check for longest matching keyword
            match = None
            for keyword in self.C_KEYWORDS:
                if text.startswith(keyword, i):
                    # Ensure it's a whole word
                    if (i + len(keyword) == len(text) or 
                        not text[i + len(keyword)].isalnum() and text[i + len(keyword)] != '_'):
                        if match is None or len(keyword) > len(match):
                            match = keyword
            
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

class SentencePieceTokenizer:
    """A wrapper around the SentencePiece library."""
    def __init__(self, model_path):
        """Initializes the tokenizer with a trained SentencePiece model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found at {model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    def tokenize(self, text):
        """Tokenizes text using the loaded SentencePiece model."""
        return self.sp.encode_as_pieces(text)

    @staticmethod
    def train(corpus_path, model_prefix, vocab_size):
        """Trains a new SentencePiece model."""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found at {corpus_path}")
            
        command = f"--input={corpus_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
        spm.SentencePieceTrainer.train(command)