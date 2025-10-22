import os
import json
import collections
import sentencepiece as spm
from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

class CharTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file=None, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", **kwargs):
        super().__init__(unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, **kwargs)

        if vocab_file is not None:
            self.vocab = self._load_vocab(vocab_file)
        else:
            # Default ASCII vocab + special tokens
            self.vocab = self._build_default_vocab()
        
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def _build_default_vocab(self):
        vocab = collections.OrderedDict()
        # Add special tokens first
        for token in [self.unk_token, self.sep_token, self.pad_token, self.cls_token, self.mask_token]:
            if token not in vocab:
                vocab[token] = len(vocab)
        # Add ASCII characters
        for i in range(128):
            char = chr(i)
            if char not in vocab:
                vocab[char] = len(vocab)
        return vocab

    def _load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def _get_token_spans(self, text):
        """Returns a list of (token, start_char_idx, end_char_idx) for each token."""
        spans = []
        current_idx = 0
        for token in self._tokenize(text):
            start_char_idx = text.find(token, current_idx)
            if start_char_idx == -1: # Should not happen if _tokenize is consistent
                start_char_idx = current_idx # Fallback
            end_char_idx = start_char_idx + len(token)
            spans.append((token, start_char_idx, end_char_idx))
            current_idx = end_char_idx
        return spans

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> tuple:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        return (vocab_file,)


class KeyCharTokenizer(CharTokenizer):
    C_KEYWORDS = {
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "int", "long", "register", "return", "short", "signed", "sizeof", "static",
        "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while"
    }

    def _build_default_vocab(self):
        vocab = super()._build_default_vocab()
        # Add C keywords after special tokens and ASCII chars
        for keyword in sorted(list(self.C_KEYWORDS)): # Sort for consistent vocab order
            if keyword not in vocab:
                vocab[keyword] = len(vocab)
        return vocab

    def _tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            # Check for longest matching keyword
            match = None
            for keyword in self.C_KEYWORDS:
                if text.startswith(keyword, i):
                    # Ensure it's a whole word (not part of an identifier)
                    if (i + len(keyword) == len(text) or 
                        not text[i + len(keyword)].isalnum() and text[i + len(keyword)] != '_'):
                        if match is None or len(keyword) > len(match): # Prefer longer keywords
                            match = keyword
            
            if match:
                tokens.append(match)
                i += len(match)
            else:
                tokens.append(text[i])
                i += 1
        return tokens


class SentencePieceTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, spm_model_file=None, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]", **kwargs):
        super().__init__(unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, **kwargs)

        if not os.path.exists(spm_model_file):
            raise FileNotFoundError(f"SentencePiece model not found at {spm_model_file}")
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_model_file)

        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    def _load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def _get_token_spans(self, text):
        """Returns a list of (token, start_char_idx, end_char_idx) for each token."""
        # For SentencePiece, we can use its internal encoding to get accurate spans
        encoded_pieces = self.sp_model.encode(text, out_type=spm.EncodeResult)
        spans = []
        for piece in encoded_pieces:
            spans.append((piece.piece, piece.begin, piece.end))
        return spans

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None) -> tuple:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)
        return (vocab_file,)

    @staticmethod
    def train(corpus_path, model_prefix, vocab_size):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file not found at {corpus_path}")
            
        command = f"--input={corpus_path} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
        spm.SentencePieceTrainer.train(command)