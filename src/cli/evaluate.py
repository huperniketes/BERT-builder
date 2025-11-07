import argparse
import json
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, BertConfig
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
import ast
import re
from collections import defaultdict, Counter

from cbert.c_ast_evaluator import CASTEvaluator, is_c_available
from cbert.trainer import TextDataset, get_tokenizer # Re-use TextDataset and get_tokenizer
from cbert.model import create_cbert_model # Import create_cbert_model
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_accuracy_mlm(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    # Only calculate accuracy on non-ignored tokens (labels != -100)
    masked_tokens = labels != -100
    correct = (predictions[masked_tokens] == labels[masked_tokens]).sum().item()
    total = masked_tokens.sum().item()
    return correct / total if total > 0 else 0

def calculate_ast_metrics(logits, labels):
    """Calculate AST tagging metrics including precision, recall, F1"""
    # Flatten predictions and labels for token-level evaluation
    predictions = torch.argmax(logits, dim=-1).cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    # Filter out padding tokens (label = -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    return accuracy, precision, recall, f1

def calculate_vi_metrics(logits, labels):
    """Calculate vulnerability identification metrics"""
    # For binary classification, use the last token's logits or pool over all tokens
    # Here we use mean pooling over sequence length
    if logits.dim() == 3:  # [batch_size, seq_len, num_classes]
        # Average pooling over sequence length, ignoring padding
        mask = labels != -100
        mask = mask.unsqueeze(-1).float()
        pooled_logits = (logits * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        pooled_logits = logits
    
    # Get probabilities for positive class
    probs = torch.softmax(pooled_logits, dim=-1)[:, 1].cpu().numpy()
    
    # Get predictions
    predictions = torch.argmax(pooled_logits, dim=-1).cpu().numpy()
    
    # Handle labels (should be binary)
    if labels.dim() > 1:
        labels = labels[:, 0]  # Take first token if sequence
    labels = labels.cpu().numpy()
    
    # Filter out any padding labels
    valid_mask = labels != -100
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    probs = probs[valid_mask]
    
    if len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate AUC-ROC if we have both classes
    try:
        if len(np.unique(labels)) > 1:
            auc = roc_auc_score(labels, probs)
        else:
            auc = 0.0
    except ValueError:
        auc = 0.0
    
    return accuracy, precision, recall, f1, auc

def calculate_perplexity(loss):
    return torch.exp(loss).item()

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def save_checkpoint(checkpoint_file, batch_idx, total_loss, total_accuracy, num_batches):
    checkpoint = {
        "batch_idx": batch_idx,
        "total_loss": total_loss,
        "total_accuracy": total_accuracy,
        "num_batches": num_batches
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)

# --- Evaluation Dataset Classes ---
class BaseEvaluationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.line_offsets = []
        self._load_line_offsets()

    def _load_line_offsets(self):
        # Try to load pre-computed offsets first
        offsets_file = self.file_path.replace('.txt', '_lineoffsets.json')
        if os.path.exists(offsets_file):
            with open(offsets_file, 'r') as f:
                data = json.load(f)
                self.line_offsets = data['line_offsets']
            logger.info(f"Loaded {len(self.line_offsets)} pre-computed line offsets from {offsets_file}")
        else:
            # Fallback to building offsets (original behavior)
            self._build_line_offsets()

    def _build_line_offsets(self):
        current_offset = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    current_offset += len(line.encode('utf-8')) # Account for skipped line's bytes
                    continue
                
                # Basic check for meaningful content (non-special tokens)
                encoded_content = self.tokenizer.encode(stripped_line, add_special_tokens=False, max_length=self.max_length, truncation=True)
                if encoded_content: # If there are any actual content tokens
                    self.line_offsets.append(current_offset)
                current_offset += len(line.encode('utf-8'))
        logger.info(f"Built {len(self.line_offsets)} line offsets from {self.file_path}")

    def __len__(self):
        return len(self.line_offsets)

    def _get_line(self, idx):
        offset = self.line_offsets[idx]
        with open(self.file_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            return f.readline().strip()

class MLMEvaluationDataset(BaseEvaluationDataset):
    def __getitem__(self, idx):
        line = self._get_line(idx)
        
        # For MLM evaluation, we need input_ids and labels (original token_ids)
        # We don't apply masking here, as the model will predict based on the input
        # and we compare against the original tokens as labels.
        token_ids = self.tokenizer.encode(line, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        
        input_ids = torch.tensor(token_ids)
        labels = input_ids.clone() # Labels are the original tokens for MLM evaluation

        return {"input_ids": input_ids, "labels": labels}

class ASTEvaluationDataset(BaseEvaluationDataset):
    def __init__(self, file_path: str, tokenizer, max_length: int, language: str = 'c'):
        super().__init__(file_path, tokenizer, max_length)
        self.language = language.lower()
        
        if self.language == 'c' and is_c_available():
            self.c_evaluator = CASTEvaluator()
            self.ast_label_map = self.c_evaluator.get_label_map()
        else:
            # Use Python AST evaluation (existing implementation)
            self.c_evaluator = None
            self.ast_label_map = {
                'FunctionDef': 0, 'ClassDef': 1, 'Name': 2, 'Constant': 3, 'Attribute': 4,
                'Call': 5, 'BinOp': 6, 'UnaryOp': 7, 'Compare': 8, 'If': 9, 'For': 10,
                'While': 11, 'Return': 12, 'Assign': 13, 'AugAssign': 14, 'AnnAssign': 15,
                'Import': 16, 'ImportFrom': 17, 'Try': 18, 'Except': 19, 'With': 20,
                'List': 21, 'Dict': 22, 'Tuple': 23, 'Set': 24, 'Lambda': 25,
                'Comprehension': 26, 'GeneratorExp': 27, 'ListComp': 28, 'DictComp': 29,
                'SetComp': 30, 'Expr': 31, 'Pass': 32, 'Break': 33, 'Continue': 34,
                'Global': 35, 'Nonlocal': 36, 'Assert': 37, 'Delete': 38, 'Raise': 39,
                'Yield': 40, 'YieldFrom': 41, 'AsyncFunctionDef': 42, 'AsyncFor': 43,
                'AsyncWith': 44, 'Await': 45, 'OTHER': 46
            }
        
        self.reverse_label_map = {v: k for k, v in self.ast_label_map.items()}
    
    def _parse_ast_labels(self, code_line):
        """Parse AST and generate token-level labels"""
        if self.language == 'c' and self.c_evaluator:
            return self.c_evaluator.get_ast_labels(code_line)
        else:
            # Use existing Python AST parsing logic
            return self._parse_python_ast_labels(code_line)
    
    def _parse_python_ast_labels(self, code_line):
        """Original Python AST parsing logic"""
        """Parse AST and generate token-level labels"""
        try:
            # Parse the code into an AST
            tree = ast.parse(code_line)
            
            # Get token-level AST node types
            token_labels = []
            tokens = self.tokenizer.tokenize(code_line)
            
            # Enhanced AST node mapping with better context awareness
            for i, token in enumerate(tokens):
                # Special tokens
                if token.startswith('##') and token.endswith('##'):
                    token_labels.append(self.ast_label_map['OTHER'])
                    continue
                
                # Context-aware mapping based on surrounding tokens
                context_tokens = tokens[max(0, i-2):i+3]  # 2 tokens before and after
                
                # Function and class definitions
                if token == 'def' and any(t in ['(', ':'] for t in context_tokens):
                    token_labels.append(self.ast_label_map['FunctionDef'])
                elif token == 'class' and any(t in ['(', ':'] for t in context_tokens):
                    token_labels.append(self.ast_label_map['ClassDef'])
                elif token in ['if', 'elif', 'else']:
                    token_labels.append(self.ast_label_map['If'])
                elif token in ['for']:
                    token_labels.append(self.ast_label_map['For'])
                elif token in ['while']:
                    token_labels.append(self.ast_label_map['While'])
                elif token in ['try']:
                    token_labels.append(self.ast_label_map['Try'])
                elif token in ['except']:
                    token_labels.append(self.ast_label_map['Except'])
                elif token in ['with']:
                    token_labels.append(self.ast_label_map['With'])
                elif token in ['return']:
                    token_labels.append(self.ast_label_map['Return'])
                elif token in ['yield']:
                    token_labels.append(self.ast_label_map['Yield'])
                elif token in ['await']:
                    token_labels.append(self.ast_label_map['Await'])
                elif token in ['import']:
                    token_labels.append(self.ast_label_map['Import'])
                elif token in ['from']:
                    token_labels.append(self.ast_label_map['ImportFrom'])
                elif token in ['pass']:
                    token_labels.append(self.ast_label_map['Pass'])
                elif token in ['break']:
                    token_labels.append(self.ast_label_map['Break'])
                elif token in ['continue']:
                    token_labels.append(self.ast_label_map['Continue'])
                elif token in ['global']:
                    token_labels.append(self.ast_label_map['Global'])
                elif token in ['nonlocal']:
                    token_labels.append(self.ast_label_map['Nonlocal'])
                elif token in ['assert']:
                    token_labels.append(self.ast_label_map['Assert'])
                elif token in ['del']:
                    token_labels.append(self.ast_label_map['Delete'])
                elif token in ['raise']:
                    token_labels.append(self.ast_label_map['Raise'])
                elif token in ['lambda']:
                    token_labels.append(self.ast_label_map['Lambda'])
                elif token in ['async']:
                    # Check if it's async function/for/with
                    if any(t in ['def', 'for', 'with'] for t in context_tokens):
                        if 'def' in context_tokens:
                            token_labels.append(self.ast_label_map['AsyncFunctionDef'])
                        elif 'for' in context_tokens:
                            token_labels.append(self.ast_label_map['AsyncFor'])
                        elif 'with' in context_tokens:
                            token_labels.append(self.ast_label_map['AsyncWith'])
                    else:
                        token_labels.append(self.ast_label_map['OTHER'])
                # Assignment operations
                elif token == '=' and i > 0:
                    token_labels.append(self.ast_label_map['Assign'])
                elif token in ['+=', '-=', '*=', '/=', '%=', '**=', '//=', '>>=', '<<=', '&=', '|=', '^=']:
                    token_labels.append(self.ast_label_map['AugAssign'])
                elif token == ':' and i > 0 and tokens[i-1] in [')', ']']:
                    # Type annotation
                    token_labels.append(self.ast_label_map['AnnAssign'])
                # Data structures
                elif token == '[' and any(t in [']', ','] for t in context_tokens):
                    token_labels.append(self.ast_label_map['List'])
                elif token == '{' and any(t in ['}', ':'] for t in context_tokens):
                    if ':' in context_tokens:
                        token_labels.append(self.ast_label_map['Dict'])
                    else:
                        token_labels.append(self.ast_label_map['Set'])
                elif token == '(' and i > 0 and tokens[i-1] in ['def', 'lambda']:
                    token_labels.append(self.ast_label_map['Lambda'])
                elif token == '(' and i > 0:
                    token_labels.append(self.ast_label_map['Call'])
                # Comprehensions
                elif token in ['for', 'in'] and any(t in ['[', '{', '('] for t in context_tokens):
                    token_labels.append(self.ast_label_map['Comprehension'])
                # Operators
                elif token in ['+', '-', '*', '/', '//', '%', '**', '<<', '>>', '|', '&', '^', '~']:
                    token_labels.append(self.ast_label_map['BinOp'])
                elif token in ['not', 'and', 'or']:
                    token_labels.append(self.ast_label_map['UnaryOp'])
                elif token in ['==', '!=', '<', '<=', '>', '>=', 'is', 'is not', 'in', 'not in']:
                    token_labels.append(self.ast_label_map['Compare'])
                # Identifiers and constants
                elif token.replace('_', '').replace('-', '').isalnum():
                    # Check if it's likely a constant
                    if (token[0].isupper() or token.replace('_', '').isupper() or 
                        token.replace('.', '').replace('_', '').isdigit()):
                        token_labels.append(self.ast_label_map['Constant'])
                    else:
                        token_labels.append(self.ast_label_map['Name'])
                else:
                    token_labels.append(self.ast_label_map['OTHER'])
            
            return token_labels
            
        except (SyntaxError, ValueError, Exception) as e:
            # If parsing fails, return default labels
            logger.debug(f"AST parsing failed for line: {code_line[:50]}... Error: {str(e)}")
            tokens = self.tokenizer.tokenize(code_line)
            return [self.ast_label_map['OTHER']] * len(tokens)
    
    def __getitem__(self, idx):
        line = self._get_line(idx)
        
        # Check if line is in JSON format (structured data)
        try:
            data = json.loads(line)
            if isinstance(data, dict) and 'code' in data and 'ast_labels' in data:
                # Use provided structured data
                code = data['code']
                ast_labels = data['ast_labels']
            else:
                # Treat as plain code
                code = line
                ast_labels = None
        except json.JSONDecodeError:
            # Plain code line
            code = line
            ast_labels = None
        
        # Generate AST labels if not provided
        if ast_labels is None:
            ast_labels = self._parse_ast_labels(code)
        
        # Tokenize code
        token_ids = self.tokenizer.encode(code, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        
        # Convert AST labels to tensor and pad/truncate
        if len(ast_labels) > self.max_length - 2:  # Account for [CLS] and [SEP]
            ast_labels = ast_labels[:self.max_length - 2]
        
        # Add special token labels (typically OTHER)
        ast_labels = [self.ast_label_map['OTHER']] + ast_labels + [self.ast_label_map['OTHER']]
        
        # Pad to max_length
        while len(ast_labels) < self.max_length:
            ast_labels.append(-100)  # Use -100 for padding (ignored in loss)
        
        return {
            "input_ids": torch.tensor(token_ids),
            "labels": torch.tensor(ast_labels)
        }

class VIEvaluationDataset(BaseEvaluationDataset):
    def __init__(self, file_path: str, tokenizer, max_length: int):
        super().__init__(file_path, tokenizer, max_length)
        # Define vulnerability patterns for detection
        self.vulnerability_patterns = {
            'sql_injection': [
                r'execute\s*\(\s*["\'].*?\+.*?["\']',  # String concatenation in SQL
                r'execute\s*\(\s*["\'].*?%.*?["\']',   # String formatting in SQL
                r'execute\s*\(\s*f["\'].*?\{.*?\}.*?["\']',  # f-strings in SQL
                r'cursor\.execute\s*\([^)]*\+',         # SQL with concatenation
                r'query\s*=\s*["\'].*?\+.*?["\']',     # Query building with concat
            ],
            'command_injection': [
                r'os\.system\s*\([^)]*\+',              # OS command with concat
                r'subprocess\.call\s*\([^)]*\+',        # Subprocess with concat
                r'eval\s*\([^)]*\+',                    # eval with concat
                r'exec\s*\([^)]*\+',                    # exec with concat
                r'popen\s*\([^)]*\+',                   # popen with concat
            ],
            'buffer_overflow': [
                r'strcpy\s*\([^)]*\)',                  # strcpy usage
                r'strcat\s*\([^)]*\)',                  # strcat usage
                r'gets\s*\([^)]*\)',                    # gets usage
                r'scanf\s*\([^)]*\)',                   # scanf without bounds
                r'sprintf\s*\([^)]*\)',                 # sprintf usage
            ],
            'xss': [
                r'innerHTML\s*=\s*[^;]*\+',             # innerHTML with concat
                r'outerHTML\s*=\s*[^;]*\+',             # outerHTML with concat
                r'document\.write\s*\([^)]*\+',         # document.write with concat
                r'eval\s*\([^)]*\+',                    # eval with user input
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\./',                 # File open with ../
                r'file\s*\([^)]*\.\./',                 # file() with ../
                r'read_file\s*\([^)]*\.\./',            # read_file with ../
                r'include\s*\([^)]*\.\./',              # include with ../
            ],
            'hardcoded_credentials': [
                r'password\s*=\s*["\'][^"\']+["\']',   # Hardcoded password
                r'api_key\s*=\s*["\'][^"\']+["\']',     # Hardcoded API key
                r'secret\s*=\s*["\'][^"\']+["\']',      # Hardcoded secret
                r'token\s*=\s*["\'][^"\']+["\']',       # Hardcoded token
            ],
            'weak_crypto': [
                r'md5\s*\([^)]*\)',                     # MD5 usage
                r'sha1\s*\([^)]*\)',                    # SHA1 usage
                r'hashlib\.md5',                        # MD5 import
                r'hashlib\.sha1',                       # SHA1 import
                r'Crypto\.Cipher\.DES',                 # DES cipher
                r'Crypto\.Cipher\.RC4',                 # RC4 cipher
            ],
            'insecure_deserialization': [
                r'pickle\.loads\s*\([^)]*\)',           # pickle.loads
                r'cPickle\.loads\s*\([^)]*\)',          # cPickle.loads
                r'marshal\.loads\s*\([^)]*\)',          # marshal.loads
                r'yaml\.load\s*\([^)]*\)',              # yaml.load (unsafe)
            ]
        }
        
    def _detect_vulnerabilities(self, code_line):
        """Detect vulnerabilities in code using pattern matching and heuristics"""
        vulnerability_score = 0
        detected_patterns = []
        
        try:
            # Pattern-based detection
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, code_line, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        vulnerability_score += len(matches)
                        detected_patterns.extend([vuln_type] * len(matches))
            
            # Enhanced heuristic checks with context awareness
            heuristics = {
                'dangerous_functions': len(re.findall(
                    r'\b(exec|eval|system|shell_exec|passthru|proc_open|popen|subprocess\.call|os\.system)\s*\(', 
                    code_line, re.IGNORECASE
                )),
                'file_inclusion': len(re.findall(
                    r'\b(include|require|include_once|require_once)\s*\([^)]*\$', 
                    code_line, re.IGNORECASE
                )),
                'sql_injection_heuristic': len(re.findall(
                    r'(select|insert|update|delete)\s+from\s+\w+\s+where.*?[\+\%]', 
                    code_line, re.IGNORECASE
                )),
                'hardcoded_secrets': len(re.findall(
                    r'(password|secret|key|token|api_key|private_key)\s*=\s*["\'][^"\']{6,}["\']', 
                    code_line, re.IGNORECASE
                )),
                'weak_random': len(re.findall(
                    r'\b(random|rand|mt_rand)\s*\(', 
                    code_line, re.IGNORECASE
                )),
                'file_operations': len(re.findall(
                    r'\b(fopen|file_get_contents|readfile|file_put_contents)\s*\([^)]*\$', 
                    code_line, re.IGNORECASE
                )),
                'header_injection': len(re.findall(
                    r'\b(header|setcookie)\s*\([^)]*\$', 
                    code_line, re.IGNORECASE
                )),
                'ldap_injection': len(re.findall(
                    r'\b(ldap_search|ldap_query)\s*\([^)]*\$', 
                    code_line, re.IGNORECASE
                )),
                'xpath_injection': len(re.findall(
                    r'\b(xpath|xpath_query)\s*\([^)]*\$', 
                    code_line, re.IGNORECASE
                )),
                'xxe': len(re.findall(
                    r'\b(DOMDocument|SimpleXMLElement|xml_parser_create)\s*\(', 
                    code_line, re.IGNORECASE
                ))
            }
            
            # Weight the heuristics differently
            heuristic_weights = {
                'dangerous_functions': 2,
                'file_inclusion': 2,
                'sql_injection_heuristic': 3,
                'hardcoded_secrets': 2,
                'weak_random': 1,
                'file_operations': 1,
                'header_injection': 2,
                'ldap_injection': 2,
                'xpath_injection': 2,
                'xxe': 2
            }
            
            for heuristic, count in heuristics.items():
                if count > 0:
                    vulnerability_score += count * heuristic_weights.get(heuristic, 1)
                    detected_patterns.extend([heuristic] * count)
            
            # Additional context-based checks
            # Check for input validation absence
            if re.search(r'\$_GET|\$_POST|\$_REQUEST|\$_COOKIE', code_line, re.IGNORECASE):
                if not re.search(r'(htmlspecialchars|strip_tags|addslashes|mysql_real_escape_string)', code_line, re.IGNORECASE):
                    vulnerability_score += 1
                    detected_patterns.append('missing_input_validation')
            
            # Check for session security issues
            if re.search(r'session_start\(\)', code_line, re.IGNORECASE):
                if not re.search(r'session_regenerate_id|session_set_cookie_params', code_line, re.IGNORECASE):
                    vulnerability_score += 1
                    detected_patterns.append('weak_session_management')
            
            # Normalize score to binary (vulnerable if score > 0)
            is_vulnerable = 1 if vulnerability_score > 0 else 0
            
            return is_vulnerable, detected_patterns
            
        except Exception as e:
            logger.debug(f"Vulnerability detection failed for line: {code_line[:50]}... Error: {str(e)}")
            return 0, []
    
    def __getitem__(self, idx):
        line = self._get_line(idx)
        
        # Check if line is in JSON format (structured data)
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                if 'code' in data and 'vulnerable' in data:
                    # Use provided structured data
                    code = data['code']
                    vulnerability_label = int(data['vulnerable'])
                elif 'code' in data:
                    # Code provided, detect vulnerability
                    code = data['code']
                    vulnerability_label, patterns = self._detect_vulnerabilities(code)
                else:
                    # Treat as plain code
                    code = line
                    vulnerability_label, patterns = self._detect_vulnerabilities(code)
            else:
                # Plain code line
                code = line
                vulnerability_label, patterns = self._detect_vulnerabilities(code)
        except json.JSONDecodeError:
            # Plain code line
            code = line
            vulnerability_label, patterns = self._detect_vulnerabilities(code)
        
        # Tokenize code
        token_ids = self.tokenizer.encode(code, add_special_tokens=True, max_length=self.max_length, padding='max_length', truncation=True)
        
        # For VI, we use a single label for the entire sequence
        # The label is repeated for all tokens (or we can use just the first token)
        labels = torch.full((len(token_ids),), vulnerability_label, dtype=torch.long)
        
        # Set padding tokens to -100 (ignored in loss)
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in token_ids]
        for i, mask_val in enumerate(attention_mask):
            if mask_val == 0:
                labels[i] = -100
        
        return {
            "input_ids": torch.tensor(token_ids),
            "labels": labels
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a C-BERT model.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the trained model directory (Hugging Face format). This directory should contain config.json, model.safetensors/pytorch_model.bin, and tokenizer files.")
    parser.add_argument("--task", type=str, default='mlm', choices=['mlm', 'ast', 'vi'], help="The evaluation task.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the pre-processed evaluation data file.")
    parser.add_argument("--output-file", type=str, default=None, help="File to save the JSON evaluation results (default: stdout).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length for tokenization.")
    parser.add_argument("--tokenizer-type", type=str, required=True, choices=['char', 'keychar', 'spe'], help="Type of tokenizer used during training.")
    parser.add_argument("--vocab-file", type=str, default=None, help="Path to the vocabulary file (required for SentencePiece tokenizer). If not provided, attempts to load from model-dir.")
    parser.add_argument("--spm-model-file", type=str, default=None, help="Path to the SentencePiece model file (required for SentencePiece tokenizer). If not provided, attempts to load from model-dir.")
    parser.add_argument("--checkpoint-file", type=str, default=None, help="Path to checkpoint file for resuming evaluation. If not provided, creates one based on output-file.")
    parser.add_argument("--use-precomputed-offsets", action="store_true", help="Use pre-computed line offsets from data preprocessing (faster startup).")
    parser.add_argument("--language", type=str, default='c', choices=['c', 'python'], help="Programming language for AST evaluation")

    args = parser.parse_args()

    # Validate SentencePiece tokenizer requirements
    if args.tokenizer_type == 'spe':
        if not args.vocab_file and not os.path.exists(os.path.join(args.model_dir, "vocab.json")):
            print("Error: --vocab-file is required for SentencePiece tokenizer or vocab.json must exist in model-dir")
            sys.exit(1)
        if not args.spm_model_file and not os.path.exists(os.path.join(args.model_dir, "spm.model")):
            print("Error: --spm-model-file is required for SentencePiece tokenizer or spm.model must exist in model-dir")
            sys.exit(1)

    # Validate file paths exist
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory does not exist: {args.model_dir}")
        sys.exit(1)
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory does not exist: {args.dataset_dir}")
        sys.exit(1)
    if args.vocab_file and not os.path.exists(args.vocab_file):
        print(f"Error: Vocab file does not exist: {args.vocab_file}")
        sys.exit(1)
    if args.spm_model_file and not os.path.exists(args.spm_model_file):
        print(f"Error: SentencePiece model file does not exist: {args.smp_model_file}")
        sys.exit(1)

    # Validate numeric arguments
    if args.batch_size <= 0:
        print("Error: --batch-size must be greater than 0")
        sys.exit(1)
    if args.max_length <= 0:
        print("Error: --max-length must be greater than 0")
        sys.exit(1)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Load Model Configuration
    config = BertConfig.from_pretrained(args.model_dir)
    if args.max_length != config.max_position_embeddings:
        logger.warning(f"Provided max_length ({args.max_length}) differs from model config max_position_embeddings ({config.max_position_embeddings}). Using config value.")
        args.max_length = config.max_position_embeddings

    # 2. Load Tokenizer
    vocab_file = args.vocab_file if args.vocab_file else os.path.join(args.model_dir, "vocab.json")
    spm_model_file = args.spm_model_file if args.spm_model_file else os.path.join(args.model_dir, "spm.model")
    
    tokenizer = get_tokenizer(args.tokenizer_type, vocab_file=vocab_file, spm_model_file=spm_model_file)
    logger.info(f"Tokenizer loaded: {type(tokenizer).__name__}")

    # 3. Load Model
    model = BertForMaskedLM.from_pretrained(args.model_dir, config=config)
    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded and set to evaluation mode.")

    # 4. Load Dataset and DataLoader
    if args.task == 'mlm':
        eval_dataset = MLMEvaluationDataset(args.dataset_dir, tokenizer, args.max_length)
    elif args.task == 'ast':
        eval_dataset = ASTEvaluationDataset(args.dataset_dir, tokenizer, args.max_length, language=args.language)
    elif args.task == 'vi':
        eval_dataset = VIEvaluationDataset(args.dataset_dir, tokenizer, args.max_length)
    else:
        raise ValueError(f"Unknown evaluation task: {args.task}")

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False) # No shuffling for evaluation
    logger.info(f"Evaluation dataset loaded with {len(eval_dataset)} samples.")

    # Setup checkpoint file
    checkpoint_file = args.checkpoint_file
    if not checkpoint_file and args.output_file:
        checkpoint_file = args.output_file.replace('.json', '_checkpoint.json')
    elif not checkpoint_file:
        checkpoint_file = 'evaluation_checkpoint.json'

    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_file)
    if checkpoint:
        start_batch = checkpoint["batch_idx"] + 1
        total_loss = checkpoint["total_loss"]
        total_accuracy = checkpoint["total_accuracy"]
        num_batches = checkpoint["num_batches"]
        logger.info(f"Resuming from batch {start_batch} (processed {num_batches} batches)")
    else:
        start_batch = 0
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        logger.info("Starting evaluation from beginning")

# Initialize metrics storage
    ast_metrics = {'precision': [], 'recall': [], 'f1': []}
    vi_metrics = {'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    # 5. Perform Evaluation
    total_batches = len(eval_dataloader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx < start_batch:
                continue
            
            try:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_loss += loss.item()
                
                # Calculate accuracy based on task
                if args.task == 'mlm':
                    total_accuracy += calculate_accuracy_mlm(logits, labels)
                elif args.task == 'ast':
                    accuracy, precision, recall, f1 = calculate_ast_metrics(logits, labels)
                    total_accuracy += accuracy
                    ast_metrics['precision'].append(precision)
                    ast_metrics['recall'].append(recall)
                    ast_metrics['f1'].append(f1)
                elif args.task == 'vi':
                    accuracy, precision, recall, f1, auc = calculate_vi_metrics(logits, labels)
                    total_accuracy += accuracy
                    vi_metrics['precision'].append(precision)
                    vi_metrics['recall'].append(recall)
                    vi_metrics['f1'].append(f1)
                    vi_metrics['auc'].append(auc)

                num_batches += 1
                
                # Progress indicator with task-specific info
                progress = (batch_idx + 1) / total_batches * 100
                avg_loss_current = total_loss / num_batches
                avg_acc_current = total_accuracy / num_batches
                
                if args.task == 'ast' and ast_metrics['f1']:
                    current_f1 = np.mean(ast_metrics['f1'][-10:]) if len(ast_metrics['f1']) >= 10 else np.mean(ast_metrics['f1'])
                    print(f"\rProgress: {batch_idx + 1}/{total_batches} ({progress:.1f}%) | Loss: {avg_loss_current:.4f} | Acc: {avg_acc_current:.4f} | F1: {current_f1:.4f}", end="", flush=True)
                elif args.task == 'vi' and vi_metrics['f1']:
                    current_f1 = np.mean(vi_metrics['f1'][-10:]) if len(vi_metrics['f1']) >= 10 else np.mean(vi_metrics['f1'])
                    current_auc = np.mean(vi_metrics['auc'][-10:]) if len(vi_metrics['auc']) >= 10 else np.mean(vi_metrics['auc'])
                    print(f"\rProgress: {batch_idx + 1}/{total_batches} ({progress:.1f}%) | Loss: {avg_loss_current:.4f} | Acc: {avg_acc_current:.4f} | F1: {current_f1:.4f} | AUC: {current_auc:.4f}", end="", flush=True)
                else:
                    print(f"\rProgress: {batch_idx + 1}/{total_batches} ({progress:.1f}%) | Loss: {avg_loss_current:.4f} | Acc: {avg_acc_current:.4f}", end="", flush=True)
                
                # Save checkpoint every 10 batches
                if batch_idx % 10 == 0:
                    save_checkpoint(checkpoint_file, batch_idx, total_loss, total_accuracy, num_batches)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    print()  # New line after progress indicator

avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0
    avg_perplexity = calculate_perplexity(torch.tensor(avg_loss)) if args.task == 'mlm' else 0.0 # Perplexity only for MLM

    # 6. Calculate Metrics
    results = {
        "task": args.task,
        "average_loss": avg_loss,
        "average_accuracy": avg_accuracy,
        "average_perplexity": avg_perplexity,
        "num_samples": len(eval_dataset)
    }
    
    # Add task-specific metrics
    if args.task == 'ast' and ast_metrics['precision']:
        results.update({
            "average_precision": float(np.mean(ast_metrics['precision'])),
            "average_recall": float(np.mean(ast_metrics['recall'])),
            "average_f1": float(np.mean(ast_metrics['f1'])),
            "num_ast_classes": len(ASTEvaluationDataset.ast_label_map),
            "ast_label_distribution": {k: v for k, v in Counter([label for batch_metrics in ast_metrics['precision'] for label in [label]]).most_common(5)} if ast_metrics['precision'] else {}
        })
    elif args.task == 'vi' and vi_metrics['precision']:
        results.update({
            "average_precision": float(np.mean(vi_metrics['precision'])),
            "average_recall": float(np.mean(vi_metrics['recall'])),
            "average_f1": float(np.mean(vi_metrics['f1'])),
            "average_auc": float(np.mean(vi_metrics['auc']))
        })
    print(f"\nEvaluation completed!")
    logger.info(f"Evaluation Results: {results}")

    # 7. Save Results
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {args.output_file}")
    else:
        print(json.dumps(results, indent=2))
    
    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("Checkpoint file removed after successful completion")

if __name__ == "__main__":
    main()
