"""
Comprehensive evaluation metrics for C-BERT model.
Includes code-specific metrics, semantic similarity, and statistical analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import ast
import re
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class CodeEvaluationMetrics:
    """Comprehensive evaluation metrics for code understanding tasks."""
    
    @staticmethod
    def bleu_score(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
        """Calculate BLEU score for code generation tasks."""
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
        
        total_score = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            if len(pred_tokens) == 0:
                continue
                
            scores = []
            for n in range(1, min(n_gram + 1, len(pred_tokens) + 1)):
                pred_ngrams = get_ngrams(pred_tokens, n)
                ref_ngrams = get_ngrams(ref_tokens, n)
                
                overlap = sum((pred_ngrams & ref_ngrams).values())
                total_pred = sum(pred_ngrams.values())
                
                if total_pred > 0:
                    scores.append(overlap / total_pred)
                else:
                    scores.append(0)
            
            if scores:
                # Geometric mean of n-gram precisions
                total_score += np.exp(np.mean(np.log(np.maximum(scores, 1e-10))))
        
        return total_score / len(predictions) if predictions else 0
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy."""
        matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
        return matches / len(predictions) if predictions else 0
    
    @staticmethod
    def token_accuracy(pred_tokens: torch.Tensor, ref_tokens: torch.Tensor, 
                      ignore_index: int = -100) -> float:
        """Calculate token-level accuracy."""
        mask = ref_tokens != ignore_index
        if mask.sum() == 0:
            return 0.0
        correct = (pred_tokens[mask] == ref_tokens[mask]).sum().item()
        return correct / mask.sum().item()
    
    @staticmethod
    def ast_similarity(code1: str, code2: str) -> float:
        """Calculate AST-based similarity between code snippets."""
        try:
            # Simple AST node counting for C-like syntax
            def count_ast_nodes(code: str) -> Dict[str, int]:
                nodes = Counter()
                # Count control structures
                nodes['if'] = len(re.findall(r'\bif\s*\(', code))
                nodes['for'] = len(re.findall(r'\bfor\s*\(', code))
                nodes['while'] = len(re.findall(r'\bwhile\s*\(', code))
                nodes['function'] = len(re.findall(r'\w+\s*\([^)]*\)\s*{', code))
                nodes['assignment'] = len(re.findall(r'\w+\s*=', code))
                return nodes
            
            nodes1 = count_ast_nodes(code1)
            nodes2 = count_ast_nodes(code2)
            
            all_keys = set(nodes1.keys()) | set(nodes2.keys())
            if not all_keys:
                return 1.0
            
            similarity = 0
            for key in all_keys:
                v1, v2 = nodes1.get(key, 0), nodes2.get(key, 0)
                if v1 + v2 > 0:
                    similarity += 2 * min(v1, v2) / (v1 + v2)
            
            return similarity / len(all_keys)
        except:
            return 0.0
    
    @staticmethod
    def semantic_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> float:
        """Calculate cosine similarity between code embeddings."""
        cos_sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=-1)
        return cos_sim.mean().item()

class VulnerabilityMetrics:
    """Metrics for vulnerability detection tasks."""
    
    @staticmethod
    def classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate precision, recall, F1, and AUC for binary classification."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions > 0.5, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels, predictions)
        except ValueError:
            auc = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

class StatisticalAnalysis:
    """Statistical analysis tools for evaluation results."""
    
    @staticmethod
    def confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for evaluation scores."""
        if len(scores) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(scores)
        sem = stats.sem(scores)
        h = sem * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        return (mean - h, mean + h)
    
    @staticmethod
    def significance_test(scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Perform statistical significance test between two sets of scores."""
        if len(scores1) < 2 or len(scores2) < 2:
            return {'p_value': 1.0, 'statistic': 0.0, 'significant': False}
        
        statistic, p_value = stats.ttest_ind(scores1, scores2)
        
        return {
            'p_value': p_value,
            'statistic': statistic,
            'significant': p_value < 0.05
        }
    
    @staticmethod
    def bootstrap_confidence(scores: List[float], n_bootstrap: int = 1000, 
                           confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if len(scores) < 2:
            return (0.0, 0.0)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (lower, upper)

class ComprehensiveEvaluator:
    """Main evaluator that combines all metrics."""
    
    def __init__(self):
        self.code_metrics = CodeEvaluationMetrics()
        self.vuln_metrics = VulnerabilityMetrics()
        self.stats = StatisticalAnalysis()
    
    def evaluate_mlm_task(self, predictions: torch.Tensor, labels: torch.Tensor,
                         pred_texts: List[str], ref_texts: List[str]) -> Dict[str, Any]:
        """Comprehensive evaluation for MLM task."""
        results = {}
        
        # Token-level accuracy
        results['token_accuracy'] = self.code_metrics.token_accuracy(predictions, labels)
        
        # Text-level metrics
        results['exact_match'] = self.code_metrics.exact_match(pred_texts, ref_texts)
        results['bleu_score'] = self.code_metrics.bleu_score(pred_texts, ref_texts)
        
        # AST similarity
        ast_similarities = [
            self.code_metrics.ast_similarity(pred, ref) 
            for pred, ref in zip(pred_texts, ref_texts)
        ]
        results['ast_similarity'] = np.mean(ast_similarities)
        
        # Statistical analysis
        if len(ast_similarities) > 1:
            ci_lower, ci_upper = self.stats.confidence_interval(ast_similarities)
            results['ast_similarity_ci'] = (ci_lower, ci_upper)
        
        return results
    
    def evaluate_vulnerability_task(self, predictions: np.ndarray, 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Comprehensive evaluation for vulnerability detection."""
        results = self.vuln_metrics.classification_metrics(predictions, labels)
        
        # Statistical analysis
        pred_scores = predictions.tolist()
        if len(pred_scores) > 1:
            ci_lower, ci_upper = self.stats.bootstrap_confidence(pred_scores)
            results['prediction_ci'] = (ci_lower, ci_upper)
        
        return results
    
    def cross_validate_scores(self, all_scores: List[List[float]]) -> Dict[str, Any]:
        """Analyze cross-validation results."""
        fold_means = [np.mean(scores) for scores in all_scores]
        
        return {
            'cv_mean': np.mean(fold_means),
            'cv_std': np.std(fold_means),
            'cv_scores': fold_means,
            'cv_ci': self.stats.confidence_interval(fold_means)
        }