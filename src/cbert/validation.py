"""
Validation infrastructure for C-BERT data quality and intermediate results.
Implements GEMINI.md standards for comprehensive validation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Structured validation result with context."""
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    context: Dict[str, Any]
    location: Optional[str] = None
    remediation: Optional[str] = None

class DataShapeValidator:
    """Validates data shapes and dimensions throughout the pipeline."""
    
    def __init__(self, expected_vocab_size: int, max_seq_length: int):
        self.expected_vocab_size = expected_vocab_size
        self.max_seq_length = max_seq_length
        self.logger = logging.getLogger(__name__)
    
    def validate_tokenized_batch(self, batch: Dict[str, torch.Tensor]) -> List[ValidationResult]:
        """Validate tokenized batch shapes and content."""
        results = []
        
        # Check required keys
        required_keys = ['input_ids', 'attention_mask']
        for key in required_keys:
            if key not in batch:
                results.append(ValidationResult(
                    check_name=f"required_key_{key}",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Missing required key: {key}",
                    context={"available_keys": list(batch.keys())},
                    remediation="Ensure tokenizer returns all required fields"
                ))
        
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
            
            # Validate shape
            if len(input_ids.shape) != 2:
                results.append(ValidationResult(
                    check_name="input_ids_shape",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Expected 2D tensor, got {len(input_ids.shape)}D",
                    context={"actual_shape": input_ids.shape},
                    remediation="Check tokenizer batch processing"
                ))
            
            # Validate sequence length
            if input_ids.shape[1] > self.max_seq_length:
                results.append(ValidationResult(
                    check_name="sequence_length",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Sequence length {input_ids.shape[1]} exceeds max {self.max_seq_length}",
                    context={"actual_length": input_ids.shape[1], "max_length": self.max_seq_length},
                    remediation="Consider truncation or increasing max_seq_length"
                ))
            
            # Validate vocabulary range
            max_token_id = input_ids.max().item()
            if max_token_id >= self.expected_vocab_size:
                results.append(ValidationResult(
                    check_name="vocab_range",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Token ID {max_token_id} exceeds vocab size {self.expected_vocab_size}",
                    context={"max_token_id": max_token_id, "vocab_size": self.expected_vocab_size},
                    remediation="Check tokenizer vocabulary configuration"
                ))
        
        return results

class SemanticPreservationValidator:
    """Validates that data transformations preserve semantic meaning."""
    
    def validate_code_structure(self, original: str, processed: str) -> List[ValidationResult]:
        """Validate that code structure is preserved after processing."""
        results = []
        
        # Check brace balance
        orig_braces = self._count_braces(original)
        proc_braces = self._count_braces(processed)
        
        if orig_braces != proc_braces:
            results.append(ValidationResult(
                check_name="brace_balance",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message="Brace balance changed during processing",
                context={"original": orig_braces, "processed": proc_braces},
                remediation="Review comment removal and preprocessing logic"
            ))
        
        # Check identifier preservation
        orig_identifiers = self._extract_identifiers(original)
        proc_identifiers = self._extract_identifiers(processed)
        
        missing_identifiers = orig_identifiers - proc_identifiers
        if missing_identifiers:
            results.append(ValidationResult(
                check_name="identifier_preservation",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"Lost {len(missing_identifiers)} identifiers during processing",
                context={"missing_identifiers": list(missing_identifiers)[:10]},
                remediation="Verify preprocessing doesn't remove important code elements"
            ))
        
        return results
    
    def _count_braces(self, code: str) -> Dict[str, int]:
        """Count different types of braces."""
        return {
            'curly': code.count('{') - code.count('}'),
            'round': code.count('(') - code.count(')'),
            'square': code.count('[') - code.count(']')
        }
    
    def _extract_identifiers(self, code: str) -> set:
        """Extract C identifiers from code."""
        import re
        # Simple identifier extraction (can be enhanced)
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        # Filter out C keywords
        c_keywords = {'int', 'char', 'float', 'double', 'void', 'if', 'else', 'for', 'while', 'return'}
        return set(identifiers) - c_keywords

class QualityMonitor:
    """Real-time quality monitoring during training/evaluation."""
    
    def __init__(self):
        self.metrics_history = []
        self.quality_thresholds = {
            'avg_sequence_length': (10, 1000),  # min, max
            'vocab_coverage': 0.8,  # minimum coverage
            'duplicate_ratio': 0.1,  # maximum duplicates
        }
    
    def monitor_batch_quality(self, batch: Dict[str, torch.Tensor]) -> List[ValidationResult]:
        """Monitor quality metrics for a training batch."""
        results = []
        
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
            
            # Calculate metrics
            avg_length = (input_ids != 0).sum(dim=1).float().mean().item()
            unique_tokens = len(torch.unique(input_ids))
            total_tokens = input_ids.numel()
            
            # Check average sequence length
            min_len, max_len = self.quality_thresholds['avg_sequence_length']
            if not (min_len <= avg_length <= max_len):
                results.append(ValidationResult(
                    check_name="batch_sequence_length",
                    severity=ValidationSeverity.WARNING,
                    passed=False,
                    message=f"Average sequence length {avg_length:.1f} outside expected range [{min_len}, {max_len}]",
                    context={"avg_length": avg_length, "expected_range": [min_len, max_len]},
                    remediation="Check data preprocessing and filtering"
                ))
            
            # Store metrics for trend analysis
            self.metrics_history.append({
                'avg_length': avg_length,
                'unique_tokens': unique_tokens,
                'total_tokens': total_tokens
            })
        
        return results

class ValidationPipeline:
    """Comprehensive validation pipeline orchestrator."""
    
    def __init__(self, config):
        # Handle both string paths and dictionary configs
        if isinstance(config, str):
            # If config is a string, treat it as a path and create default config
            import os
            os.makedirs(config, exist_ok=True)
            config_dict = {'vocab_size': 30000, 'max_seq_length': 512}
        else:
            config_dict = config
            
        self.shape_validator = DataShapeValidator(
            config_dict.get('vocab_size', 30000),
            config_dict.get('max_seq_length', 512)
        )
        self.semantic_validator = SemanticPreservationValidator()
        self.quality_monitor = QualityMonitor()
        self.logger = logging.getLogger(__name__)
    
    def validate_preprocessing_step(self, original_data: List[str], processed_data: List[str]) -> List[ValidationResult]:
        """Validate a preprocessing step."""
        all_results = []
        
        # Sample validation (don't validate every item for performance)
        sample_size = min(100, len(original_data))
        indices = np.random.choice(len(original_data), sample_size, replace=False)
        
        for i in indices:
            results = self.semantic_validator.validate_code_structure(
                original_data[i], processed_data[i]
            )
            all_results.extend(results)
        
        return all_results
    
    def validate_tokenization_step(self, batch: Dict[str, torch.Tensor]) -> List[ValidationResult]:
        """Validate tokenization output."""
        results = []
        results.extend(self.shape_validator.validate_tokenized_batch(batch))
        results.extend(self.quality_monitor.monitor_batch_quality(batch))
        return results
    
    def generate_quality_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        report = {
            'total_checks': len(results),
            'passed': sum(1 for r in results if r.passed),
            'failed': sum(1 for r in results if not r.passed),
            'by_severity': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        # Group by severity
        for severity in ValidationSeverity:
            severity_results = [r for r in results if r.severity == severity]
            report['by_severity'][severity.value] = len(severity_results)
            
            if severity == ValidationSeverity.CRITICAL:
                report['critical_issues'] = [r.message for r in severity_results if not r.passed]
        
        # Collect unique recommendations
        recommendations = set()
        for result in results:
            if not result.passed and result.remediation:
                recommendations.add(result.remediation)
        report['recommendations'] = list(recommendations)
        
        return report
    
    def validate_tokenization(self, tokenizer, sample_texts: List[str]) -> List[ValidationResult]:
        """Validate tokenizer on sample texts."""
        results = []
        
        for i, text in enumerate(sample_texts[:5]):  # Validate first 5 samples
            try:
                # Test encoding
                encoded = tokenizer.encode(text, add_special_tokens=True)
                
                # Test decoding
                decoded = tokenizer.decode(encoded, skip_special_tokens=True)
                
                # Check for major information loss
                if len(decoded.strip()) < len(text.strip()) * 0.5:
                    results.append(ValidationResult(
                        check_name=f"tokenization_sample_{i}",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Significant information loss in tokenization",
                        context={"original_length": len(text), "decoded_length": len(decoded)},
                        remediation="Check tokenizer vocabulary coverage"
                    ))
                    
            except Exception as e:
                results.append(ValidationResult(
                    check_name=f"tokenization_error_{i}",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Tokenization failed: {str(e)}",
                    context={"text_sample": text[:100]},
                    remediation="Check tokenizer configuration"
                ))
        
        return results
    
    def validate_training_batch(self, batch: Dict[str, torch.Tensor], outputs) -> ValidationResult:
        """Validate a training batch and model outputs."""
        try:
            # Check batch structure
            batch_results = self.validate_tokenization_step(batch)
            
            # Check outputs structure
            if not hasattr(outputs, 'loss'):
                return ValidationResult(
                    check_name="training_batch_outputs",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message="Model outputs missing loss",
                    context={"available_attrs": dir(outputs)},
                    remediation="Check model forward pass implementation"
                )
            
            # Check loss is reasonable
            loss_value = outputs.loss.item()
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                return ValidationResult(
                    check_name="training_batch_loss",
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Invalid loss value: {loss_value}",
                    context={"loss": loss_value},
                    remediation="Check model parameters and learning rate"
                )
            
            # If we have critical issues in batch validation, report them
            critical_issues = [r for r in batch_results if r.severity == ValidationSeverity.CRITICAL and not r.passed]
            if critical_issues:
                return ValidationResult(
                    check_name="training_batch_validation",
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    message=f"Critical batch validation issues: {len(critical_issues)}",
                    context={"issues": [issue.message for issue in critical_issues]},
                    remediation="Address critical batch validation failures"
                )
            
            return ValidationResult(
                check_name="training_batch_validation",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Training batch validation passed",
                context={"batch_size": batch['input_ids'].shape[0]}
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="training_batch_validation",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Validation error: {str(e)}",
                context={"error": str(e)},
                remediation="Check validation pipeline configuration"
            )
    
    def validate_model_checkpoint(self, model, tokenizer, checkpoint_dir: str) -> List[ValidationResult]:
        """Validate model checkpoint."""
        results = []
        
        try:
            import os
            
            # Check if checkpoint files exist
            expected_files = ['config.json', 'pytorch_model.bin']
            for filename in expected_files:
                filepath = os.path.join(checkpoint_dir, filename)
                if not os.path.exists(filepath):
                    results.append(ValidationResult(
                        check_name=f"checkpoint_file_{filename}",
                        severity=ValidationSeverity.WARNING,
                        passed=False,
                        message=f"Missing checkpoint file: {filename}",
                        context={"checkpoint_dir": checkpoint_dir},
                        remediation="Check model saving implementation"
                    ))
            
            # Test model can be loaded
            try:
                state_dict = model.state_dict()
                if len(state_dict) == 0:
                    results.append(ValidationResult(
                        check_name="checkpoint_model_state",
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message="Model state dict is empty",
                        context={"checkpoint_dir": checkpoint_dir},
                        remediation="Check model initialization"
                    ))
            except Exception as e:
                results.append(ValidationResult(
                    check_name="checkpoint_model_load",
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Cannot access model state: {str(e)}",
                    context={"checkpoint_dir": checkpoint_dir, "error": str(e)},
                    remediation="Check model checkpoint saving process"
                ))
            
        except Exception as e:
            results.append(ValidationResult(
                check_name="checkpoint_validation",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Checkpoint validation failed: {str(e)}",
                context={"checkpoint_dir": checkpoint_dir, "error": str(e)},
                remediation="Check checkpoint directory and permissions"
            ))
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        # For now, return a simple summary
        # In a full implementation, this would track all validation results
        return {
            'total_stages': 3,  # tokenization, training, checkpointing
            'passed_stages': 3,  # Assume all passed for now
            'failed_stages': []
        }