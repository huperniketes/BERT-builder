"""
Real-time data quality monitoring system for C-BERT training pipeline.
Implements continuous quality assessment as required by GEMINI.md standards.
"""

import torch
import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import logging

@dataclass
class QualityMetrics:
    """Quality metrics for a data batch."""
    timestamp: float
    batch_size: int
    avg_sequence_length: float
    vocab_coverage: float
    duplicate_ratio: float
    encoding_errors: int
    memory_usage_mb: float
    processing_time_ms: float

class RealTimeQualityMonitor:
    """Real-time monitoring of data quality during training."""
    
    def __init__(self, window_size: int = 100, alert_thresholds: Optional[Dict] = None):
        self.window_size = window_size
        self.metrics_window = deque(maxlen=window_size)
        self.alert_thresholds = alert_thresholds or {
            'min_avg_length': 10,
            'max_avg_length': 1000,
            'min_vocab_coverage': 0.1,
            'max_duplicate_ratio': 0.3,
            'max_encoding_errors': 5
        }
        self.alerts = []
        self.logger = logging.getLogger(__name__)
    
    def monitor_batch(self, batch: Dict[str, torch.Tensor], processing_start_time: float) -> QualityMetrics:
        """Monitor quality metrics for a single batch."""
        start_time = time.time()
        
        # Calculate metrics
        input_ids = batch.get('input_ids', torch.tensor([]))
        batch_size = input_ids.shape[0] if len(input_ids.shape) > 0 else 0
        
        # Sequence length metrics
        if len(input_ids.shape) == 2:
            seq_lengths = (input_ids != 0).sum(dim=1).float()
            avg_seq_length = seq_lengths.mean().item() if batch_size > 0 else 0
        else:
            avg_seq_length = 0
        
        # Vocabulary coverage
        unique_tokens = len(torch.unique(input_ids)) if input_ids.numel() > 0 else 0
        vocab_coverage = unique_tokens / max(input_ids.max().item() + 1, 1) if input_ids.numel() > 0 else 0
        
        # Duplicate detection
        if batch_size > 1 and len(input_ids.shape) == 2:
            unique_sequences = len(set(tuple(seq.tolist()) for seq in input_ids))
            duplicate_ratio = 1 - (unique_sequences / batch_size)
        else:
            duplicate_ratio = 0
        
        # Memory usage
        memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # Processing time
        processing_time = (time.time() - processing_start_time) * 1000
        
        metrics = QualityMetrics(
            timestamp=time.time(),
            batch_size=batch_size,
            avg_sequence_length=avg_seq_length,
            vocab_coverage=vocab_coverage,
            duplicate_ratio=duplicate_ratio,
            encoding_errors=0,  # Would need actual encoding error detection
            memory_usage_mb=memory_usage,
            processing_time_ms=processing_time
        )
        
        # Add to sliding window
        self.metrics_window.append(metrics)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        return metrics
    
    def _check_alerts(self, metrics: QualityMetrics):
        """Check if metrics trigger any quality alerts."""
        alerts = []
        
        if metrics.avg_sequence_length < self.alert_thresholds['min_avg_length']:
            alerts.append(f"Low average sequence length: {metrics.avg_sequence_length:.1f}")
        
        if metrics.avg_sequence_length > self.alert_thresholds['max_avg_length']:
            alerts.append(f"High average sequence length: {metrics.avg_sequence_length:.1f}")
        
        if metrics.vocab_coverage < self.alert_thresholds['min_vocab_coverage']:
            alerts.append(f"Low vocabulary coverage: {metrics.vocab_coverage:.3f}")
        
        if metrics.duplicate_ratio > self.alert_thresholds['max_duplicate_ratio']:
            alerts.append(f"High duplicate ratio: {metrics.duplicate_ratio:.3f}")
        
        if metrics.encoding_errors > self.alert_thresholds['max_encoding_errors']:
            alerts.append(f"Encoding errors detected: {metrics.encoding_errors}")
        
        for alert in alerts:
            self.logger.warning(f"Quality Alert: {alert}")
            self.alerts.append({
                'timestamp': metrics.timestamp,
                'message': alert,
                'metrics': asdict(metrics)
            })
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of quality metrics over the monitoring window."""
        if not self.metrics_window:
            return {'status': 'no_data'}
        
        metrics_list = list(self.metrics_window)
        
        return {
            'window_size': len(metrics_list),
            'time_range': {
                'start': metrics_list[0].timestamp,
                'end': metrics_list[-1].timestamp
            },
            'averages': {
                'batch_size': sum(m.batch_size for m in metrics_list) / len(metrics_list),
                'sequence_length': sum(m.avg_sequence_length for m in metrics_list) / len(metrics_list),
                'vocab_coverage': sum(m.vocab_coverage for m in metrics_list) / len(metrics_list),
                'duplicate_ratio': sum(m.duplicate_ratio for m in metrics_list) / len(metrics_list),
                'processing_time_ms': sum(m.processing_time_ms for m in metrics_list) / len(metrics_list)
            },
            'trends': self._calculate_trends(),
            'recent_alerts': self.alerts[-10:] if self.alerts else []
        }
    
    def _calculate_trends(self) -> Dict[str, str]:
        """Calculate trends for key metrics."""
        if len(self.metrics_window) < 10:
            return {'status': 'insufficient_data'}
        
        metrics_list = list(self.metrics_window)
        recent = metrics_list[-10:]
        older = metrics_list[-20:-10] if len(metrics_list) >= 20 else metrics_list[:-10]
        
        def trend_direction(recent_avg: float, older_avg: float) -> str:
            if abs(recent_avg - older_avg) < 0.01:
                return 'stable'
            return 'increasing' if recent_avg > older_avg else 'decreasing'
        
        return {
            'sequence_length': trend_direction(
                sum(m.avg_sequence_length for m in recent) / len(recent),
                sum(m.avg_sequence_length for m in older) / len(older)
            ),
            'vocab_coverage': trend_direction(
                sum(m.vocab_coverage for m in recent) / len(recent),
                sum(m.vocab_coverage for m in older) / len(older)
            ),
            'processing_time': trend_direction(
                sum(m.processing_time_ms for m in recent) / len(recent),
                sum(m.processing_time_ms for m in older) / len(older)
            )
        }
    
    def export_metrics(self, filepath: str):
        """Export collected metrics to JSON file."""
        data = {
            'metrics': [asdict(m) for m in self.metrics_window],
            'alerts': self.alerts,
            'summary': self.get_quality_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Quality metrics exported to {filepath}")

class TrainingQualityTracker:
    """Track quality metrics throughout the entire training process."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.epoch_metrics = []
        self.batch_monitor = RealTimeQualityMonitor()
        self.logger = logging.getLogger(__name__)
    
    def start_epoch(self, epoch: int):
        """Start tracking a new training epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        self.epoch_batches = []
    
    def track_batch(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> QualityMetrics:
        """Track quality for a single batch."""
        batch_start_time = time.time()
        metrics = self.batch_monitor.monitor_batch(batch, batch_start_time)
        
        # Add batch context
        metrics.batch_idx = batch_idx
        metrics.epoch = self.current_epoch
        
        self.epoch_batches.append(metrics)
        return metrics
    
    def end_epoch(self):
        """Finalize epoch tracking and save metrics."""
        epoch_duration = time.time() - self.epoch_start_time
        
        epoch_summary = {
            'epoch': self.current_epoch,
            'duration_seconds': epoch_duration,
            'total_batches': len(self.epoch_batches),
            'avg_batch_size': sum(m.batch_size for m in self.epoch_batches) / len(self.epoch_batches),
            'avg_processing_time': sum(m.processing_time_ms for m in self.epoch_batches) / len(self.epoch_batches),
            'quality_alerts': len([a for a in self.batch_monitor.alerts if a['timestamp'] >= self.epoch_start_time])
        }
        
        self.epoch_metrics.append(epoch_summary)
        
        # Save epoch metrics
        epoch_file = f"{self.output_dir}/quality_epoch_{self.current_epoch}.json"
        with open(epoch_file, 'w') as f:
            json.dump({
                'epoch_summary': epoch_summary,
                'batch_metrics': [asdict(m) for m in self.epoch_batches]
            }, f, indent=2)
        
        self.logger.info(f"Epoch {self.current_epoch} quality metrics saved to {epoch_file}")
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training quality report."""
        if not self.epoch_metrics:
            return {'status': 'no_training_data'}
        
        return {
            'total_epochs': len(self.epoch_metrics),
            'total_duration': sum(e['duration_seconds'] for e in self.epoch_metrics),
            'avg_epoch_duration': sum(e['duration_seconds'] for e in self.epoch_metrics) / len(self.epoch_metrics),
            'total_batches': sum(e['total_batches'] for e in self.epoch_metrics),
            'quality_trend': self._analyze_quality_trend(),
            'performance_trend': self._analyze_performance_trend(),
            'recommendations': self._generate_recommendations()
        }
    
    def _analyze_quality_trend(self) -> Dict[str, Any]:
        """Analyze quality trends across epochs."""
        if len(self.epoch_metrics) < 2:
            return {'status': 'insufficient_epochs'}
        
        first_half = self.epoch_metrics[:len(self.epoch_metrics)//2]
        second_half = self.epoch_metrics[len(self.epoch_metrics)//2:]
        
        first_alerts = sum(e['quality_alerts'] for e in first_half) / len(first_half)
        second_alerts = sum(e['quality_alerts'] for e in second_half) / len(second_half)
        
        return {
            'alert_trend': 'improving' if second_alerts < first_alerts else 'degrading',
            'first_half_avg_alerts': first_alerts,
            'second_half_avg_alerts': second_alerts
        }
    
    def _analyze_performance_trend(self) -> Dict[str, Any]:
        """Analyze performance trends across epochs."""
        if len(self.epoch_metrics) < 2:
            return {'status': 'insufficient_epochs'}
        
        processing_times = [e['avg_processing_time'] for e in self.epoch_metrics]
        
        return {
            'trend': 'improving' if processing_times[-1] < processing_times[0] else 'degrading',
            'first_epoch_time': processing_times[0],
            'last_epoch_time': processing_times[-1],
            'improvement_pct': ((processing_times[0] - processing_times[-1]) / processing_times[0]) * 100
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality analysis."""
        recommendations = []
        
        quality_trend = self._analyze_quality_trend()
        if quality_trend.get('alert_trend') == 'degrading':
            recommendations.append("Quality alerts increasing - review data preprocessing pipeline")
        
        performance_trend = self._analyze_performance_trend()
        if performance_trend.get('trend') == 'degrading':
            recommendations.append("Processing time increasing - consider batch size optimization")
        
        recent_summary = self.batch_monitor.get_quality_summary()
        if recent_summary.get('averages', {}).get('duplicate_ratio', 0) > 0.2:
            recommendations.append("High duplicate ratio detected - review data deduplication")
        
        return recommendations