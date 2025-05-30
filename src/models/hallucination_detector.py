import torch
import torch.nn as nn
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
from typing import Dict, List, Tuple, Optional
import logging

class HallucinationDetector:
    """
    Advanced hallucination detection system for ASR models.
    Implements multiple detection strategies for comprehensive analysis.
    """
    
    def __init__(self, model_name: str = "openai/whisper-base", confidence_threshold: float = 0.7):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize detection components
        self.attention_analyzer = AttentionAnalyzer()
        self.temporal_validator = TemporalValidator()
        self.confidence_scorer = ConfidenceScorer()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def transcribe_with_analysis(self, audio_path: str) -> Dict:
        """
        Transcribe audio and perform comprehensive hallucination analysis
        """
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            # Generate transcription with attention weights
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    return_dict_in_generate=True,
                    output_attentions=True,
                    output_scores=True,
                    max_length=448
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )[0]
            
            # Perform hallucination analysis
            analysis_results = self._analyze_hallucination_risk(
                audio, transcription, outputs, inputs
            )
            
            return {
                'transcription': transcription,
                'hallucination_risk': analysis_results['overall_risk'],
                'confidence_score': analysis_results['confidence'],
                'attention_anomalies': analysis_results['attention_issues'],
                'temporal_consistency': analysis_results['temporal_score'],
                'detected_hallucinations': analysis_results['hallucination_segments'],
                'mitigation_suggestions': analysis_results['suggestions']
            }
            
        except Exception as e:
            self.logger.error(f"Error in transcription analysis: {e}")
            return {
                'transcription': '',
                'hallucination_risk': 1.0,
                'error': str(e)
            }
    
    def _analyze_hallucination_risk(self, audio: np.ndarray, transcription: str, 
                                   outputs, inputs) -> Dict:
        """
        Comprehensive hallucination risk analysis
        """
        # 1. Attention pattern analysis
        attention_score = self.attention_analyzer.analyze_patterns(
            outputs.attentions, audio.shape[0]
        )
        
        # 2. Confidence scoring
        confidence_score = self.confidence_scorer.calculate_confidence(
            outputs.scores, outputs.sequences
        )
        
        # 3. Temporal consistency check
        temporal_score = self.temporal_validator.validate_consistency(
            transcription, audio
        )
        
        # 4. Detect specific hallucination types
        hallucination_segments = self._detect_hallucination_types(
            transcription, audio, attention_score
        )
        
        # 5. Calculate overall risk
        overall_risk = self._calculate_overall_risk(
            attention_score, confidence_score, temporal_score, hallucination_segments
        )
        
        # 6. Generate mitigation suggestions
        suggestions = self._generate_mitigation_suggestions(
            overall_risk, hallucination_segments
        )
        
        return {
            'overall_risk': overall_risk,
            'confidence': confidence_score,
            'attention_issues': attention_score,
            'temporal_score': temporal_score,
            'hallucination_segments': hallucination_segments,
            'suggestions': suggestions
        }
    
    def _detect_hallucination_types(self, transcription: str, audio: np.ndarray, 
                                   attention_score: float) -> List[Dict]:
        """
        Detect specific types of hallucinations
        """
        hallucinations = []
        
        # Type 1: Repetition loops
        repetitions = self._detect_repetitions(transcription)
        if repetitions:
            hallucinations.extend(repetitions)
        
        # Type 2: Phantom words in silence
        phantom_words = self._detect_phantom_words(transcription, audio)
        if phantom_words:
            hallucinations.extend(phantom_words)
        
        # Type 3: Language switching
        lang_switches = self._detect_language_switches(transcription)
        if lang_switches:
            hallucinations.extend(lang_switches)
        
        return hallucinations
    
    def _detect_repetitions(self, transcription: str) -> List[Dict]:
        """Detect repetitive patterns in transcription"""
        words = transcription.split()
        repetitions = []
        
        # Check for immediate repetitions
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                repetitions.append({
                    'type': 'repetition',
                    'text': words[i],
                    'position': i,
                    'severity': 'medium'
                })
        
        # Check for longer repetitive patterns
        for pattern_length in range(2, min(6, len(words) // 2)):
            for i in range(len(words) - pattern_length * 2 + 1):
                pattern = words[i:i + pattern_length]
                next_pattern = words[i + pattern_length:i + pattern_length * 2]
                
                if pattern == next_pattern:
                    repetitions.append({
                        'type': 'pattern_repetition',
                        'text': ' '.join(pattern),
                        'position': i,
                        'severity': 'high'
                    })
        
        return repetitions
    
    def _detect_phantom_words(self, transcription: str, audio: np.ndarray) -> List[Dict]:
        """Detect words generated during silence periods"""
        # Simple silence detection (can be enhanced)
        rms_energy = librosa.feature.rms(y=audio)[0]
        silence_threshold = np.percentile(rms_energy, 10)
        
        # This is a simplified implementation
        # In practice, you'd need more sophisticated audio-text alignment
        phantom_words = []
        
        if len(transcription.strip()) > 0 and np.mean(rms_energy) < silence_threshold * 2:
            phantom_words.append({
                'type': 'phantom_words',
                'text': transcription,
                'severity': 'high',
                'reason': 'text_in_silence'
            })
        
        return phantom_words
    
    def _detect_language_switches(self, transcription: str) -> List[Dict]:
        """Detect unexpected language switches"""
        # Simplified language detection
        # In practice, use proper language detection libraries
        switches = []
        
        # Check for common non-English patterns in English transcription
        non_english_patterns = ['señor', 'señora', 'merci', 'danke', 'grazie']
        words = transcription.lower().split()
        
        for pattern in non_english_patterns:
            if pattern in words:
                switches.append({
                    'type': 'language_switch',
                    'text': pattern,
                    'severity': 'medium',
                    'detected_language': 'non_english'
                })
        
        return switches
    
    def _calculate_overall_risk(self, attention_score: float, confidence_score: float,
                              temporal_score: float, hallucinations: List) -> float:
        """Calculate overall hallucination risk score"""
        
        # Weight different factors
        attention_weight = 0.3
        confidence_weight = 0.4
        temporal_weight = 0.2
        hallucination_weight = 0.1
        
        # Calculate hallucination penalty
        hallucination_penalty = min(len(hallucinations) * 0.2, 1.0)
        
        # Combine scores (lower is better for risk)
        risk_score = (
            (1 - attention_score) * attention_weight +
            (1 - confidence_score) * confidence_weight +
            (1 - temporal_score) * temporal_weight +
            hallucination_penalty * hallucination_weight
        )
        
        return min(max(risk_score, 0.0), 1.0)
    
    def _generate_mitigation_suggestions(self, risk_score: float, 
                                       hallucinations: List) -> List[str]:
        """Generate suggestions for mitigating detected hallucinations"""
        suggestions = []
        
        if risk_score > 0.7:
            suggestions.append("High hallucination risk detected - manual review recommended")
        
        if any(h['type'] == 'repetition' for h in hallucinations):
            suggestions.append("Apply repetition filtering in post-processing")
        
        if any(h['type'] == 'phantom_words' for h in hallucinations):
            suggestions.append("Check audio quality and silence detection")
        
        if any(h['type'] == 'language_switch' for h in hallucinations):
            suggestions.append("Verify input language and model language settings")
        
        if risk_score > 0.5:
            suggestions.append("Consider using beam search with repetition penalty")
            suggestions.append("Apply confidence-based filtering")
        
        return suggestions


class AttentionAnalyzer:
    """Analyzes attention patterns for anomaly detection"""
    
    def analyze_patterns(self, attention_weights, audio_length: int) -> float:
        """Analyze attention patterns for anomalies"""
        # Simplified attention analysis
        # In practice, implement sophisticated attention pattern recognition
        return 0.8  # Placeholder


class TemporalValidator:
    """Validates temporal consistency of transcriptions"""
    
    def validate_consistency(self, transcription: str, audio: np.ndarray) -> float:
        """Validate temporal consistency"""
        # Simplified temporal validation
        # In practice, implement audio-text alignment validation
        return 0.7  # Placeholder


class ConfidenceScorer:
    """Calculates confidence scores for transcriptions"""
    
    def calculate_confidence(self, scores, sequences) -> float:
        """Calculate confidence score from model outputs"""
        if scores is None:
            return 0.5
        
        # Calculate average confidence from logits
        confidences = []
        for score in scores:
            # Convert logits to probabilities
            probs = torch.softmax(score, dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            confidences.extend(max_probs.cpu().numpy())
        
        return float(np.mean(confidences)) if confidences else 0.5