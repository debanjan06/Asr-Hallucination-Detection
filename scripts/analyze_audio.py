#!/usr/bin/env python3
"""
Audio Analysis Script for Hallucination Detection
Usage: python analyze_audio.py <audio_file_path>
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hallucination_detector import HallucinationDetector
import json

def analyze_audio_file(audio_path: str, output_file: str = None):
    """
    Analyze an audio file for hallucinations
    """
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file '{audio_path}' not found")
        return
    
    print(f"🔄 Analyzing audio file: {audio_path}")
    print("=" * 50)
    
    # Initialize detector
    try:
        detector = HallucinationDetector()
        print("✅ Hallucination detector initialized")
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return
    
    # Analyze audio
    try:
        result = detector.transcribe_with_analysis(audio_path)
        
        # Display results
        print(f"\n📝 Transcription:")
        print(f"   {result['transcription']}")
        
        print(f"\n🎯 Hallucination Risk: {result['hallucination_risk']:.3f}")
        print(f"🔢 Confidence Score: {result['confidence_score']:.3f}")
        
        if result['detected_hallucinations']:
            print(f"\n⚠️  Detected Issues:")
            for issue in result['detected_hallucinations']:
                print(f"   • {issue['type']}: '{issue['text']}' (severity: {issue['severity']})")
        else:
            print(f"\n✅ No hallucinations detected")
        
        if result['mitigation_suggestions']:
            print(f"\n💡 Suggestions:")
            for suggestion in result['mitigation_suggestions']:
                print(f"   • {suggestion}")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to: {output_file}")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze audio for ASR hallucinations")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    analyze_audio_file(args.audio_file, args.output)

if __name__ == "__main__":
    main()