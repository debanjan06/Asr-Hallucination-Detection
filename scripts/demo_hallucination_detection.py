#!/usr/bin/env python3
"""
Demo script for hallucination detection without requiring audio files
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.hallucination_detector import HallucinationDetector
import numpy as np
import json

def demo_repetition_detection():
    """Demo repetition detection capabilities"""
    print("🔄 Testing Repetition Detection")
    print("=" * 40)
    
    detector = HallucinationDetector()
    
    test_cases = [
        {
            "name": "Simple Repetition",
            "text": "hello hello world",
            "expected": "Should detect 'hello' repetition"
        },
        {
            "name": "Pattern Repetition", 
            "text": "the quick brown the quick brown fox",
            "expected": "Should detect pattern repetition"
        },
        {
            "name": "Multiple Repetitions",
            "text": "this is this is a test test case",
            "expected": "Should detect multiple repetitions"
        },
        {
            "name": "Normal Speech",
            "text": "the quick brown fox jumps over the lazy dog",
            "expected": "Should detect no repetitions"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}: '{case['text']}'")
        repetitions = detector._detect_repetitions(case['text'])
        
        if repetitions:
            print(f"   ✅ Detected {len(repetitions)} repetition(s):")
            for rep in repetitions:
                print(f"      • {rep['type']}: '{rep['text']}' (severity: {rep['severity']})")
        else:
            print(f"   ✅ No repetitions detected")
        
        print(f"   📝 Expected: {case['expected']}")

def demo_language_switching():
    """Demo language switching detection"""
    print("\n🔄 Testing Language Switching Detection")
    print("=" * 40)
    
    detector = HallucinationDetector()
    
    test_cases = [
        "hello world señor garcia",
        "thank you merci beaucoup", 
        "good morning guten tag",
        "this is normal english text"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{case}'")
        switches = detector._detect_language_switches(case)
        
        if switches:
            print(f"   ⚠️  Detected {len(switches)} language switch(es):")
            for switch in switches:
                print(f"      • '{switch['text']}' ({switch['detected_language']})")
        else:
            print(f"   ✅ No language switches detected")

def demo_risk_calculation():
    """Demo overall risk calculation"""
    print("\n🔄 Testing Risk Calculation")
    print("=" * 40)
    
    detector = HallucinationDetector()
    
    # Simulate different scenarios
    scenarios = [
        {
            "name": "Low Risk Scenario",
            "attention_score": 0.9,
            "confidence_score": 0.8,
            "temporal_score": 0.85,
            "hallucinations": []
        },
        {
            "name": "Medium Risk Scenario", 
            "attention_score": 0.6,
            "confidence_score": 0.7,
            "temporal_score": 0.5,
            "hallucinations": [{"type": "repetition"}]
        },
        {
            "name": "High Risk Scenario",
            "attention_score": 0.3,
            "confidence_score": 0.4,
            "temporal_score": 0.2,
            "hallucinations": [
                {"type": "repetition"},
                {"type": "phantom_words"},
                {"type": "language_switch"}
            ]
        }
    ]
    
    for scenario in scenarios:
        risk = detector._calculate_overall_risk(
            scenario["attention_score"],
            scenario["confidence_score"], 
            scenario["temporal_score"],
            scenario["hallucinations"]
        )
        
        risk_level = "LOW" if risk < 0.3 else "MEDIUM" if risk < 0.6 else "HIGH"
        print(f"\n📊 {scenario['name']}: Risk = {risk:.3f} ({risk_level})")
        print(f"   Attention: {scenario['attention_score']:.2f}")
        print(f"   Confidence: {scenario['confidence_score']:.2f}")
        print(f"   Temporal: {scenario['temporal_score']:.2f}")
        print(f"   Hallucinations: {len(scenario['hallucinations'])}")

def demo_mitigation_suggestions():
    """Demo mitigation suggestion generation"""
    print("\n🔄 Testing Mitigation Suggestions")
    print("=" * 40)
    
    detector = HallucinationDetector()
    
    scenarios = [
        {
            "name": "High Risk with Repetitions",
            "risk_score": 0.8,
            "hallucinations": [{"type": "repetition"}, {"type": "pattern_repetition"}]
        },
        {
            "name": "Medium Risk with Phantom Words",
            "risk_score": 0.6,
            "hallucinations": [{"type": "phantom_words"}]
        },
        {
            "name": "Language Switch Issue",
            "risk_score": 0.5,
            "hallucinations": [{"type": "language_switch"}]
        }
    ]
    
    for scenario in scenarios:
        suggestions = detector._generate_mitigation_suggestions(
            scenario["risk_score"],
            scenario["hallucinations"]
        )
        
        print(f"\n💡 {scenario['name']} (Risk: {scenario['risk_score']:.1f}):")
        for suggestion in suggestions:
            print(f"   • {suggestion}")

def main():
    print("🧠 ASR Hallucination Detection System Demo")
    print("=" * 50)
    print("This demo showcases the detection capabilities without requiring audio files.")
    print()
    
    try:
        # Run all demos
        demo_repetition_detection()
        demo_language_switching()
        demo_risk_calculation()
        demo_mitigation_suggestions()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\n📋 System Capabilities Demonstrated:")
        print("   ✅ Repetition pattern detection")
        print("   ✅ Language switching identification") 
        print("   ✅ Risk assessment calculation")
        print("   ✅ Mitigation strategy suggestion")
        print("\n🚀 Next Steps:")
        print("   • Test with real audio: python scripts/analyze_audio.py <audio_file>")
        print("   • Explore notebooks: jupyter notebook notebooks/01_hallucination_analysis.ipynb")
        print("   • Run full system test: python scripts/test_hallucination_detection.py")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")

if __name__ == "__main__":
    main()