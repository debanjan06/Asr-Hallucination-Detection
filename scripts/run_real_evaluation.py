#!/usr/bin/env python3
"""
Run real audio dataset evaluation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.real_audio_evaluator import RealAudioEvaluator

def main():
    print("🚀 Real Audio Dataset Evaluation")
    print("=" * 50)
    print("This evaluation will:")
    print("  • Download real speech datasets (LibriSpeech, CommonVoice)")
    print("  • Test ASR performance on actual speech")
    print("  • Evaluate hallucination detection on real cases")
    print("  • Generate comprehensive analysis")
    print()
    
    # Ask user for confirmation
    response = input("Download and test with real datasets? (y/n): ").lower().strip()
    if response != 'y':
        print("Evaluation cancelled. Use python scripts/run_evaluation.py for synthetic testing.")
        return
    
    try:
        # Run comprehensive evaluation
        evaluator = RealAudioEvaluator()
        results, report = evaluator.run_comprehensive_evaluation(max_samples_per_dataset=5)
        
        print("\n🎉 Real audio evaluation completed!")
        print("\n📊 Summary of Results:")
        
        # Print summary
        for dataset_name, dataset_results in results.items():
            successful = len([r for r in dataset_results if r.get("success", False)])
            total = len(dataset_results)
            print(f"  • {dataset_name}: {successful}/{total} successful evaluations")
        
        print(f"\n📁 Check results/real_audio_evaluation/ for detailed analysis!")
        
    except Exception as e:
        print(f"❌ Real audio evaluation failed: {e}")
        print("\nFallback options:")
        print("1. Check internet connection for dataset downloads")
        print("2. Use: python scripts/run_evaluation.py (synthetic data)")
        print("3. Ensure datasets package is installed: pip install datasets")

if __name__ == "__main__":
    main()