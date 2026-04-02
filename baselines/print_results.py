#!/usr/bin/env python3
"""
Compute speedup against UAutomizer baseline results.

Speedup is defined as:
- If the results don't match (different result for same filename): speedup = 0
- If the results match: speedup = uautomizer_time_taken / input_time_taken
- Final metric is the geometric mean speedup over all problems in the input JSON
"""

import json
import argparse
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


def load_json_results(file_path: str) -> List[Dict]:
    """Load results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)


def generated_invariant_correctness_and_timeout(input_result: Dict[str, List[Dict]]) -> float:
    """
    Print the mean % of generated invariants that are correct averaged across all samples and problems
    """
    correct_count = 0
    timeout_count = 0
    false_count = 0
    total_count = 0
    for filename, input_result_list in input_result.items():
        for input_result in input_result_list:
            if input_result['assert_verification_result'] is not None and input_result['assert_verification_result'].get('result', '') == 'TRUE':
                correct_count += 1
                break
            elif input_result['assert_verification_result'] is not None and input_result['assert_verification_result'].get('result', '') == 'FALSE':
                false_count += 1
                break
            if (input_result['assert_verification_result'] is not None and input_result['assert_verification_result'].get('result', '') == 'TIMEOUT') or (input_result['assume_verification_result'] is not None and input_result['assume_verification_result'].get('result', '') == 'TIMEOUT'):
                timeout_count += 1
                break
        total_count += 1
    return correct_count / total_count, timeout_count / total_count, false_count / total_count

def compute_speedup(input_results: Dict[str, List[Dict]], baseline_results: List[Dict]) -> Tuple[float, float, float, float, List[float], List[float], List[Tuple[float, bool]], List[Tuple[float, bool]]]:
    """
    Compute speedup for each problem and return geometric mean speedup with stats.
    
    Returns:
        tuple: (geometric_mean_speedup, result_stats, pct_consistent, pct_consistent_and_faster, mean_speedup_consistent_and_faster, speedups, speedups_raw, timing_data, baseline_timing_data)
        where speedups uses max(..., 1.0) for metrics, speedups_raw uses baseline_time/input_time for table/plot,
        timing_data is a list of (time, is_solved) tuples for input results,
        and baseline_timing_data is a list of (time, is_solved) tuples for baseline results
    """
    fn_key = 'filename'
    time_key = 'time_taken'

    # Create lookup dict for baseline results by filename
    baseline_lookup = {result[fn_key]: result for result in baseline_results}
    
    speedups = []  # With max(..., 1.0) for metrics
    speedups_raw = []  # Without max for table/plot
    consistent_and_faster_speedups = []
    result_stats = {'TRUE': 0, 'FALSE': 0, 'UNKNOWN': 0, 'TIMEOUT': 0}
    consistent_count = 0
    consistent_and_faster_count = 0
    timing_data = []  # List of (time, is_solved) tuples for input results
    baseline_timing_data = []  # List of (time, is_solved) tuples for baseline results
    
    total_problems = len(input_results)
    
    for filename, input_result_list in input_results.items():
        for sample_result in input_result_list:
            generation_time = sample_result.get('generation_time', 0.0)
            assume_verification_time = sample_result.get('assume_verification_time', 0.0)
            assert_verification_time = sample_result.get('assert_verification_time', 0.0)
            assume_verification_result = sample_result.get('assume_verification_result')
            assert_verification_result = sample_result.get('assert_verification_result')
            
            if assume_verification_time == 0.0 and assert_verification_time == 0.0:
                sample_result['calculated_time_taken'] = 0.0
            else:
                # Scenario 2: If assume returns FALSE and (assert was KILLED or assume is faster/equal to assert),
                # use assume time only since aggregated result will be FALSE
                if (assume_verification_result is not None and 
                    assume_verification_result.get('result') == 'FALSE' and
                    (assert_verification_result is None or 
                     assert_verification_result.get('result') == 'KILLED' or
                     assume_verification_time <= assert_verification_time)):
                    # Use assume time only (early termination scenario)
                    sample_result['calculated_time_taken'] = assume_verification_time + generation_time
                else:
                    # Calculate total time for this sample: max(assume, assert) + generation
                    sample_result['calculated_time_taken'] = max(assume_verification_time, assert_verification_time) + generation_time
        
        # Find matching baseline result
        if filename not in baseline_lookup:
            raise ValueError(f"Filename {filename} not found in baseline results")
        
        baseline_result = baseline_lookup[filename]
        baseline_result_value = baseline_result['result']
        
        # Filter samples that are consistent with baseline
        consistent_samples = [sample for sample in input_result_list if sample['result'] == baseline_result_value]
        # Filter samples that are correct (assert verification TRUE)
        correct_samples = [
            sample for sample in input_result_list
            if sample.get('assert_verification_result') is not None
            and sample.get('assert_verification_result', {}).get('result') == 'TRUE'
        ]
        consistent_correct_samples = [
            sample for sample in correct_samples
            if sample.get('result') == baseline_result_value
        ]
        
        # Select the sample with minimum calculated time that is consistent and fastest
        if consistent_samples:
            input_result = min(consistent_samples, key=lambda x: x.get('calculated_time_taken', float('inf')))
        else:
            # if no consistent samples, select the fastest one
            input_result = min(input_result_list, key=lambda x: x.get('calculated_time_taken', float('inf')))

        input_result_value = input_result['result']
        input_time = input_result.get('calculated_time_taken', 0.0)
        
        result_stats[input_result_value] += 1
        
        # Track timing data: solved means TRUE or FALSE (not UNKNOWN or TIMEOUT)
        is_solved = input_result_value in ['TRUE', 'FALSE']
        timing_data.append((input_time, is_solved))
        
        # Baseline result and time (already retrieved above)
        baseline_time = baseline_result[time_key]
        
        # Track baseline timing data: solved means TRUE or FALSE (not UNKNOWN or TIMEOUT)
        baseline_is_solved = baseline_result_value in ['TRUE', 'FALSE']
        baseline_timing_data.append((baseline_time, baseline_is_solved))
        
        # Count consistent results
        if input_result_value == baseline_result_value:
            consistent_count += 1
        # For speedup, require correct + consistent sample
        if consistent_correct_samples:
            speedup_sample = min(consistent_correct_samples, key=lambda x: x.get('calculated_time_taken', float('inf')))
            speedup_time = speedup_sample.get('calculated_time_taken', 0.0)
            speedup_raw = baseline_time / speedup_time if speedup_time > 0 else 1.0  # Raw speedup for table/plot
            speedup = max(speedup_raw, 1.0)  # Capped speedup for metrics
            if speedup_raw > 1.0:
                consistent_and_faster_count += 1
                consistent_and_faster_speedups.append(speedup)
        else:
            speedup = 1.0
            speedup_raw = 1.0
        
        speedups.append(speedup)
        speedups_raw.append(speedup_raw)


    result_stats['SUCCESS'] = result_stats['TRUE'] + result_stats['FALSE'] 
    result_stats = {k : str(round(v / total_problems * 100, 2)) + '%' for k, v in result_stats.items()}
    
    # Compute geometric mean for speedups
    if speedups:
        # Geometric mean = (product of all values)^(1/n)
        # Use log space to avoid overflow: exp(mean(log(values)))
        log_speedups = [math.log(speedup) for speedup in speedups]
        mean_speedup = math.exp(sum(log_speedups) / len(log_speedups))
    else:
        mean_speedup = 1.0
    
    if consistent_and_faster_speedups:
        log_consistent_speedups = [math.log(speedup) for speedup in consistent_and_faster_speedups]
        mean_speedup_consistent_and_faster = math.exp(sum(log_consistent_speedups) / len(log_consistent_speedups))
    else:
        mean_speedup_consistent_and_faster = 1.0
    pct_consistent = (consistent_count / total_problems) * 100 if total_problems > 0 else 0.0
    pct_consistent_and_faster = (consistent_and_faster_count / total_problems) * 100 if total_problems > 0 else 0.0
    return mean_speedup, result_stats, pct_consistent, pct_consistent_and_faster, mean_speedup_consistent_and_faster, speedups, speedups_raw, timing_data, baseline_timing_data



def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Compute speedup against UAutomizer baseline')
    parser.add_argument('input', type=str, help='Input JSON file to compare')
    args = parser.parse_args()
    
    
    baseline = str(Path(__file__).parent.resolve()) + "/../Dataset/timing_uautomizer.json"
    
    input_results = load_json_results(args.input)
    baseline_results = load_json_results(baseline)
    
    pct_correct_invariants, pct_timeout_invariants, pct_false_invariants = generated_invariant_correctness_and_timeout(input_results) 
    pct_correct_invariants = pct_correct_invariants * 100
    pct_timeout_invariants = pct_timeout_invariants * 100
    pct_false_invariants = pct_false_invariants * 100
    print(f"\n\n% of generated invariants that are correct across all problems and samples: {pct_correct_invariants:.1f}%\n\n")
    print(f"\n\n% of generated invariants that are false across all problems and samples: {pct_false_invariants:.1f}%\n\n")
    print(f"\n\n% of generated invariants that timed out across all problems and samples: {pct_timeout_invariants:.1f}%\n\n")
    
    # Compute speedup
    mean_speedup, result_stats, pct_consistent, pct_consistent_and_faster, mean_speedup_consistent_and_faster, speedups, speedups_raw, timing_data, baseline_timing_data = compute_speedup(input_results, baseline_results)
    
    # Print final results
    print(f'##### Best-of-N results #####')
    print(f"Geometric mean speedup: {mean_speedup:.3f}x ({len(input_results)} problems)")
    print(f"Geometric mean speedup (consistent and faster only): {mean_speedup_consistent_and_faster:.3f}x")
    print(f"% of each verification result across all problems: {result_stats}")
    print(f"% problems consistent with baseline: {pct_consistent:.1f}%")
    print(f"% problems consistent and faster: {pct_consistent_and_faster:.1f}%")
    
    # Calculate counts for different speedup thresholds (using raw speedups for table/plot)
    # Note: >1.0 means strictly faster, ≥ for other thresholds
    thresholds = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    threshold_labels = ['>1.0', '≥1.2', '≥1.4', '≥1.6', '≥1.8', '≥2.0']
    counts = []
    for i, threshold in enumerate(thresholds):
        if i == 0:  # Strictly greater than 1.0
            count = sum(1 for s in speedups_raw if s > threshold)
        else:  # Greater than or equal for all others
            count = sum(1 for s in speedups_raw if s >= threshold)
        counts.append(count)
    
    # Create and print table
    table_data = [[label, count, f"{count/len(speedups)*100:.1f}%"] for label, count in zip(threshold_labels, counts)]
    print(f'\n##### Speedup Distribution Table #####')
    print(tabulate(table_data, headers=['Speedup Threshold', 'Number of Instances', 'Percentage'], tablefmt='grid'))

if __name__ == "__main__":
    main() 