#!/usr/bin/env python3
"""
Batch run LLM-based invariant generation and UAutomizer verification on C files
This script processes all C files in the evaluation directory using:
1. LLM to generate loop invariants 
2. UAutomizer to verify the modified programs
"""

import os
import subprocess
import time
import json
import re
import tempfile
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import concurrent.futures
import threading
import signal
import queue
import random

# Add the current directory to Python path to import inference.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import get_client
from sglang.utils import terminate_process, wait_for_server, launch_server_cmd


def get_uautomizer_path() -> str:
    """Get the path to uautomizer tool"""
    # Get path relative to this script
    script_dir = Path(__file__).parent.resolve()
    uautomizer_path = script_dir / "../tools/uautomizer/Ultimate.py"
    uautomizer_path = uautomizer_path.resolve()

    if not os.path.exists(uautomizer_path):
        raise FileNotFoundError(f"UAutomizer not found at: {uautomizer_path}")

    return str(uautomizer_path)


def get_spec_path() -> str:
    """Get the path to the specification file"""
    # Get path relative to this script
    script_dir = Path(__file__).parent.resolve()
    spec_path = script_dir / "../Dataset/properties/unreach-call.prp"
    spec_path = spec_path.resolve()
    return str(spec_path)


def get_esbmc_path() -> str:
    """Get the path to ESBMC wrapper tool"""
    # Get path relative to this script
    script_dir = Path(__file__).parent.resolve()
    esbmc_path = script_dir / "../tools/esbmc/esbmc-wrapper.py"
    esbmc_path = esbmc_path.resolve()

    if not os.path.exists(esbmc_path):
        raise FileNotFoundError(f"ESBMC wrapper not found at: {esbmc_path}")

    return str(esbmc_path)


class InvariantGenerationResult:
    def __init__(self, filename: str, success: bool, result: str = "", 
                 time_taken: float = 0.0, error: str = "", 
                 llm_response: str = "", invariants_count: int = 0,
                 overall_time_taken: float = 0.0, sample_id: int = -1,
                 assume_verification_result: Optional[Dict] = None,
                 assert_verification_result: Optional[Dict] = None,
                 generation_time: float = 0.0,
                 assume_verification_time: float = 0.0,
                 assert_verification_time: float = 0.0):
        self.filename = filename
        self.success = success
        self.result = result
        self.time_taken = time_taken  # Max of assume and assert verification time
        self.error = error
        self.llm_response = llm_response
        self.invariants_count = invariants_count
        self.overall_time_taken = overall_time_taken  # Deprecated: will be calculated in print_results.py
        self.sample_id = sample_id
        self.assume_verification_result = assume_verification_result  # Raw assume verification result
        self.assert_verification_result = assert_verification_result  # Raw assert verification result
        self.generation_time = generation_time  # LLM generation time for this sample
        self.assume_verification_time = assume_verification_time  # Time taken for assume verification
        self.assert_verification_time = assert_verification_time  # Time taken for assert verification

class BatchInvariantProcessor:
    def __init__(self, c_files_dir: str, output_file: str, timeout: int, 
                 max_workers: int = 8, client=None, prompts=None, 
                 enable_cot: bool = False,
                 max_new_tokens: int = 2048, temperature: float = 0.0,
                 best_of_n: int = 1,
                 server_process=None, verifier: str = 'uautomizer',
                 test_gt_invariants: bool = False,
                 reload_results_file: Optional[str] = None,
                 num_shots: int = 0
                 ):
        self.c_files_dir = c_files_dir
        self.output_file = output_file
        self.timeout = timeout
        self.max_workers = max_workers
        self.client = client
        self.prompts = prompts
        self.enable_cot = enable_cot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.best_of_n = best_of_n
        self.server_process = server_process
        self.verifier = verifier
        self.test_gt_invariants = test_gt_invariants
        self.reload_results_file = reload_results_file
        self.num_shots = num_shots
        
        # Thread-safe result storage - changed to dict of lists
        self.results = {}  # filename -> list of sample results
        self.results_lock = threading.Lock()
        self.generation_results = {}  # Store generation results by filename
        self.generation_results_lock = threading.Lock()
        self.verification_results = {}  # Store verification results by (filename, sample_id, verification_type)
        self.verification_results_lock = threading.Lock()
        
        # Load pre-computed original program verification results
        self.original_results = self._load_original_verification_results()
        
        # Load ground truth invariants if in GT mode
        if self.test_gt_invariants:
            self.gt_invariants = self._load_gt_invariants()
        
        # Load invariants from previous results file if specified
        if self.reload_results_file:
            self.reloaded_invariants = self._load_reloaded_invariants()
        else:
            self.reloaded_invariants = None
    
    def _load_original_verification_results(self) -> Dict[str, Dict]:
        """Load pre-computed original program verification results from JSON file"""
        # Load timing results from Dataset/timing_uautomizer.json
        script_dir = Path(__file__).parent.resolve()
        results_file = script_dir / "../Dataset/timing_uautomizer.json"
        results_file = results_file.resolve()
        
        with open(results_file, 'r') as f:
            results_list = json.load(f)
        
        # Convert list to dictionary keyed by filename
        results_dict = {}
        for result in results_list:
            filename = result['filename']
            # Compute success from result if not present
            success = result.get('success', result['result'] in ['TRUE', 'FALSE'])
            results_dict[filename] = {
                'success': success,
                'result': result['result'],
                'time_taken': result['time_taken'],
                'stderr': result.get('error', '')
            }
        
        print(f"✅ Loaded {len(results_dict)} pre-computed original verification results", flush=True)
        return results_dict

    def _load_gt_invariants(self) -> Dict[str, List[Dict]]:
        """Load ground truth invariants from timing_uautomizer.json"""
        # Load GT invariants from Dataset/timing_uautomizer.json
        script_dir = Path(__file__).parent.resolve()
        timing_file = script_dir / "../Dataset/timing_uautomizer.json"
        timing_file = timing_file.resolve()
            
        print(f"📄 Loading GT invariants from: {timing_file}", flush=True)
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)
        
        # Extract invariants from timing file
        # Format: [{"filename": "...", "result": "...", "time_taken": ..., "invariants": [{"line": ..., "invariant": "..."}]}, ...]
        gt_invariants = {}
        for entry in timing_data:
            filename = entry['filename']
            if 'invariants' in entry and entry['invariants']:
                # Convert format from timing file to expected format
                # Timing file: {"line": 32, "invariant": "0 < l"}
                # Expected format: {"line": 32, "invariant": "0 < l"} (same, but we'll use 'line' as 'line_after')
                gt_invariants[filename] = entry['invariants']
        
        print(f"✅ Loaded {len(gt_invariants)} ground truth invariants", flush=True)
        return gt_invariants
    
    def _load_reloaded_invariants(self) -> Dict[str, List[Dict]]:
        """Load invariants from a previous results JSON file"""
        print(f"📄 Loading invariants from previous results file: {self.reload_results_file}", flush=True)
        
        if not os.path.exists(self.reload_results_file):
            raise FileNotFoundError(f"Reload results file not found: {self.reload_results_file}")
        
        with open(self.reload_results_file, 'r') as f:
            results_data = json.load(f)
        
        # Extract invariants from model_response fields
        reloaded_invariants = {}
        
        for filename, sample_results in results_data.items():
            if not isinstance(sample_results, list):
                continue
            
            file_invariants = []
            for sample_result in sample_results:
                if 'model_response' not in sample_result:
                    continue
                
                model_response = sample_result['model_response']
                # Handle quoted strings (remove surrounding quotes if present)
                # JSON may store strings with quotes as part of the value
                if isinstance(model_response, str):
                    model_response = model_response.strip()
                    if model_response.startswith('"') and model_response.endswith('"'):
                        model_response = model_response[1:-1]
                sample_id = sample_result.get('sample_id', len(file_invariants))
                generation_time = sample_result.get('generation_time', 0.0)
                
                # Extract invariants from the model response using existing function
                extracted_invariants = extract_invariants_from_response(model_response)
                
                # Validate and add valid invariants
                for inv in extracted_invariants:
                    if check_valid_invariant_operation(inv['condition']):
                        file_invariants.append({
                            'sample_id': sample_id,
                            'llm_response': model_response,
                            'invariant': inv,
                            'generation_time': generation_time
                        })
                        break  # Only take first valid invariant per sample
            
            if file_invariants:
                reloaded_invariants[filename] = file_invariants
        
        print(f"✅ Loaded invariants for {len(reloaded_invariants)} files from previous results", flush=True)
        return reloaded_invariants
    
    def _generate_reloaded_invariants_for_file(self, c_file: str) -> Dict:
        """Generate invariants using reloaded data for a single file"""
        print(f"🔄 Loading invariants from previous results for: {c_file}", flush=True)
        
        c_file_path = os.path.join(self.c_files_dir, c_file)
        
        # Check if reloaded invariants exist for this file
        if c_file not in self.reloaded_invariants:
            print(f"⚠️  No reloaded invariants found for {c_file}, will verify original program only", flush=True)
            return {
                'filename': c_file,
                'success': True,
                'samples': [],
                'total_samples_generated': 0,
                'samples_selected': 0,
                'error': 'No reloaded invariants found',
                'generation_time': 0.0  # No generation time when reloading (already generated previously)
            }
        
        reloaded_inv_list = self.reloaded_invariants[c_file]
        
        # Find valid loop insertion points for this file
        c_code_with_line_numbers = read_c_file_with_line_numbers(c_file_path)
        valid_insertion_points = find_loop_invariant_insertion_points(c_code_with_line_numbers)
        
        if not valid_insertion_points:
            print(f"⚠️  No valid loop insertion points found for {c_file}, will verify original program only", flush=True)
            return {
                'filename': c_file,
                'success': True,
                'samples': [],
                'total_samples_generated': 0,
                'samples_selected': 0,
                'error': 'No valid loop insertion points found',
                'generation_time': 0.0  # No generation time when reloading (already generated previously)
            }
        
        # Filter invariants by valid insertion points
        valid_points_set = set(valid_insertion_points)
        valid_samples = []
        
        for sample_data in reloaded_inv_list:
            inv = sample_data['invariant']
            if inv['line_after'] in valid_points_set:
                valid_samples.append(sample_data)
            else:
                print(f"⚠️  Filtering out reloaded invariant at line {inv['line_after']} (not at valid loop insertion point)", flush=True)
        
        # Use all valid samples (no limit)
        selected_samples = valid_samples
        
        print(f"🔄 Loaded {len(selected_samples)} valid reloaded invariants for {c_file} (filtered from {len(reloaded_inv_list)} total)", flush=True)
        
        # Calculate total generation time from all reloaded samples (sum of individual generation times)
        total_generation_time = sum(sample.get('generation_time', 0.0) for sample in selected_samples)
        
        result = {
            'filename': c_file,
            'success': True,
            'samples': selected_samples,
            'total_samples_generated': len(valid_samples),
            'samples_selected': len(selected_samples),
            'error': '',
            'generation_time': total_generation_time  # Sum of generation times from reloaded samples
        }
        
        return result
        
    def generate_invariants_for_file(self, c_file: str) -> Dict:
        """Generate invariants for a single file - Phase 1"""
        if self.reload_results_file and self.reloaded_invariants is not None:
            return self._generate_reloaded_invariants_for_file(c_file)
        elif self.test_gt_invariants:
            return self._generate_gt_invariants_for_file(c_file)
        else:
            return self._generate_llm_invariants_for_file(c_file)
    
    def _generate_gt_invariants_for_file(self, c_file: str) -> Dict:
        """Generate invariants using ground truth data for a single file"""
        print(f"🎯 Loading GT invariants for: {c_file}", flush=True)
        
        c_file_path = os.path.join(self.c_files_dir, c_file)
        
        # Check if GT invariants exist for this file
        if c_file not in self.gt_invariants:
            print(f"⚠️  No GT invariants found for {c_file}, will verify original program only", flush=True)
            return {
                'filename': c_file,
                'success': True,
                'samples': [],
                'total_samples_generated': 0,
                'samples_selected': 0,
                'error': 'No GT invariants found',
                'generation_time': 0.0  # No generation time for GT mode
            }
        
        gt_inv_list = self.gt_invariants[c_file]
        
        # Find valid loop insertion points for this file
        c_code_with_line_numbers = read_c_file_with_line_numbers(c_file_path)
        valid_insertion_points = find_loop_invariant_insertion_points(c_code_with_line_numbers)
        
        if not valid_insertion_points:
            print(f"⚠️  No valid loop insertion points found for {c_file}, will verify original program only", flush=True)
            return {
                'filename': c_file,
                'success': True,
                'samples': [],
                'total_samples_generated': 0,
                'samples_selected': 0,
                'error': 'No valid loop insertion points found'
            }
        
        # Convert GT invariants to the expected format and filter by valid insertion points
        # GT format from timing file: {"line": 33, "invariant": "0 <= count && 0 <= i"}
        all_samples_invariants = []
        valid_points_set = set(valid_insertion_points)
        
        for i, gt_inv in enumerate(gt_inv_list):
            invariant = {
                'line_after': gt_inv['line'],  
                'condition': gt_inv['invariant']
            }
            
            # Only include GT invariants that are at valid loop insertion points
            if gt_inv['line'] in valid_points_set and check_valid_invariant_operation(gt_inv['invariant']):
                all_samples_invariants.append({
                    'sample_id': i,
                    'llm_response': f"GT invariant: {gt_inv['invariant']}",
                    'invariant': invariant,
                    'generation_time': 0.0  # No generation time for GT
                })
            else:
                print(f"⚠️  Filtering out GT invariant at line {gt_inv['line']} (not at valid loop insertion point)", flush=True)
        
        # For GT mode, use all valid invariants
        selected_samples = all_samples_invariants
        
        print(f"🎯 Loaded {len(selected_samples)} valid GT invariants for {c_file} (filtered from {len(gt_inv_list)} total)", flush=True)
        
        result = {
            'filename': c_file,
            'success': True,
            'samples': selected_samples,
            'total_samples_generated': len(all_samples_invariants),
            'samples_selected': len(selected_samples),
            'error': '',
            'generation_time': 0.0  # No generation time for GT mode
        }
        
        return result
    
    def _generate_llm_invariants_for_file(self, c_file: str) -> Dict:
        """Generate invariants using LLM for a single file - original logic"""
        print(f"🤖 Generating invariants for: {c_file}", flush=True)
        
        c_file_path = os.path.join(self.c_files_dir, c_file)
        
        
        # Step 1: Read file with line numbers
        c_code_with_line_numbers = read_c_file_with_line_numbers(c_file_path)
        
        # Step 2: Create prompt and get LLM response
        messages = create_messages(c_code_with_line_numbers, self.prompts, self.enable_cot, self.num_shots)
        start_time = time.time()
        # Generate completion using the shared client
        responses = self.client.generate_completion(
            prompt = None,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            n=self.best_of_n, enable_thinking=self.enable_cot
        )
        end_time = time.time()
        total_generation_time = end_time - start_time
        # Calculate per-sample time (for individual sample tracking)
        # Safety check: ensure best_of_n is at least 1 to avoid division by zero
        generation_time_per_sample = total_generation_time / max(self.best_of_n, 1)
        
        # Step 3: Extract invariants from all samples and deduplicate
        all_samples_invariants = []
        seen_invariants = set()  # For deduplication based on string match
        
        for i, llm_response in enumerate(responses):
            sample_invariants = extract_invariants_from_response(llm_response)
            
            # Validate against insertion points
            insertion_points = find_loop_invariant_insertion_points(c_code_with_line_numbers)
            valid_invariant = validate_invariant_insertions(sample_invariants, insertion_points)
            
            # Deduplicate based on string representation
            if valid_invariant is not None:
                inv_str = f"{valid_invariant['line_after']}:{valid_invariant['condition']}"
                if inv_str not in seen_invariants:
                    seen_invariants.add(inv_str)
                    all_samples_invariants.append({
                        'sample_id': i,
                        'llm_response': llm_response,
                        'invariant': valid_invariant,
                        'generation_time': generation_time_per_sample
                    })
        
        # Use all generated samples (no limit)
        selected_samples = all_samples_invariants
        
        print(f"🎯 Generated {len(all_samples_invariants)} unique samples, selected {len(selected_samples)} for verification", flush=True)
        
        # If no valid samples, still return success but with empty samples
        result = {
            'filename': c_file,
            'success': True,
            'samples': selected_samples,
            'total_samples_generated': len(all_samples_invariants),
            'samples_selected': len(selected_samples),
            'error': '',
            'generation_time': total_generation_time  # Total generation time for all samples (wall-clock time)
        }
        
        total_invariants = len(selected_samples)  # Each sample now has exactly one invariant
        print(f"✅ Generated invariants for {c_file}: {total_invariants} total valid invariants across {len(selected_samples)} samples ({generation_time_per_sample:.2f}s per sample)", flush=True)
        return result
            


    def run_smart_verification(self, c_file: str, generation_result: Dict, original_result: Dict, verification_tasks: List[Tuple]) -> None:
        """
        Run verification with parallel assume and assert queries.
        Both queries run with timeout of 1.2x original_baseline_time.
        """
        if len(verification_tasks) == 0:
            return
            
        original_time = original_result['time_taken']
        # Use 1.2x original time as timeout for both assume and assert queries
        verification_timeout = 1.2 * original_time
        # Cap at absolute timeout limit
        verification_timeout = min(verification_timeout, self.timeout)
        
        print(f"🔄 Parallel verification for {c_file}: timeout={verification_timeout:.2f}s (1.2x original={original_time:.2f}s)", flush=True)
            
        # Shared state for process management (thread-safe)
        process_info_lock = threading.Lock()
        process_info = {}  # verification_type -> {'process': process, 'start_time': time, 'completed': bool}
        results_queue = queue.Queue()
        
        def run_verification_with_monitoring(sample_id: int, verification_type: str, file_path: str, temp_files: List[str]):
            """Run verification with process monitoring and timeout"""
            start_time = time.time()
            
            # Use temporary files for stdout/stderr to prevent deadlock from pipe buffer overflow
            # We use w+b (binary) to avoid encoding issues during subprocess write, and decode later
            stdout_f = tempfile.TemporaryFile(mode='w+b')
            stderr_f = tempfile.TemporaryFile(mode='w+b')
            
            try:
                # Store process info for potential termination
                if self.verifier == 'uautomizer':
                    verifier_path = get_uautomizer_path()
                    command = [
                        "python3", "-u", "Ultimate.py",
                        "--spec", get_spec_path(),
                        "--file", file_path,
                        "--architecture", "64bit",
                        "--full-output"
                    ]
                else:  # esbmc
                    verifier_path = get_esbmc_path()
                    command = [
                        "python3", "-u", os.path.basename(verifier_path),
                        "-p", get_spec_path(),
                        "-s", "kinduction",
                        "--arch", "64",
                        file_path
                    ]
                
                verifier_dir = os.path.dirname(verifier_path)
                
                # Don't use os.chdir in parallel execution - use cwd parameter instead
                process = subprocess.Popen(
                    command,
                    stdout=stdout_f,
                    stderr=stderr_f,
                    cwd=verifier_dir,
                    preexec_fn=os.setsid
                )
                
                # Store process info thread-safely
                with process_info_lock:
                    process_info[(sample_id, verification_type)] = {
                        'process': process,
                        'start_time': start_time,
                        'completed': False
                    }
                
                # Monitor for timeout
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    
                    # Check if timeout exceeded
                    if elapsed >= verification_timeout:
                        end_time = time.time()
                        # Kill this process
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            time.sleep(1)
                            if process.poll() is None:
                                os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        except:
                            pass
                        
                        # Cleanup temp files
                        for temp_file in temp_files:
                            try:
                                os.unlink(temp_file)
                            except:
                                pass
                        
                        kill_reason = f'Process killed due to timeout ({elapsed:.2f}s >= {verification_timeout:.2f}s)'
                        results_queue.put({
                            'sample_id': sample_id,
                            'verification_type': verification_type,
                            'success': False,
                            'result': 'TIMEOUT',
                            'time_taken': elapsed,
                            'stderr': kill_reason
                        })
                        return
                        
                    time.sleep(0.001)  # Small sleep to avoid busy waiting
                
                # Process completed normally
                # Read output from temp files
                stdout_f.seek(0)
                stderr_f.seek(0)
                stdout = stdout_f.read().decode('utf-8', errors='replace')
                stderr = stderr_f.read().decode('utf-8', errors='replace')
                
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Mark as completed thread-safely
                with process_info_lock:
                    if (sample_id, verification_type) in process_info:
                        process_info[(sample_id, verification_type)]['completed'] = True
                
                # Parse result
                verification_result = "UNKNOWN"
                if process.returncode == 0:
                    if self.verifier == 'uautomizer':
                        # UAutomizer parsing: get the last non-empty line
                        stdout_lines = stdout.split('\n')
                        for line in reversed(stdout_lines):
                            if line.strip():
                                verification_result = line.strip()
                                break
                    else:  # esbmc
                        # ESBMC parsing: look for TRUE/FALSE keywords
                        if "TRUE" in stdout:
                            verification_result = "TRUE"
                        elif "FALSE" in stdout:
                            verification_result = "FALSE"
                
                result = {
                    'sample_id': sample_id,
                    'verification_type': verification_type,
                    'success': process.returncode == 0,
                    'result': verification_result,
                    'time_taken': elapsed_time,
                    'stderr': stderr
                }
                
                results_queue.put(result)
                
                # Cleanup temp files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
            
            finally:
                # Ensure log files are closed and deleted
                stdout_f.close()
                stderr_f.close()
                        

        
        # Start verification threads
        threads = []
        for c_file_task, sample_id, verification_type, file_path, temp_files in verification_tasks:
            thread = threading.Thread(
                target=run_verification_with_monitoring,
                args=(sample_id, verification_type, file_path, temp_files)
            )
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        collected_results = {}
        while not results_queue.empty():
            result = results_queue.get()
            sample_id = result['sample_id']
            verification_type = result['verification_type']
            
            # Store result thread-safely
            with self.verification_results_lock:
                self.verification_results[(c_file, sample_id, verification_type)] = result
            
            collected_results[(sample_id, verification_type)] = result
            
            if result['result'] != 'KILLED':
                print(f"✅ Sample {sample_id} {verification_type} verification completed for {c_file}: {result['result']} ({result['time_taken']:.2f}s)", flush=True)
            else:
                print(f"🚫 Sample {sample_id} {verification_type} verification killed for {c_file} (optimization)", flush=True)


    def aggregate_file_results(self, c_file: str) -> List[InvariantGenerationResult]:
        """Aggregate verification results across all samples and return all sample results"""
        generation_result = self.generation_results[c_file]
        if not generation_result['success'] or generation_result['samples_selected'] == 0:
            # Use original file result only
            original_result = self.verification_results[(c_file, -1, 'original')]
            # Get generation time from result, default to 0.0 if not present
            generation_time = generation_result.get('generation_time', 0.0)
            return [InvariantGenerationResult(
                filename=c_file,
                success=original_result['success'],
                result=original_result['result'],
                time_taken=original_result['time_taken'],
                error=generation_result.get('error', '') + "; " + original_result.get('stderr', ''),
                llm_response="No valid samples generated",
                invariants_count=0,
                overall_time_taken=original_result['time_taken'],
                sample_id=-1,
                assume_verification_result=None,
                assert_verification_result=None,
                generation_time=generation_time,
                assume_verification_time=0.0,
                assert_verification_time=0.0
            )]
        
        # Get original result (applies to all samples)
        original_result = self.verification_results[(c_file, -1, 'original')]
        
        # Process each sample and collect all results
        all_sample_results = []
        
        for sample in generation_result['samples']:
            sample_id = sample['sample_id']
            generation_time = sample['generation_time']
            llm_response = sample['llm_response']
            invariants_count = 1  # Each sample now has exactly one invariant
            
            # Check if enhanced verification was attempted for this sample
            assume_key = (c_file, sample_id, 'assume')
            assert_key = (c_file, sample_id, 'assert')
            
            if assume_key not in self.verification_results or assert_key not in self.verification_results:
                # Verification not attempted for this sample, use original result
                all_sample_results.append(InvariantGenerationResult(
                    filename=c_file,
                    success=original_result['success'],
                    result=original_result['result'],
                    time_taken=original_result['time_taken'],
                    error=original_result.get('stderr', ''),
                    llm_response=llm_response,
                    invariants_count=invariants_count,
                    overall_time_taken=original_result['time_taken'],
                    sample_id=sample_id,
                    assume_verification_result=None,
                    assert_verification_result=None,
                    generation_time=generation_time,
                    assume_verification_time=0.0,
                    assert_verification_time=0.0
                ))
                continue
            
            assume_result = self.verification_results[assume_key]
            assert_result = self.verification_results[assert_key]
            
            # Extract individual verification times
            assume_verification_time = assume_result['time_taken']
            assert_verification_time = assert_result['time_taken']
            max_verification_time = max(assume_verification_time, assert_verification_time)
            
            # Aggregate assume and assert results for this sample
            aggregated_result = aggregate_verification_results(assume_result, assert_result)
            
            # Determine final result and success
            if aggregated_result in ["UNKNOWN", "TIMEOUT", "KILLED"]:
                # Use original result if aggregated result is not definitive
                final_result = original_result['result']
                final_success = original_result['success']
                final_time_taken = original_result['time_taken']
            else:
                # Enhanced result is TRUE or FALSE, use it
                final_result = aggregated_result
                final_success = aggregated_result in ["TRUE", "FALSE"]
                final_time_taken = max_verification_time
            
            all_sample_results.append(InvariantGenerationResult(
                filename=c_file,
                success=final_success,
                result=final_result,
                time_taken=final_time_taken,
                error=assume_result.get('stderr', '') + "; " + assert_result.get('stderr', '') if final_result == aggregated_result else original_result.get('stderr', ''),
                llm_response=llm_response,
                invariants_count=invariants_count,
                overall_time_taken=0.0,  # Will be calculated in print_results.py
                sample_id=sample_id,
                assume_verification_result=assume_result,
                assert_verification_result=assert_result,
                generation_time=generation_time,
                assume_verification_time=assume_verification_time,
                assert_verification_time=assert_verification_time
            ))
        
        # Return all sample results instead of selecting the best one
        if not all_sample_results:
            # Fallback to original result
            return [InvariantGenerationResult(
                filename=c_file,
                success=original_result['success'],
                result=original_result['result'],
                time_taken=original_result['time_taken'],
                error=original_result.get('stderr', ''),
                llm_response="No sample results available",
                invariants_count=0,
                overall_time_taken=original_result['time_taken'],
                sample_id=-1,
                assume_verification_result=None,
                assert_verification_result=None,
                generation_time=0.0,
                assume_verification_time=0.0,
                assert_verification_time=0.0
            )]
        
        # Find the best sample for logging purposes (but return all)
        # Use a temporary calculation for logging
        best_sample_idx = min(range(len(all_sample_results)), 
                             key=lambda i: all_sample_results[i].generation_time + max(
                                 all_sample_results[i].assume_verification_time,
                                 all_sample_results[i].assert_verification_time
                             ))
        best_result = all_sample_results[best_sample_idx]
        best_total_time = best_result.generation_time + max(
            best_result.assume_verification_time,
            best_result.assert_verification_time
        )
        
        print(f"✅ Selected best sample {best_sample_idx} for {c_file}: {best_result.result} ({best_total_time:.2f}s)", flush=True)
        
        return all_sample_results


    def run_two_phase_processing(self, c_files: List[str]):
        """Run two-phase processing: first generate all invariants in parallel, then verify files sequentially (with samples in parallel within each file)"""
        
        
        # Phase 1: Generate invariants for all files in parallel
        print(f"🚀 Phase 1: Generating invariants for {len(c_files)} files in parallel...", flush=True)
        
        def generate_wrapper(c_file: str):
            result = self.generate_invariants_for_file(c_file)
            with self.generation_results_lock:
                self.generation_results[c_file] = result
            return result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            generation_futures = [executor.submit(generate_wrapper, c_file) for c_file in c_files]
            
            # Wait for all generation tasks to complete
            completed = 0
            for future in concurrent.futures.as_completed(generation_futures):
                completed += 1
                future.result()
                
                # Show progress every 5 files
                if completed % 5 == 0:
                    print(f"Generation Progress: {completed}/{len(c_files)} files completed", flush=True)
                elif completed == len(c_files):  # Final completion
                    print(f"Generation Progress: {completed}/{len(c_files)} files completed (FINAL)", flush=True)
                    

        
        print(f"✅ Phase 1 completed: All invariants generated", flush=True)
        
        # Cleanup SGLang server after generation phase since it's no longer needed
        if self.server_process:
            print("🧹 Cleaning up LLM server after generation phase...", flush=True)
            terminate_process(self.server_process)
            self.server_process = None
        
        # Phase 2: Process all files sequentially for verification (but samples within each file in parallel)
        print(f"🚀 Phase 2: Processing verification for {len(c_files)} files sequentially...", flush=True)
        
        def process_file_verification(c_file: str, file_idx: int):
            """Process verification for a single file"""
            print(f"📁 Processing verification for: {c_file} ({file_idx+1}/{len(c_files)})", flush=True)
            
            # Get the already generated result
            generation_result = self.generation_results[c_file]
            
            # Step 1: Create verification tasks for this file
            c_file_path = os.path.join(self.c_files_dir, c_file)
            verification_tasks = []
            
            # Step 1.1: Load pre-computed original verification result
            original_result = self.original_results[c_file].copy()
            original_result['verification_type'] = 'original'
            original_result['c_file'] = c_file
            
            # Store the original result (applies to all samples)
            with self.verification_results_lock:
                self.verification_results[(c_file, -1, 'original')] = original_result
            
            print(f"✅ Loaded original result for {c_file}: {original_result['result']} ({original_result['time_taken']:.2f}s)", flush=True)

            if not generation_result['success'] or generation_result['samples_selected'] == 0:
                # Only use original file result (already loaded above)
                print(f"📊 Using original verification result only for {c_file}", flush=True)
                verification_tasks = []  # No additional verification needed
            else:
                # Create verification tasks for assume and assert versions for each sample
                samples = generation_result['samples']
                total_tasks = len(samples) * 2  # assume + assert per sample
                print(f"✏️  Creating {total_tasks} verification tasks for {c_file} across {len(samples)} samples", flush=True)
                
                for sample in samples:
                    sample_id = sample['sample_id']
                    invariant = sample['invariant']
                    
                    # Create temporary files for assume and assert versions for this sample
                    temp_assume_path = insert_invariant_into_program(c_file_path, invariant)
                    temp_assert_path = insert_invariant_as_assertion_and_remove_final_assert(c_file_path, invariant)
                    
                    # Add verification tasks for this sample (include sample_id)
                    verification_tasks.append((c_file, sample_id, 'assume', temp_assume_path, [temp_assume_path]))
                    verification_tasks.append((c_file, sample_id, 'assert', temp_assert_path, [temp_assert_path]))
            
            # Step 2: Run verification tasks in parallel for this file (if any)
            if len(verification_tasks) > 0:
                print(f"🚀 Running {len(verification_tasks)} verification tasks for {c_file} (samples in parallel)...", flush=True)
                self.run_smart_verification(c_file, generation_result, original_result, verification_tasks)
            else:
                print(f"🚀 No additional verification needed for {c_file} (using original result only)", flush=True)
            
            # Step 3: Aggregate results for this file
            results = self.aggregate_file_results(c_file)
            with self.results_lock:
                self.results[c_file] = results
            
            # Find best result for logging (minimum overall time)
            def calc_total_time(r):
                return max(r.assume_verification_time, r.assert_verification_time) + r.generation_time
            best_result = min(results, key=lambda r: calc_total_time(r))
            best_total_time = calc_total_time(best_result)
            print(f"✅ Completed processing {c_file}: {best_result.result} ({best_total_time:.2f}s total)", flush=True)
            
            return c_file
        
        # Process all files sequentially
        for file_idx, c_file in enumerate(c_files):
            process_file_verification(c_file, file_idx)
            
            # Show progress every 5 files
            completed = file_idx + 1
            if completed % 5 == 0:
                print(f"Verification Progress: {completed}/{len(c_files)} files completed", flush=True)
            elif completed == len(c_files):  # Final completion
                print(f"Verification Progress: {completed}/{len(c_files)} files completed (FINAL)", flush=True)

        
        print(f"\n✅ Two-phase processing completed for all {len(c_files)} files", flush=True)

def filter_gt_files_with_valid_invariants(directory: str, gt_invariants: Dict[str, List[Dict]]) -> List[str]:
    """Filter GT files to only include those with valid invariants at loop insertion points"""
    valid_files = []
    
    for c_file in os.listdir(directory):
        if not c_file.endswith('.c'):
            continue
            
        if c_file not in gt_invariants:
            continue
            
        c_file_path = os.path.join(directory, c_file)
        
        try:
            # Find valid loop insertion points for this file
            c_code_with_line_numbers = read_c_file_with_line_numbers(c_file_path)
            valid_insertion_points = find_loop_invariant_insertion_points(c_code_with_line_numbers)
            
            if not valid_insertion_points:
                print(f"⚠️  No valid loop insertion points found for {c_file}, skipping", flush=True)
                continue
            
            # Check if any GT invariants are at valid insertion points
            valid_points_set = set(valid_insertion_points)
            has_valid_invariant = False
            
            for gt_inv in gt_invariants[c_file]:
                if (gt_inv['line'] in valid_points_set and 
                    check_valid_invariant_operation(gt_inv['invariant'])):
                    has_valid_invariant = True
                    break
            
            if has_valid_invariant:
                valid_files.append(c_file)
            else:
                print(f"⚠️  No valid GT invariants at loop insertion points for {c_file}, skipping", flush=True)
                
        except Exception as e:
            print(f"⚠️  Error processing {c_file}: {e}, skipping", flush=True)
            continue
    
    return sorted(valid_files)

def find_c_files(directory: str, num_problems: int = -1, test_gt_invariants: bool = False, 
                  available_gt_files: Optional[List[str]] = None) -> List[str]:
    """Find all .c files in the directory, with optional filtering for GT mode"""
    c_files = []
    for file in os.listdir(directory):
        if file.endswith('.c'):
            c_files.append(file)
    
    # Sort for consistent ordering
    c_files = sorted(c_files)
    
    # In GT mode, filter to only files that have ground truth invariants
    if test_gt_invariants and available_gt_files is not None:
        c_files = [f for f in c_files if f in available_gt_files]
        print(f"📊 Filtered to {len(c_files)} files with GT invariants", flush=True)
    
    # Random selection with seed 42 if requested and in GT mode
    if test_gt_invariants and num_problems > 0 and num_problems < len(c_files):
        random.seed(42)
        c_files = random.sample(c_files, num_problems)
        c_files = sorted(c_files)  # Sort again for consistent ordering after sampling
        print(f"🎲 Randomly selected {len(c_files)} files with seed 42", flush=True)
    
    return c_files

def read_c_file_with_line_numbers(file_path):
    """Read a C file and return it with line numbers in the format requested"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    formatted_lines = []
    for i, line in enumerate(lines, 1):
        # Strip newline and add line number comment
        line_content = line.rstrip('\n')
        formatted_lines.append(f"{line_content} // line {i}")
    
    return '\n'.join(formatted_lines)

def find_loop_invariant_insertion_points(c_code_with_line_numbers: str) -> List[int]:
    """
    Find where loop invariants should be inserted for each loop in the program.
    
    Logic:
    - Always insert at the beginning of each loop (one insertion point per loop)
    
    Returns:
        List of line numbers where invariants should be inserted after that line.
    """
    lines = c_code_with_line_numbers.split('\n')
    insertion_points = []
    
    # Find all loops and their line numbers
    loops = []  # (start_line, end_line, first_body_line)
    
    # Track bracket depth to find loop boundaries
    bracket_depth = 0
    current_loop_stack = []  # Stack of (loop_start_line, first_body_line, entry_bracket_depth)
    
    for i, line in enumerate(lines, 1):
        line_content = line.split(' // line')[0].strip()  # Remove line number comment
        
        # Check for loop keywords (more precise matching)
        stripped_line = line_content.strip()
        
        # Check if this is actually a loop statement (not just containing the keywords)
        is_loop = False
        if (stripped_line.startswith('while') and '(' in stripped_line) or \
           (stripped_line.startswith('for') and '(' in stripped_line) or \
           (stripped_line.startswith('do') and ('{' in stripped_line or len(stripped_line.strip()) <= 3)):
            is_loop = True
        
        if is_loop:
            # Found a loop start
            loop_start_line = i
            
            # For 'do' loops, the body starts immediately
            if stripped_line.startswith('do'):
                first_body_line = i + 1
            else:
                # For while/for loops, find the opening brace
                if '{' in line_content:
                    first_body_line = i + 1
                else:
                    # Opening brace might be on next line
                    first_body_line = i + 1
                    # Look ahead to find the opening brace
                    for j in range(i, min(i + 3, len(lines))):
                        if j < len(lines) and '{' in lines[j]:
                            first_body_line = j + 2  # Line after the brace
                            break
            
            # Record the bracket depth when we enter this loop
            entry_bracket_depth = bracket_depth
            current_loop_stack.append((loop_start_line, first_body_line, entry_bracket_depth))
        
        # Track bracket depth
        if '{' in line_content:
            bracket_depth += line_content.count('{')
        if '}' in line_content:
            closing_count = line_content.count('}')
            bracket_depth -= closing_count
            
            # Check if we closed any loops by returning to their entry bracket depth
            loops_to_close = []
            for idx, (loop_start, first_body, entry_depth) in enumerate(current_loop_stack):
                if bracket_depth <= entry_depth:
                    loops_to_close.append(idx)
                    loops.append((loop_start, i, first_body))
            
            # Remove closed loops from stack (in reverse order to maintain indices)
            for idx in reversed(loops_to_close):
                current_loop_stack.pop(idx)
    
    # For each loop, always insert at the beginning
    loops_by_start = sorted(loops, key=lambda x: x[0])
    
    for loop_start, loop_end, first_body_line in loops_by_start:
        # Always insert after the line before the loop body
        if first_body_line > 1:
            insertion_points.append(first_body_line - 1)
    
    # Remove duplicates and sort
    insertion_points = list(set(insertion_points))
    insertion_points.sort()
    
    return insertion_points

def check_valid_invariant_operation(condition: str) -> bool:
    """
    Check if the invariant condition is valid (no ++, --, +=, -=, =)
    Valid operators: ==, <=, >=, !=, <, >, &&, ||, !, +, -, *, /, %, etc.
    Invalid operators: =, +=, -=, *=, /=, %=, ++, --
    """
    # Check for obvious invalid operations
    invalid_ops = ['++', '--', '+=', '-=', '*=', '/=', '%=']
    for op in invalid_ops:
        if op in condition:
            return False
    
    # Check for standalone assignment operator '='
    # We need to distinguish '=' from '==', '<=', '>=', '!='
    i = 0
    while i < len(condition):
        if condition[i] == '=':
            # Check character before '='
            char_before = condition[i-1] if i > 0 else None
            # Check character after '='
            char_after = condition[i+1] if i < len(condition) - 1 else None
            
            # Valid cases: ==, <=, >=, !=
            if char_before in ['=', '<', '>', '!'] or char_after == '=':
                i += 1
                continue
            else:
                # This is a standalone '=' which is assignment
                return False
        i += 1
    
    return True

def validate_invariant_insertions(invariants: List[Dict], valid_insertion_points: List[int]) -> Optional[Dict]:
    """
    Validate invariants and return the first valid one, or None if none are valid.
    
    Args:
        invariants: List of invariant dictionaries with 'line_after' key
        valid_insertion_points: List of line numbers where invariants can be inserted after
        
    Returns:
        The first valid invariant dict, or None if no valid invariants
    """
    # If no invariants generated, return None (will verify original program only)
    if len(invariants) == 0:
        print(f"📊 No invariants generated, will verify original program only", flush=True)
        return None
    
    # If more than one invariant, take the first valid one
    if len(invariants) > 1:
        print(f"⚠️  Generated {len(invariants)} invariants, taking the first valid one...", flush=True)
    
    # Check each invariant until we find a valid one
    valid_points_set = set(valid_insertion_points)
    
    for i, inv in enumerate(invariants):
        inv_point = inv['line_after']
        
        if inv_point in valid_points_set and check_valid_invariant_operation(inv['condition']):
            if len(invariants) > 1:
                print(f"✅ Using invariant {i+1}/{len(invariants)} at insertion point after line: {inv_point}", flush=True)
            else:
                print(f"✅ Valid invariant at insertion point after line: {inv_point}", flush=True)
            return inv
        else:
            print(f"⚠️  Invariant {i+1}/{len(invariants)} has invalid insertion point after line: {inv_point}, trying next...", flush=True)
    
    # No valid invariants found
    print(f"⚠️  No valid invariants found (expected insertion points: {valid_insertion_points}), will verify original program only", flush=True)
    return None

def create_messages(c_code_with_line_numbers, prompts, enable_cot=False, num_shots=0):
    """Create messages for LLM using prompts dictionary"""
    # Select prompt type based on enable_cot
    if enable_cot:
        system_prompt = prompts["cot_system_prompt"]
        user_prompt = prompts["cot_user_prompt"]
    else:
        system_prompt = prompts["std_system_prompt"]
        user_prompt = prompts["std_user_prompt"]
    
    # Find insertion points for loop invariants
    insertion_points = find_loop_invariant_insertion_points(c_code_with_line_numbers)
    
    # Format insertion points for the prompt
    if insertion_points:
        points_text = '\n'.join([f'After line {x}' for x in insertion_points])
    else:
        raise ValueError("No loops found in the program")
    
    messages = []
    
    few_shot_examples = prompts["few_shot_examples"][:num_shots] if num_shots > 0 else []
    for few_shot_example in few_shot_examples:
        messages.append({
            "role": "user",
            "content": system_prompt + '\n\n' + user_prompt.replace("{PROGRAM}", few_shot_example["program"]).replace("{POINTS}", few_shot_example["point"])
        })
        messages.append({
            "role": "assistant",
            "content": few_shot_example["response"]
        })
    
    messages.append({
        "role": "user",
        "content": system_prompt + '\n\n' + user_prompt.replace("{PROGRAM}", c_code_with_line_numbers).replace("{POINTS}", points_text)
    })

    return messages

def extract_invariants_from_response(response):
    """
    Extract invariant insertions from LLM response.
    Handles both simple and complex conditions with nested parentheses.
    """
    if '/*@' in response and '@*/' in response:
        response = response.split('/*@')[1].split('@*/')[0]
    
    invariants = []
    pattern_start = r"After line (\d+), insert assume\("
    
    for match in re.finditer(pattern_start, response, re.IGNORECASE):
        line_after = int(match.group(1))
        start_pos = match.end()  # Position after "assume("
        
        # Find the matching closing parenthesis for the assume statement
        condition, end_pos = extract_balanced_condition(response, start_pos)
        
        if condition is not None:
            # Check if there's a semicolon immediately after
            remaining = response[end_pos:end_pos+10].strip()
            if remaining.startswith(';') or end_pos >= len(response):
                invariants.append({
                    'line_after': line_after,
                    'condition': condition
                })
    
    
    return invariants

def extract_balanced_condition(text, start_pos):
    """
    Extract a balanced condition starting from start_pos in text.
    Returns (condition, end_pos) where end_pos is the position after the last ')'
    """
    if start_pos >= len(text):
        return None, start_pos
    
    paren_count = 0
    condition_chars = []
    pos = start_pos
    
    while pos < len(text):
        char = text[pos]
        
        if char == '(':
            paren_count += 1
            condition_chars.append(char)
        elif char == ')':
            if paren_count == 0:
                # This is the closing parenthesis for the assume statement
                break
            else:
                paren_count -= 1
                condition_chars.append(char)
        else:
            condition_chars.append(char)
        
        pos += 1
    
    if pos < len(text) and text[pos] == ')':
        # Found the matching closing parenthesis
        condition = ''.join(condition_chars).strip()
        return condition, pos + 1
    else:
        # No matching closing parenthesis found
        return None, pos

def insert_invariant_into_program(original_file_path, invariant):
    """Insert assume statement into the program and save to temporary file"""
    with open(original_file_path, 'r') as f:
        lines = f.readlines()
    
    # Insert invariant after specified line
    line_after = invariant['line_after'] 
    condition = invariant['condition']
        
    # Insert after line_after (which is at index line_after-1 in 0-based indexing)
    # So we insert at index line_after (which puts it after line_after)
    if line_after <= len(lines):
        # Get indentation from the line after which we're inserting
        line_after_content = lines[line_after - 1] if line_after > 0 else ""
        indent = ""
        for char in line_after_content:
            if char in [' ', '\t']:
                indent += char
            else:
                break
        
        assume_stmt = f"{indent}__VERIFIER_assume({condition});\n"
        lines.insert(line_after, assume_stmt)
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.c', prefix='invariant_test_')
    with os.fdopen(temp_fd, 'w') as f:
        f.writelines(lines)
    
    return temp_path

def insert_invariant_as_assertion_and_remove_final_assert(original_file_path, invariant):
    """Insert assert statement into the program and remove final assertions"""
    with open(original_file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove final assertions - look for __VERIFIER_assert or assert calls (but not function definitions)
    lines_to_remove = []
    for i, line in enumerate(lines):
        stripped_line = line.strip()
        # Only remove if it's an actual function call ending with semicolon, not a function definition
        if (('__VERIFIER_assert(' in stripped_line and stripped_line.endswith(');')) or 
            (stripped_line.startswith('assert(') and stripped_line.endswith(');'))):
            lines_to_remove.append(i)
    
    # Remove lines in reverse order to maintain indices
    for i in reversed(lines_to_remove):
        lines.pop(i)
    
    # Insert invariant as assertion after specified line
    line_after = invariant['line_after']
    condition = invariant['condition']
    
        # Adjust line_after if lines were removed before it
    adjusted_line_after = line_after
    for removed_line in lines_to_remove:
        if removed_line < line_after:
            adjusted_line_after -= 1
            
    # Insert after adjusted_line_after
    if adjusted_line_after <= len(lines):
        # Get indentation from the line after which we're inserting
        line_after_content = lines[adjusted_line_after - 1] if adjusted_line_after > 0 else ""
        indent = ""
        for char in line_after_content:
            if char in [' ', '\t']:
                indent += char
            else:
                break
        
        assert_stmt = f"{indent}__VERIFIER_assert({condition});\n"
        lines.insert(adjusted_line_after, assert_stmt)
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.c', prefix='invariant_assert_test_')
    with os.fdopen(temp_fd, 'w') as f:
        f.writelines(lines)
    
    return temp_path

def aggregate_verification_results(assume_result, assert_result):
    """
    Aggregate verification results according to the truth table:
    (FALSE, *) -> FALSE
    (TRUE, TRUE) -> TRUE
    (*, FALSE) (except handled by first row) -> UNKNOWN
    (TIMEOUT, TRUE) -> TIMEOUT
    (TRUE, TIMEOUT) -> TIMEOUT
    (UNKNOWN, TRUE) -> UNKNOWN
    (TRUE, UNKNOWN) -> UNKNOWN
    (UNKNOWN, UNKNOWN) -> UNKNOWN
    (TIMEOUT, UNKNOWN) -> TIMEOUT
    (UNKNOWN, TIMEOUT) -> TIMEOUT
    (TIMEOUT, TIMEOUT) -> TIMEOUT
    (KILLED, *) -> UNKNOWN (treat killed processes as unknown)
    (*, KILLED) -> UNKNOWN (treat killed processes as unknown)
    """
    r_assume = assume_result['result']
    r_assert = assert_result['result']
    
    # Handle KILLED results - treat as UNKNOWN
    if r_assume == "KILLED" or r_assert == "KILLED":
        return "UNKNOWN"
    
    # First row: (FALSE, *) -> FALSE
    if r_assume == "FALSE":
        return "FALSE"
    
    # (TRUE, TRUE) -> TRUE
    if r_assume == "TRUE" and r_assert == "TRUE":
        return "TRUE"
    
    # (*, FALSE) (except handled by first row) -> UNKNOWN
    if r_assert == "FALSE":
        return "UNKNOWN"
    
    # All timeout cases
    if r_assume == "TIMEOUT" or r_assert == "TIMEOUT":
        return "TIMEOUT"
    
    # All remaining cases with UNKNOWN
    return "UNKNOWN"

def run_uautomizer_verification(c_file_path, timeout=300):
    """Run UAutomizer verification on the file"""
    # Get paths using helper functions
    uautomizer_path = get_uautomizer_path()
    
    # Command similar to batch_run_uautomizer.py
    command = [
        "python3", "-u", "Ultimate.py",
        "--spec", get_spec_path(),
        "--file", c_file_path,
        "--architecture", "64bit",
        "--full-output"
    ]
    
    start_time = time.time()
    
    try:
        # Get uautomizer directory
        uautomizer_dir = os.path.dirname(uautomizer_path)
        
        # Run with timeout - use cwd parameter instead of os.chdir
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=uautomizer_dir,
            preexec_fn=os.setsid  # Create new process group
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Parse result
        verification_result = "UNKNOWN"
        if process.returncode == 0:
            stdout_lines = stdout.split('\n')
            for line in reversed(stdout_lines):
                if line.strip():
                    verification_result = line.strip()
                    break
        return {
            'success': process.returncode == 0,
            'result': verification_result,
            'time_taken': elapsed_time,
            'stderr': stderr
        }
        
    except subprocess.TimeoutExpired:
        # Kill process group
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(2)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except Exception:
            try:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
                process.wait()
            except:
                pass
        
        return {
            'success': False,
            'result': 'TIMEOUT',
            'time_taken': timeout,
            'stderr': 'Process timed out'
        }
    except Exception as e:
        return {
            'success': False,
            'result': 'ERROR',
            'time_taken': time.time() - start_time,
            'stderr': str(e)
        }

def run_esbmc_verification(c_file_path, timeout=300):
    """Run ESBMC verification on the file"""
    # Get paths relative to this script
    script_dir = Path(__file__).parent.resolve()
    esbmc_path = script_dir / "../tools/esbmc/esbmc"
    esbmc_path = esbmc_path.resolve()
    
    if not os.path.exists(esbmc_path):
        raise FileNotFoundError(f"ESBMC not found at: {esbmc_path}")
    
    # Command for ESBMC
    command = [
        str(esbmc_path),
        c_file_path,
        "--property-file", get_spec_path(),
        "--k-induction",
        "--arch", "64bit"
    ]
    
    start_time = time.time()
    
    try:
        # Run with timeout - use cwd parameter instead of os.chdir
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Parse result
        verification_result = "UNKNOWN"
        if process.returncode == 0:
            if "TRUE" in stdout:
                verification_result = "TRUE"
            elif "FALSE" in stdout:
                verification_result = "FALSE"
        
        return {
            'success': process.returncode == 0,
            'result': verification_result,
            'time_taken': elapsed_time,
            'stderr': stderr
        }
        
    except subprocess.TimeoutExpired:
        # Kill process group
        try:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(2)
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except Exception:
            try:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
                process.wait()
            except:
                pass
        
        return {
            'success': False,
            'result': 'TIMEOUT',
            'time_taken': timeout,
            'stderr': 'Process timed out'
        }
    except Exception as e:
        return {
            'success': False,
            'result': 'ERROR',
            'time_taken': time.time() - start_time,
            'stderr': str(e)
        }


def save_results(results: Dict[str, List[InvariantGenerationResult]], output_file: str):
    """Save results to JSON file in the format {filename: [sample_results]}"""
    results_dict = {}
    for filename, sample_results in results.items():
        results_dict[filename] = []
        for result in sample_results:
            # Strip time_taken from raw verification results before saving (since we have top-level times)
            assume_result = None
            if result.assume_verification_result is not None:
                assume_result = {k: v for k, v in result.assume_verification_result.items() if k != 'time_taken'}
            
            assert_result = None
            if result.assert_verification_result is not None:
                assert_result = {k: v for k, v in result.assert_verification_result.items() if k != 'time_taken'}
            
            results_dict[filename].append({
                'filename': result.filename,
                'success': result.success,
                'result': result.result,
                'verify_time_taken': result.time_taken,  # Max of assume and assert verification time
                'error': result.error,
                'model_response': result.llm_response,  # Include LLM response
                'invariants_count': result.invariants_count,
                'sample_id': result.sample_id,
                'assume_verification_result': assume_result,  # Raw assume verification result (without time_taken)
                'assert_verification_result': assert_result,  # Raw assert verification result (without time_taken)
                'generation_time': result.generation_time,  # LLM generation time for this sample
                'assume_verification_time': result.assume_verification_time,  # Time taken for assume verification
                'assert_verification_time': result.assert_verification_time   # Time taken for assert verification
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}", flush=True)

def print_summary(results: Dict[str, List[InvariantGenerationResult]]):
    """Print summary statistics"""
    # Flatten all results to get overall statistics
    all_results = []
    for filename, sample_results in results.items():
        all_results.extend(sample_results)
    
    total = len(all_results)
    successful = sum(1 for r in all_results if r.success)
    failed = sum(1 for r in all_results if not r.success)
    
    # Count by result type (four categories as requested)
    true_count = sum(1 for r in all_results if r.result == "TRUE")
    false_count = sum(1 for r in all_results if r.result == "FALSE")
    unknown_count = sum(1 for r in all_results if r.result == "UNKNOWN")
    timeout_count = sum(1 for r in all_results if r.result == "TIMEOUT")
    
    # Calculate average overall time for all files (max(assume, assert) + generation)
    def calc_overall_time(r):
        return max(r.assume_verification_time, r.assert_verification_time) + r.generation_time
    
    avg_overall_time = sum(calc_overall_time(r) for r in all_results) / total if total > 0 else 0
    
    # Calculate average overall time for solved instances (TRUE + FALSE)
    solved_results = [r for r in all_results if r.result in ["TRUE", "FALSE"]]
    solved_count = len(solved_results)
    solved_avg_overall_time = sum(calc_overall_time(r) for r in solved_results) / solved_count if solved_count > 0 else 0
    
    # Calculate average verification time for all files
    avg_verification_time = sum(r.time_taken for r in all_results) / total if total > 0 else 0
    solved_avg_verification_time = sum(r.time_taken for r in solved_results) / solved_count if solved_count > 0 else 0
    
    print("\n" + "="*60, flush=True)
    print("📊 DETAILED SUMMARY STATISTICS", flush=True)
    print("="*60, flush=True)
    print(f"Total files processed: {len(results)}", flush=True)
    print(f"Successful runs: {successful}", flush=True)
    print(f"Failed runs: {failed}", flush=True)
    print(flush=True)
    print("🔍 VERIFICATION RESULTS:", flush=True)
    print(f"  TRUE: {true_count}", flush=True)
    print(f"  FALSE: {false_count}", flush=True)
    print(f"  UNKNOWN: {unknown_count}", flush=True)
    print(f"  TIMEOUT: {timeout_count}", flush=True)
    print(flush=True)
    print("⏱️  OVERALL TIMING STATISTICS (LLM + Verification):", flush=True)
    print(f"  Average overall time per file (all): {avg_overall_time:.2f}s", flush=True)
    print(f"  Solved instances (TRUE + FALSE): {solved_count}", flush=True)
    print(f"  Average overall time per solved instance: {solved_avg_overall_time:.2f}s", flush=True)
    print(flush=True)
    print("🔧 VERIFICATION-ONLY TIMING STATISTICS:", flush=True)
    print(f"  Average verification time per file (all): {avg_verification_time:.2f}s", flush=True)
    print(f"  Average verification time per solved instance: {solved_avg_verification_time:.2f}s", flush=True)
    print("="*60, flush=True)
    
    # Additional detailed breakdown
    if true_count > 0:
        true_overall_times = [calc_overall_time(r) for r in all_results if r.result == "TRUE"]
        true_verification_times = [r.time_taken for r in all_results if r.result == "TRUE"]
        true_avg_overall_time = sum(true_overall_times) / len(true_overall_times)
        true_avg_verification_time = sum(true_verification_times) / len(true_verification_times)
        print(f"  TRUE instances average overall time: {true_avg_overall_time:.2f}s")
        print(f"  TRUE instances average verification time: {true_avg_verification_time:.2f}s")
    
    if false_count > 0:
        false_overall_times = [calc_overall_time(r) for r in all_results if r.result == "FALSE"]
        false_verification_times = [r.time_taken for r in all_results if r.result == "FALSE"]
        false_avg_overall_time = sum(false_overall_times) / len(false_overall_times)
        false_avg_verification_time = sum(false_verification_times) / len(false_verification_times)
        print(f"  FALSE instances average overall time: {false_avg_overall_time:.2f}s")
        print(f"  FALSE instances average verification time: {false_avg_verification_time:.2f}s")
    
    if timeout_count > 0:
        timeout_overall_times = [calc_overall_time(r) for r in all_results if r.result == "TIMEOUT"]
        timeout_verification_times = [r.time_taken for r in all_results if r.result == "TIMEOUT"]
        timeout_avg_overall_time = sum(timeout_overall_times) / len(timeout_overall_times)
        timeout_avg_verification_time = sum(timeout_verification_times) / len(timeout_verification_times)
        print(f"  TIMEOUT instances average overall time: {timeout_avg_overall_time:.2f}s")
        print(f"  TIMEOUT instances average verification time: {timeout_avg_verification_time:.2f}s")
    
    print("="*60)

def main():
    """Main function to run the batch invariant generation pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Batch invariant generation and verification')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Coder-7B-Instruct',
                        help='Model name for LLM inference (default: Qwen/Qwen2.5-Coder-7B-Instruct)')
    parser.add_argument('--inference_client', type=str, default='sglang',
                        help='Inference client to use (default: sglang)')
    parser.add_argument('--num_problems', type=int, default=-1,
                        help='Number of problems to evaluate (-1 means all problems, default: -1)')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of parallel workers for LLM inference (default: 8)')
    
    
    ### generation args
    parser.add_argument('--enable_cot', action='store_true', default=False,
                        help='Use Chain-of-Thought (CoT) versions of prompts (default: False)')
    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='Maximum number of new tokens to generate (default: 8192)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for generation (default: 0.0)')
    parser.add_argument('--best_of_n', type=int, default=1,
                        help='Number of samples to generate for best-of-N sampling (default: 1)')
    parser.add_argument('--verifier', type=str, default='uautomizer', choices=['uautomizer', 'esbmc'],
                        help='Verifier to use for program verification (default: uautomizer)')
    parser.add_argument('--test_gt_invariants', action='store_true', default=False,
                        help='Use ground truth invariants from Dataset/timing_uautomizer.json instead of LLM generation (default: False)')
    parser.add_argument('--reload_results', type=str, default=None,
                        help='Path to a previous results JSON file to reload invariants from (skips generation phase, default: None)')
    parser.add_argument('--num_shots', type=int, default=0,
                        help='Number of few-shot examples to include (default: 0)')
    args = parser.parse_args()
    
    # Configuration - all C files from evaluation_all
    script_dir = Path(__file__).parent.resolve()
    c_files_dir = str((script_dir / "../Dataset/evaluation_all").resolve())
    
    if args.test_gt_invariants:
        model_short_name = "gt_invariants"
    else:
        model_short_name = args.model_name.split('/')[-1]
    
    # Generate output file based on model name
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = str(results_dir / f"{model_short_name}_cot={args.enable_cot}_best_of_n={args.best_of_n}_num_shots={args.num_shots}_temperature={args.temperature}_verifier={args.verifier}_invariant_generation_results.json")

    timeout = 600  # 600 seconds timeout (10 minutes)
    
    # Load prompts from YAML file
    prompt_file = "prompt.yaml"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = yaml.safe_load(f)
    print(f"✅ Loaded prompts from {prompt_file}", flush=True)

    
    # Force stdout and stderr to be unbuffered for real-time output
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    print("🚀 Starting batch invariant generation and verification", flush=True)
    print(f"🤖 Model: {args.model_name}", flush=True)
    print(f"📁 Source directory: {c_files_dir}", flush=True)
    print(f"⏱️  Timeout per file: {timeout}s", flush=True)
    print(f"🔧 Parallel workers: {args.max_workers}", flush=True)
    print(f"🧠 Enable CoT prompts: {args.enable_cot}", flush=True)
    print(f"🎯 Best of N sampling: {args.best_of_n}", flush=True)
    print(f"🔍 Verifier: {args.verifier}", flush=True)
    print(f"🎯 GT Invariants: {args.test_gt_invariants}", flush=True)
    if args.test_gt_invariants:
        script_dir = Path(__file__).parent.resolve()
        gt_file = script_dir / "../Dataset/timing_uautomizer.json"
        print(f"📄 GT Invariants file: {gt_file}", flush=True)
    if args.reload_results:
        print(f"🔄 Reload Results: {args.reload_results}", flush=True)
    print(f"📊 Number of problems: {'All' if args.num_problems == -1 else args.num_problems}", flush=True)
    print(f"💾 Output file: {output_file}", flush=True)
    
    # Initialize SGLang client once for all files (skip if in GT mode or reloading)
    client = None
    server_process = None
    sglang_addr = None
    port = None
    try:
        # Initialize the SGLang client (skip if using GT invariants or reloading results)
        if not args.test_gt_invariants and not args.reload_results:
            if args.inference_client == 'sglang':
                print("🤖 Initializing LLM client...", flush=True)
                command_str = f"python -m sglang.launch_server --model-path {args.model_name} --host 0.0.0.0" 
                if '30B-A3B' in args.model_name:
                    command_str += " --context-length 8192"
                server_process, port = launch_server_cmd(command_str)
                wait_for_server(f"http://localhost:{port}")
                sglang_addr = f"http://localhost:{port}"
            
            client = get_client(args.inference_client, model_name=args.model_name, sglang_addr=sglang_addr)
            if port is not None:
                print(f"✅ LLM client initialized on port {port}", flush=True)
            else:
                print(f"✅ LLM client initialized", flush=True)
        else:
            if args.test_gt_invariants:
                print("🎯 GT mode: Skipping LLM client initialization", flush=True)
            elif args.reload_results:
                print("🔄 Reload mode: Skipping LLM client initialization", flush=True)
        
        # Create a temporary processor to load GT invariants if needed
        available_gt_files = None
        
        if args.test_gt_invariants:
            temp_processor = BatchInvariantProcessor(
                c_files_dir, output_file, timeout,
                max_workers=args.max_workers,
                client=None,  # No client needed for loading GT data
                prompts=None,  # No prompts needed for GT mode
                test_gt_invariants=True
            )
            # Filter GT files to only include those with valid invariants at loop insertion points
            available_gt_files = filter_gt_files_with_valid_invariants(c_files_dir, temp_processor.gt_invariants)
            print(f"📊 Filtered to {len(available_gt_files)} files with valid GT invariants at loop insertion points", flush=True)
        
        # Find C files
        c_files = find_c_files(c_files_dir, args.num_problems, args.test_gt_invariants, available_gt_files)
        print(f"📋 Found {len(c_files)} C files", flush=True)
        
        if not c_files:
            print("❌ No C files found in the directory", flush=True)
            return
        
        # Limit number of files if specified (for non-GT mode)
        if args.num_problems > 0:
            c_files = c_files[:args.num_problems]
            print(f"📊 Processing first {len(c_files)} files", flush=True)
        
        # Create a BatchInvariantProcessor instance
        processor = BatchInvariantProcessor(
            c_files_dir, output_file, timeout, 
            max_workers=args.max_workers, 
            client=client, 
            prompts=prompts, 
            enable_cot=args.enable_cot, 
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            best_of_n=args.best_of_n,
            server_process=server_process,
            verifier=args.verifier,
            test_gt_invariants=args.test_gt_invariants,
            reload_results_file=args.reload_results,
            num_shots=args.num_shots
        )
        
        # Run two-phase processing:
        # - Phase 1: Generate invariants for all files in parallel
        # - Phase 2: Verify files sequentially (samples within each file run in parallel)
        processor.run_two_phase_processing(c_files)
        
        
        # Save final results
        save_results(processor.results, output_file)
        
        # Print summary
        print_summary(processor.results)
        
        # Print failed files
        failed_files = []
        for filename, sample_results in processor.results.items():
            for result in sample_results:
                if not result.success:
                    failed_files.append(result)
        
        if failed_files:
            print(f"\n❌ Failed files ({len(failed_files)}):")
            for result in failed_files:
                print(f"  - {result.filename}: {result.result} ({result.error})")
        
        # Print timeout files
        timeout_files = []
        for filename, sample_results in processor.results.items():
            for result in sample_results:
                if result.result == "TIMEOUT":
                    timeout_files.append(result)
        
        if timeout_files:
            print(f"\n⏰ Timeout files ({len(timeout_files)}):")
            for result in timeout_files:
                total_time = max(result.assume_verification_time, result.assert_verification_time) + result.generation_time
                print(f"  - {result.filename}: {total_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        return
    finally:
        # Cleanup SGLang server if it still exists
        if server_process:
            print("🧹 Cleaning up LLM server...", flush=True)
            terminate_process(server_process)

if __name__ == "__main__":
    main() 