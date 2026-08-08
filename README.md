# Quokka: Accelerating Program Verification with LLMs via Invariant Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2509.21629-b31b1b.svg)](https://www.arxiv.org/abs/2509.21629) [![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/license/apache-2-0) 

Quokka is the official repository for the paper "Quokka: Accelerating Program Verification with LLMs via Invariant Synthesis". 

## Install Dependencies

### Create and Activate a Conda Environment
```bash
conda create -y -n invenv python=3.11.4
conda activate invenv
```

### Install Uautomizer (and optionally ESBMC)
```
./build.sh
```

### Install Requirements

```bash
cd baselines/
pip install -r reqs.txt
```

### Set API Keys

If you use a hosted inference client, export the corresponding API key before running experiments.

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export TOGETHER_API_KEY=your_together_api_key
```

You only need to set the key for the client you plan to use. If you use the default `sglang` client with a local model server, no API key is required.

## Run Experiments

```bash
cd baselines
python batch_invariant_generation.py --max_workers 1 --model_name gpt-5.2 --inference_client openai --max_new_tokens 200 --best_of_n 2 --temperature 0.7
```
Results will be stored as a JSON in the `results/` folder within `baselines/`. By default the inference client is sglang and can be set to openai, anthropic, together (for together AI) to run models.

## Print Results

`baselines/print_results.py` summarizes Quokka result JSONs and prints model-comparison tables.

```bash
python baselines/print_results.py [PATH_TO_RESULT_JSON_FILE_OR_DIRECTORY]
```

Examples:

```bash
# summarize one result JSON
python baselines/print_results.py baselines/results/<RESULT_JSON>.json

# summarize every result JSON in baselines/results/
python baselines/print_results.py baselines/results

# change timeout thresholds used for #Ext, #Slv, and PAR
python baselines/print_results.py baselines/results --timeouts 30 500

# also print the LaTeX table
python baselines/print_results.py baselines/results --latex

# also print the detailed per-run diagnostic summary
python baselines/print_results.py baselines/results --detailed
```

By default it prints the compare-models table:

- `#Corr`: number of problems with at least one verifier-confirmed synthesized invariant (`assert_verification_result.result == TRUE`)
- `#Ext@T`: number of problems solved by Quokka within timeout `T` that are not solved by the baseline within `T`
- `#Slv@T`: number of problems solved by Quokka within timeout `T`
- `PAR@T`: penalized average runtime at timeout `T`

The baseline file is inferred from the result filename suffix `verifier=...` and can be overridden with `--baseline`. Timeout cutoffs can be changed with `--timeouts`.

Useful optional flags:

- `--detailed`: also print the per-run diagnostic summary
- `--latex`: also print a LaTeX table for the same compare-models summary
- `--timeouts T1 T2 ...`: set the timeout cutoffs used for reporting
- `--baseline PATH`: override the baseline JSON path
- `--verifier {auto,uautomizer,esbmc}`: choose which baseline to use when it cannot be inferred from the filename

## Dataset

- `Dataset/evaluation_all/`: the benchmark C programs used for evaluation.
- `Dataset/properties/unreach-call.prp`: the safety property passed to the verifier.
- `Dataset/timing_uautomizer.json`: UAutomizer results on the original benchmark programs. Each entry records:
  - `filename`: benchmark filename
  - `result`: original-program verification result
  - `time_taken`: runtime on the original program
  - `invariants`: when available, ground-truth loop invariants as objects of the form `{"line": <source line>, "invariant": <C boolean expression>}`. The `line` value identifies the loop-location line used for invariant insertion.
- `Dataset/timing_esbmc.json`: ESBMC results on the original benchmark programs, with `filename`, `result`, and `time_taken` for each benchmark.

## Citation

If our research inspires you, please cite our paper:

```bibtex
@inproceedings{wei2026quokka,
  title={Quokka: Accelerating Program Verification with {LLM}s via Invariant Synthesis},
  author={Wei, Anjiang and Sun, Tianran and Suresh, Tarun and Wu, Haoze and Wang, Ke and Aiken, Alex},
  booktitle={Third Conference on Language Modeling},
  year={2026},
  url={https://openreview.net/forum?id=YvyAcGrpcJ}
}
```

## Acknowledgement

- [LEMUR](https://github.com/ai-ar-research/Lemur-program-verification)

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0). 
