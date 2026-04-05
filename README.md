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

## Run Experiments

```bash
cd baselines
python batch_invariant_generation.py --max_workers 1 --model_name gpt-5.2 --inference_client openai --max_new_tokens 200 --best_of_n 2 --temperature 0.7
```
Results will be stored as a JSON in the `results/` folder within `baselines/`. By default the inference client is sglang and can be set to openai, anthropic, together (for together AI) to run models.

## Print Results

`baselines/print_results.py` contains the code to print the results. 

```bash
python baselines/print_results.py [PATH_TO_RESULT_JSON_FILE]
```

## Dataset

TBD

## Citation

If our research inspires you, please cite our paper:

```bibtex
@inproceedings{wei2026quokka,
  title={Quokka: Accelerating Program Verification with LLMs via Invariant Synthesis},
  author={Wei, Anjiang and Sun, Tianran and Suresh, Tarun and Wu, Haoze and Wang, Ke and Aiken, Alex},
  year={2026},
  eprint={2509.21629},
  archivePrefix={arXiv},
  primaryClass={cs.PL},
  url={https://arxiv.org/abs/2509.21629},
}
```

## Acknowledgement

- [LEMUR](https://github.com/ai-ar-research/Lemur-program-verification)

## License

This project is licensed under the [Apache License 2.0](https://opensource.org/license/apache-2-0). 