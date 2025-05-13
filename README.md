### Generating data
`cd generation/generate_explanation/`
####Get the training data for LPFT.
`python LPFT_train_data.py`
####Get the candidate paths for LPFT.
`python candidate_paths.py`
### Generating explanations

`cd generation/generate_explanation/`

`python main_transe.py --dataset <dataset> --facts_to_explain_path <target_path> --candidate_path_dict<candidate_path>
`

`python main_conve.py --dataset <dataset> --facts_to_explain_path <target_path> --candidate_path_dict<candidate_path>
`

`python main_complex.py --dataset <dataset> --facts_to_explain_path <target_path> --candidate_path_dict<candidate_path>
`
Acceptable values for `--dataset` are `WN18RR` or `FB15k237`.
