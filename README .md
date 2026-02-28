# vulnsc

## introduction for each folders
models: store the source code of baselines
scripts: store the scripts of vulnsc that are used to enhance the datasets.
data: store the dataset that are original and enhanced by vulnsc.

## command for running vulnsc to enhance dataset Devign.
python main.py 


## command for running baselines for experiments.
cd models
./run_devign_codellama.sh
./run_devign_deepseek.sh
./run_devign_gpt4o.sh
./run_devign_mixtral.sh

