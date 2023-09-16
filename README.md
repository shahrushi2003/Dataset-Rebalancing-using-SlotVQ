# Dataset Rebalancing using SlotVQ


## How to run

This code uses [vqtorch](https://github.com/minyoungg/vqtorch/tree/main) for vector quantization. In order to run this code, you will first need to set up this module by running the following code setup:
```
pip3 install -r requirements.txt

# Cloning the repo with different name
git clone https://github.com/minyoungg/vqtorch vqtorch_folder

pip3 install -e vqtorch_folder
```

Then, edit the config files from `config.py` and run the following command to start the experiment:
```
python main.py
```
