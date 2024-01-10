# GALILEO (NeurIPS'23 Spotlight)

The official code of "Adversarial Counterfactual Environment Model Learning". 

We provide a faithful offline dynamics model learning techniques based on the adversarial model learning paradigm. 

![](./resources/Neurips-galileo-poster.png)

## quickstart

install
```
pip install -e .
pip install -r requirements.txt
```

run

```
cd run_scripts
python main.py --data_type d4rl --env_name hopper --data_train_type medium
```

view your results

1. the tensorboard logs are in ./RLA_LOG/log folder;
2. you can manager your experiment result via RLAssistant (see: https://github.com/polixir/RLAssistant)
