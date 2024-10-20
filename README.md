# DeepEmotion: Contrastive Learning for fMRI Classification of Emotions

### Setup
#### Package Dependencies
Deep Emotion uses Poetry for handling package dependencies. To get started, run

```
git clone git@github.com:lungerjo/DeepEmotion.git
cd DeepEmotion
pip install poetry
poetry install
```

To activate the poetry environment, run
```
poetry shell
```
and ``exit`` to exit.

#### Datalad
The dataset is hosted on [datalad](https://www.datalad.org) stored in ``raw``. To get started with datalad, follow the [install documentation](https://handbook.datalad.org/en/latest/intro/installation.html#install-datalad). Once you have installed, you can fetch the dataset with 
```datalad get data/raw```
and any individual file within with 
```datalad get <path/to/data>```







