# DeepEmotion: 3DCNNs Learn Naturalistic Emotion from 7t fMRI

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
The dataset is hosted on [datalad](https://www.datalad.org) stored in ``raw`` and ``annotations``. To get started with datalad, follow the [install documentation](https://handbook.datalad.org/en/latest/intro/installation.html#install-datalad). Once you have installed, you can fetch the raw and annotation datasets with with 
```
datalad get data/raw
datalad get data/annotations
```
This will populate the directories with symbolic links. To get the data, run 
```datalad get <path/to/data>```
or
```datalad get -f <path/to/data/dir>```







