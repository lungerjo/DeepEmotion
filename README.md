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
The dataset is hosted on [datalad](https://www.datalad.org) stored in ``raw``. To download the dataset, navigate to the ``data`` directory and run
```
datalad get -r raw
```
This will download about 250gb of fMRI data to your machine. You have been warned.


