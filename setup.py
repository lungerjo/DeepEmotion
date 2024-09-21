# setup.py
from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="DeepEmotion: fMRI Classification for Emotion Representation",
    version="0.1.0",
    description="training classification models on fMRI from Forrest Gump.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joshua Lunger, Mason Hu",
    author_email="j.lunger@mail.utoronto.ca",
    url="https://github.com/lungerjo/DeepEmotion",  # Update with your repository
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        # Core Libraries
        "torch>=1.9.0",
        "torchvision>=0.10.0",

        # Training Frameworks
        "pytorch-lightning>=1.4.0",
        "hydra-core>=1.1.0",
        "omegaconf>=2.1.0",
        "torchmetrics>=0.6.0",

        # Data Processing
        "numpy>=1.20.0",
        "pandas>=1.2.0",
        "nibabel>=3.2.0",       # For fMRI data handling

        # Logging and Visualization
        "tensorboard>=2.5.0",
        "loguru>=0.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",

        # Hyperparameter Optimization (Optional)
        # "optuna>=2.8.0",

        # Experiment Tracking (Optional)
        # "wandb>=0.12.0",        # Weights & Biases

        # Testing
        "pytest>=6.2.0",

        # Additional Utilities
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "optuna>=2.8.0",
            "wandb>=0.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3.12.6",  # Update as necessary
        "License :: OSI Approved :: MIT License",  # Update if using a different license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
