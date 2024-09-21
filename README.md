# DeepEmotion: Contrastive Learning for fMRI Classification of Emotions

### Setup
Deep Emotion uses the native python venv client for handling package dependencies. To get started, run

```
git clone git@github.com:lungerjo/DeepEmotion.git
cd DeepEmotion
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Extending DeepEmotion

When extending or contributing to the DeepEmotion project, ensure each script includes the following header for consistent configuration management:

```
import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)  # Adjust as needed based on script location (should point to /DeepEmotion)
os.environ['PROJECT_ROOT'] = PROJECT_ROOT
CONFIG_PATH = str(Path(PROJECT_ROOT) / "src" / "configs")
```

Any globals should be defined under `src/configs/<config>.yaml`
