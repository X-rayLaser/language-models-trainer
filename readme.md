# Status: Work in progress

# Installation

Open a terminal, clone the repository into the current working directory
```
git clone https://github.com/X-rayLaser/language-models-trainer.git
```

Go inside the repository directory that was just created
```
cd language-models-trainer
```

Create a virtual environment created using Python 3 executable.
```
virtualenv --python=/path/to/python3/executable venv
```
For Linux based OS you can get the path to the Python executable with this command:
```
which python3
```

Activate the virtual environment. For Linux based OS:
```
. venv/bin/activate
```

Install Python modules used by the project
```
pip install -r requirements.txt
```

# Usage

## Training LSTM model
Train LSTM language model on sentences taken from fiction text in Brown corpus
```
python train.py --capacity=32 --genre='fiction' --epochs=100
```

Train LSTM model on paragraphs taken from fiction text in Brown corpus
```
python train.py --capacity=32 --genre='fiction' --epochs=100 --paras=True
```

Train LSTM model on sentences taken from the whole Brown corpus
```
python train.py --capacity=32 --epochs=100
```

To see all available options, run
```
python train.py --help
```
