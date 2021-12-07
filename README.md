# nama

Using deep learning to find similar personal names

## Initial Setup
    make create_environment
    conda activate nama
    make requirements
    conda install -c conda-forge faiss-cpu

### If you want to develop, also do the following
    nbstripout --install   # automatically strip notebook output before commit
    pytest                 # run tests

## Using nama
### Starting the server and online server documentation 
    uvicorn src.server.server:app --reload
    http://localhost:8000/docs

### Using Fastec2 for managing remote jupyter notebooks
* Instructions: https://www.fast.ai/2019/02/15/fastec2/
* pip install git+https://github.com/fastai/fastec2.git
* ssh-add default.pem
* fe2 launch < name > base 80 r5.2xlarge  # 80Gb disk, 64Gb memory
* ./remote-install.sh < ip >
* fe2 connect < name > 8888
  * cd nama
  * conda activate nama
  * jupyter notebook
* fe2 stop < name >       # stopped instances can be re-started with fe2 start < name >
* fe2 terminate < name >  # releases instance name and associated disk

### Logging into Weights and Biases
* run `wandb login`
* login information will be added to your ~/.netrc file
* copy just that information from ~/.netrc to ~/.netrc.wandb. This file will be added to machines launched with FastEC2

## Data
### locations
* raw data can be found at s3://familysearch-names/raw 
* large interim data files can be found at s3://familysearch-names/interim

### descriptions
* tree_hr_given-similar.csv + tree-hr-surname-similar.csv - likely-similar name pairs
  * name - preferred name in the tree
  *	alt_name - name in the record (pairs are omitted when alt_name is the same as name)
  * frequency - how many times these two names appear together
  *	reverse_frequency - how many times the alt_name is the name in the tree and the name is the name in the record
  *	sum_name_frequency - how many times the (tree) name appears as a (tree) name
  *	total_name_frequency - how many times the (tree) name appears as a (tree) name or as an alt_name
  *	total_alt_name_frequency - how many times the alt_name appears as an alt_name or a (tree) name
  *	ordered_prob - frequency / sum_name_frequency
  *	unordered_prob - (frequency + reverse_frequency) / the total number of times either name appears as a name or an alt_name
  *	similarity - 1.0 - (levenshtein distance between the name and alt_name / max(len(name), len)alt_name)))


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── tests              <- Tests for source code
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
