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

### Notebooks

Run notebooks in the order listed

* 00_snorkel - Experiment with snorkel (ignore, we tried using snorkel but it didn't work well)
* 00_snorkel_names - Use snorkel to generate training data (ignore this as well)
* 10_clean - Clean the raw name pairs from FamilySearch (pairs of tree <-> record name) and separate into given and surnames
* 11_clean - Clean the preferred tree names from FamilySearch 
* 12_analyze_preferred - Review given name abbreviations (optional)
* 20_generate_pairs - Generate pairs of best-matching name pieces from multi-word given or surnames
* 30_aggregate - Aggregate pairs of matching tree <-> record name pieces and compute counts, probabilities, and similarities
* 31_aggregate_preferred - Aggregate preferred (tree) names and compute counts
* 32_analyze_aggregate_preferred - Get an idea of how much mass is in the top Nk names (optional)
* 40_filter - Create lists of pairs that are similarly-spelled (used to create the model) and pairs that are dissimilar
* 41_generate_nicknames - Create a file of dissimilar given name (potential nickname) pairs for human review
* 45_train_test_split - Split similar name pairs into training and test sets
* 46_demo_dataset - Create a very small demo dataset to play with (optional)
* 47_bad_pair_generator - Generate hard (near) negative pairs (ignore - not used)
* 47a_filter_bad_pairs - Remove pairs that were identified as bad pairs during a manual review of borderline pairs
* 48_weighted_actual_names_to_csv - Write weighted-actual name pairs for human review (optional)
* 49_augment_dataset - Augment the training dataset with other matching pairs based upon names having the same code or levenshtein similarity
* 50_autoencoder - Generate an autoencoder model based upon similar name pairs (not used)
* 51_autoencoder_triplet - Generate a triplet-loss model based upon the autoencoder and near-negatives (not used)
* 52_glove - Generate a glove model (not used)
* 60_swivel_tune - Run hyperparameter tuning on swivel model (optional)
* 61_swivel - Train a swivel model (takes a long, long, long time)
* 62_swivel_encoder_tune - Tune a swivel-based encoder model (not used)
* 63_swivel_encoder - Train a swivel-based encoder model (not used)
* 64_analyze_scores - Compare swivel and levenshtein scores (optional)
* 65_tfidf - Train a TfidfVectorizer to filter names sent to levenshtein
* 66_ensemble - Train an ensemble model over swivel + levenshtein
* 70_compare_similarity - Compare the ensemble model to levenshtein and other floating-score algorithms (optional)
* 71_analyze_embeddings - Visualize swivel vectors in 2d space (optional)
* 72_analyze_names - Graph various statistics of names and name-coders (optional)
* 80_cluster_tune - Run hyperparameter tuning on clustering model (optional)
* 81_cluster - Train a clustering model (optional)
* 82_cluster_levenshtein - Not sure what this does. Very out of date. (ignore)
* 90_compare_clusters - # Compare our clusters to the clusters formed by various coders (soundex, nysiis, etc.) (optional)
* 91_compare_clusters_old_algo - Get Statistics for the old clusters (optional)
* 97_analyze_nicknames - Review nicknames (optional)
* 98_given_surname_freq - Generate how likely a name is to be a given vs surname (optional)
* 99_precompute - Pre-compute embeddings and cluster-scores so they can be cached (optional)

### Server

The server is currently a bit out of date.

#### Starting the server and online server documentation 
    uvicorn src.server.server:app --reload
    http://localhost:8000/docs

#### Using docker
    docker build -t nama .
    docker run --rm -d --name nama -p 8080:8080 nama

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

### Using gcloud
* Use their deeplearning ami, and don't try to install another version of pytorch
* don't use conda

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
