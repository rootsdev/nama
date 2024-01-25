# nama

Using deep learning to find similar personal names

[Presentation Slides](https://docs.google.com/presentation/d/1xg0PYlfd14vQjgzsGm6woGYEcWljI_4T3C5S6-XqNvk/edit?usp=sharing)

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

* 10_clean - clean the raw name pairs from FamilySearch (pairs of tree <-> record name) and separate into given and surnames
  * input: tree-hr-raw
  * output: tree-hr-names
* 11_clean_preferred - generate given and surname preferred names from FamilySearch
  * input: pref-names-raw
  * output: pref-names-interim
* 20_generate_pairs - generate pairs from best-matching name pieces
  * input: tree-hr-names
  * output: tree-hr-pairs
* 30_aggregate - aggregate pairs of matching tree <-> record name pieces and compute counts, probabilities, and similarities
  * input: tree-hr-pairs
  * output: tree-hr-parquet-v2
* 31_aggregate_preferred - aggregate preferred names
  * input: pref-names-interim
  * output: pref-names
* 40_filter - convert the tree-hr-attach parquet files into similar and dissimilar name pairs
  * input: tree-hr-parquet-v2
  * output: similar-v2, dissimilar-v2
* 100_train_test_split - split similar names into train and test sets, removing bad pairs
  * input: similar-v2, pref-names, bad-pairs
  * output: train-v2, test-v2
* 200_generate_triplets - generate triplets from training data
  * input: train-v2
  * output: triplets
* 204_generate_subword_tokenizer - train a subword tokenizer
  * input: triplets, pref-names, train-v2
  * output: subword-tokenizer
* 205_generate_common_non_negatives - generate pairs of names that are not negative examples
  * input: std-buckets, pref-names, triplets, given-nicknames
  * output: common-non-negatives
* 206_analyze_triplets - review triplets (optional)
  * input: triplets, pref-names, common-non-negatives,
* 207_augment_triplets - augment triplets with additional triplets
  * input: triplets, pref-names, common-non-negatives, subword-tokenizer
  * output: triplets-augmented
* 220_create_language_model_dataset - create large datasets to train roberta masked language model
  * input: pref-names, tree-hr-parquet(-v2)?
  * output: all-tree-pref-names, all-tree-hr-names
* 221_train_language_model - train a roberta masked language model in preparation for training the name-pair cross-encoder
  * input: all-tree-hr-names-sample, pref-names
  * output: roberta
* 222_train_cross_encoder - train a cross-encoder model (based on sentence bert) that takes a pair of names and outputs a similarity score
  * input: roberta, triplets-augmented
  * output: cross-encoder
* 223_generate_triplets_from_cross_encoder - generate various datasets of triplets for training the bi-encoder from the cross-encoder, cause the bi-encoder needs a lot of data
  * input: pref-names, train-v2, common-non-negatives, std-buckets, cross-encoder
  * output: cross-encoder-triplets-0 and cross-encoder-triplets-common (run twice)
* 224_train_bi_encoder - train a bi-encoder model
  * input: cross-encoder-triplets-common-0-augmented, subword-tokenizer
  * output: bi-encoder
* 230_eval_bi_encoder - evaluate a bi-encoder model, used to pick hyperparameters
  * input: subword-tokenizer, bi-encoder, pref-names, triplets, common-non-negatives
* 240_create_clusters_from_buckets - split buckets into clusters using the cross encoder; clusters in the same bucket form a super-cluster
  * input: std-buckets, subword-tokenizer, cross-encoder, bi-encoder, pref-names
  * output: clusters, super-clusters
* 241_augment_clusters - augment clusters with additional names that were not in any cluster
  * input: clusters, subword-tokenizer, pref-names, cross-encoder, bi-encoder
  * output: augmented-clusters
* 242_nearby_clusters - compute nearby clusters for each cluster using the bi-encoder followed by the cross-encoder (deprecated)
  * input: augmented-clusters, subword-tokenizer, bi-encoder, cross-encoder
  * output: nearby-clusters
* 243_compress_clusters - compress cluster and super-cluster files so we can check them into git
  * input: augmented-clusters, super-clusters
  * output: compressed versions of augmented-clusters and super-clusters
* 245_eval_coder - compare the precision and recall of nama to familysearch and other coders
  * input: augmented-clusters, super-clusters, subword-tokenizer, bi-encoder, train-v2, test-v2, query-names, pref-names, given-nicknames
* 250_create_phonebook - create the phonebook for surnames
  * input: augmented-clusters, super-clusters, pref-names or hr-names
  * output: phonebook
* 251_save_bi_encoder_weights - save the bi-encoder weights so we can use them in fs-nama (java)
  * input: subword-tokenizer, bi-encoder
  * output: bi-encoder-weights

### Files

* all-tree-hr-names-sample - 10m sample of all-tree-hr names
  * f"../data/processed/all-tree-hr-{given_surname}-sample-10m.txt"
* all-tree-hr-names - all tree preferred names
  * f"../data/processed/all-tree-hr-{given_surname}.txt"
* all-tree-pref-names - all tree preferred names
  * f"../data/processed/all-tree-preferred-{given_surname}.txt"
* augmented-clusters - similar names from the same bucket that have been augmented with additional frequent names that were not in any bucket
  * f"../data/processed/clusters_{given_surname}-{scorer}-{linkage}-{similarity_threshold}-{cluster_freq_normalizer}-augmented.json"
* bad-pairs - pairs of names that are not similar (Clorinda reviewed)
  * f"s3://familysearch-names/interim/{given_surname}_variants_clorinda_reviewed.tsv"
  * I don't recall exactly how the borderline pairs that went to review were generated, but most likely we simply identified similar-v2 training pairs that had low levenshtein similarity. We don't have a notebook for this.
* bi-encoder - model to convert a tokenized name to a vector
  * f"../data/models/bi_encoder-{given_surname}-{model_type}.pth" 
* bi-encoder-weights - json file containing bi-encoder token and position weights
  * f"../data/models/bi_encoder-{given_surname}-{model_type}-weights.json"
* clusters - similar names from the same bucket
  * f"../data/processed/clusters_{given_surname}-{scorer}-{linkage}-{similarity_threshold}-{cluster_freq_normalizer}.json"
* common-non-negatives - pairs of names that may be similar (are not negative)
  * f"../data/processed/common_{given_surname}_non_negatives.csv" 
* cross-encoder - model to evaluate the similarity of two names
  * f"../data/models/cross-encoder-{given_surname}-10m-265-same-all"
* cross-encoder-triplets-0 - triplets generated from cross-encoder with num_easy_negs=0
  * f"../data/processed/cross-encoder-triplets-{given_surname}-0.csv"
* cross-encoder-triplets-common - triplets generated from cross-encoder with num_easy_negs='common'
  * f"../data/processed/cross-encoder-triplets-{given_surname}-common.csv"
* cross-encoder-triplets-common-0-augmented = cross-encoder-triplets-common + cross-encoder-triplets-0 + triplets-augmented 
  * f"../data/processed/cross-encoder-triplets-{given_surname}-common-0-augmented.csv"
* dissimilar-v2 - pairs of names from tree-record attachments that are probably not similar
  * f"s3://familysearch-names/processed/tree-hr-{given_surname}-dissimilar-v2.csv.gz" 
* given-nicknames - nicknames for given names (hand curated from a variety of sources)
  * f"../references/givenname_nicknames.csv"
* hr-names - names from historical records - Richard provides this by zcat'ing all files into a single file
  * f"../data/processed/hr-{given_surname}-aggr.csv.gz"
* nearby-clusters - for each cluster, list the nearby clusters
  * f"../data/processed/nearby_clusters_{given_surname}-{scorer}-{linkage}-{similarity_threshold}-{cluster_freq_normalizer}.json"
* phonebook - map surname clusters to partitions
  * f"s3://familysearch-names/processed/phonebook.csv"
* pref-names - preferred names from the tree
  * f"../data/processed/tree-preferred-{given_surname}-aggr.csv.gz"
* pref-names-interim 
  * f"s3://familysearch-names/interim/tree-preferred-{given_surname}/"
* pref-names-raw
  * f"s3://familysearch-names/raw/tree-preferred/"
* query-names - sample of queried names to be evaluated
  * f"../data/processed/query-names-{given_surname}-v2.csv.gz"
* roberta - roberta-based language model for names
  * f"../data/models/roberta-{given_surname}-10m-{vocab_size}"
* similar-v2 - train-v2 + test-v2 before bad pairs have been removed (same as tree-hr-{given_surname}-similar.csv.gz)
  * f"s3://familysearch-names/processed/tree-hr-{given_surname}-similar-v2.csv.gz"
* std-buckets - original name buckets
  * f"../references/std_{given_surname}.txt"
* subword-tokenizer - tokenize names into subwords
  * f"../data/models/fs-{given_surname}-subword-tokenizer-2000f.json"  
* super-clusters - sets of clusters that were in the same bucket
  * f"../data/processed/super_clusters_{given_surname}-{scorer}-{linkage}-{similarity_threshold}-{cluster_freq_normalizer}.json"
* test-v2 - tree-record frequency (test set, which has rarer names)
  * f"../data/processed/tree-hr-{given_surname}-test-v2.csv.gz"
* train-v2 - tree-record frequency
  * f"../data/processed/tree-hr-{given_surname}-train-v2.csv.gz"
* tree-hr-names - names from tree-record attachments
  * f"s3://familysearch-names/interim/tree-hr-{given_surname}/"
* tree-hr-pairs - pairs of names from tree-record attachments
  * f"s3://familysearch-names/interim/tree-hr-{given_surname}-pairs/"
* tree-hr-parquet - local copy of tree-hr-names
  * f"../data/tree-hr-{given_surname}/*.parquet"
* tree-hr-parquet-v2 - aggregated pairs of tree-hr with similarity scores
  * f"s3://familysearch-names/interim/tree-hr-{given_surname}-aggr-v2.parquet"
* tree-hr-raw - raw tree-record attachments
  * f"s3://familysearch-names/raw/tree-hr/"
* triplets - triplets generated from train-v2
  * f"../data/processed/tree-hr-{given_surname}-triplets-v2-1000.csv.gz" 
* triplets-augmented - triplets generated from train-v2, augmented
  * f"../data/processed/tree-hr-{given_surname}-triplets-v2-1000-augmented.csv.gz" 

## Future work

We could consider using the swivel output as input to train the bi-encoder (notebook 224) instead of
cross-encoder-triplets-common-0-augmented. The cross-encoder-triplets-common-0-augmented file has an 
unfortunate "bump" in scores at 0.4. That is, a lot of name pairs in the training data are considered
0.4 similar due to the way the training data was generated. This may make the bi-encoder less-accurate
than it could be if it were trained instead with the scores from the swivel model. (Estimate 1 week.)

We could consider using the swivel output to re-train the original classifier. (Estimate 1 month.)

## Archive

**The information in this system describes a previous version of nama.**

The previous version has been superceded by the current version described above, but the information below may still be useful.

### Notebooks

Run notebooks in the order listed

* 00_snorkel - Experiment with snorkel (ignore, we tried using snorkel but it didn't work well)
* 00_snorkel_names - Use snorkel to generate training data (ignore this as well)
* 10_clean - Clean the raw name pairs from FamilySearch (pairs of tree <-> record name) and separate into given and surnames
  * input:
    * hr_raw: "s3://familysearch-names/raw/tree-hr/"
  * output:
    * hr_names[given]: "s3://familysearch-names/interim/tree-hr-given/"
    * hr_names[surname]: "s3://familysearch-names/interim/tree-hr-surname/"
* 11_clean - Clean the preferred tree names from FamilySearch
  * input: 
    * pref: "s3://familysearch-names/raw/tree-preferred/"
  * output:
    * pref_given: "s3://familysearch-names/interim/tree-preferred-given/"
    * pref_surname: "s3://familysearch-names/interim/tree-preferred-surname/"
* 12_analyze_preferred - Review given name abbreviations (optional)
  * input:
    * pref_given: "s3://familysearch-names/interim/tree-preferred-given/"
* 20_generate_pairs - Generate pairs of best-matching name pieces from multi-word given or surnames
  * input:
    * hr_names: "s3://familysearch-names/interim/tree-hr-{given_surname}/"
  * output:
    * hr_pairs: "s3://familysearch-names/interim/tree-hr-{given_surname}-pairs/"
* 30_aggregate - Aggregate pairs of matching tree <-> record name pieces and compute counts, probabilities, and similarities
  * input:
    * hr_pairs: "s3://familysearch-names/interim/tree-hr-{given_surname}-pairs/"
  * output:
    * hr_aggr: "s3://familysearch-names/interim/tree-hr-{given_surname}-aggr.parquet"
* 31_aggregate_preferred - Aggregate preferred (tree) names and compute counts
  * input:
    * pref: "s3://familysearch-names/interim/tree-preferred-{given_surname}/"
  * output:
    * pref_aggr: "s3://familysearch-names/interim/tree-preferred-{given_surname}-aggr.csv.gz"
* 32_analyze_aggregate_preferred - Get an idea of how much mass is in the top Nk names (optional)
  * input:
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
* 40_filter - Create lists of pairs that are similarly-spelled (used to create the model) and pairs that are dissimilar
  * input:
    * hr_aggr: "s3://familysearch-names/interim/tree-hr-{given_surname}-aggr.parquet"
  * output:
    * hr_similar: "s3://familysearch-names/processed/tree-hr-{given_surname}-similar.csv.gz"
    * hr_dissimilar: "s3://familysearch-names/processed/tree-hr-{given_surname}-dissimilar.csv.gz" 
* 41_generate_nicknames - Create a file of dissimilar given name (potential nickname) pairs for human review
  * input:
    * hr_dissimilar_given: "s3://familysearch-names/processed/tree-hr-given-dissimilar.csv.gz"
  * output:
    * hr_possible_nicknames: "s3://familysearch-names/processed/tree-hr-nicknames.csv.gz"
* 45_train_test_split - Split similar name pairs into training and test sets
  * input:
    * hr_similar: "s3://familysearch-names/processed/tree-hr-{given_surname}-similar.csv.gz"
  * output:
    * hr_train_unfiltered: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-unfiltered.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
* 46_demo_dataset - Create a very small demo dataset to play with (optional)
  * input:
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * hr_train_unfiltered: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-unfiltered.csv.gz"
  * output:
    * hr_demo_output: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-demo.csv.gz"  
* 47_bad_pair_generator - Generate hard (near) negative pairs (ignore - not used)
* 47a_filter_bad_pairs - Remove pairs that were identified as bad pairs during a manual review of borderline pairs
  * input:
    * hr_train_unfiltered: s3://familysearch-names/processed/tree-hr-{given_surname}-train-unfiltered.csv.gz"
    * bad_pairs: "s3://familysearch-names/interim/{given_surname}_variants_clorinda_reviewed.tsv"
  * output:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
* 48_weighted_actual_names_to_csv - Write weighted-actual name pairs for human review (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
  * output:
    * hr_train_weighted_actuals: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-{size}-weighted-actuals.csv"
    * hr_test_weighted_actuals: "s3://familysearch-names/processed/tree-hr-{given_surname}-test-weighted-actuals.csv"
* 49_augment_dataset - Augment the training dataset with other matching pairs based upon names having the same code or levenshtein similarity
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
  * output:
    * hr_train_augments: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-augments.csv.gz"
    * hr_train_augmented: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-augmented.csv.gz"
* 50_autoencoder - Generate an autoencoder model based upon similar name pairs (not used)
* 51_autoencoder_triplet - Generate a triplet-loss model based upon the autoencoder and near-negatives (not used)
* 52_glove - Generate a glove model (not used)
* 60_swivel_tune - Run hyperparameter tuning on swivel model (optional)
  * input:
    * hr_train_augmented: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-augmented.csv.gz"
* 61_swivel - Train a swivel model (takes a long, long, long time)
  * input:
    * hr_train_augmented: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-augmented.csv.gz"
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
  * output:
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
* 62_swivel_encoder_tune - Tune a swivel-based encoder model (not used)
* 63_swivel_encoder - Train a swivel-based encoder model (not used)
* 64_analyze_scores - Compare swivel and levenshtein scores (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
* 65_tfidf - Train a TfidfVectorizer to filter names sent to levenshtein
  * input:
    * hr_train_augmented: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-augmented.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
  * output:
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
* 66_ensemble - Train an ensemble model over swivel + levenshtein
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
  output:
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
* 70_compare_similarity - Compare the ensemble model to levenshtein and other floating-score algorithms (optional)
  * input:
    * hr_train_augmented: "s3://familysearch-names/processed/tree-hr-{given_surname}-train-augmented.csv.gz"
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
* 71_analyze_embeddings - Visualize swivel vectors in 2d space (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
* 72_analyze_names - Analyze name frequencies and codes (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
* 80_cluster_tune - Run hyperparameter tuning on clustering model (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
* 81_cluster - Train a clustering model
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * hr_aggr: "s3://familysearch-names-private/hr-preferred-{given_surname}-aggr.csv.gz"
      * comes from FamilySearch, not nama 
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
  * output:
    * clusters: "s3://nama-data/data/models/fs-{given_surname}-cluster-names.csv"
    * cluster_partitions: "s3://nama-data/data/models/fs-{given_surname}-cluster-partitions.csv"
* 82_cluster_levenshtein - Not sure what this does. Very out of date. (ignore)
* 90_compare_clusters - Compare our clusters to the clusters formed by various coders (soundex, nysiis, etc.) (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
    * clusters: "s3://nama-data/data/models/fs-{given_surname}-cluster-names.csv"
    * cluster_partitions: "s3://nama-data/data/models/fs-{given_surname}-cluster-partitions.csv"
* 91_compare_clusters_old_algo - Get Statistics for the old clusters (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
* 92_compare_oov_approaches - Compare our approach to handling out of vocab names to four simpler approaches (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * hr_test: "s3://familysearch-names/processed/tree-hr-{given_surname}-test.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
    * clusters: "s3://nama-data/data/models/fs-{given_surname}-cluster-names.csv"
    * cluster_scores: "s3://nama-data/data/processed/fs-{given_surname}-cluster-scores-{vocab_size}-{embed_dim}-precomputed.jsonl.gz"
      * see 99_precompute
    * hr_aggr: "s3://familysearch-names/interim/tree-hr-{given_surname}-aggr.parquet"
* 97_analyze_nicknames - Review nicknames (optional)
  * input:
    * nicknames: "../references/givenname_nicknames.csv"
* 98_given_surname_freq - Generate how likely a name is to be a given vs surname (optional)
  * input:
    * pref_aggr[given]: "s3://familysearch-names/processed/tree-preferred-given-aggr.csv.gz"
    * pref_aggr[surname]: "s3://familysearch-names/processed/tree-preferred-surname-aggr.csv.gz"
  * output:
    * given_surname_freq: "s3://familysearch-names/processed/tree-preferred-given-surname-freq.csv.gz"
* 99_precompute - Pre-compute embeddings and cluster-scores so they can be cached (optional)
  * input:
    * hr_train: "s3://familysearch-names/processed/tree-hr-{given_surname}-train.csv.gz"
    * pref_aggr: "s3://familysearch-names/processed/tree-preferred-{given_surname}-aggr.csv.gz"
    * swivel_vocab: "s3://nama-data/data/models/fs-{given_surname}-swivel-vocab-{vocab_size}-augmented.csv"
    * swivel_model: "s3://nama-data/data/models/fs-{given_surname}-swivel-model-{vocab_size}-{embed_dim}-augmented.pth"
    * tfidf_model: "s3://nama-data/data/models/fs-{given_surname}-tfidf.joblib"
    * ensemble_model: "s3://nama-data/data/models/fs-{given_surname}-ensemble-model-{vocab_size}-{embed_dim}-augmented-{negative_multiplier}.joblib"
    * clusters: "s3://nama-data/data/models/fs-{given_surname}-cluster-names.csv"
  * output:
    * cluster_scores: "s3://nama-data/data/processed/fs-{given_surname}-cluster-scores-{vocab_size}-{embed_dim}-precomputed.jsonl.gz"
    * embeddings: "s3://nama-data/data/processed/fs-{given_surname}-embeddings-{vocab_size}-{embed_dim}-precomputed.jsonl.gz"
    * cluster_embeddings: "s3://nama-data/data/processed/fs-{given_surname}-cluster-embeddings-{vocab_size}-{embed_dim}-precomputed.jsonl.gz"

### Server

The server is currently a out of date.

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
    │   ├── data           <- Functions to download or generate data
    │   ├── eval           <- Functions to evaluate models
    │   ├── features       <- Functions to turn raw data into features for modeling
    │   ├── models         <- Functions to train models and then use trained models to make predictions
    │   ├── server         <- Simple server
    │   └── visualization  <- empty for now
    ├── tests              <- Tests for source code
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
