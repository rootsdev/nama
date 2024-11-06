from collections import Counter, defaultdict

import cologne_phonetics
import jellyfish
import pandas as pd
import phonetics
from mpire import WorkerPool
from pyphonetics import RefinedSoundex

from nama.data.match import levenshtein_similarity
from nama.models.utils import remove_padding


def get_tree2records_record2trees_name_counts(input_names, record_name_frequencies):
    """construct name_counts, tree2records, and record2trees"""
    # TODO once we have removed Clorinda's name pairs from training, don't exclude frequent but dissimilar names anymore
    name_counts = Counter()
    for input_name, rnfs in zip(input_names, record_name_frequencies):
        for name, frequency in rnfs:
            name_counts[input_name] += frequency
            name_counts[name] += frequency
    freq_names = {name for name, _ in name_counts.most_common(5000)}

    tree2records = defaultdict(Counter)
    record2trees = defaultdict(Counter)
    for input_name, rnfs in zip(input_names, record_name_frequencies):
        input_name_no_padding = remove_padding(input_name)
        if len(input_name_no_padding) == 0:
            continue
        for name, frequency in rnfs:
            name_no_padding = remove_padding(name)
            if len(name_no_padding) == 0:
                continue
            # exclude frequent dissimilar pairs
            if (
                input_name in freq_names
                and name in freq_names
                and levenshtein_similarity(input_name_no_padding, name_no_padding) <= 0.65
            ):
                continue
            tree2records[input_name][name] += frequency
            record2trees[name][input_name] += frequency
    return tree2records, record2trees, name_counts


#
# define strategies based upon co-occurrence (tree2records and record2trees)
#
def tree_record_strategy(name, tree2records, record2trees):
    """tree->name => tree<->name"""
    c = Counter()
    c += tree2records[name]
    c += record2trees[name]
    return c


def get_weight(name, occurrence, name_counts):
    return occurrence / name_counts[name]


def tree_record_pair_strategy(name, tree2records, record2trees, name_counts, threshold=0.0, score=1):
    """tree->name, tree->alt_name => name<->alt_name"""
    c = Counter()
    for tree_name, occurrence in record2trees[name].items():
        if get_weight(name, occurrence, name_counts) < threshold:
            continue
        for alt_name, alt_occurrence in tree2records[tree_name].items():
            if get_weight(tree_name, alt_occurrence, name_counts) < threshold:
                continue
            if alt_name != name:
                c[alt_name] += score
    return c


def record_tree_pair_strategy(name, tree2records, record2trees, name_counts, threshold=0, score=1):
    """name->record, alt_name->record => name<->alt_name"""
    c = Counter()
    for record_name, occurrence in tree2records[name].items():
        if get_weight(name, occurrence, name_counts) < threshold:
            continue
        for alt_name, alt_occurrence in record2trees[record_name].items():
            if get_weight(record_name, alt_occurrence, name_counts) < threshold:
                continue
            if alt_name != name:
                c[alt_name] += score
    return c


def tree_record_record_strategy(name, tree2records, record2trees, name_counts, threshold=0, score=1):
    """name->record, record(as tree name)->alt_record => name<->alt_record"""
    c = Counter()
    for record_name, occurrence in tree2records[name].items():
        if get_weight(name, occurrence, name_counts) < threshold:
            continue
        for alt_name, alt_occurrence in tree2records[record_name].items():
            if get_weight(record_name, alt_occurrence, name_counts) < threshold:
                continue
            if alt_name != name:
                c[alt_name] += score
    for tree_name, occurrence in record2trees[name].items():
        if get_weight(name, occurrence, name_counts) < threshold:
            continue
        for alt_name, alt_occurrence in record2trees[tree_name].items():
            if get_weight(tree_name, alt_occurrence, name_counts) < threshold:
                continue
            if alt_name != name:
                c[alt_name] += score
    return c


#
# define strategies based upon coders
#
def get_codes(coder, names, multiple_codes=False):
    codes = defaultdict(list)
    for name in names:
        result = coder(remove_padding(name))
        if multiple_codes:
            for code in result:
                if code:
                    codes[code].append(name)
        else:
            codes[result].append(name)
    return codes


def get_code_matches(name, coder, codes, multiple_codes=False):
    result = coder(remove_padding(name))
    if multiple_codes:
        names = set()
        for code in result:
            if code:
                names.update(codes[code])
        return list(names)
    else:
        return codes[result]


def code_strategy(name, coder, codes, multiple_codes=False, score=1):
    c = Counter()
    for alt_name in get_code_matches(name, coder, codes, multiple_codes=multiple_codes):
        if alt_name != name:
            c[alt_name] = score
    return c


#
# define strategy based upon levenshtein
#
def get_levenshtein_matches(name, names, threshold=0.65):
    matches = {}
    name = remove_padding(name)
    for n in names:
        variant = remove_padding(n)
        score = levenshtein_similarity(name, variant)
        if score >= threshold:
            matches[n] = score
    return matches


def lev_strategy(name, names, threshold, score=1):
    c = Counter()
    for alt_name, _ in get_levenshtein_matches(name, names, threshold).items():
        if alt_name != name:
            c[alt_name] = score
    return c


def print_coder_stats(coder_name, codes):
    print(coder_name, len(codes), sum(len(v) for v in codes.values()))


def add_variants(variants_for_name, learner_name, variants):
    """add the variants generated by this learner"""
    for variant in variants:
        variants_for_name[variant][learner_name] = 1


def init_learners(all_learners):
    def fn():
        return {learner: 0 for learner in all_learners}

    return fn


def augment_dataset_batch(shared, name_batch, _):
    all_learners, tree2records, record2trees, name_counts, all_names, threshold, discount = shared
    init_dict = init_learners(all_learners)

    # get codes (repeats for each batch, but its pretty fast and may be better than passing the codes around)
    # caverphone = CaverphoneTwo()
    refined_soundex = RefinedSoundex()
    cologne = lambda n: [result[1] for result in cologne_phonetics.encode(n)]

    nysiis_codes = get_codes(jellyfish.nysiis, all_names)
    # caverphone_codes = get_codes(caverphone._pre_process, all_names)
    refined_soundex_codes = get_codes(refined_soundex.phonetics, all_names)
    dmetaphone_codes = get_codes(phonetics.dmetaphone, all_names, True)
    cologne_codes = get_codes(cologne, all_names, True)
    metaphone_codes = get_codes(jellyfish.metaphone, all_names)

    name_pairs = []
    for name in name_batch:
        variants_for_name = defaultdict(init_dict)

        # add variants for learners based upon the various strategies
        add_variants(
            variants_for_name,
            "tree_record_pair_1e-3",
            tree_record_pair_strategy(
                name, tree2records=tree2records, record2trees=record2trees, name_counts=name_counts, threshold=1e-3
            ),
        )
        add_variants(variants_for_name, "code_nysiis", code_strategy(name, coder=jellyfish.nysiis, codes=nysiis_codes))

        # add_variants(variants_for_name,
        #              "code_caverphone",
        #              code_strategy(name, coder=caverphone._pre_process, codes=caverphone_codes))

        add_variants(
            variants_for_name,
            "code_refined_soundex",
            code_strategy(name, coder=refined_soundex.phonetics, codes=refined_soundex_codes),
        )

        add_variants(
            variants_for_name,
            "code_dmetaphone",
            code_strategy(name, coder=phonetics.dmetaphone, codes=dmetaphone_codes, multiple_codes=True),
        )

        add_variants(
            variants_for_name,
            "code_cologne",
            code_strategy(name, coder=cologne, codes=cologne_codes, multiple_codes=True),
        )

        add_variants(
            variants_for_name, "code_metaphone", code_strategy(name, coder=jellyfish.metaphone, codes=metaphone_codes)
        )

        add_variants(variants_for_name, "lev_65", lev_strategy(name, names=all_names, threshold=0.65))

        add_variants(variants_for_name, "lev_70", lev_strategy(name, names=all_names, threshold=0.70))

        add_variants(variants_for_name, "lev_75", lev_strategy(name, names=all_names, threshold=0.75))

        add_variants(variants_for_name, "lev_80", lev_strategy(name, names=all_names, threshold=0.80))

        add_variants(variants_for_name, "lev_85", lev_strategy(name, names=all_names, threshold=0.85))

        add_variants(variants_for_name, "lev_90", lev_strategy(name, names=all_names, threshold=0.90))

        # gather all variants into name_pairs
        for variant, learners in variants_for_name.items():
            learners["tree_name"] = name
            learners["record_name"] = variant
            name_pairs.append(learners)

    # sum learners for each pair and use that as the frequency to add
    name_pairs_df = pd.DataFrame(name_pairs)
    name_pairs_df["frequency"] = name_pairs_df[all_learners].sum(axis=1)
    name_pairs_df = name_pairs_df[["tree_name", "record_name", "frequency"]]
    # filter out any below threshold
    if threshold != 0:
        name_pairs_df = name_pairs_df[name_pairs_df["frequency"] >= threshold]
    # subtract discount
    if discount != 0:
        name_pairs_df["frequency"] -= discount

    return name_pairs_df


def generate_augmented_name_pairs(
    input_names,
    record_name_frequencies,
    candidate_names,
    threshold=3,
    discount=1,
    batch_size=5000,
    n_jobs=0,
):
    all_names = {name for name in input_names if len(remove_padding(name)) > 0}.union({
        name for name in candidate_names if len(remove_padding(name)) > 0
    })

    # get tree2records, record2trees, name_counts
    tree2records, record2trees, name_counts = get_tree2records_record2trees_name_counts(
        input_names, record_name_frequencies
    )

    # apply variant-name strategies to freq_names to get possible variants
    all_learners = [
        "tree_record_pair_1e-3",
        # "tree_record_pair_1e-4",
        # "tree_record_pair_1e-5",
        # "record_tree_pair_1e-3",
        # "record_tree_pair_1e-4",
        # "record_tree_pair_1e-5",
        # "tree_record_record_1e-3",
        # "tree_record_record_1e-4",
        # "tree_record_record_1e-5",
        "code_nysiis",
        # "code_caverphone",
        "code_refined_soundex",
        "code_dmetaphone",
        "code_cologne",
        "code_metaphone",
        "lev_65",
        "lev_70",
        "lev_75",
        "lev_80",
        "lev_85",
        "lev_90",
    ]

    # construct batches
    all_names_list = list(all_names)
    name_batches = []
    for start in range(0, len(all_names_list), batch_size):
        # need a dummy second parameter to keep pool.map from iterating through the batch
        name_batches.append((all_names_list[start : start + batch_size], None))

    # get augmented name pairs
    with WorkerPool(
        n_jobs=n_jobs,
        shared_objects=(all_learners, tree2records, record2trees, name_counts, all_names, threshold, discount),
    ) as pool:
        results = pool.map(augment_dataset_batch, name_batches, progress_bar=True)

    return pd.concat(results)


def augment_dataset(names_df, augments_df, multiplier=4):
    names_df = names_df.copy()
    tree_names = set(names_df["tree_name"])

    # prefer existing tree_name's in the first position
    augment_tree_name = []
    augment_record_name = []
    augment_frequency = []
    print("Creating augments_df")
    for ix, (tree_name, record_name, frequency) in enumerate(
        zip(augments_df["tree_name"], augments_df["record_name"], augments_df["frequency"])
    ):
        if ix % 100_000_000 == 0:
            print("   ", ix)
        if tree_name not in tree_names and record_name in tree_names:
            tree_name, record_name = record_name, tree_name
        tree_names.add(tree_name)
        augment_tree_name.append(tree_name)
        augment_record_name.append(record_name)
        augment_frequency.append(frequency)
    augments_df = pd.DataFrame({
        "tree_name": augment_tree_name,
        "record_name": augment_record_name,
        "frequency": augment_frequency,
    })

    print("Multiply frequency by", multiplier)
    names_df["frequency"] *= multiplier
    print("Creating names_df")
    names_df = pd.concat([names_df, augments_df]).groupby(["tree_name", "record_name"]).sum().reset_index()
    return names_df
