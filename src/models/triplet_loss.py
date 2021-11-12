from typing import List, Tuple

import numpy as np
import random
import torch
from collections import defaultdict
from tqdm import tqdm, trange

from src.models import utils
from src.eval import metrics
from src.eval.encoder import eval_encoder


def get_near_negatives(
    input_names: list, weighted_actual_names_list: List[List[Tuple[str, float, int]]], candidate_names, k=50
):
    """
    Return near-negatives for all input names
    :param input_names: list of names to get near-negatives for
    :param weighted_actual_names_list: list of lists of [name2, weight, ?] for each name1 in input_names
    :param candidate_names: other names that we can pull near-negatives from
    :param k: how many near-negatives to return
    :return: dict of name -> list of near-negative names
    """
    all_names_unpadded = set([utils.remove_padding(name) for name in (input_names + candidate_names.tolist())])
    near_negatives = defaultdict(list)
    for name, positives in tqdm(zip(input_names, weighted_actual_names_list), total=len(input_names)):
        positive_names = set(utils.remove_padding(n) for n, _, _ in positives)
        near_negatives[name] = [
            utils.add_padding(n)
            for n in utils.get_k_near_negatives(utils.remove_padding(name), positive_names, all_names_unpadded, k)
        ]
    return near_negatives


class TripletDataLoader:
    """
    Data loader for triplet loss: return triplets: (anchor name tensor, positive name tensor, near-negative name tensor)
    """

    def __init__(
        self,
        input_names,
        weighted_actual_names_list,
        near_negatives,
        char_to_idx_map,
        max_name_length,
        batch_size,
        shuffle,
    ):
        name_pairs = []
        for input_name, positives in zip(input_names, weighted_actual_names_list):
            for pos_name, _, _ in positives:
                name_pairs.append([input_name, pos_name])
        self.name_pairs = np.array(name_pairs)
        self.near_negatives = near_negatives
        self.char_to_idx_map = char_to_idx_map
        self.max_name_length = max_name_length
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.name_pairs)
        self.ix = 0
        return self

    def __next__(self):
        if self.ix >= self.name_pairs.shape[0]:
            raise StopIteration
        # return a batch of input_names, pos_names, neg_names
        input_names = self.name_pairs[self.ix : self.ix + self.batch_size, 0]
        pos_names = self.name_pairs[self.ix : self.ix + self.batch_size, 1]
        neg_names = np.apply_along_axis(
            lambda row: np.array(random.choice(self.near_negatives[row[0]]), object), 1, input_names.reshape(-1, 1)
        )
        # not very efficient to re-convert over and over, but it's convenient to do it here
        input_tensor, _ = utils.convert_names_to_model_inputs(input_names, self.char_to_idx_map, self.max_name_length)
        pos_tensor, _ = utils.convert_names_to_model_inputs(pos_names, self.char_to_idx_map, self.max_name_length)
        neg_tensor, _ = utils.convert_names_to_model_inputs(neg_names, self.char_to_idx_map, self.max_name_length)
        self.ix += self.batch_size
        return input_tensor, pos_tensor, neg_tensor


def train_triplet_loss(
    model,
    input_names_train,
    weighted_actual_names_list_train,
    near_negatives_train,
    input_names_test,
    weighted_actual_names_list_test,
    candidate_names_test,
    candidate_names_train,
    candidate_names_all,
    char_to_idx_map,
    max_name_length,
    num_epochs=100,
    batch_size=128,
    margin=0.1,
    k=100,
    device="cpu",
):
    """
    Train triplet loss
    :param model: from autoencoder
    :param input_names_train: list of names (name1)
    :param weighted_actual_names_list_train: list of lists of (name, weight, frequency) for each name in input_names
    :param near_negatives_train: list of lists of near-negative names for each name in input_names
    :param input_names_test: list of names for testing
    :param weighted_actual_names_list_test: actuals for testing
    :param candidate_names_test: candidate names (name2) for testing
    :param candidate_names_train: candidate names (name2) for training
    :param candidate_names_all: all candidate names (name2)
    :param char_to_idx_map: map characters to ids
    :param max_name_length: max name length
    :param num_epochs: number of epochs to train
    :param batch_size: batch size
    :param margin: margin between positive and negative similarity
    :param k: when reporting AUC, calculate for this many candidates
    :param device: cpu or gpu device
    """
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)

    candidate_names_train_X, _ = utils.convert_names_to_model_inputs(
        candidate_names_train, char_to_idx_map, max_name_length
    )
    input_names_test_X, _ = utils.convert_names_to_model_inputs(input_names_test, char_to_idx_map, max_name_length)
    candidate_names_test_X, _ = utils.convert_names_to_model_inputs(
        candidate_names_test, char_to_idx_map, max_name_length
    )

    data_loader = TripletDataLoader(
        input_names_train,
        weighted_actual_names_list_train,
        near_negatives_train,
        char_to_idx_map,
        max_name_length,
        batch_size,
        shuffle=True,
    )

    model.to(device)
    model.device = device

    with trange(num_epochs) as pbar:
        for _ in pbar:
            # train on gpu if there is one
            model.train()
            losses = []
            for i, (anchor_tensor, pos_tensor, neg_tensor) in enumerate(data_loader):
                # Clear out gradient
                model.zero_grad()

                # Compute forward pass
                anchor_encoded = model(anchor_tensor, just_encoder=True)
                pos_encoded = model(pos_tensor, just_encoder=True)
                neg_encoded = model(neg_tensor, just_encoder=True)

                loss = loss_fn(anchor_encoded, pos_encoded, neg_encoded)
                losses.append(loss.detach().item())
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                candidate_names_train_encoded = eval_encoder(model, candidate_names_train_X, batch_size)
                input_names_test_encoded = eval_encoder(model, input_names_test_X, batch_size)
                candidate_names_test_encoded = eval_encoder(model, candidate_names_test_X, batch_size)
                candidate_names_all_encoded = np.vstack((candidate_names_train_encoded, candidate_names_test_encoded))
                best_matches = utils.get_best_matches(
                    input_names_test_encoded,
                    candidate_names_all_encoded,
                    candidate_names_all,
                    num_candidates=k,
                    metric="euclidean",
                )
            # calc test AUC
            auc = metrics.get_auc(
                weighted_actual_names_list_test,
                best_matches,
                min_threshold=0.0,
                max_threshold=10.0,
                step=0.01,
                distances=True,
            )
            print("test AUC", auc)

            # Update loss value on progress bar
            pbar.set_postfix({"loss": sum(losses) / len(losses), "auc": auc})
