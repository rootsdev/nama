from src.eval.encoder import eval_encoder
from src.models.utils import build_token_idx_maps, convert_names_to_model_inputs, get_best_matches


def eval_triplet(triplet_model, input_names, candidate_names, k, batch_size=512):
    MAX_NAME_LENGTH = 30
    char_to_idx_map, idx_to_char_map = build_token_idx_maps()

    # Get embeddings for input names
    input_names_X, _ = convert_names_to_model_inputs(input_names, char_to_idx_map, MAX_NAME_LENGTH)
    input_names_encoded = eval_encoder(triplet_model, input_names_X, batch_size)

    # Get embeddings for candidate names
    candidate_names_X, _ = convert_names_to_model_inputs(candidate_names, char_to_idx_map, MAX_NAME_LENGTH)
    candidate_names_encoded = eval_encoder(triplet_model, candidate_names_X, batch_size)

    return get_best_matches(
        input_names_encoded, candidate_names_encoded, candidate_names, num_candidates=k, metric="euclidean"
    )
