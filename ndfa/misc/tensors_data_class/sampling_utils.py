__author__ = "Elad Nachmias"
__email__ = "eladnah@gmail.com"
__date__ = "2021-12-21"

import torch
import dataclasses
from typing import List, Optional, Collection

import numpy as np

from .misc import get_random_seed_per_example, CollateData
from .tensors_data_class import TensorsDataClass
from ndfa.nn_utils.modules.params.sampling_params import SamplingParams, DistributionInfoParams  # TODO: put this in TensorsDataClass module


__all__ = ['apply_sampling_on_inputs']


def sample_by_distribution_params(distribution: DistributionInfoParams, rng: np.random.RandomState):
    if distribution.distribution_type == DistributionInfoParams.DistributionType.Constant:
        return distribution.distribution_params[0]
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Normal:
        return rng.normal(*distribution.distribution_params)
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Gamma:
        return rng.gamma(*distribution.distribution_params)
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Exponential:
        return rng.exponential(*distribution.distribution_params)
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Uniform:
        return rng.uniform(*distribution.distribution_params)
    else:
        raise ValueError(f'Unsupported distribution type `{distribution.distribution_type}`.')


def make_distribution_closer_to_one(
        distribution: DistributionInfoParams, factor: float) -> DistributionInfoParams:
    if distribution.distribution_type == DistributionInfoParams.DistributionType.Constant:
        assert len(distribution.distribution_params) == 1
        initial_value, = distribution.distribution_params
        assert distribution.distribution_params[0] < 1. + 2. * np.finfo(float).eps
        cur_value = min(1., initial_value + (1 - initial_value) * (1. - factor))
        return dataclasses.replace(distribution, distribution_params=(cur_value,))
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Normal:
        assert len(distribution.distribution_params) == 2
        initial_mu, initial_sigma = distribution.distribution_params
        assert initial_mu < 1. + 2. * np.finfo(float).eps
        assert initial_sigma > 0
        cur_mu = min(1., initial_mu + (1 - initial_mu) * (1. - factor))
        # cur_sigma = max(distribution.distribution_params[1] * factor, np.finfo(float).eps)
        cur_sigma = max(initial_sigma * ((1 - cur_mu) / (1 - initial_mu)) ** 0.5, np.finfo(float).eps)
        return dataclasses.replace(distribution, distribution_params=(cur_mu, cur_sigma))
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Gamma:
        raise NotImplementedError  # TODO: impl!
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Exponential:
        raise NotImplementedError  # TODO: impl!
    elif distribution.distribution_type == DistributionInfoParams.DistributionType.Uniform:
        raise NotImplementedError  # TODO: impl!
    else:
        raise ValueError(f'Unsupported distribution type `{distribution.distribution_type}`.')


def sample_sample_size_by_distribution_params_with_decay(
        total_nr_items: int, sampling_distribution_params: DistributionInfoParams,
        sample_rate_decay_factor: Optional[float], rng: np.random.RandomState) -> int:
    if sample_rate_decay_factor is not None:
        sampling_distribution_params = make_distribution_closer_to_one(
            distribution=sampling_distribution_params, factor=sample_rate_decay_factor)
    sampling_rate = max(1 / total_nr_items, min(1, sample_by_distribution_params(
        distribution=sampling_distribution_params, rng=rng)))
    nr_items_to_sample = max(1, min(total_nr_items, round(sampling_rate * total_nr_items)))
    return nr_items_to_sample


def apply_sampling_on_inputs(
        inputs: List[TensorsDataClass],
        field_names: Collection[str],
        sequences_per_example_sampling,
        sequences_sampling_initial_seed_salt: str,
        collate_data: CollateData) -> List[TensorsDataClass]:
    if sequences_per_example_sampling is None or \
            not (collate_data.is_training or sequences_per_example_sampling.sample_in_eval):
        return inputs
    assert sequences_per_example_sampling.min_nr_items_to_sample_by_rate is None or \
           sequences_per_example_sampling.max_nr_items is None or \
           sequences_per_example_sampling.min_nr_items_to_sample_by_rate <= \
           sequences_per_example_sampling.max_nr_items
    random_seed_per_example = get_random_seed_per_example(
        batch_dependent_seed=True,
        example_dependent_seed=True,
        initial_seed_salt=sequences_sampling_initial_seed_salt,
        collate_data=collate_data)
    fixed_inputs_dicts = []
    for example_idx, inp in enumerate(inputs):
        fixed_input_dict = {}
        fixed_inputs_dicts.append(fixed_input_dict)
        for field_name in field_names:
            tensors = getattr(inp, field_name)
            nr_tensors = tensors.size(0) if isinstance(tensors, torch.Tensor) else len(tensors)
            if nr_tensors < 1:
                continue
            if sequences_per_example_sampling.min_nr_items_to_sample_by_rate is not None and \
                    nr_tensors < sequences_per_example_sampling.min_nr_items_to_sample_by_rate:
                continue
            random_state = np.random.RandomState(random_seed_per_example[example_idx])
            nr_sequences_to_sample_per_example = nr_tensors
            if sequences_per_example_sampling.distribution_for_rate_to_sample_by is not None:
                decay_factor = sequences_per_example_sampling.sample_rate_train_decay_factor
                sample_rate_decay_factor = \
                    None if decay_factor is None or not collate_data.is_training else \
                        (1 - decay_factor) ** (collate_data.train_progress_info.epoch_nr - 1)
                sample_sample_size_by_distribution_params_with_decay(
                    total_nr_items=nr_tensors,
                    sampling_distribution_params=
                    sequences_per_example_sampling.distribution_for_rate_to_sample_by,
                    sample_rate_decay_factor=sample_rate_decay_factor,
                    rng=random_state)
                sampling_rate = max(1 / nr_tensors, min(1, sample_by_distribution_params(
                    distribution=sequences_per_example_sampling.distribution_for_rate_to_sample_by,
                    rng=random_state)))
                nr_sequences_to_sample_per_example = max(1, min(nr_tensors, round(sampling_rate * nr_tensors)))
            if sequences_per_example_sampling.max_nr_items is not None:
                nr_sequences_to_sample_per_example = \
                    min(nr_sequences_to_sample_per_example, sequences_per_example_sampling.max_nr_items)
            if nr_tensors > nr_sequences_to_sample_per_example:
                sampled_items_indices = random_state.choice(
                    nr_tensors, size=nr_sequences_to_sample_per_example, replace=False)
                if isinstance(tensors, torch.Tensor):
                    sampled_items = torch.index_select(
                        tensors, 0, torch.LongTensor(sampled_items_indices))  # TODO: check!
                    assert sampled_items.shape[1:] == tensors.shape[1:]
                    assert sampled_items.size(0) == nr_sequences_to_sample_per_example
                else:
                    assert isinstance(tensors, (list, tuple))
                    sampled_items = [tensors[index] for index in sampled_items_indices]
                    assert len(sampled_items) == nr_sequences_to_sample_per_example
                fixed_input_dict[field_name] = sampled_items
    sampled_inputs = [
        dataclasses.replace(orig_inp, **fixed_inp_dict)
        for orig_inp, fixed_inp_dict in zip(inputs, fixed_inputs_dicts)]
    return sampled_inputs
