import os

# make sure an openai api key is set
if "OPENAI_API_KEY" not in os.environ:
    # warn the user and set the key to a fake
    print("OPENAI_API_KEY not set, setting to fake key")
    os.environ["OPENAI_API_KEY"] = "fake_key"

from tqdm import tqdm
from typing import Union, List, Dict, Optional

from neuron_explainer.activations.activations import (
    ActivationRecordSliceParams,
)
from neuron_explainer.explanations.explainer import (
    TokenActivationPairExplainer,
    HARMONY_V4_MODELS,
)

HARMONY_V4_MODELS.append("meta-llama/Meta-Llama-3-8B-Instruct")
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.calibrated_simulator import (
    UncalibratedNeuronSimulator,
)
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.activations.activation_records import calculate_max_activation
from neuron_explainer.explanations.scoring import simulate_and_score

from .dataclasses import NeuronRecords
from .local_api_client import LocalApiClient


def _create_explainer(
    model_name: str,
    prompt_format: PromptFormat,
) -> TokenActivationPairExplainer:
    explainer = TokenActivationPairExplainer(
        model_name=model_name,
        prompt_format=prompt_format,
        max_concurrent=1,
    )

    if model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        # replace with our client
        explainer.client = LocalApiClient(
            model_name=model_name,
            max_concurrent=1,
            cache=False,
        )

    return explainer


def _create_simulator(
    model_name: str,
    prompt_format: PromptFormat,
    explanation: str,
) -> ExplanationNeuronSimulator:
    # Simulates neuron behavior based on an explanation.
    explain_simulator = ExplanationNeuronSimulator(
        model_name,
        explanation,
        max_concurrent=1,
        prompt_format=prompt_format,
    )

    if model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        # replace with our client
        # TODO can our model handle PromptFormat.INSTRUCTION_FOLLOWING?
        explain_simulator.api_client = LocalApiClient(
            model_name=model_name,
            max_concurrent=1,
            cache=False,
        )

    # initialize simulator
    # Pass through the activations without trying to calibrate.
    return UncalibratedNeuronSimulator(explain_simulator)


async def interpret_neuron_record(
    neuron_record,
    n_examples_per_split: int = 5,
    simulator_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",  # "text-davinci-003",
    simulator_prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
    # either explainer or explainer_model_name should be provided
    explainer: Optional[TokenActivationPairExplainer] = None,
    explainer_model_name: Optional[
        str
    ] = "meta-llama/Meta-Llama-3-8B-Instruct",  # "gpt-4",
    explainer_prompt_format: Optional[PromptFormat] = PromptFormat.HARMONY_V4,
) -> Dict[str, Union[int, str, float]]:
    """
    Interpret the features of the model for a single neuron

    Args:
        neuron_record: NeuronRecord
            The neuron record to interpret
        n_examples_per_split: int
            Number of examples to use for each split
        simulator_model_name: str
            The model name to use for the simulator
        simulator_prompt_format: PromptFormat
            The prompt format to use for the simulator (as the simulator model might require a different prompt format)
        explainer: Optional[TokenActivationPairExplainer]
            The explainer to use (if not provided, explainer_model_name should be provided)
        explainer_model_name: Optional[str]
        explainer_prompt_format: Optional[PromptFormat]

    Returns:
        Dict[str, Union[int, str, float]]: The interpretation result
        will have keys: "neuron_id", "explanation", "score", "activation_count", "layer_index"
    """

    if explainer is None:
        assert (
            explainer_model_name is not None
        ), "Either explainer or explainer_model_name should be provided"
        explainer = _create_explainer(explainer_model_name, explainer_prompt_format)

    # info about which feature we are evaluating
    neuron_id: int = neuron_record.neuron_id.neuron_index
    layer_index: int = neuron_record.neuron_id.layer_index
    activation_count: int = neuron_record.activation_count

    # Grab the activation records we'll need.
    slice_params = ActivationRecordSliceParams(
        n_examples_per_split=n_examples_per_split
    )
    train_activation_records = neuron_record.train_activation_records(
        activation_record_slice_params=slice_params
    )
    valid_activation_records = neuron_record.valid_activation_records(
        activation_record_slice_params=slice_params
    )
    prompt_kwargs = {
        "all_activation_records": train_activation_records,
        "max_tokens_for_completion": 60,
        "max_activation": calculate_max_activation(train_activation_records),
    }
    explainer_prompt = explainer.make_explanation_prompt(**prompt_kwargs)
    # rename max_tokens_for_completion to max_tokens in prompt_kwargs
    # generate_explanations(uses different kwargs than make_explanation_prompt)
    prompt_kwargs["max_tokens"] = prompt_kwargs.pop("max_tokens_for_completion")
    explanations = await explainer.generate_explanations(num_samples=1, **prompt_kwargs)

    if len(explanations) > 1:
        print(f"Multiple explanations for neuron {neuron_id}")
        print("Taking the first explanation")
        print(explanations)

    explanation = explanations[0]

    simulator = _create_simulator(
        simulator_model_name, simulator_prompt_format, explanation
    )

    scored_simulation = await simulate_and_score(simulator, valid_activation_records)
    score = scored_simulation.get_preferred_score()

    result = {
        "neuron_id": neuron_id,
        "layer_index": layer_index,
        "explanation": explanation,
        "score": score,
        "activation_count": activation_count,
        "explainer_prompt": explainer_prompt,
    }

    return result


async def interpret_neuron_records(
    neuron_records: NeuronRecords,
    top_k: int = 20,
    eval_num_features: int = 150,
    explainer_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",  # "gpt-4",
    explainer_prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
    simulator_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",  # "text-davinci-003",
    simulator_prompt_format: PromptFormat = PromptFormat.HARMONY_V4,
) -> List[Dict[str, Union[int, str, float]]]:
    """
    Interpret the features of the model for multiple neurons

    Args:
        neuron_records: NeuronRecords
        top_k: int
            The minimum number of times a neuron should be activated to be considered
        eval_num_features: int
            The number of features to evaluate
        explainer_model_name: str
            The model name to use for the explainer
        explainer_prompt_format: PromptFormat
            The prompt format to use for the explainer (as the explainer model might require a different prompt format)
        simulator_model_name: str
            The model name to use for the simulator
        simulator_prompt_format: PromptFormat
            The prompt format to use for the simulator (as the simulator model might require a different prompt format)

    Returns:
        List[Dict[str, Union[int, str, float]]]: The interpretation results
        Each result will have keys: "neuron_id", "explanation", "score", "activation_count"
    """
    neuron_records = neuron_records.records
    assert (
        len(neuron_records) > eval_num_features
    ), "Number of features should be greater than eval_num_features"

    # initialize explainer
    explainer = _create_explainer(explainer_model_name, explainer_prompt_format)

    if explainer_model_name in ["meta-llama/Meta-Llama-3-8B-Instruct"]:
        # replace with our client
        explainer.client = LocalApiClient(
            model_name=explainer_model_name,
            max_concurrent=1,
            cache=False,
        )

    # sort by activation count
    neuron_records.sort(key=lambda x: x.activation_count, reverse=True)
    # drop all that are not activated at least top_k times
    neuron_records = [x for x in neuron_records if x.activation_count > top_k]

    assert len(neuron_records) > eval_num_features, "Not enough features to evaluate"

    # select records (take high frequency neurons & low frequency neurons)
    neuron_records_to_evaluate = (
        neuron_records[: eval_num_features // 2]
        + neuron_records[-eval_num_features // 2 :]
    )

    results = []

    for neuron_record in tqdm(neuron_records_to_evaluate, desc="Interpreting Features"):
        try:
            result = await interpret_neuron_record(
                neuron_record,
                n_examples_per_split=5,
                simulator_model_name=simulator_model_name,
                simulator_prompt_format=simulator_prompt_format,
                explainer=explainer,
            )
            results.append(result)
        except Exception as e:
            print(f"Error for neuron {neuron_record.neuron_id}:")
            print(e)
            continue

    return results
