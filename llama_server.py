import argparse
import time
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any, List, Optional
import json

app = Flask(__name__)

model = None
tokenizer = None
"""
SMAPLE:
client.completions.create(
    model="davinci-002",
    prompt="Translate this German text to English: 'Ich bin ein Berliner'",
    max_tokens=0,
    echo=True,
    logprobs=15
)

---->
Completion(
id='cmpl-9X4xTlUsVTVaaVBfZPnE8i2dmbcrD',
choices=[CompletionChoice(finish_reason='length', index=0, logprobs=Logprobs(text_offset=[0, 9, 14, 21, 26, 29, 37, 38, 40, 43, 47, 51, 58, 60], token_logprobs=[None, -3.0610795, -10.573809, -3.7063658, -2.365942, -0.5000815, -2.5796685, -6.3122582, -2.836824, -1.9714774, -1.942698, -2.1503136, -0.007854446, -0.9208847], tokens=['Translate', ' this', ' German', ' text', ' to', ' English', ':', " '", 'Ich', ' bin', ' ein', ' Berlin', 'er', "'"], top_logprobs=[None, {' this': -3.0610795, ' to': -2.3953376, '\n\n': -3.1084037, ' the': -3.252296, ' into': -3.4042563}, {' German': -10.573809, ' page': -0.33744776, ' for': -3.4311252, ' post': -3.5867524, ' text': -4.142465, ' Page': -4.1822867}, {' text': -3.7063658, ' to': -1.9002755, '-': -2.944812, ' translation': -3.597767, ' document': -3.655453, ' word': -3.6742814}, {' to': -2.365942, ' into': -0.62988764, ':': -3.0473557, '\n\n': -3.247099, ':\n\n': -3.255372}, {' English': -0.5000815, ':': -1.9808974, ':\n\n': -3.567329, ' another': -4.100801, ' an': -4.21462}, {':': -2.5796685, '\n\n': -1.4703956, ' (': -2.112596, ' Translate': -2.8901496, ':\n\n': -2.8972383}, {" '": -6.3122582, ' Ich': -2.7060394, ' "': -3.4053984, ' Hal': -3.6897955, ' Die': -3.8494797, ' (': -3.9004383}, {'Ich': -2.836824, 'Die': -3.452189, 'The': -3.6653285, 'Der': -3.7949643, 'Das': -3.8457198}, {' bin': -1.9714774, ' habe': -1.7523392, ' möchte': -2.9595213, ' kann': -3.2676687, ' suche': -3.8232946}, {' ein': -1.942698, ' der': -3.2705407, ' eine': -3.3548398, ' nicht': -3.3629804, ' sehr': -3.4843717}, {' Berlin': -2.1503136, ' De': -3.4964697, ' Mann': -3.686618, ' Mens': -3.823023, ' Schwe': -4.3781166}, {'er': -0.007854446, 'erin': -5.75041, "'": -7.487542, '-B': -7.867202, 'ner': -8.48324}, {"'": -0.9208847, "'\n\n": -1.8898875, ".'": -2.4000297, "'.": -2.7712417, ".'\n\n": -3.1839662}]), text="Translate this German text to English: 'Ich bin ein Berliner'")], created=1717670911, model='davinci-002', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=None, prompt_tokens=14, total_tokens=14)
)



"""
# sample api response for tlogporps=15 and max_tokens=0


# Completion(id='cmpl-9X4xTlUsVTVaaVBfZPnE8i2dmbcrD', choices=[CompletionChoice(finish_reason='length', index=0, logprobs=Logprobs(text_offset=[0, 9, 14, 21, 26, 29, 37, 38, 40, 43, 47, 51, 58, 60], token_logprobs=[None, -3.0610795, -10.573809, -3.7063658, -2.365942, -0.5000815, -2.5796685, -6.3122582, -2.836824, -1.9714774, -1.942698, -2.1503136, -0.007854446, -0.9208847], tokens=['Translate', ' this', ' German', ' text', ' to', ' English', ':', " '", 'Ich', ' bin', ' ein', ' Berlin', 'er', "'"], top_logprobs=[None, {' this': -3.0610795, ' to': -2.3953376, '\n\n': -3.1084037, ' the': -3.252296, ' into': -3.4042563}, {' German': -10.573809, ' page': -0.33744776, ' for': -3.4311252, ' post': -3.5867524, ' text': -4.142465, ' Page': -4.1822867}, {' text': -3.7063658, ' to': -1.9002755, '-': -2.944812, ' translation': -3.597767, ' document': -3.655453, ' word': -3.6742814}, {' to': -2.365942, ' into': -0.62988764, ':': -3.0473557, '\n\n': -3.247099, ':\n\n': -3.255372}, {' English': -0.5000815, ':': -1.9808974, ':\n\n': -3.567329, ' another': -4.100801, ' an': -4.21462}, {':': -2.5796685, '\n\n': -1.4703956, ' (': -2.112596, ' Translate': -2.8901496, ':\n\n': -2.8972383}, {" '": -6.3122582, ' Ich': -2.7060394, ' "': -3.4053984, ' Hal': -3.6897955, ' Die': -3.8494797, ' (': -3.9004383}, {'Ich': -2.836824, 'Die': -3.452189, 'The': -3.6653285, 'Der': -3.7949643, 'Das': -3.8457198}, {' bin': -1.9714774, ' habe': -1.7523392, ' möchte': -2.9595213, ' kann': -3.2676687, ' suche': -3.8232946}, {' ein': -1.942698, ' der': -3.2705407, ' eine': -3.3548398, ' nicht': -3.3629804, ' sehr': -3.4843717}, {' Berlin': -2.1503136, ' De': -3.4964697, ' Mann': -3.686618, ' Mens': -3.823023, ' Schwe': -4.3781166}, {'er': -0.007854446, 'erin': -5.75041, "'": -7.487542, '-B': -7.867202, 'ner': -8.48324}, {"'": -0.9208847, "'\n\n": -1.8898875, ".'": -2.4000297, "'.": -2.7712417, ".'\n\n": -3.1839662}]), text="Translate this German text to English: 'Ich bin ein Berliner'")], created=1717670911, model='davinci-002', object='text_completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=None, prompt_tokens=14, total_tokens=14))


# OpenAI API hits BadRequestError: Error code: 400 - {‘error’: {‘message’: “You requested a length 0 completion, but did not specify a ‘logprobs’ parameter to see the probability of each token. Either request a completion with length greater than 1, or set ‘logprobs’ to 0 (to see the probabilities of each token you submitted) or more (to also see the probabilities over alternative tokens).”, ‘type’: ‘invalid_request_error’, ‘param’: None, ‘code’: None}} [Error reference: https://platform.openai.com/docs/guides/error-codes/api-errors 9]


def pretty_print_messages(messages):
    for message in messages:
        print()
        print(f"------- {message['role']} -------")
        print(message["content"])
        print("----------------")


def pretty_print_json(response_object):
    # Convert the response object to a pretty formatted string
    pretty_response = json.dumps(response_object, indent=2)
    # Print the pretty formatted string
    print(pretty_response)


def load_model(model_id, load_in_4bit=False, load_in_8bit=False, device="cuda"):
    model_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    if load_in_4bit:
        model_kwargs["quantization_config"] = {
            "load_in_4bit": True,
            # "bnb_4bit_use_double_quant":True,
            # "bnb_4bit_quant_type":"nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
    elif load_in_8bit:
        model_kwargs["quantization_config"] = {"load_in_8bit": True}

    kwargs = {}
    if not load_in_4bit and not load_in_8bit:
        kwargs["device"] = device

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        **model_kwargs,
    )
    return model, tokenizer


def handle_gen_output(
    model,
    tokenizer,
    input_ids_len: int,
    sequence: torch.LongTensor,
    logprobs_count: int,
    echo: bool,
) -> Dict[str, Any]:
    response = {}

    if logprobs_count > 0:
        # pass through model again
        output = model(sequence.unsqueeze(0))
        # output is a transformers.modeling_outputs.CausalLMOutput
        logits = output.logits[0]
        # apply log_softmax
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        # the long probs of the actual tokens chosen
        # Starts with None
        token_logprobs: List[Optional[float]] = []
        token_logprobs.append(None)
        for i in range(1, len(sequence)):
            # we want to know the logit why it was selected, i.e. at step i -1
            # that is why we start at 1 and add a None before
            token_logprobs.append(logits[i - 1, sequence[i]].item())

        # the tokens chosen
        tokens: List[str] = []
        # compute the text offsets for each token
        # starts with 0
        text_offsets: List[int] = []
        text_offsets.append(0)
        for i in range(len(sequence)):
            decode_to_here = tokenizer.decode(
                sequence[: i + 1], skip_special_tokens=True
            )
            text_offsets.append(len(decode_to_here))
            tokens.append(decode_to_here[text_offsets[-2] : text_offsets[-1]])
        # remove the last text offset as is is just the length of the sequence
        text_offsets.pop(-1)

        # calcutae the top logprobs
        # Also Starts with None
        # key will be the token string representation and the value will be the log probability
        top_logprobs: List[Optional[Dict[str:float]]] = []
        top_logprobs.append(None)
        for i in range(1, len(sequence)):
            top_k = logits[i - 1].topk(logprobs_count)

            current_tok_logprobs = {}
            for k, v in zip(top_k.indices.tolist(), top_k.values.tolist()):
                # get the token str based on the fully decoded sequence

                ids_with_candidate_token = sequence.clone()
                ids_with_candidate_token[i] = k
                ids_with_candidate_token = ids_with_candidate_token[: i + 1]

                new_tok = tokenizer.decode(
                    ids_with_candidate_token, skip_special_tokens=True
                )
                new_tok = new_tok[text_offsets[i] :]

                current_tok_logprobs[new_tok] = v

            top_logprobs.append(current_tok_logprobs)

        response["logprobs"] = {
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "tokens": tokens,
            "text_offset": text_offsets,
        }

    if not echo:
        sequence = sequence[input_ids_len:]

        # also slice the logprobs
        if "logprobs" in response:
            response["logprobs"]["token_logprobs"] = response["logprobs"][
                "token_logprobs"
            ][input_ids_len:]
            response["logprobs"]["top_logprobs"] = response["logprobs"]["top_logprobs"][
                input_ids_len:
            ]
            response["logprobs"]["tokens"] = response["logprobs"]["tokens"][
                input_ids_len:
            ]
            response["logprobs"]["text_offset"] = response["logprobs"]["text_offsets"][
                input_ids_len:
            ]

    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    response["message"] = {
        "role": "assistant",  # TODO. should this be parsed?
        "content": generated_text,
    }

    return response


@torch.no_grad()
def handle(request):
    data = request.json
    #print("----HAndlinG____")
    #pretty_print_messages(data["messages"])
    messages = data["messages"]
    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 1.0)
    top_p = data.get("top_p", 1.0)
    logprobs_count = data.get("logprobs", 0)
    echo = data.get("echo", False)
    n = data.get("n", 1)

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    if max_tokens > 0:
        outputs = model.generate(
            input_ids,
            num_return_sequences=n,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        assert 0 == max_tokens
        outputs = input_ids.repeat(n, 1)

    # output is a transformers.generation.GenerateDecoderOnlyOutput
    # output.sequences is a longtensor of shape (n_samples, seqlen (input + output))

    # somehwat like: https://platform.openai.com/docs/api-reference/making-requests
    result = {
        "object": "text_completion",
        "created": int(time.time()),
        "model": model.config.name_or_path,
        "choices": [],
    }

    for i in range(n):
        # TODO: finish_reason
        choice = handle_gen_output(
            model,
            tokenizer,
            input_ids.shape[-1],
            outputs[i],
            logprobs_count,
            echo,
        )
        choice["index"] = i
        result["choices"].append(choice)

    return result


@app.route("/v1/completions", methods=["POST"])
def generate_text():
    try:
        result = handle(request)
        return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the text generation server.")
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Load the model in 4 bits."
    )
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Load the model in 8 bits."
    )
    parser.add_argument(
        "--test", action="store_true", help="Test a simple API call and shut down."
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run the model on (default: cuda)."
    )
    parser.add_argument(
        "--model_id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model ID to load (default: meta-llama/Meta-Llama-3-8B-Instruct).",
    )

    args = parser.parse_args()

    model, tokenizer = load_model(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device=args.device,
    )

    if args.test:
        test_payload = {
            "messages":
            # [
            #     {
            #         "role": "system",
            #         "content": "You are a pirate chatbot who always responds in pirate speak!",
            #     },
            #     {"role": "user", "content": "Who are you?"},
            # ],
            [
                {
                    "role": "user",
                    "content": "This is a test \t with a tab",
                }
            ]
        }
        with app.test_client() as client:
            print("Testing the API...")
            response = client.post("/v1/completions", json=test_payload)
            # print(response.json)
            pretty_print_json(response.json)
    else:
        app.run(host="0.0.0.0", port=5001)
