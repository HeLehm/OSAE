import argparse
import time
from flask import Flask, request, jsonify
import transformers
import torch

app = Flask(__name__)

pipeline = None


def load_model(model_id, load_in_4bit=False, load_in_8bit=False, device="cuda"):
    model_kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    if load_in_4bit:
        model_kwargs["quantization_config"] = {"load_in_4bit": True}
    elif load_in_8bit:
        model_kwargs["quantization_config"] = {"load_in_8bit": True}

    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
        device=device,
    )


@app.route("/v1/completions", methods=["POST"])
def generate_text():
    try:
        data = request.json
        messages = data["messages"]
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.6)
        top_p = data.get("top_p", 0.9)


        # Create the prompt using the chat template
        prompt = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(prompt)

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        # Generate the response
        outputs = pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        generated_text = outputs[0]["generated_text"][len(prompt) :]

        # Format the response
        result = {
            "id": "cmpl-xxx",
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [{"text": generated_text, "index": 0}],
        }
        return jsonify(result), 200
    except Exception as e:
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

    pipeline = load_model(
        args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        device=args.device,
    )

    if args.test:
        test_payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a pirate chatbot who always responds in pirate speak!",
                },
                {"role": "user", "content": "Who are you?"},
            ],
            "max_tokens": 256,
            "temperature": 0.6,
            "top_p": 0.9,
        }
        with app.test_client() as client:
            print("Testing the API...")
            response = client.post("/v1/completions", json=test_payload)
            print(response.json)
    else:
        app.run(host="0.0.0.0", port=5001)
