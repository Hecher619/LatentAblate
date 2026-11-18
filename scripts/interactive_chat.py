import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="LatentAblate: Live Terminal Chat Shell")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to liberated model directory")
    args = parser.parse_args()

    print("=" * 60)
    print(f"💬 Loading Liberated Model from {args.model_dir}...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print("\n✅ Model loaded successfully! Type your prompt below (type 'exit' to quit).\n" + "-" * 50)

    history = []
    while True:
        try:
            user_input = input("\nUser: ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                print("Exiting chat shell. Goodbye!")
                break

            history.append({"role": "user", "content": user_input})
            
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompt_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = user_input

            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

            print("Assistant: ", end="", flush=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(response)
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break

if __name__ == "__main__":
    main()
