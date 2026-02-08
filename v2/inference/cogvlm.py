"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
import torch
import json

from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import pandas as pd
import yaml
from adapter.dct_adapter import DCTAdapter


from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from utils.utils import evaluate_on_mmvetv2, process_images_for_question, evaluate_on_mmvetv2_preference_subset
from adapter.adapter_helper_functions import inject_adapters_vlm, freeze_model_except_adapters

from huggingface_hub import snapshot_download
import accelerate.big_modeling
from contextlib import contextmanager
from accelerate.utils import parse_flag_from_env
import torch.nn as nn

# Monkeypatch init_on_device to fix TypeError: Parameter.__new__() got an unexpected keyword argument '_is_hf_initialized'
@contextmanager
def custom_init_on_device(device: torch.device, include_buffers=None):
    if include_buffers is None:
        include_buffers = parse_flag_from_env("ACCELERATE_INIT_INCLUDE_BUFFERS", False)

    if include_buffers:
        with device:
            yield
        return

    old_register_parameter = nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            # Fix: remove _is_hf_initialized if present
            if "_is_hf_initialized" in kwargs:
                del kwargs["_is_hf_initialized"]
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    # Patch tensor creation
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    try:
        nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

accelerate.big_modeling.init_on_device = custom_init_on_device


class CogVLM:
    def __init__(
        self,
        model_name="zai-org/cogvlm-chat-hf", # OLD -> THUDM/cogvlm-chat-hf
        tokenizer_name="lmsys/vicuna-7b-v1.5",
        image_first=False,
        system_message="You are a helpful assistant, dedicated to delivering comprehensive and meticulous responses.",
        chat_format=True,
        quant=None,
    ):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        quantization_config = None
        if quant == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        if args.bf16:
            self.torch_type = torch.bfloat16
        else:
            self.torch_type = torch.float16

        print(
            "========Use torch type as:{} with device:{}========\n\n".format(
                self.torch_type, self.DEVICE
            )
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.torch_type,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
        )
        
        # Monkeypatch for compatibility with newer transformers which removed _extract_past_from_model_output
        if not hasattr(model, "_extract_past_from_model_output"):
            def _extract_past_from_model_output(self, outputs, **kwargs):
                if hasattr(outputs, "past_key_values"):
                    return outputs.past_key_values
                return None
            import types
            model._extract_past_from_model_output = types.MethodType(_extract_past_from_model_output, model)

        self.model = model.eval()
        self.system_message = system_message
        self.chat_format = chat_format

    def get_response(self, image_folder, prompt="What's in this image?") -> str:
        images = []
        text_queries = []
        queries = prompt.split("<IMG>")
        for query in queries:
            query = query.strip()
            if query.endswith((".jpg", ".png", ".jpeg")):
                images.append(os.path.join(image_folder, query))
                text_queries.append("<IMAGE>")
            else:
                text_queries.append(query)
        text_query = "".join(text_queries)
        image = process_images_for_question(images).convert("RGB")
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=text_query, history=None, images=[image]
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.DEVICE),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.DEVICE),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.DEVICE),
            "images": (
                [[input_by_model["images"][0].to(self.DEVICE).to(self.torch_type)]]
                if image is not None
                else None
            ),
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [input_by_model["cross_images"][0].to(self.DEVICE).to(self.torch_type)]
            ]

        # add any transformers params here.
        gen_kwargs = {
            "max_new_tokens": 2048,
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": False
        }  # "temperature": 0.9
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            # outputs = outputs[:, inputs["input_ids"].shape[1] :]
            # decode only the new tokens? generate returns all tokens by default unless...
            # if model follows HF standard, generate returns [input_ids + new_tokens]
            # So we slice.
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("</s>")[0].strip()
        output_text = response
        return output_text


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant", choices=[4], type=int, default=None, help="quantization bits"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="zai-org/cogvlm-chat-hf",
        help="pretrained ckpt",
    )
    parser.add_argument(
        "--local_tokenizer",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="tokenizer path",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--mmvetv2_path",
        type=str,
        default="data/mm-vet-v2",
        help="Download mm-vet.zip and `unzip mm-vet.zip` and change the path here",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="results_cogvlm",
    )
    parser.add_argument(
        "--image_first",
        action="store_true",
        help="whether <image>text",
    )
    parser.add_argument(
        "--chat_format",
        action="store_true",
        help="whether to use chat format",
    )
    parser.add_argument(
        "--preference_optimize",
        default=True,
        action="store_true",
        help="whether to use preference optimization",
    )
    parser.add_argument(
        "--openai_config_path",
        type=str,
        default="config/openai_config.yaml",
    )
    parser.add_argument(
        "--adapter_config_path",
        type=str,
        default="config/cogvlm_config.yaml",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()

    model = CogVLM(model_name=args.model_name, tokenizer_name=args.local_tokenizer, image_first=args.image_first, quant=args.quant)
    if args.image_first:
        args.model_name = args.model_name + "-image-first"
    if args.chat_format:
        args.model_name = args.model_name + "-chat-format"
    # print(args)

    # For original model 
    if not args.preference_optimize:
        original_model_results_path = evaluate_on_mmvetv2(args, model, is_pref_result=False, adapter_config=None)

   
    elif args.preference_optimize:
        
        # Load the OpenAI API config & adapter config
        with open(args.adapter_config_path, "r") as f:
            adapter_config = yaml.safe_load(f) 


        if os.path.exists(args.openai_config_path):
            with open(args.openai_config_path, "r") as file:
                openai_config = yaml.safe_load(file)
            openai_api_key = openai_config.get("OPENAI_API_KEY")
            if openai_api_key:
                os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            raise ValueError("OpenAI API config not found")

        # TODO: Unfreeze the following later
        original_model_results_path = evaluate_on_mmvetv2(args, model, is_pref_result=False, adapter_config=adapter_config)       
        # original_model_results_path = "results_cogvlm/zai-org--cogvlm-chat-hf.json"
        # Adapter Injection
        adapter_params_json = adapter_config.get("adapter").get("params")
        adapter_layers_json = adapter_config.get("adapter").get("layers")

        model.model = inject_adapters_vlm(
            model.model, 
            DCTAdapter, 
            adapter_params_json, 
            adapter_layers_json 
        )

        # Move adapters to same dtype/device
        base_dtype = torch.float16
        base_device = next(model.model.parameters()).device

        for name, param in model.model.named_parameters():
            if "adapter" in name:
                param.data = param.data.to(device=base_device, dtype=base_dtype)

        
        model.model = model.model.to(device=model.DEVICE)
        freeze_model_except_adapters(model.model)


        # Finetune the model adapters on preference data
        injected_model_results_path = evaluate_on_mmvetv2_preference_subset(args, model, adapter_config, original_model_results_path)

        # Finally get the response 
        model.eval()
        with torch.inference_mode():
            evaluate_on_mmvetv2(args, model, is_pref_result=True)
    


