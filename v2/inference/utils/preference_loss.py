import torch 
from PIL import Image
from pprint import pprint
from utils.vlm_two_statements_evaluation import evaluate_two_statements
import base64
# from utils.model_tokenization import tokenize_prompt_relevant_model
import openai

def compute_preference_loss(
    model_name, 
    adapted_model, 
    adapted_model_tokenizer,
    original_response_dicts,  # Now accepts a list of dicts for batch processing
    forward_pass_outputs,
    
    config,
    device="cuda"):
    """
    Computes the preference loss between the adapted model and the original model.
    Supports batched processing for efficiency.
    
    Args:
        adapted_model: The model with adapters
        adapted_model_tokenizer: Tokenizer for the model
        original_response_dicts: List of dicts containing original model responses for each sample in batch
        forward_pass_outputs: Forward pass outputs from the model
        config: Configuration dictionary
        device: Device to run computations on
    
    Returns:
        Mean loss across the batch
    """
    
    batch_size = len(original_response_dicts)
    batch_losses = []
    sum_score = 0.0
    sum_log_prob_diff = 0.0
    
    # Process each sample in the batch
    for batch_idx in range(batch_size):
        original_response_dict = original_response_dicts[batch_idx]
        sample_loss = 0

        # Response A is the adapted model response 
        # Response B is the original model response
        original_answer = original_response_dict['original_model_generated_output']
     
        inputs = tokenize_prompt_relevant_model(
            prompt=original_response_dict['prompt'],
            model_name=model_name,
            tokenizer=adapted_model_tokenizer, 
            model=adapted_model
            
        )
        inputs = {k: v.to(adapted_model.device) for k, v in inputs.items()}

        adapted_model.eval()

        with torch.inference_mode():
            # Generate response for this sample
            generated_ids = adapted_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=config["generation"]["max_new_tokens"],
                do_sample=config['generation']['do_sample'], 
                temperature=config['generation']['temperature'],
                top_p=config['generation']['top_p'],
                num_beams=1,
            )
        
        input_text = adapted_model_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        output_text = adapted_model_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        adapted_model_generated_output = output_text[len(input_text):]
        adapted_answer = adapted_model_generated_output
        
       

        data_input = {
            "id": original_response_dict["index"],
            "query": original_response_dict['prompt'],
            "response_A": adapted_answer,
            "response_B": original_answer,
        }

        # Evaluate using GPT-4o
        json_judgment = evaluate_two_statements(
            data_input,
            judge_model_name=config["judge"]["model_name"],
        )

        print(f"\n[Batch {batch_idx+1}/{batch_size}] Evaluation:")
        pprint(json_judgment)

        # A -> Adapted 
        eval_A = json_judgment.get("evaluation_A", {"score": None, "explanation": "Not parsed"})
        eval_A_score = eval_A.get("score", None)

        # B -> Original
        eval_B = json_judgment.get("evaluation_B", {"score": None, "explanation": "Not parsed"})
        eval_B_score = eval_B.get("score", None)

        # Extract logits for this specific sample in the batch
        logits = forward_pass_outputs.logits[batch_idx:batch_idx+1]  # Keep batch dimension
        
        original_answer_ids = tokenize_prompt_relevant_model(
            prompt=original_answer, 
            model_name=model_name, 
            tokenizer=adapted_model_tokenizer,
            model=adapted_model
        )
        
        original_answer_ids = original_answer_ids['input_ids']
            
        original_answer_ids = original_answer_ids.to(device) 
        
        min_seq_len = min(logits.size(1), original_answer_ids.size(1))
        original_answer_ids = original_answer_ids[:, :min_seq_len]
        logits_trimmed = logits[:, :min_seq_len, :]
        adapted_log_prob = torch.max(logits_trimmed, dim=-1).values

        original_log_prob = logits_trimmed.gather(dim=2, index=original_answer_ids.unsqueeze(-1)).squeeze(-1)

        log_probA = adapted_log_prob.mean()
        log_probB = original_log_prob.mean()

        # Handle None scores
        if eval_A_score is None:
            eval_A_score = 0.0
        if eval_B_score is None:
            eval_B_score = 0.0
            
        if eval_A_score is not None and eval_B_score is not None:
            sum_score += (eval_A_score - eval_B_score)
            sum_log_prob_diff += (log_probA - log_probB)
            # Working version
            # if eval_A_score > eval_B_score:
            #     # If adapted model is better, we want to maximize its log prob
            #     sample_loss = -log_probA + log_probB
            # else:
            #     # If original model is better, we want to minimize its log prob
            #     sample_loss = -log_probB + log_probA
        

        # print(f"[Sample {batch_idx+1}] Adapted: {eval_A_score}, Original: {eval_B_score}, Loss: {sample_loss}")
        # batch_losses.append(sample_loss)
    
    if sum_score < 0:
        # Original model is better on average, minimize its log prob
        total_loss = sum_log_prob_diff
    else:
        # Adapted model is better on average, minimize its log prob
        total_loss = -sum_log_prob_diff
    # Average loss across the batch
    # mean_loss = torch.stack(batch_losses).mean()
    print(f"\n[Batch Summary] Mean Loss: {total_loss.item():.4f}")
    
    return total_loss

def tokenize_prompt_relevant_model(prompt, 
            model_name, 
            tokenizer,
            model):
    r