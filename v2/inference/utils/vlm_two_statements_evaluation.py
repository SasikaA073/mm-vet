from typing import Dict, Any, Optional, List
import os
import openai
import time
import re 
from datasets import Dataset # This is Huggingface Dataset not Pytorch dataset
import json 
class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

class GenericLLMEvaluator: # Removed inheritance from BaseEvaluator
    """Generic LLM evaluator - Refactored to be OpenCompass independent."""

    def __init__(
        self,
        judge_cfg: ConfigDict,
        prompt_template: ConfigDict, # Expecting a dict with 'template_text' and 'input_columns'
        dataset_cfg: Optional[ConfigDict] = None, # Less critical now, primarily uses test_set
        pred_postprocessor: Optional[callable] = None, # Function to postprocess predictions
        dict_postprocessor: Optional[callable] = None, # Function to postprocess judge's raw output dict
        output_path: str = "./eval_results/generic_eval.json", # Default output path
        max_retries: int = 3,
        retry_delay: int = 10,
        keep_predictions: bool = False, # Added from original, though not heavily used here
    ) -> None:
        
        self.judge_cfg = judge_cfg
        self.prompt_template_text = prompt_template.template # a string with placeholders like {query}, {prediction}
        self.prompt_input_columns = prompt_template.get('input_columns', ['query', 'prediction', 'reference'])


        self.dataset_cfg = dataset_cfg # Retained but its role is diminished
        self.pred_postprocessor = pred_postprocessor
        self.dict_postprocessor = dict_postprocessor
        self.output_path = output_path
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.keep_predictions = keep_predictions # Retained from original
        self.judge_system_prompt_content = judge_cfg.get('system_prompt_content') # Added for system/developer prompt


        self._out_dir = os.path.dirname(self.output_path)
        os.makedirs(self._out_dir, exist_ok=True)

        # Initialize OpenAI client
        # API key and base should be set as environment variables: OPENAI_API_KEY, OPENAI_API_BASE
        # Or passed directly in judge_cfg (though env vars are more common)
        self.openai_client = openai.OpenAI(
            api_key=judge_cfg.get('key'),
            base_url=judge_cfg.get('openai_api_base', os.getenv("OPENAI_API_BASE"))
        )
        self.judge_model_name = judge_cfg.get('model') 
        self.judge_temperature = judge_cfg.get('temperature')
        self.judge_max_tokens = judge_cfg.get('max_out_len')
        self.judge_query_per_second = judge_cfg.get('query_per_second', 1) # Simple rate limiting

      
        
        

    def _format_prompt(self, data_item: Dict[str, Any]) -> str:
        """Formats the prompt string using data_item."""
        # Ensure all necessary columns for the prompt are present in the data_item
        # Missing columns will be replaced with a placeholder string.
        prompt_values = {col: data_item.get(col, f"[{col} not found]") for col in self.prompt_input_columns}
        try:
            return self.prompt_template_text.format(**prompt_values)
        except KeyError as e:
            print(f"Missing key {e} in prompt formatting for item: {data_item}. Input columns: {self.prompt_input_columns}")
            # Return a modified prompt indicating the error, or raise
            return f"Error: Missing key {e} for prompt. Data: { {k:v for k,v in prompt_values.items() if k in self.prompt_input_columns} }"


    def _get_judge_response(self, prompt: str) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                # Simple delay based on QPS, more sophisticated rate limiting might be needed for high volume
                time.sleep(1.0 / self.judge_query_per_second)

                messages_for_api = []
                if self.judge_system_prompt_content:
                    messages_for_api.append({"role": "system", "content": self.judge_system_prompt_content})
                messages_for_api.append({"role": "user", "content": prompt})

                response = self.openai_client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=messages_for_api,
                    temperature=self.judge_temperature,
                    max_tokens=self.judge_max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                self.logger.warning(f"OpenAI API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff might be better
                else:
                    print("Max retries reached for OpenAI API call.")
                    return f"Error: Max retries reached for judge. Last error: {e}"
        return None # Should be unreachable if max_retries > 0

    def score(
        self,
        dummy_predictions: List[str], # Model's predictions
        references: Optional[List[str]] = None, # Ground truth references (can be part of test_set)
        test_set: Optional[Dataset] = None, # HF Dataset with columns for prompt
    ) -> Dict:
        """
        Score predictions using an LLM judge.

        Args:
            predictions: List of model predictions.
            references: List of ground truths. (Optional if all info is in test_set)
            test_set: Huggingface Dataset containing all necessary columns to format the judge's prompt.
                      Typically includes 'query', 'prediction', and 'reference'.
                      The 'prediction' column from this test_set will be IGNORED; the passed 'predictions' list is used.
        """
        

        if len(dummy_predictions) != len(test_set):
            raise ValueError(f"Length of predictions ({len(dummy_predictions)}) must match length of test_set ({len(test_set)}).")

        all_results = []
        processed_predictions = self.pred_postprocess(dummy_predictions)

        for i, example in enumerate(test_set):
            # Override the 'prediction' in the example with the one from the processed_predictions list
            current_data_item = example.copy() # Make a mutable copy
            current_data_item['prediction'] = processed_predictions[i]
            # Ensure 'reference' is also present if needed by prompt
            if 'reference' not in current_data_item and references and i < len(references):
                 current_data_item['reference'] = references[i]


            prompt_text = self._format_prompt(current_data_item)
            if prompt_text.startswith("Error:"): # Handle prompt formatting errors
                judge_response_text = prompt_text # Propagate error
            else:
                judge_response_text = self._get_judge_response(prompt_text)

            if judge_response_text is None: # Should not happen if _get_judge_response returns error string
                judge_response_text = "Error: Failed to get response from judge."

            # The dict_postprocessor is responsible for parsing the judge_response_text
            # (e.g., extracting a score, explanation) into a dictionary.
            # If None, the raw text is kept under a 'judge_response' key.
            if self.dict_postprocessor:
                # The postprocessor might expect the raw response and other context.
                # Adjust signature if it needs more than just the text.
                # For now, assume it takes the text and returns a dict.
                try:
                    eval_result = self.dict_postprocessor(judge_response_text)
                    if not isinstance(eval_result, dict): # Ensure it's a dict
                        eval_result = {'judge_output': eval_result, 'raw_judge_response': judge_response_text}
                except Exception as post_e:
                    print(f"Error in dict_postprocessor for judge response '{judge_response_text[:100]}...': {post_e}")
                    eval_result = {'error_postprocessing': str(post_e), 'raw_judge_response': judge_response_text}

            else:
                eval_result = {'raw_judge_response': judge_response_text}

            # Include original data for reference, if desired (controlled by keep_predictions or similar logic)
            # For now, just keeping the judge's processed output and raw response.
            # The 'id' or index might be useful to include.
            # The OpenCompass format often has a 'details' structure.
            # We'll create a simpler list of results here.
            # The calling script (evaluation.py) will be responsible for aggregating.
            
            # Add original query, model prediction, reference for context in output
            result_item = {
                'id': example.get('id', i), # Use 'id' if present in dataset, else index
                'query': example.get('query', 'N/A'),
                'model_prediction': processed_predictions[i],
                'reference': example.get('reference', 'N/A' if not references else references[i]),
                'judge_evaluation': eval_result # This contains score etc. from postprocessor
            }
            all_results.append(result_item)

        # Save all raw judge outputs and processed scores
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4)
            print(f"Saved detailed judge outputs to {self.output_path}")
        except Exception as e:
            print(f"Failed to save judge outputs to {self.output_path}: {e}")

        # The final score aggregation (e.g., averaging scores) should be done
        # by the caller (evaluation.py) based on the contents of `all_results`.
        # This `score` method returns the detailed results for flexibility.
        # For compatibility with how evaluation.py uses it, we can return a dict with 'details'
        # and a summary if the postprocessor provides a clear numeric score.
        
        summary_scores = []
        for res_item in all_results:
            # Assuming dict_postprocessor (like TextSummarizer) puts a 'score' key in judge_evaluation
            if 'judge_evaluation' in res_item and isinstance(res_item['judge_evaluation'], dict) and 'score' in res_item['judge_evaluation']:
                try:
                    summary_scores.append(float(res_item['judge_evaluation']['score']))
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse score from: {res_item['judge_evaluation']['score']}")

        final_output = {"details": all_results}
        if summary_scores:
            final_output["average_score"] = sum(summary_scores) / len(summary_scores) if summary_scores else 0
            final_output["num_scored"] = len(summary_scores)
        
        return final_output


    def pred_postprocess(self, predictions: List[str]) -> List[str]:
        if self.pred_postprocessor is None:
            return predictions
        else:
            # Assuming pred_postprocessor is a callable that takes a string and returns a string
            return [self.pred_postprocessor(pred) for pred in predictions]

    # Removed output_postprocess as its logic is now more integrated or handled by dict_postprocessor
    # and final aggregation in the calling script.

    # Removed default_judge_cfg property, as judge config is now directly passed and processed in __init__.
    # The calling script (evaluation.py) will be responsible for constructing this judge_cfg.
# Prompt template for evaluating two statements

# Post-processor for the judge's output when evaluating two statements
def two_statement_postprocessor(judge_response_text: str) -> Dict[str, Any]:
    """
    Parses the judge's raw text output to extract evaluations and scores for two responses.
    """
    output = {
        "response_A": {"score": None, "explanation": None},
        "response_B": {"score": None, "explanation": None},
        "raw_judge_response": judge_response_text,
        "parsing_errors": []  # To store specific parsing issues
    }

    try:
        # Explanation for Response A
        exp_a_match = re.search(
            r"Evaluation for Response A:\s*(.*?)\s*Score for Response A:",
            judge_response_text, re.DOTALL | re.IGNORECASE
        )
        if exp_a_match:
            output["response_A"]["explanation"] = exp_a_match.group(1).strip()
        else:
            output["parsing_errors"].append("Could not parse explanation for Response A.")

        # Score for Response A
        score_a_match = re.search(
            r"Score for Response A:\s*(\d+(?:\.\d+)?)(?:/\d+)?", # Matches "7" or "7.5" or "7/10"
            judge_response_text, re.IGNORECASE
        )
        if score_a_match:
            try:
                output["response_A"]["score"] = float(score_a_match.group(1))
            except ValueError:
                output["parsing_errors"].append(f"Failed to parse score for Response A from '{score_a_match.group(1)}'.")
        else:
            output["parsing_errors"].append("Could not parse score for Response A.")

        # Explanation for Response B
        exp_b_match = re.search(
            r"Evaluation for Response B:\s*(.*?)\s*Score for Response B:",
            judge_response_text, re.DOTALL | re.IGNORECASE
        )
        if exp_b_match:
            output["response_B"]["explanation"] = exp_b_match.group(1).strip()
        else:
            output["parsing_errors"].append("Could not parse explanation for Response B.")

        # Score for Response B
        score_b_match = re.search(
            r"Score for Response B:\s*(\d+(?:\.\d+)?)(?:/\d+)?",
            judge_response_text, re.IGNORECASE
        )
        if score_b_match:
            try:
                output["response_B"]["score"] = float(score_b_match.group(1))
            except ValueError:
                output["parsing_errors"].append(f"Failed to parse score for Response B from '{score_b_match.group(1)}'.")
        else:
            output["parsing_errors"].append("Could not parse score for Response B.")
        
        # Final check if no structured info was found despite no specific regex error
        if not output["response_A"]["explanation"] and output["response_A"]["score"] is None and \
           not output["response_B"]["explanation"] and output["response_B"]["score"] is None and \
           not output["parsing_errors"]:
            output["parsing_errors"].append("Judge response did not match expected structure, but no specific regex pattern failed.")

    except Exception as e:
        # This catches errors in the regex execution itself or other unexpected errors
        output["parsing_errors"].append(f"Critical post-processing error: {str(e)}")

    return output


# TWO_STATEMENT_PROMPT_TEMPLATE = {
#     "template": """[Instruction]
# You are an impartial judge. You will be given a user's query and two responses (Response A and Response B) generated by AI assistants.
# Your task is to evaluate each response individually based on the query, considering helpfulness, relevance, accuracy, depth, creativity, and level of detail.
# Provide a separate evaluation (a short explanation) and a score (on a scale of 1 to 10) for each response.

# Please structure your output strictly as follows:

# Evaluation for Response A:
# [Your detailed evaluation of Response A]
# Score for Response A: [Score for A, e.g., 7/10 or 7]

# Evaluation for Response B:
# [Your detailed evaluation of Response B]
# Score for Response B: [Score for B, e.g., 9/10 or 9]

# Ensure your output strictly follows this format. Do not add any other text before or after this structured evaluation.

# [User Query]
# {query}

# [Response A]
# {response_A}

# [Response B]
# {response_B}
# """,
#     "input_columns": ["query", "response_A", "response_B"], # Columns expected from the input data
# }


TWO_STATEMENT_PROMPT_TEMPLATE = {
    "template": """[Instruction]
Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.
Provide a separate evaluation (a short explanation) and a score (on a scale of 1 to 10) for each response. Avoid giving the same score to both responses unless they are perfectly identical in text.

Please structure your output strictly as follows:

Evaluation for Response A:
[Your detailed evaluation of Response A]
Score for Response A: [Score for A, e.g., 7/10 or 7]

Evaluation for Response B:
[Your detailed evaluation of Response B]
Score for Response B: [Score for B, e.g., 9/10 or 9]

Ensure your output strictly follows this format. Do not add any other text before or after this structured evaluation.

[User Query]
{query}

[Image]
{image}

[Response A]
{response_A}

[Response B]
{response_B}
""",
    "input_columns": ["query", "image","response_A", "response_B"], # Columns expected from the input data
}

# 





def evaluate_two_statements(
    input_dictionary: Dict[str, str],
    # openai_config_path: str = "config/openai_config.yaml", # Relative to VLM_DCT_Adapter
    judge_model_name,
    judge_system_prompt = None,
    eval_output_dir = "./eval_results_two_statements"
) -> Dict[str, Any]:
    """
    Evaluates two text statements (response_A, response_B) against a query using an LLM judge.

    Args:
        input_dictionary: A dictionary with keys "query", "response_A", "response_B".
        openai_config_path: Path to the OpenAI configuration YAML file.
        judge_model_name: The name of the LLM judge model to use (e.g., "gpt-4o").
        judge_system_prompt: Optional system prompt for the judge.
        eval_output_dir: Directory where the GenericLLMEvaluator will save its detailed JSON output.

    Returns:
        A dictionary containing the original query, responses, and the judge's parsed evaluation
        (scores and explanations for each response), raw judge response, and any parsing errors.
    """
    

    os.makedirs(eval_output_dir, exist_ok=True)
   

    # Configure the judge
    judge_cfg_dict = {
        "model": judge_model_name,
        "key": os.environ.get("OPENAI_API_KEY"),
        # "openai_api_base": openai_params.get("base_url"),
        "temperature": 0.0,
        "max_out_len": 1024, # Judge might need more tokens for two explanations
        "query_per_second": 1, # Adjust as needed
        "system_prompt_content": judge_system_prompt
    }

    # Configure the prompt template
    prompt_template_config = ConfigDict(TWO_STATEMENT_PROMPT_TEMPLATE)
    judge_config = ConfigDict(judge_cfg_dict)

    # Path for GenericLLMEvaluator's detailed output file
    evaluator_internal_output_path = os.path.join(
        eval_output_dir,
        f"judge_details_{judge_model_name.replace('/', '_')}.jsonl"
    )

    # Initialize the GenericLLMEvaluator
    evaluator = GenericLLMEvaluator(
        judge_cfg=judge_config,
        prompt_template=prompt_template_config,
        dict_postprocessor=two_statement_postprocessor,
        output_path=evaluator_internal_output_path, # GenericLLMEvaluator will save its raw output here
    )
    
    data_for_dataset = {
        "query": [input_dictionary["query"]],
        "response_A": [input_dictionary["response_A"]],
        "response_B": [input_dictionary["response_B"]],
        "id": [input_dictionary["id"]] # Add a dummy ID
    }
    eval_hf_dataset = Dataset.from_dict(data_for_dataset)

    # The `predictions` argument for `evaluator.score()` is usually a list of single model outputs.
    # Since our prompt template gets 'response_A' and 'response_B' from the dataset columns,
    # we can pass a dummy list for `predictions` that matches the dataset length (which is 1).
    dummy_predictions = ["N/A"]

    print(f"Sending request to judge model: {judge_model_name} for query: '{input_dictionary['query'][:50]}...'")
    
    evaluation_results = {}
    try:
        evaluation_results = evaluator.score(
            dummy_predictions=dummy_predictions,
            test_set=eval_hf_dataset
        )
    except Exception as e:
        print(f"Error during evaluator.score: {e}")
        return {
            "error": f"Evaluation failed: {str(e)}",
            "query": input_dictionary["query"],
            "response_A_text": input_dictionary["response_A"],
            "response_B_text": input_dictionary["response_B"],
        }

    # Process the results
    # `evaluation_results['details']` is a list of results, one for each item in test_set.
    # We have only one item.
    if evaluation_results and evaluation_results.get("details") and len(evaluation_results["details"]) > 0:
        judge_output_parsed = evaluation_results["details"][0].get("judge_evaluation", {})
        
        # Construct the final return dictionary
        final_output = {
            "query": input_dictionary["query"],
            "response_A_text": input_dictionary["response_A"],
            "response_B_text": input_dictionary["response_B"],
            "evaluation_A": judge_output_parsed.get("response_A", {"score": None, "explanation": "Not parsed"}),
            "evaluation_B": judge_output_parsed.get("response_B", {"score": None, "explanation": "Not parsed"}),
            "raw_judge_response": judge_output_parsed.get("raw_judge_response", "Not available"),
            "parsing_errors": judge_output_parsed.get("parsing_errors", ["No evaluation details found."])
        }
        # If parsing errors were significant, indicate it
        if not final_output["evaluation_A"]["explanation"] and not final_output["evaluation_A"]["score"] and \
           not final_output["evaluation_B"]["explanation"] and not final_output["evaluation_B"]["score"]:
            if "Judge response did not match expected structure, but no specific regex pattern failed." not in final_output["parsing_errors"] and \
               any("Critical post-processing error" not in err for err in final_output["parsing_errors"]):
                 final_output["parsing_errors"].append("Failed to extract structured scores or explanations for either response.")
        
        return final_output
    else:
        print("Evaluation results were empty or not in the expected format.")
        return {
            "error": "Evaluation produced no details.",
            "query": input_dictionary["query"],
            "response_A_text": input_dictionary["response_A"],
            "response_B_text": input_dictionary["response_B"],
            "raw_evaluation_results": evaluation_results # include for debugging
        }

