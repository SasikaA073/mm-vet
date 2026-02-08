import math
import os
import json
import base64
from PIL import Image, ImageDraw
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch 
from pprint import pprint
from PIL import Image
from utils.preference_loss import compute_preference_loss


def process_images_for_question(images, key=None):
    images = [Image.open(path) for path in images]  #
    if not images:
        return  #
    n = len(images)
    grid_cols = math.ceil(math.sqrt(n))
    grid_rows = math.ceil(n / grid_cols)

    #
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    cell_width = max_width + 20  # add gap
    cell_height = max_height + 30  #

    #
    collage_width = cell_width * grid_cols
    collage_height = cell_height * grid_rows
    collage = Image.new("RGB", (collage_width, collage_height), "white")
    draw = ImageDraw.Draw(collage)

    for index, img in enumerate(images):
        row, col = divmod(index, grid_cols)
        x = col * cell_width + (cell_width - img.width) // 2
        y = row * cell_height + (cell_height - img.height - 10) // 2  #
        collage.paste(img, (x, y + 20))  #

        # add img id
        draw.text((x + img.width // 2, y), str(index + 1), fill="black")

    return collage


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def evaluate_on_mmvetv2(args, model, is_pref_result=True, adapter_config=None):
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)

    model_name = args.model_name.replace("/", "--")
    
    if is_pref_result: 
        model_name += "_pref"   
    results_path = os.path.join(args.result_path, f"{model_name}.json")
    image_folder = os.path.join(args.mmvetv2_path, "images")
    meta_data = os.path.join(args.mmvetv2_path, "mm-vet-v2.json")

    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    with open(meta_data, "r") as f:
        data = json.load(f)
    
    if adapter_config:
        args.num_samples = adapter_config.get('data').get('num_train_samples')
        data = dict(list(data.items())[:args.num_samples])

    
    for i in range(len(data)):    
        id = f"v2_{i}"
        if id in results:
            continue
        prompt = data[id]["question"].strip()
        print(id)
        print(f"Prompt: {prompt}")
        try:
            response = model.get_response(image_folder, prompt)
        except Exception as e:
            print(f"Error processing {id}: {e}")
            import traceback
            traceback.print_exc()
            response = ""
        print(f"Response: {response}")
        results[id] = response
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    return results_path

class PreferenceDataset(Dataset):
    def __init__(self, original_mmvet_dict, original_responses_dict, model, transfroms=None, args=None):
        self.original_mmvet_dict = original_mmvet_dict

        # {'v2_0': {'added_in': 'v1',
        #   'answer': '-1<AND>-5',
        #   'capability': ['ocr', 'math'],
        #   'question': 'What is x in the equation?<IMG>v2_0_0.png'},}
        
        self.original_model_responses_dict = original_responses_dict
        self.data_samples = []

        self.transforms = transfroms
        self.model = model 

        for sample_id, sample_data_dict in self.original_mmvet_dict.items(): 
            sample_data_dict['original_model_response'] = self.original_model_responses_dict[sample_id]
            
            question = sample_data_dict['question']
            question_text, img_name = question.split("<IMG>")
            
            img_path = os.path.join(args.mmvetv2_path, "images", img_name) 
            img = Image.open(img_path).convert("RGB")
            
            sample_data_dict['question_text'] = question_text 
            sample_data_dict['img_name'] = img_name 
            sample_data_dict['img'] = img 

            self.data_samples.append(sample_data_dict)

    def __getitem__(self, idx):   
        data_sample = self.data_samples[idx]

    
        data_sample['question']
        data_sample['answer']

        query = data_sample['question']
        inputs = self.model.model.build_conversation_input_ids(self.model.tokenizer, query=query, history=[], images=[data_sample['img']])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}

        return data_sample 
    
    def __len__(self):
        return len(self.data_samples)


def evaluate_on_mmvetv2_preference_subset(args, model, adapter_config, original_model_results_path):
    """
    model : CogVLM instance
    """
    if os.path.exists(args.result_path) is False:
        os.makedirs(args.result_path)

    model_name = args.model_name.replace("/", "--")
    
    image_folder = os.path.join(args.mmvetv2_path, "images")
    meta_data = os.path.join(args.mmvetv2_path, "mm-vet-v2.json")

    if os.path.exists(original_model_results_path):
        with open(original_model_results_path, "r") as f:
            original_model_results = json.load(f)
    else:
        print("Original model responses missing!")
        raise ValueError

    with open(meta_data, "r") as f:
        data = json.load(f)

    original_mmvet2_data_subset_dict = dict(list(data.items())[:args.num_samples])

    original_model_responses_subset_dict = dict(list(original_model_results.items())[:args.num_samples])
    

    pref_data_subset = PreferenceDataset(
        original_mmvet_dict = original_mmvet2_data_subset_dict,
        original_responses_dict = original_model_responses_subset_dict,
        model=model,
        args=args
        
    )

    def mmvetv2_collate_fn(batch):
        a = batch 
        print(batch )
        return 


    pref_dataloader = DataLoader(
        pref_data_subset, 
        batch_size = adapter_config.get("training").get("batch_size"),
        shuffle=False, 
        num_workers=0,
        collate_fn=mmvetv2_collate_fn
    )

    sample_batch = next(iter(pref_dataloader))
    # Vertify the trainable model params
    print("Trainable Parameters:")
    for name, params in model.model.named_parameters():
        if params.requires_grad:
            print("-", name)
        
    # Define the optimizer 
    optimizer = torch.optim.AdamW(
        [p for p in model.model.parameters() if p.requires_grad], 
        lr=adapter_config.get("training").get("learning_rate"),
        weight_decay=0.1
    )

    # TODO: Optimize preference
    for epoch in range(adapter_config.get("training").get("num_epochs")):
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(pref_dataloader, desc="Training Adapters")):
            train_one_epoch(model, pref_dataloader, optimizer, batch, batch_idx, args=args)
        print("Training completed!")
        

def train_one_epoch(model, optimizer, batch, adapter_config,batch_idx, args):
    input_ids = batch["input_ids"].to(model.DEVICE)
    attention_mask = batch["attention_mask"].to(model.DEVICE)
    labels = batch["labels"].to(model.DEVICE)

    outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    original_response_dicts_batch = [
            original_responses_dict[idx] for idx in batch["indices"] # TODO: Will have to fix this!!!
        ]
        
    loss = compute_preference_loss(
        model_name=model.model_name, 
        adapted_model=model.model,
        adapted_model_tokenizer=model.tokenizer,
        original_response_dicts=original_response_dicts_batch,
        forward_pass_outputs=outputs,
        config=adapter_config,
        device=model.model.device
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item() * BATCH_SIZE # TODO: I have to normalize here right?
    
    # Calculate Loss 
    avg_epoch_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}")

    # Save checkoint every 5 epochs (ending in 4, 9, etc) or last epoch
    if epoch % 5 == 4 or epoch == num_epochs - 1:
        checkpoint_dir = f"checkpoints/{model_name}_dct_adapter_epoch_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        adapted_model.save_pretrained(checkpoint_dir)
        adapted_tokenizer.save_pretrained(checkpoint_dir)
        
        adapter_state_dict = {}
        for name, param in adapted_model.named_parameters():
            if 'adapter' in name or '.0.weight' in name or '.1.' in name:
                adapter_state_dict[name] = param.cpu()
        
        torch.save(adapter_state_dict, f"{checkpoint_dir}/adapter_weights.pt")
        
        adapter_config = {
            'adapter_params': adapter_params_json,
            'adapter_layers': adapter_layers_json
        }
        with open(f"{checkpoint_dir}/adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        
        print(f"Model and adapters saved to {checkpoint_dir}")

    
def mmvetv2_collate_fn(batch):
    pass    