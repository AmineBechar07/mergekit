import torch
import tqdm
import transformers
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM,AutoTokenizer, AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from mergekit.common import ModelReference
from mergekit.moe.config import Expert

import torch.nn as nn


def get_hidden_states(
    model: Union[MistralForCausalLM, LlamaForCausalLM],
	@@ -115,6 +117,7 @@ def get_gate_params(
        embed = model_ref.lazy_loader(lazy_unpickle=lazy_unpickle).get_tensor(
            "model.embed_tokens.weight"
        )


        def _do_it(tokenized):
            return get_cheap_embedding(
	@@ -140,6 +143,18 @@ def _do_it(tokenized):
            return get_hidden_states(
                model, tokenized=tokenized, average=mode == "hidden_avg"
            )
    elif mode == "model_based":
            prompt_mapper_model = PromptMapperModel.load_from_checkpoint(
                "path/to/prompt_mapper_model.ckpt"
            )
            prompt_mapper_model.eval()

            def _do_it(tokenized):
                with torch.no_grad():
                    expert_probs = prompt_mapper_model(tokenized)
                    expert_probs = expert_probs.permute(1, 0, 2)  # (num_layers, batch_size, num_experts)
                    expert_probs = expert_probs.mean(dim=1)  # (num_layers, num_experts)
                    return expert_probs

    gate_vecs = []
    for expert in tqdm.tqdm(experts, desc="expert prompts"):
	@@ -148,7 +163,7 @@ def _do_it(tokenized):
            hidden_states -= _do_it(
                tokenize_prompts(expert.negative_prompts, tokenizer)
            )
    
        hidden_states /= hidden_states.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        gate_vecs.append(hidden_states)
    gate_vecs = torch.stack(gate_vecs, dim=0)  # (num_expert, num_layer, hidden_size)
	@@ -187,3 +202,45 @@ def warn_degenerate_gates(gate_vecs: torch.Tensor, threshold: float = 5.0):
            "- your prompts may be too similar."
        )
        logging.warning("One or more experts will be underutilized in your model.")



class PromptMapperModel(nn.Module):
    def __init__(self, model_name, num_experts, num_layers):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.projection = nn.Linear(self.encoder.config.hidden_size, num_experts * num_layers)

    def forward(self, prompts):
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        outputs = self.encoder(**inputs)
        sequence_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token representation
        logits = self.projection(sequence_output)
        logits = logits.view(-1, self.num_layers, self.num_experts)
        probs = nn.Softmax(dim=-1)(logits)
        return probs

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model_name, num_experts, num_layers):
        model = cls(model_name, num_experts, num_layers)
        model.load_state_dict(torch.load(checkpoint_path))
        return model

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def train(self, mode=True):
        self.training = mode
        self.encoder.train(mode=mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def to(self, device):
        self.encoder = self.encoder.to(device)
        self.projection = self.projection.to(device)
        return self
