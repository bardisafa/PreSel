import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import random
import torch
import argparse
import transformers
import tokenizers
import sys
from tqdm import tqdm

sys.path.append("./LLaVA")

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset

from llava import conversation as conversation_lib
from llava.model import *
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
import re

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)

def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
  
@dataclass
class Args:
    data_path: str
    model_path: str
    lazy_preprocess: bool
    is_multimodal: bool
    image_folder: str
    image_aspect_ratio: str
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    
@dataclass
class DataArguments:
    data_path: str = field(
        default="/data/01/bsafaei/Projects/datasets/AL/random/round1.json", metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default="/data/01/bsafaei/Projects/datasets/coco/train2017")
    image_aspect_ratio: str = "square"

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    
def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            else:
                image = processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            
        data_dict['unique_idx']=self.list_data_dict[i]['unique_idx']
        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        unique_indices=[instance['unique_idx'] for instance in instances]
        batch['unique_indices']=unique_indices
        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Data Arguments")

    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/01/bsafaei/Projects/datasets/AL/random/round1.json",
        help="Path to the training data."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/02/bsafaei/checkpoints/llava_rnd1_perc5_random",
        help="Path to the Model."
    )
    parser.add_argument(
        "--lazy_preprocess",
        type=bool,
        default=True,
        help="Whether to use lazy preprocessing."
    )
    parser.add_argument(
        "--is_multimodal",
        type=bool,
        default=True,
        help="Whether the data is multimodal."
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/data/01/bsafaei/Projects/datasets/coco/train2017",
        help="Path to the image folder."
    )
    parser.add_argument(
        "--image_aspect_ratio",
        type=str,
        default="square",
        help="Aspect ratio of the images."
    )

    return parser.parse_args()

def clear_gpu_memory():
    import gc
    import torch

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()

def inference(args, attn_implementation=None):
    # args = parse_args()
    
    # Set the seed for reproducibility
    seed = 12345
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # model.config.use_cache = False

    # Loading Model
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_name=model_name,
        model_base=None
    )
    args.image_processor = image_processor
    args.mm_use_im_start_end = ( model.config.mm_use_im_start_end )
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    data_collator = data_module['data_collator']
    dataset = data_module['train_dataset']

    compute_dtype = (
        torch.float16
    )
    
    model.eval()
    model.cuda()
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, collate_fn=data_collator, shuffle=False)
    results = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            unique_indices = batch.pop("unique_indices")
            assert "inputs_embeds" not in batch
            # images = batch["images"].cuda().to(dtype=compute_dtype)
            # input_ids = batch["input_ids"].cuda().to(dtype=torch.long)
            # labels = batch["labels"].cuda().to(dtype=torch.long)
            # attention_mask = batch["attention_mask"].cuda().to(dtype=torch.long)
            # past_key_values = batch.get("past_key_values")
            # position_ids = batch.get("position_ids")
            
            images = batch["images"].to("cuda:0",dtype=compute_dtype)
            input_ids = batch["input_ids"].to("cuda:0",dtype=torch.long)
            labels = batch["labels"].to("cuda:0",dtype=torch.long)
            attention_mask = batch["attention_mask"].to("cuda:0",dtype=torch.long)
            past_key_values = batch.get("past_key_values")
            position_ids = batch.get("position_ids")
        
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )
            batch["input_ids"] = input_ids
            batch["position_ids"] = position_ids
            batch["attention_mask"] = attention_mask
            batch["past_key_values"] = past_key_values
            batch["inputs_embeds"] = inputs_embeds
            batch["labels"] = labels
            
            outputs = model(**batch)
            
            # Calculate per-sample loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels.to(shift_logits.device)

            bsz, max_seq_len, vocab_size = shift_logits.shape
            shift_probs = torch.softmax(shift_logits.view(-1, vocab_size), dim=-1)
            shift_labels = shift_labels.view(-1)

            valid_mask = shift_labels > -10
            shift_labels = shift_labels * valid_mask
            
            per_sample_loss = -torch.log(shift_probs[range(0, shift_labels.shape[0]), shift_labels])
            per_sample_loss = per_sample_loss.view(bsz, max_seq_len) * valid_mask.view(bsz, max_seq_len)
            per_sample_loss = per_sample_loss.sum(dim=1) / valid_mask.view(bsz, max_seq_len).sum(dim=1)
            
            ## Collect the results
            for i in range(bsz):
                image_index = unique_indices[i]
                loss_value = per_sample_loss[i].item()
                results.append({"unique_idx": image_index, "loss": loss_value})
                # print (f"Image index: {image_index}, Loss: {loss_value}")
    # Save results to a JSON file
    output_file = "/data/02/bsafaei/checkpoints/inference_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # Clear GPU memory after inference
    clear_gpu_memory()
    return results
# if __name__ == "__main__":
#     inference(attn_implementation="flash_attention_2")
