import os
import time
from collections import OrderedDict
from typing import Any, List, Optional

import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from tensorizer import TensorDeserializer
from tensorizer.utils import no_init_or_tensor
from transformers import (AutoConfig, AutoModelForCausalLM,
                          GPTNeoXForCausalLM, AutoTokenizer)

# from config import HUGGINGFACE_MODEL_NAME, load_tokenizer
from subclass import YieldingCausalLM

DEFAULT_MODEL = "databricks/dolly-v2-12b"
CACHE_DIR = "pretrained_weights"
TOKENIZER_PATH = './tokenizer'

# To download tensorizer weights instead of load them from a local source, `REMOTE_PATH_TO_TENSORIZER_WEIGHTS` to their URL
REMOTE_PATH_TO_TENSORIZER_WEIGHTS = None
PATH_TO_TENSORIZER_WEIGHTS = REMOTE_PATH_TO_TENSORIZER_WEIGHTS if REMOTE_PATH_TO_TENSORIZER_WEIGHTS else './tensorized_models/dolly-v2-12b-fp16.tensors'

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
INTRO_BLURB = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction_text}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction_text="{instruction_text}",
    response_key=RESPONSE_KEY,
)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        if weights is not None and weights.name == 'weights':
            # bugfix
            weights = None
        if weights is None and PATH_TO_TENSORIZER_WEIGHTS:
            print('Loading tensorized weights from public path')
            self.model = self.load_tensorizer(weights=PATH_TO_TENSORIZER_WEIGHTS)

        elif (hasattr(weights, 'filename') and 'tensors' in weights.filename) or str(weights).endswith(".tensors"):
            self.model = self.load_tensorizer(weights)
        else:
            self.model = self.load_huggingface_model(weights=weights)

        self.tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_PATH
        )

    def load_huggingface_model(self, weights=None):
        st = time.time()
        print(f'loading weights from {weights} w/o tensorizer')
        model = YieldingCausalLM.from_pretrained(
            weights, cache_dir=CACHE_DIR
        ).to("cuda:0")
        print(f'weights loaded in {time.time() - st}')
        return model


    def load_tensorizer(self, weights):
        st = time.time()
        print(f'deserializing weights from {weights}')
        config = AutoConfig.from_pretrained(DEFAULT_MODEL)

        model = no_init_or_tensor(
            lambda: YieldingCausalLM.from_pretrained(
                None, config=config, state_dict=OrderedDict()
            )
        )
        des = TensorDeserializer(weights, plaid_mode=True)
        des.load_into_module(model)
        print(f'weights loaded in {time.time() - st}')
        return model

    def predict(
        self,
        prompt: str = Input(description=f"Input Prompt."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        decoding: str = Input(
            description="Choose a decoding method",
            choices=["top_p", "top_k"],
            default="top_p",
        ),
        # num_beams: int = Input(
        #     description="Valid if you choose beam_search. Number of beams for beam search. 1 means no beam search.",
        #     default=1,
        # ),
        top_k: int = Input(
            description="Valid if you choose top_k decoding. The number of highest probability vocabulary tokens to keep for top-k-filtering",
            default=50,
        ),
        top_p: float = Input(
            description="Valid if you choose top_p decoding. When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1.2,
        ),
    ) -> ConcatenateIterator[str]:

            
        prompt_text = PROMPT_FOR_GENERATION_FORMAT.format(instruction_text=prompt)

        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to("cuda:0")

        do_sample = False if decoding == "beam_search" else True
        with torch.inference_mode():
            first_token_yielded = False
            prev_ids = []
            for output in self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=do_sample,
                    #num_beams=num_beams,
                    temperature=temperature,
                    top_p=top_p if decoding == "top_p" else 1,
                    top_k=top_k if decoding == "top_k" else 50,
                    repetition_penalty=repetition_penalty,
            ):
                cur_id = output.item()
                # in order to properly handle spaces, we need to do our own tokenizing. Fun!
                # we're building up a buffer of sub-word / punctuation tokens until we hit a space, and then yielding whole words + punctuation.
                cur_token = self.tokenizer.convert_ids_to_tokens(cur_id)

                # skip initial newline, which this almost always yields. hack - newline id = 13.
                if not first_token_yielded and not prev_ids and cur_id == 187:
                    continue

                # Space is represented as "Ġ". 
                # Yield previous IDs if we hit a space
                # or if the current token includes a space
                # at its start (e.g. ' is' -> 'Ġis')
                if cur_token.startswith("Ġ"):
                    if prev_ids:
                        yield self.tokenizer.decode(prev_ids)
                        prev_ids = []

                    prev_ids = [cur_id]
                    continue

                # End token
                if cur_token == "### End":
                    break

                prev_ids.append(cur_id)

            if prev_ids:
                yield self.tokenizer.decode(prev_ids)
                prev_ids = []


