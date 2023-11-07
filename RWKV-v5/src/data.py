from lightning import LightningDataModule

from torch.utils.data import DataLoader

import wandb
from datasets import load_from_disk, load_dataset, Dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from multiprocessing import cpu_count
num_cpus = cpu_count()
num_workers = cpu_count() if cpu_count() < 8 else 8

# Get the script directory
import os
SRC_DIR = os.path.dirname(os.path.realpath(__file__))

# World tokenizer
from .dataflow.trie_tokenizer import world_tokenizer_encode
import numpy as np

# delay pattern
from . import delay_pattern

# We have to extract out the prepare function to be "outside the class"
# else it will not be hashed / serialized properly, and will produce the following error:
#
# ```
# Parameter 'function'=<function RWKVDataModule.prepare_data.<locals>.map_tokenizer at 0x7f7672c5e340> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.
# ```
def prepare_data_static(**kargs):

    # Check if skip_datapath_setup is enabled
    # useful for extra large datasets
    if kargs["skip_datapath_setup"] == True:
        return

    # Special handling of world_add_endoftext_token (if enabled)
    if kargs["world_add_endoftext_token"]:
        world_add_endoftext_token = True
    else:
        world_add_endoftext_token = False

    # Source data processing
    if kargs["source"] is not None:
        # Special handling for binidx
        #--------------------------------

        if kargs["tokenizer"] == "binidx":
            from .dataflow.binidx import MMapIndexedDataset

            # Load the MMapIndexedDataset from the source path
            mmap_dataset = MMapIndexedDataset(kargs["source"])
            mmap_dataset_len = mmap_dataset.__len__()

            # Torch dataset generator wrapper
            def gen():
                for idx in range(mmap_dataset_len):
                    # cast to supported types, note that np.int32 limit is 2,147,483,647 
                    # - so unless we have a tokenizer that exceeds this, it should be ok
                    tokens = np.array(mmap_dataset.get(idx), dtype=np.int32)
                    # binidx doesn't support multi-dimensional data so we unfold
                    tokens = tokens.reshape(-1, kargs["n_channel"])
                    yield {
                        'input_ids': tokens,
                        'token_type_ids': [[0] * kargs["n_channel"]] * len(tokens),
                        'attention_mask': [[1] * kargs["n_channel"]] * len(tokens)
                    }

            # Load the huggingface dataset from the generator
            src_dataset = Dataset.from_generator(gen)

            # Train/test split
            test_split = kargs["test_split"]
            # The minimum test size is 1, if not we will get errors in the trainer?
            if test_split <= 0 or test_split <= 0.0:
                test_split = 1
            split_dataset = src_dataset.train_test_split(
                test_size=test_split,shuffle=kargs["test_split_shuffle"],
                seed=42 #Fixed seed, to prevent train/test reshuffling between test runs
            )

            # Save the dataset to disk
            split_dataset.save_to_disk(kargs["data_path"])
            # Does nothing else (done)
            return

        # Reverting back to general purpose HF dataset / tokenizer handling
        #--------------------------------

        load_dataset_params = {
            'path': kargs["source"],
            'num_proc': num_cpus
        }

        # Handle advance params (if set)
        if kargs["source_data_dir"] is not None:
            load_dataset_params['data_dir'] = kargs["source_data_dir"]
        if kargs["source_dataset_params"] is not None:
            source_dataset_params = kargs["source_dataset_params"]
            for k, v in source_dataset_params.items():
                load_dataset_params[k] = v

        # Load the dataset
        src_dataset = load_dataset(**load_dataset_params)

        # If for some reason the dataset is a "test" only split, and missing a "train" split, we remap it as a "train" split
        if "train" not in src_dataset.keys():
            if "test" in src_dataset.keys():
                src_dataset["train"] = src_dataset["test"]
                del src_dataset["test"]
            else:
                raise ValueError('Dataset must have a "train" split')

        # If an int value is used, it is interprated as document count
        # If a floating value (<1.0) is used, it is interprated as a percentage of the dataset
        if kargs["dataset_offset"] > 0 or kargs["dataset_length"] > 0:
            # src dataset length
            train_length = len(src_dataset["train"])

            # Compute the offset position
            offset_val = kargs["dataset_offset"]

            # If offset is a float, we will use it as a percentage
            if offset_val < 0:
                offset_val = 0
            if offset_val > 0 and offset_val < 1.0:
                offset_val = int(train_length * offset_val) # Rounded down value

            # Compute the length position
            length_val = kargs["dataset_length"]
            if length_val < 0:
                length_val = train_length - offset_val
            if length_val > 0 and length_val < 1.0:
                length_val = int(train_length * length_val)
            if length_val > (train_length - offset_val):
                length_val = (train_length - offset_val)

            # Get the subset of the dataset
            src_dataset["train"] = src_dataset["train"].select(range(offset_val, offset_val + length_val))
        
        def add_type_and_mask(x):
            ret = dict(x)
            for i in [
                ("token_type_ids", 0),
                ("attention_mask", 1),
            ]:
                if i[0] not in x:
                    ret[i[0]] = [[i[1] * len(x["input_ids"][0])] * len(x["input_ids"])]
            return ret
        src_dataset = src_dataset.map(add_type_and_mask, batched=True, 
                                      batch_size=kargs["text_rechunk_size"]*10,
                                      num_proc=num_cpus)
        
        # Remove all features, except input_ids, token_type_ids, attention_mask and extra
        # as the metadata/etc columns may cause problems down the line (when passed to the trainer)
        dataset_features = src_dataset["train"].features
        dataset_features_to_remove = {k: v for k, v in dataset_features.items() if k not in ["input_ids", "token_type_ids", "attention_mask", "extra"]}
        src_dataset = src_dataset.remove_columns(list(dataset_features_to_remove.keys()))
        
        # Remove empty datasets (it causes an error otherwise)
        # and perform min/max length filtering (if configured)
        def dataset_filter(x):
            row_length = len(x["input_ids"])
            if row_length <= 0:
                return False
            if kargs["min_token_size"] > 0 and row_length < kargs["min_token_size"]:
                return False
            if kargs["max_token_size"] > 0 and row_length > kargs["max_token_size"]:
                return False
            return True
        src_dataset = src_dataset.filter(dataset_filter, num_proc=num_cpus)

        if kargs["start_padding"]:
            def apply_start_padding(x):
                ret = dict(x)
                for i in [
                    ("input_ids", kargs["padding_idx"]),
                    ("token_type_ids", 0),
                    ("attention_mask", 0),
                ]:
                    ret[i[0]] = [i[1] * len(x[i[0]][])] + x[i[0]]
                return ret
            src_dataset = src_dataset.map(apply_start_padding, batched=True, 
                                        batch_size=kargs["text_rechunk_size"]*10,
                                        num_proc=num_cpus)

        if kargs["delay_pattern_enable"]:
            def apply_delay_pattern(x):
                ret = dict(x)
                for i in [
                    ("input_ids", kargs["padding_idx"]),
                    ("token_type_ids", 0),
                    ("attention_mask", 0),
                ]:
                    ret[i[0]] = delay_pattern.apply(x[i[0]], kargs["delay_pattern_groups"], i[1])
                return ret
            src_dataset = src_dataset.map(apply_delay_pattern, batched=True, 
                                        batch_size=kargs["text_rechunk_size"]*10,
                                        num_proc=num_cpus)
        
        # Perform a sort by length
        if kargs["sort_by_length"]:
            sort_asc = kargs["sort_asc"]
            
            def add_length(example):
                example["length"] = len(example['input_ids'])
                return example
            
            src_dataset = src_dataset.map(add_length)
            
            # sort by length (not sorting the columns, just the rows)
            src_dataset = src_dataset.sort("length", reverse=not sort_asc)

        # Perform rechunking after filtering, if source is not a "text" based 
        # dataset and text_rechunk_force is enabled
        if kargs["source"] != "text" and kargs["text_rechunk_size"] > 0 and kargs["text_rechunk_force"]:
            src_dataset = src_dataset.map(rechunk_text, batched=True, 
                                        batch_size=kargs["text_rechunk_size"]*2,
                                        num_proc=num_cpus)

        # Check if the dataset does not have a test split
        # and if so, perform the split
        if 'test' not in src_dataset.keys():
            test_split = kargs["test_split"]
            # The minimum test size is 1, if not we will get errors in the trainer?
            if test_split <= 0 or test_split <= 0.0:
                test_split = 1
            src_dataset = src_dataset['train'].train_test_split(
                test_size=test_split,shuffle=kargs["test_split_shuffle"],
                seed=42 #Fixed seed, to prevent train/test reshuffling between test runs
            )
        
        # Save the dataset to disk
        src_dataset.save_to_disk(kargs["data_path"])


class RWKVDataModule(LightningDataModule):
    def __init__(
        self, 
        # load_from_disk(dataset_path) param
        data_path: str,
        # load_dataset(path) param
        source: str = None,
        # load_dataset(data_dir) param
        source_data_dir: str = None,
        # Additional dataset params
        source_dataset_params: dict = None,
        # Test split of source data, if it was not already done
        test_split: float = 0.01,
        test_split_shuffle: bool = False,
        # Text rechunking size
        text_rechunk_size: int = 4096,
        text_rechunk_force: bool = False,
        # ---
        # Tokenizer settings
        # ---
        tokenizer: str = None,
        autoTokenizer = None,

        # Add <|endoftext|> string token to the world tokenizer, at index 0
        # this was missing from the original world trie_tokenizer
        world_add_endoftext_token: bool = True,

        # ---
        # HF dataset conversion helpers
        # ---
        # Min / Max token size filtering
        min_token_size: int = -1,
        max_token_size: int = -1,
        
        # Sort by length
        sort_by_length: bool = False,
        sort_asc: bool = True,

        # Dataset offset and limit controls
        dataset_offset: int = -1,
        dataset_length: int = -1,
        
        # Custom 'text' column to support, mostly used for dataset where the 
        # desired train data is in another column (eg. 'code')
        custom_text_key: str = None,
        # Multi column merging support, used for instruct/input/output datasets
        # or similar varients where the input and output are in different columns
        # and need to be merged
        multi_column_keys: list = None,
        multi_column_prefix: list = None,
        multi_column_suffix: list = None,
        multi_column_train_mask: list = None,
        multi_column_separator: str = None,
        # prompt/completion format masking support
        disable_prompt_completion_mask: bool = False,
        # Skip database setup checks if datapath exists, ignored if using preload_datapath.py
        skip_datapath_setup: bool = False,
        
        # multichannel i/o (for binidx only)
        n_channel: int = 1,
        
        # other multichannel
        start_padding: bool = True,
        delay_pattern: bool = False,
        delay_pattern_groups: int = 1,
        padding_idx: int = 1024,
    ):
        # Capture the init parameters
        self._init_locals = locals()
        del self._init_locals["self"]
        del self._init_locals["__class__"]
        
        super().__init__()
        self.data_path = data_path
        self._loaded_dataset = None

        # Log to wandb
        if wandb.run is not None:
            wandb.config.update({ "data":dict(self._init_locals) })
    
    # Called once for initial setup
    def prepare_data(self):
        prepare_data_static(**self._init_locals)
    
    # Setup process that is universal
    def _internal_setup(self):
        if self._loaded_dataset is None:
            self._loaded_dataset = load_from_disk(self.data_path).with_format('torch')

    # Called once for every process in DDP
    def setup(self, stage):
        self._internal_setup()

    # Return the train dataloader
    def train_dataloader(self):
        self._internal_setup()
        return DataLoader(self._loaded_dataset['train'], num_workers=num_workers)
    
    # Return the validation dataloader
    def val_dataloader(self):
        self._internal_setup()
        return DataLoader(self._loaded_dataset['test'], num_workers=num_workers)
