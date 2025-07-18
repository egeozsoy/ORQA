# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import types

from llamafactory.model.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from llamafactory.model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

# Step 2: Create fake modules to override transformers' imports
local_qwen2_vl_processing = types.ModuleType("transformers.models.qwen2_vl.processing_qwen2_vl")
local_qwen2_vl_processing.Qwen2VLProcessor = Qwen2VLProcessor

local_qwen2_vl_image_processing = types.ModuleType("transformers.models.qwen2_vl.image_processing_qwen2_vl")
# <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>
local_qwen2_vl_image_processing.Qwen2VLImageProcessor = Qwen2VLImageProcessor

# Step 3: Inject into sys.modules to override the original references
sys.modules["transformers.models.qwen2_vl.processing_qwen2_vl"] = local_qwen2_vl_processing
sys.modules["transformers.models.qwen2_vl.image_processing_qwen2_vl"] = local_qwen2_vl_image_processing

from llamafactory.train.tuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
