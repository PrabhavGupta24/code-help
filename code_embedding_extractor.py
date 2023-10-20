# coding=utf-8
# Copyright 2018-2023 EvaDB
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
import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.udfs.abstract.abstract_udf import AbstractUDF
from evadb.udfs.decorators.decorators import forward, setup
from evadb.udfs.decorators.io_descriptors.data_types import PandasDataframe
from evadb.udfs.gpu_compatible import GPUCompatible

from transformers import AutoTokenizer, AutoModel



class CodeEmbeddingExtractor(AbstractUDF, GPUCompatible):
    @setup(cacheable=False, udf_type="FeatureExtraction", batchable=False)
    def setup(self):
        self.model_name = "Lazyhope/RepoSim"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def to_device(self, device: str) -> GPUCompatible:
        self.model = self.model.to(device)
        return self

    @property
    def name(self) -> str:
        return "CodeEmbeddingExtractor"

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.ANYTYPE],
                column_shapes=[(1)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["embeddings"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(1, 768)],
            )
        ],
    )
    # def forward(self, df: pd.DataFrame) -> pd.DataFrame:
    #     def _forward(row: pd.Series) -> np.ndarray:
    #         data = row
    #         embedded_list = self.model.encode(data)
    #         return embedded_list

    #     ret = pd.DataFrame()
    #     ret["features"] = df.apply(_forward, axis=1)
    #     return ret
    
    
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        def _forward(row: pd.Series) -> np.ndarray:
            data = row
            inputs = self.tokenizer(data, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).detach().numpy()

        ret = pd.DataFrame()
        ret["embeddings"] = df.apply(_forward, axis=1)
        return ret
