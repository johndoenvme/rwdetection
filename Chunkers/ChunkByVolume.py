from typing import List
import pandas as pd
from Misc.ParsedData import ParsedData

SLICE_VOLUME = 2 ** 20


class ChunkByVolume:

    def __init__(self, parsed_data: List[ParsedData]):
        self.parsed_data = parsed_data

    @staticmethod
    def create_chunk_indices_series(data: pd.DataFrame) -> pd.Series:
        target_sum = SLICE_VOLUME

        nlb_series: pd.Series = data['NLB']

        chunk_series = pd.Series([-1] * nlb_series.size, dtype=int)
        indices: List[int] = []
        current_chunk_index = 0
        current_sum = 0

        for index, value in nlb_series.items():
            current_sum += value
            indices.append(index)
            if current_sum > target_sum:
                chunk_series[indices] = current_chunk_index
                current_chunk_index += 1
                indices.clear()
                current_sum = 0

        return chunk_series

