import pandas as pd
import numpy as np
from Misc.ParsedData import ParsedData

block_size = 512
blocks_in_mb = 2 ** 11
blocks_in_gb = 2 ** 21
static_disk_size_in_blocks = block_size * blocks_in_gb
us_to_sec = 10 ** 6
res_0bit = 2 ** 0
res_8bit = 2 ** 8
res_16bit = 2 ** 16
res_24bit = 2 ** 24
res_32bit = 2 ** 32
all_bytes_res = [res_0bit, res_8bit, res_16bit, res_24bit, res_32bit]
dtypes = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'uint24': np.uint32,
    'uint32': np.uint32,
    'float16': np.float16,
    'float32': np.float32,
}
to_bytes = [dtypes['uint8'], dtypes['uint16'], dtypes['uint24'], dtypes['uint32']]

ts = 'Timestamp'
dt = 'time_lapse'  # Time difference between consecutive commands
opc = 'OpCode'
slba = 'SLBA'
nlb = 'NLB'
war = 'WaR'
waw = 'WaW'
raw = 'RaW'
rar = 'RaR'

historic_mean_dt = 0

MAX_NLB = 4096


class CLTTokenizer:

    def __init__(self):
        self.token_parts = ['quantized_time_lapse', 'quantized_nlb', 'quantized_opc', 'quantized_slba_msb_dynamic',
                            'quantized_slba_lsb', 'quantized_war', 'quantized_rar', 'quantized_raw']

        self.quantizations = [4, 4, 1, 4, 2, 1, 1, 1]
        self.byte_res = 2
        self.dtype = 'uint16'
        self.diminish_factor = 0

        self.tokenizers = {
            'opc': self._generate_token_part_opcode,
            'war': self._generate_token_part_war,
            'waw': self._generate_token_part_waw,
            'raw': self._generate_token_part_raw,
            'rar': self._generate_token_part_rar,
            'quantized_slba_msb_dynamic': self._generate_quantized_slba_msb_dynamic,
            'quantized_slba_lsb': self._generate_quantized_slba_lsb,
            'quantized_nlb': self._generate_quantized_nlb,
            'quantized_time_lapse': self._generate_quantized_time_lapse,
            'quantized_opc': self._generate_quantized_opcode,
            'quantized_war': self._generate_quantized_war,
            'quantized_rar': self._generate_quantized_rar,
            'quantized_raw': self._generate_quantized_raw,
            'quantized_waw': self._generate_quantized_waw,
        }

        self.cur_rec_id = -1
        self.prev_chunk_last_ts = 0
        self.cur_disk_size_in_blocks = -1
        self.exp_mean_dt = historic_mean_dt

    def generate_per_chunk(self, parsed_data: ParsedData) -> np.ndarray:
        chunk_data = parsed_data.data
        self.cur_disk_size_in_blocks = parsed_data.metadata
        if not self.quantizations:
            # noinspection PyArgumentList
            tokens = np.hstack([self.tokenizers[part](chunk_data) for part in self.token_parts])
            tokens = tokens.ravel().astype(to_bytes[self.byte_res - 1]) if self.byte_res else tokens
        else:
            # noinspection PyArgumentList
            tokens = self._merge_bits([self.tokenizers[part](chunk_data, quant)
                                       for part, quant in zip(self.token_parts, self.quantizations)]).astype(self.dtype)
        self.prev_chunk_last_ts = chunk_data[ts].iloc[-1]
        return tokens

    def _generate_token_part_opcode(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(df[opc].astype(np.uint8), dtype=self.dtype)[:, np.newaxis]

    def _generate_token_part_war(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(~df[war].isna() & (df[war] > 0), dtype=self.dtype)[:, np.newaxis]

    def _generate_token_part_waw(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(~df[waw].isna() & (df[waw] > 0), dtype=self.dtype)[:, np.newaxis]

    def _generate_token_part_raw(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(~df[raw].isna() & (df[raw] > 0), dtype=self.dtype)[:, np.newaxis]

    def _generate_token_part_rar(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(~df[rar].isna() & (df[rar] > 0), dtype=self.dtype)[:, np.newaxis]

    def _generate_quantized_slba_msb_dynamic(self, df: pd.DataFrame, n_bits: int) -> np.ndarray:
        return np.array(df[slba].astype(np.int64) // (self.cur_disk_size_in_blocks // (2 ** n_bits)),
                        dtype=self.dtype)[:, np.newaxis]

    def _generate_quantized_slba_lsb(self, df: pd.DataFrame, n_bits: int) -> np.ndarray:
        return np.array(df[slba].astype(np.int64) // (blocks_in_mb * 2) % (2 ** n_bits),
                        dtype=self.dtype)[:, np.newaxis]

    def _generate_quantized_nlb(self, df: pd.DataFrame, n_bits: int = 4) -> np.ndarray:
        arr = np.array(np.floor(np.log2(df[nlb].clip(1, MAX_NLB).astype(np.uint16))), dtype=int)
        arr[df[nlb] == 1024] = 13  # using the extra capacity to map some "special" and common NLB values
        arr[df[nlb] == 256] = 14
        arr[df[nlb] == 32] = 15
        return arr.astype(self.dtype)[:, np.newaxis]

    def _generate_quantized_time_lapse(self, df: pd.DataFrame, n_bits: int = 4) -> np.ndarray:
        # cur_mean = df[dt].mean()
        # self.exp_mean_dt = self.exp_mean_dt if self.exp_mean_dt else cur_mean
        # self.exp_mean_dt = self.diminish_factor * self.exp_mean_dt + (1 - self.diminish_factor) * cur_mean
        return np.array(np.log2(self.get_dt(df).astype(np.float64) * us_to_sec / 10 + 1).round().clip(0, 2 ** n_bits - 1),
                        dtype=self.dtype)[:, np.newaxis]

    def _generate_quantized_opcode(self, df: pd.DataFrame, n_bits: int = 1) -> np.ndarray:
        return self._generate_token_part_opcode(df)

    def _generate_quantized_war(self, df: pd.DataFrame, n_bits: int = 1) -> np.ndarray:
        return self._generate_token_part_war(df)

    def _generate_quantized_rar(self, df: pd.DataFrame, n_bits: int = 1) -> np.ndarray:
        return self._generate_token_part_rar(df)

    def _generate_quantized_raw(self, df: pd.DataFrame, n_bits: int = 1) -> np.ndarray:
        return self._generate_token_part_raw(df)

    def _generate_quantized_waw(self, df: pd.DataFrame, n_bits: int = 1) -> np.ndarray:
        return self._generate_token_part_waw(df)

    def _generate_index(self, index: int, length: int) -> np.ndarray:
        return np.array([index] * length, dtype=self.dtype)

    def _to_bytes(self, tok: np.ndarray, n_bytes: int) -> np.ndarray:
        arr = np.array([(tok // all_bytes_res[i * self.byte_res]) % self.byte_res for i in range(n_bytes)])
        return arr[:, np.newaxis] if arr.ndim == 1 else arr.T

    def _merge_bits(self, proc_token_parts: list) -> np.ndarray:
        index = 0
        counter = 0
        token_size = 8 * self.byte_res
        total_tok_len = sum(self.quantizations)
        index_size = int(np.ceil(np.log2(np.ceil(total_tok_len / token_size))))
        group_size = token_size - index_size
        n_tokens = int(np.ceil(total_tok_len / group_size))
        group_fill = int(np.ceil(total_tok_len / n_tokens))
        arr = np.zeros((proc_token_parts[0].shape[0], n_tokens))
        for i, (tok, tok_len) in enumerate(zip(proc_token_parts, self.quantizations)):
            arr[:, counter // token_size] += tok.squeeze() * 2 ** (counter % token_size)
            counter += tok_len
            if index_size and (counter % token_size == group_fill or i == len(proc_token_parts) - 1):
                arr[:, counter // token_size] += self._generate_index(index, len(tok)) * 2 ** group_fill
                counter = (counter // token_size + 1) * token_size
                index += 1
        return arr.ravel()

    def get_dt(self, df: pd.DataFrame) -> pd.Series:
        if dt in df:
            tl = df[dt]
        else:
            dt0 = df[ts].iloc[0] - self.prev_chunk_last_ts if self.prev_chunk_last_ts else 0
            tl = df[ts].diff().fillna(dt0).clip(0, np.inf)
        return tl

