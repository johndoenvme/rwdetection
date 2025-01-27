import numpy as np
import pandas as pd
from Misc.Constants import NVMeOpCodeREAD, NVMeOpCodeWRITE
from scipy.stats import entropy
from Misc.ParsedData import ParsedData


class DeftPunkFeaturizer:

    def __init__(self):
        pass

    def generate_per_chunk(self, parsed_data: ParsedData) -> np.array:
        df = parsed_data.data
        #  Distinguish between all command types
        RAW = df['RaW'] > 0
        WAR = df['WaR'] > 0
        RAR = df['RaR'] > 0
        WAW = df['WaW'] > 0
        R = df['OpCode'] == NVMeOpCodeREAD
        W = df['OpCode'] == NVMeOpCodeWRITE
        IO_types = [R, W, RAW, WAR, RAR, WAW]

        timelapses = [df[R].iloc[-1]['Timestamp'] - df[R].iloc[0]['Timestamp'] if R.any() else 0,  # Read
                      df[W].iloc[-1]['Timestamp'] - df[W].iloc[0]['Timestamp'] if W.any() else 0,  # Write
                      df.iloc[-1]['Timestamp'] - df.iloc[0]['Timestamp']]  # Total

        #  DeftPunk's IO Dependency on Block
        IO_dependency_counts = [x.sum() for x in IO_types[2:]]  # DeftPunk: IO Counts of RAW/WAR/RAR/WAW
        IO_dependency_GB = [df[x].NLB.sum() * 512e-9 for x in
                            IO_types[2:]]  # DeftPunk: IO Bytes (GB, actually) of RAW/WAR/RAR/WAW

        #  DeftPunk's IO Statistics
        IO_statistics_counts = [R.sum(), W.sum(), len(df)]  # IO Counts of R / W / R+W
        IO_statistics_GB = [df[R].NLB.sum() * 512e-9, df[W].NLB.sum() * 512e-9,
                            df.NLB.sum() * 512e-9]  # (Giga)bytes of R / W / R+W
        IO_statistics_Size = [x / y if y != 0 else 0 for x, y in
                              zip(IO_statistics_GB, IO_statistics_counts)]  # Size of R / W / R+W
        IO_statistics_GBps = [x / y if y != 0 else 0 for x, y in
                              zip(IO_statistics_GB, timelapses)]  # (Giga)Bps of R / W / R+W

        #  DeftPunk's Working Set Size (WSS)
        df["covered_blocks"] = df.apply(self.covered_blocks, axis=1)  # Calculate covered blocks for each row
        WSS = [len(set.union(*df[x]["covered_blocks"])) if len(df[x]) > 0 else 0 for x in IO_types]

        #  DeftPunk's IO Offset Statistics
        IO_offset_statistics = [df.SLBA.var() * (512e-9 ** 2),
                                np.sqrt(df.SLBA.var()) / (df.SLBA.mean() + 1e-5)]  # VAR(SLBA) CoV(SLBA)
        IO_offset_statistics += [entropy(df[x].SLBA.to_numpy()) for x in IO_types]  # Entropy(SLBA) of all IO types

        #  DeftPunk's Access on LBA Head Region
        disk_heads = [df.SLBA * 512e-9 < 0.1, df.SLBA * 512e-9 < 1.0]  # The disk's first [100MB, 1GB]
        Access_LBA_Head_region = [len(df[x & y]) for x in disk_heads for y in [R, W]]  # IO count
        Access_LBA_Head_region += [df[x & y].NLB.sum() * 512e-9 for x in disk_heads for y in [R, W]]  # IO (Giga)Bytes

        deft_punk_features = np.concatenate([IO_dependency_counts,  # 4
                                             IO_dependency_GB,  # 4
                                             IO_statistics_counts,  # 3
                                             IO_statistics_GB,  # 3
                                             IO_statistics_Size,  # 3
                                             IO_statistics_GBps,  # 3
                                             WSS,  # 6
                                             IO_offset_statistics,  # 8 (2 + 6)
                                             Access_LBA_Head_region])  # 8

        deft_punk_features = np.nan_to_num(deft_punk_features)  # Remove NaNs

        return deft_punk_features

    @staticmethod
    def covered_blocks(row):
        """
        Calculates the blocks covered between SLBA and SLBA+NLB
        """
        return set(range(int(row.SLBA), int(row.SLBA) + int(row.NLB)))
