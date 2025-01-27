import numpy as np
import pandas as pd
from Misc.Constants import NVMeOpCodeREAD, NVMeOpCodeWRITE
from Misc.ParsedData import ParsedData


class RFFeaturizer:

    def __init__(self):
        self.nlb_histogram_bins = [1, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025]

    def generate_per_chunk(self, parsed_data: ParsedData) -> np.array:
        df = parsed_data.data
        write_nlbs = np.array(df[df['OpCode'] == NVMeOpCodeWRITE]['NLB'])
        writes_war_lapses = np.array(df[df['OpCode'] == NVMeOpCodeWRITE]['WaR Lapse'])
        writes_war_lengths = np.array(df[df['OpCode'] == NVMeOpCodeWRITE]['WaR'])
        fractions = writes_war_lengths / write_nlbs
        war_lapses = writes_war_lapses[fractions > 0]

        read_nlbs = []
        war_lengths = []

        accum_read = 0
        accum_write = 0
        accum_war = 0

        for i in range(len(df)):
            row = df.iloc[i]

            if row['OpCode'] == NVMeOpCodeWRITE:
                accum_write += row['NLB']
                fraction = row['WaR'] / row['NLB']
                if fraction > 0:
                    accum_war += row['WaR']
                    war_lengths.append(row['NLB'])
            elif row['OpCode'] == NVMeOpCodeREAD:
                accum_read += row['NLB']
                read_nlbs.append(row['NLB'])

        hist_r = np.histogram(np.clip(read_nlbs, 1, self.nlb_histogram_bins[-1]), self.nlb_histogram_bins)[0]
        hist_war = np.histogram(np.clip(war_lengths, 1, self.nlb_histogram_bins[-1]), self.nlb_histogram_bins)[0]

        if len(war_lapses) > 0:
            dev = np.std(war_lapses) / np.mean(war_lapses)
            if np.mean(war_lapses) == 0:
                dev = None
        else:
            dev = None

        return np.concatenate([[accum_read / np.sum(df['NLB'])],
                               [accum_war / np.sum(df['NLB'])],
                               [dev if dev is not None else 0],
                               hist_r,
                               hist_war])
