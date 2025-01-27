from typing import List
import numpy as np
import pandas as pd
from Misc.ParsedData import ParsedData

token_width = 50
number_of_tokens = 100

class PLTTokenizer:

    def __init__(self):

        self.number_of_tokens = number_of_tokens
        self.token_width = token_width
        self.diminish_factor = 0.8
        self.last_datasrc_id = 0
        self.avg_dt = None
        self.avg_war_lapse = None
        self.avg_rar_lapse = None
        self.embedding_dim = 181
        self.bin_nlb = 13
        self.bin_hists = 15

    @staticmethod
    def Lqnorm_weighted(df: pd.DataFrame, x: str, weight: str, q: int = 2) -> tuple:
        w = df[weight] / df[weight].mean()
        x_mean = np.mean(df[x] * w)
        x_std = (np.mean((np.abs(df[x] - x_mean) ** q) * w)) ** (1 / q)
        return x_mean, x_std

    @staticmethod
    def get_plt_labels(chunk_data):
        #  Tag per chunk

        df = chunk_data.copy()

        cumsum_nlb = df['NLB'].cumsum()
        vmin = cumsum_nlb.min()
        vmax = cumsum_nlb.max() + 1
        dv = ((vmax - vmin) - token_width / 512e-6) / (number_of_tokens - 1)  # in nlb units
        v0s = np.arange(start=vmin, stop=vmax, step=dv)[:number_of_tokens]

        df['dt'] = df['Timestamp'].diff()

        channels = []
        for v0 in v0s:
            condition = (cumsum_nlb >= v0) \
                        & (cumsum_nlb < v0 + token_width / 512e-6)

            df_token = df[condition]

            rw_NLB_read, rw_NLB_write = np.nan, np.nan
            rw_NLB_war, rw_NLB_rar = np.nan, np.nan

            condition_read = df_token['OpCode'] == 0
            NLB_read = df_token[condition_read]['NLB'].to_numpy().sum()
            NLB_write = df_token[~condition_read]['NLB'].to_numpy().sum()
            chunk_volume = NLB_read + NLB_write

            if 'WaR' in df_token.columns:
                condition_war = df_token['WaR'] > 0

            if 'RaR' in df_token.columns:
                condition_rar = df_token['RaR'] > 0

            if 'Label' in df_token.columns:
                condition_rw = df_token['Label'] == 1.0
                rw_NLB_read = df_token[condition_read & condition_rw]['NLB'].to_numpy().sum()
                rw_NLB_write = df_token[~condition_read & condition_rw]['NLB'].to_numpy().sum()

                if 'WaR' in df_token.columns:
                    rw_NLB_war = df_token[condition_war & condition_rw]['WaR'].to_numpy().sum()
                if 'RaR' in df_token.columns:
                    rw_NLB_rar = df_token[condition_rar & condition_rw]['RaR'].to_numpy().sum()

            channels.append([rw_NLB_read / chunk_volume, rw_NLB_write / chunk_volume,
                             rw_NLB_war / chunk_volume, rw_NLB_rar / chunk_volume])

        channels = np.array(channels).T

        return channels

    def generate_per_chunk(self, parsed_data: ParsedData) -> np.array:
        eps = 1e-10

        df = parsed_data.data.copy()

        cumsum_nlb = df['NLB'].cumsum()
        vmin = cumsum_nlb.min()
        vmax = cumsum_nlb.max() + 1
        dv = ((vmax - vmin) - self.token_width / 512e-6) / (self.number_of_tokens - 1)  # in nlb units
        v0s = np.arange(start=vmin, stop=vmax, step=dv)[:self.number_of_tokens]

        df['dt'] = df['Timestamp'].diff()
        slba0, dslba = self.Lqnorm_weighted(df=df, x='SLBA', weight='NLB')

        dt0, _ = self.Lqnorm_weighted(df=df.iloc[1:, :], x='dt', weight='NLB')
        Tchunk0 = df.Timestamp.max() - df.Timestamp.min()

        if self.avg_dt is None:
            self.avg_dt = dt0
            self.avg_Tchunk = Tchunk0
        else:
            self.avg_dt = (self.diminish_factor * self.avg_dt + (1 - self.diminish_factor) * dt0)
            self.avg_Tchunk = (self.diminish_factor * self.avg_Tchunk + (1 - self.diminish_factor) * Tchunk0)

        n = np.linspace(0, 12, self.bin_nlb)
        bins_normalized_dt = np.linspace(0, 3, self.bin_hists)
        bins_normalized_slba = np.linspace(-3, 3, self.bin_hists)
        vol_token = self.token_width / 512e-6

        channels = []
        for v0 in v0s:
            condition = (cumsum_nlb >= v0) \
                        & (cumsum_nlb < v0 + self.token_width / 512e-6)

            df_token = df[condition]

            if df_token.empty:
                features = [0] * self.embedding_dim
            else:
                # volume of commands in token
                frac = df_token.NLB.sum() / vol_token
                frac_r = df_token[df_token.OpCode == 0].NLB.sum() / vol_token
                frac_w = df_token[df_token.OpCode == 1].NLB.sum() / vol_token
                frac_war = df_token.WaR.sum() / vol_token
                frac_rar = df_token.RaR.sum() / vol_token

                # number of commands in token
                len_r = len(df_token[df_token.OpCode == 0])
                len_w = len(df_token[df_token.OpCode == 1])
                len_war = len(df_token[df_token.WaR > 0])
                len_rar = len(df_token[df_token.RaR > 0])

                features_simple = [frac, frac_r, frac_w, frac_war, frac_rar,
                                   len_r, len_w, len_war, len_rar]

                # nlb histograms in token
                hist_nlb = np.hstack(
                    [np.histogram(np.clip(np.log2(df_token[df_token.OpCode == 0].NLB), n[0], n[-1]), bins=n)[0],
                     np.histogram(np.clip(np.log2(df_token[df_token.OpCode == 1].NLB), n[0], n[-1]), bins=n)[0]])
                hist_nlb = hist_nlb / (hist_nlb.sum() + eps)

                hist_war = \
                    np.histogram(np.clip(np.log2(df_token[df_token.WaR > 0].WaR), n[0], n[-1]), bins=n)[0]
                hist_rar = \
                    np.histogram(np.clip(np.log2(df_token[df_token.RaR > 0].RaR), n[0], n[-1]), bins=n)[0]
                hist_rest = np.histogram(
                    np.clip(np.log2(
                        df_token[(~(df_token.WaR > 0)) & (~(df_token.RaR > 0))].NLB), n[0], n[-1]),
                    bins=n)[0]
                hist_afters = np.hstack([hist_war, hist_rar, hist_rest])
                hist_afters = hist_afters / (hist_afters.sum() + eps)

                features_hists_nlb = list(hist_nlb) + list(hist_afters)

                # dt histograms in token
                dt_normalized = np.clip(df_token.dt / self.avg_dt, 0, bins_normalized_dt[-1])
                hist_normalized_dt = np.histogram(dt_normalized, weights=df_token.NLB, bins=bins_normalized_dt)[0]
                hist_normalized_dt = list(hist_normalized_dt / (hist_normalized_dt.sum() + eps))

                # lapses histograms in token
                war_lapses_normalized = np.clip(
                    df_token[df_token.WaR > 0]['WaR Lapse'] / (10 * self.avg_dt), 0, bins_normalized_dt[-1])
                rar_lapses_normalized = np.clip(
                    df_token[df_token.RaR > 0]['RaR Lapse'] / (10 * self.avg_dt), 0, bins_normalized_dt[-1])

                hist_normalized_war_lapse = np.histogram(
                    war_lapses_normalized,
                    weights=df_token[df_token.WaR > 0].WaR,
                    bins=bins_normalized_dt)[0]
                hist_normalized_rar_lapse = np.histogram(
                    rar_lapses_normalized,
                    weights=df_token[df_token.RaR > 0].RaR,
                    bins=bins_normalized_dt)[0]
                hist_lapses = list(hist_normalized_war_lapse / (hist_normalized_war_lapse.sum() + eps)) + \
                              list(hist_normalized_rar_lapse / (hist_normalized_rar_lapse.sum() + eps))

                # slba histograms in token
                normalized_slbas_r = np.clip((df_token[df_token.OpCode == 0].SLBA - slba0) / dslba,
                                             bins_normalized_slba[0], bins_normalized_slba[-1])
                normalized_slbas_w = np.clip((df_token[df_token.OpCode == 1].SLBA - slba0) / dslba,
                                             bins_normalized_slba[0], bins_normalized_slba[-1])

                hist_normalized_slbar = np.histogram(
                    normalized_slbas_r, weights=df_token[df_token.OpCode == 0].NLB, bins=bins_normalized_slba)[0]
                hist_normalized_slbaw = np.histogram(
                    normalized_slbas_w, weights=df_token[df_token.OpCode == 1].NLB, bins=bins_normalized_slba)[0]

                hist_slba = np.hstack([hist_normalized_slbar, hist_normalized_slbaw])
                hist_slba = list(hist_slba / (hist_slba.sum() + eps))

                # slba(over-actions) histograms in token
                normalized_slbas_war = np.clip((df_token[df_token.WaR > 0].SLBA - slba0) / dslba,
                                               bins_normalized_slba[0], bins_normalized_slba[-1])
                normalized_slbas_rar = np.clip((df_token[df_token.RaR > 0].SLBA - slba0) / dslba,
                                               bins_normalized_slba[0], bins_normalized_slba[-1])
                normalized_slbas_rest = np.clip(
                    (df_token[(~(df_token.WaR > 0)) & (~(df_token.RaR > 0))].SLBA - slba0) / dslba,
                    bins_normalized_slba[0], bins_normalized_slba[-1])

                hist_normalized_slbawar = np.histogram(
                    normalized_slbas_war, weights=df_token[df_token.WaR > 0].NLB, bins=bins_normalized_slba)[0]
                hist_normalized_slbarar = np.histogram(
                    normalized_slbas_rar, weights=df_token[df_token.RaR > 0].NLB, bins=bins_normalized_slba)[0]
                hist_normalized_slbarest = np.histogram(
                    normalized_slbas_rest,
                    weights=df_token[(~(df_token.WaR > 0)) & (~(df_token.RaR > 0))].NLB,
                    bins=bins_normalized_slba)[0]

                hist_slba_over = np.hstack(
                    [hist_normalized_slbawar, hist_normalized_slbarar, hist_normalized_slbarest]
                )
                hist_slba_over = list(hist_slba_over / (hist_slba_over.sum() + eps))

                features = features_simple + \
                           features_hists_nlb + \
                           hist_normalized_dt + \
                           hist_slba + \
                           hist_slba_over + \
                           hist_lapses

            channels.append(features)

        channels = np.array(channels).T

        # normalize commands' number by the read+write number of commands, averaged across tokens
        mean_token_command_number = channels[5:7, :].sum(axis=0).mean()
        channels[5:9, :] /= mean_token_command_number

        return channels.astype(np.float16)
