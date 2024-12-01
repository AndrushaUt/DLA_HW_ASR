import re
from string import ascii_lowercase

import torch

from pyctcdecode import (
    Alphabet, 
    BeamSearchDecoderCTC,
    build_ctcdecoder,
)

import numpy as np
from collections import defaultdict

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, librispeech_vocab_path=None, lm_path=None,**kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """
        if librispeech_vocab_path is not None:
            with open(librispeech_vocab_path) as f:
                unigrams = [t.lower() for t in f.read().strip().split("\n")]
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if lm_path:
            self.decoder = build_ctcdecoder(self.vocab, lm_path, unigrams)
        else:
            self.decoder = BeamSearchDecoderCTC(Alphabet(self.vocab, False), None)

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_char_ind:
                continue
            elif ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_char_ind = ind
        return "".join(decoded)
    
    def ctc_beam_search_decode(self, inds: torch.Tensor, beam_size: int) -> str:
        return self.decoder.decode(inds, beam_width=beam_size)
    
    def ctc_beam_search(self, probs, beam_size=10) -> str:
        probs = np.exp(probs)
        dp = {('', self.EMPTY_TOK) : 1.0, }
        for prob in probs:
            dp = self.expand_and_merge_path(dp, prob)
            dp = self.truncate_paths(dp, beam_size)
        dp = [(prefix, proba) for (prefix, _), proba in sorted(dp.items(), key=lambda x : -x[1])][0][0]
        print(dp)
        return dp
    
    def expand_and_merge_path(self, dp, next_token_probs):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(next_token_probs):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char:
                    new_prefix = prefix
                else:
                    if cur_char != self.EMPTY_TOK:
                        new_prefix = prefix + cur_char
                    else:
                        new_prefix = prefix
                new_dp[(new_prefix, cur_char)] += v * next_token_prob
        return new_dp

    def truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key = lambda x : -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
