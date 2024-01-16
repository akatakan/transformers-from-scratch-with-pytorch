import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_dst, src_lang, dst_lang, max_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.src_lang = src_lang
        self.dst_lang = dst_lang

        self.bos_token = torch.Tensor([tokenizer_src.token_to_id('[BOS]')],dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id('[EOS]')],dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id('[PAD]')],dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_dst_pair = self.ds[index]
        src_text = src_dst_pair['translation'][self.src_lang]
        dst_text = src_dst_pair['translation'][self.dst_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_dst.encode(dst_text).ids

        enc_num_padding_tokens = self.max_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
            raise ValueError("Sentence is too long")
        
        encoder_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]* enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.bos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token]*dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.max_len
        assert decoder_input.size(0) == self.max_len
        assert label.size(0) == self.max_len

        return{
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), 1).type(torch.int)
    return mask == 0