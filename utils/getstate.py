import os
import rwkv
import torch
import re
from safetensors import safe_open
from safetensors.torch import save_file, save_model
from collections import OrderedDict
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help="RWKV model")
    parser.add_argument('-o', '--output', required=True, help="The state")
    args = parser.parse_args()

    state_raw = torch.load(args.input)
    time_state = OrderedDict()
    for i, tensor_name in enumerate(state_raw):
        if "att.time_state" in tensor_name:
            time_state[tensor_name] = state_raw[tensor_name].transpose(1, 2).contiguous()
    save_file(time_state, args.output)


if __name__ == "__main__":
    main()
