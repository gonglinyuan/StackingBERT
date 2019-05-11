import collections
import sys

import torch


def main():
    ckpt = torch.load(sys.argv[1])

    lst = []
    for k, v in ckpt['model'].items():
        k_split = k.split('.')
        if k_split[0] == 'encoder' and k_split[1] == 'layers':
            l_id = int(k_split[2])
            k_split[2] = str(l_id + ckpt['args'].encoder_layers)
            new_k = '.'.join(k_split)
            lst.append([new_k, v.clone()])
            k_split[2] = str(l_id + ckpt['args'].encoder_layers * 2)
            new_k = '.'.join(k_split)
            lst.append([new_k, v.clone()])
            k_split[2] = str(l_id + ckpt['args'].encoder_layers * 3)
            new_k = '.'.join(k_split)
            lst.append([new_k, v.clone()])
    for k, v in lst:
        ckpt['model'][k] = v

    ckpt['args'].encoder_layers *= 4
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
