import sys

import torch


def main():
    ckpt = torch.load(sys.argv[1])
    lst = []
    for k, v in ckpt['model'].items():
        k_split = k.split('.')
        if k_split[0] == 'encoder' and k_split[1] == 'layers':
            id = int(k_split[2])
            k_split[2] = str(id + ckpt['args'].encoder_layers)
            new_k = '.'.join(k_split)
            lst.append([new_k, v.clone()])
    for k, v in lst:
        k_split = k.split('.')
        if k_split[-2] in ['fc2', 'out_proj']:
            ckpt['model'][k] = torch.zeros_like(v)
        elif k_split[-1].endswith('bias'):
            ckpt['model'][k] = torch.zeros_like(v)
        else:
            # Kaiming normal
            std = v.size(0) ** -0.5
            ckpt['model'][k] = torch.randn_like(v) * std
    ckpt['args'].encoder_layers *= 2
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
