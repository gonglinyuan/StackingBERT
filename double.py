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
    for k, v in lst:
        ckpt['model'][k] = v

    if len(sys.argv) > 3 and sys.argv[3] == '--double-optimizer':
        print('doubling the optimizer')
        new_optimizer_state = collections.OrderedDict()
        new_optimizer_state['state'] = collections.OrderedDict()
        new_optimizer_state['param_groups'] = [collections.OrderedDict()]
        for k in ['betas', 'eps', 'weight_decay', 'amsgrad']:
            new_optimizer_state['param_groups'][0][k] = ckpt['last_optimizer_state']['param_groups'][0][k]
        new_optimizer_state['param_groups'][0]['lr'] = 1e-7
        new_optimizer_state['param_groups'][0]['params'] = []
        head, layers, tail = [], [], []
        cnt = 0
        for k, v in ckpt['last_optimizer_state']['state'].items():
            if cnt < 2:
                head.append(v)
                print(f"head {v['exp_avg'].shape}")
            elif cnt < 2 + ckpt['args'].encoder_layers * 8:
                layers.append(v)
                print(f"layers {v['exp_avg'].shape}")
            else:
                tail.append(v)
                print(f"tail {v['exp_avg'].shape}")
            cnt += 1
        cnt = 0
        for it in head:
            it['step'] = 0
            new_optimizer_state['state'][cnt] = it
            new_optimizer_state['param_groups'][0]['params'].append(cnt)
            cnt += 1
        for it in layers:
            it['step'] = 0
            new_optimizer_state['state'][cnt] = it
            new_optimizer_state['param_groups'][0]['params'].append(cnt)
            cnt += 1
        for it in layers:
            it['step'] = 0
            new_optimizer_state['state'][cnt] = it
            new_optimizer_state['param_groups'][0]['params'].append(cnt)
            cnt += 1
        for it in tail:
            it['step'] = 0
            new_optimizer_state['state'][cnt] = it
            new_optimizer_state['param_groups'][0]['params'].append(cnt)
            cnt += 1
        ckpt['last_optimizer_state'] = new_optimizer_state

    ckpt['args'].encoder_layers *= 2
    torch.save(ckpt, sys.argv[2])


if __name__ == '__main__':
    main()
