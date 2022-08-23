import torch
from torch import optim as optim


try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def split_weights(model, joint_optimizer_lrs, weight_decay=None, skip_list=[]):
    res_params = []
    if weight_decay is None:
        res_params = \
        [{'params': model.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
        {'params': model.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        ]
        # if hasattr(model, 'last_layer'):
        #     res_params.append({'params': model.last_layer.parameters(), 'lr': joint_optimizer_lrs['prototype_vectors']})
        if hasattr(model, 'prototype_vectors'):
            res_params.append({'params': model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']})
        if hasattr(model, 'prototype_vectors_global'):
            res_params.append({'params': model.prototype_vectors_global, 'lr': joint_optimizer_lrs['prototype_vectors']})
    else:
        for module_name in ['features', 'add_on_layers', 'last_layer']:
            module = getattr(model, module_name)
            for name, param in module.named_parameters():
                decay, no_decay = [], []
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            res_params.append({'params': no_decay, 'lr': joint_optimizer_lrs[module_name], 'weight_decay': 0.})
            res_params.append({'params': decay, 'lr': joint_optimizer_lrs[module_name], 'weight_decay': weight_decay})
        res_params.append({'params': model.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors'], 'weight_decay': weight_decay})
    return res_params


def create_optimizer(args, model, joint_optimizer_lrs=None, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay
    # if weight_decay and filter_bias_and_bn:
    #     skip = {}
    #     if hasattr(model, 'no_weight_decay'):
    #         skip = model.no_weight_decay()
    #     parameters = add_weight_decay(model, weight_decay, skip)
    #     weight_decay = 0.
    # else:
    #     parameters = model.parameters()

    if joint_optimizer_lrs is not None:
        # parameters = split_weights(model, joint_optimizer_lrs, weight_decay)
        parameters = split_weights(model, joint_optimizer_lrs)
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
