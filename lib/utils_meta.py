from collections import OrderedDict
import torch
import torch.nn.functional as F


def gradient_update_bn_parameters(loss,
                               params=None,
                               model=None,
                               step_size=1e-3,
                               first_order=True):

    #if not isinstance(model, MetaModule):
    #    raise ValueError('The model must be an instance of `torchmeta.modules.'
    #                     'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    bn_params = OrderedDict()
    for name in params.keys():
        if name.find('norm') >= 0:
            bn_params[name] = params[name]

    grads = torch.autograd.grad(loss,
                                bn_params.values(), create_graph=True, retain_graph=True)
    grads = OrderedDict(zip(bn_params.keys(), grads))

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for name in params.keys():
            if name.find('norm') >= 0:
                updated_params[name] = params[name] - step_size[name] * grads[name].detach()
            else:
                updated_params[name] = params[name]

    else:
        for name in params.keys():
            if name.find('norm') >= 0:
                updated_params[name] = params[name] - step_size * grads[name].detach()
            else:
                updated_params[name] = params[name]

    return updated_params


def gradient_update_other_parameters(loss,
                               params=None,
                               model=None,
                               step_size=0.5,
                               first_order=False):

    #if not isinstance(model, MetaModule):
    #    raise ValueError('The model must be an instance of `torchmeta.modules.'
    #                     'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    other_params = OrderedDict()
    for name in params.keys():
        if name.find('norm') < 0:
            other_params[name] = params[name]

    grads = torch.autograd.grad(loss,
                                other_params.values(),
                                create_graph=not first_order)
    grads = OrderedDict(zip(other_params.keys(), grads))

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for name in params.keys():
            if name.find('norm') < 0:
                updated_params[name] = params[name] - step_size[name] * grads[name]
            else:
                updated_params[name] = params[name]

    else:
        for name in params.keys():
            if name.find('norm') < 0:
                updated_params[name] = params[name] - step_size * grads[name]
            else:
                updated_params[name] = params[name]

    return updated_params


@torch.no_grad()
def update_client_parameters(server, client, alpha=0.99):
    for param_s, param_c in zip(server.parameters(), client.parameters()):
        param_c.data = param_c.data * alpha + param_s.data * (1. - alpha)


def communication(server_params, client_params, alpha_server=0.99, alpha_client=0.99, side="server"):
    
    #server_params = server.meta_named_parameters()
    #client_params = client.meta_named_parameters()

    if side == "server":
        for name in server_params.keys():
            if name.find('norm') >= 0:
                server_params[name] = alpha_server * server_params[name] + (1. - alpha_server) * client_params[name]
        return server_params
    elif side == "client":
        for name in client_params.keys():
            if name.find('norm') >= 0:
                client_params[name] = alpha_client * client_params[name] + (1. - alpha_client) * server_params[name]
        return client_params
    else:
        print("Unknown side. Choose server or client")
        sys.exit()


def info_nce_loss(z1, z2, temperature=0.1):

    n_views = 2
    batch_size = z1.shape[0]
    device = z1.get_device()
    features = torch.cat((z1, z2), 0)

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels



