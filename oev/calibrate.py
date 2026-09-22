import torch


def apply_temperature(logits, temperature):
    return logits / temperature


def fit_temperature_from_logits(logits, labels, max_iter=200, lr=0.05):
    logits = logits.detach().float().cpu()
    labels = labels.detach().cpu()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=max_iter, lr=lr)
    labels = labels.long()

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())
