import torch


class FitnessEWMALogger:
    """Logger for the fitness values of data batches"""

    def __init__(self, data_gen, model, criterion):
        self.ewma_alpha = 1
        self.iter_counter = 1
        self.num_batches = len(data_gen)
        self.ewma = torch.zeros(self.num_batches)
        # FIXME
        # sum of losses per batch for the current iteration
        self.current_losses = torch.zeros(self.num_batches).to(torch.device("cpu"))
        # count of evaluations per batch for the current iteration
        self.current_counts = torch.zeros(self.num_batches).to(torch.device("cpu"))
        self.set_initial_losses(data_gen, model, criterion)

    def set_initial_losses(self, data_gen, model, criterion):
        # XXX this is really ugly
        model.cuda()
        for batch_idx, (b_x, y) in data_gen:
            out = model(b_x)
            loss = criterion(out, y).item()
            self.ewma[batch_idx] = loss
            if batch_idx >= self.num_batches - 1:
                break

    def update_batch(self, batch_idx, loss):
        self.current_losses[batch_idx] += loss
        self.current_counts[batch_idx] += 1
        return loss - self.ewma[batch_idx]  # individual's fitness

    def update_after_iteration(self):
        self.ewma *= 1 - self.ewma_alpha
        # calculate normal average for each batch and include it in the EWMA
        self.ewma += self.ewma_alpha * (self.current_losses / self.current_counts)
        # reset stats for the new iteration
        self.current_losses = torch.zeros(self.num_batches)
        # XXX ones to prevent 0 / 0
        self.current_counts = torch.ones(self.num_batches)
        self.ewma_alpha = 1 / (self.iter_counter ** (1 / 3))
        self.iter_counter += 1
