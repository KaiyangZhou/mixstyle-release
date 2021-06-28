from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops import random_mixstyle, crossdomain_mixstyle


@TRAINER_REGISTRY.register()
class Vanilla2(TrainerX):
    """Vanilla baseline.

    Slightly modified for mixstyle.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        mix = cfg.TRAINER.VANILLA2.MIX

        if mix == 'random':
            self.model.apply(random_mixstyle)

        elif mix == 'crossdomain':
            self.model.apply(crossdomain_mixstyle)

        else:
            raise NotImplementedError

    def forward_backward(self, batch):
        input, label = self.parse_batch_train(batch)
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
