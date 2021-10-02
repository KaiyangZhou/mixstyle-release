import torch
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
            print('MixStyle: random mixing')

        elif mix == 'crossdomain':
            self.model.apply(crossdomain_mixstyle)
            print('MixStyle: cross-domain mixing')

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
    
    @torch.no_grad()
    def vis(self):
        self.set_model_mode('eval')
        output_dir = self.cfg.OUTPUT_DIR
        source_domains = self.cfg.DATASET.SOURCE_DOMAINS
        print('Source domains:', source_domains)

        out_embed = []
        out_domain = []
        out_label = []

        split = self.cfg.TEST.SPLIT
        data_loader = self.val_loader if split == 'val' else self.test_loader

        print('Extracting style features')

        for batch_idx, batch in enumerate(data_loader):
            input = batch['img'].to(self.device)
            label = batch['label']
            domain = batch['domain']
            impath = batch['impath']

            # model should directly output features or style statistics
            raise NotImplementedError
            output = self.model(input)
            output = output.cpu().numpy()
            out_embed.append(output)
            out_domain.append(domain.numpy())
            out_label.append(label.numpy()) # CLASS LABEL

            print('processed batch-{}'.format(batch_idx + 1))

        out_embed = np.concatenate(out_embed, axis=0)
        out_domain = np.concatenate(out_domain, axis=0)
        out_label = np.concatenate(out_label, axis=0)
        print('shape of feature matrix:', out_embed.shape)
        out = {
            'embed': out_embed,
            'domain': out_domain,
            'dnames': source_domains,
            'label': out_label
        }
        out_path = osp.join(output_dir, 'embed.pt')
        torch.save(out, out_path)
        print('File saved to "{}"'.format(out_path))
