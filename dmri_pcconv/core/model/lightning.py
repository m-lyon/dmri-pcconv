'''Base Lightning Model'''

import torch
import lightning.pytorch as pl


class BaseLightningModule(pl.LightningModule):
    '''Base Lightning Module with training functions implemented'''

    @property
    def loss_func(self):
        '''Loss function'''
        return torch.nn.functional.l1_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters())
        return optimizer

    def training_step(self, train_batch, *args):
        # pylint: disable=arguments-differ
        inputs, dmri_out = train_batch
        dmri_out_inf = self(*inputs)

        loss = self.loss_func(dmri_out, dmri_out_inf)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, *args):
        # pylint: disable=arguments-differ
        inputs, dmri_out = val_batch
        dmri_out_inf = self(*inputs)
        loss = self.loss_func(dmri_out, dmri_out_inf)

        if not self.trainer.sanity_checking:
            self.log('val_loss', loss, sync_dist=True)
        return loss

    def predict_step(self, batch, *args):
        # pylint: disable=arguments-differ
        inputs = tuple(batch)
        return self(*inputs)

    def test_step(self, batch, *args):
        # pylint: disable=arguments-differ
        inputs, dmri_out = batch
        dmri_out_inf = self(*inputs)
        return self.loss_func(dmri_out, dmri_out_inf)
