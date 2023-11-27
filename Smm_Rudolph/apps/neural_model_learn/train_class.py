import pytorch_lightning as pl
from rudolph.model import get_rudolph_model
import torch
from rudolph.model.utils import get_attention_mask


class Rudolph_(pl.LightningModule):
    vae = None
    def __init__(self, model_name, device, args):
        super().__init__()
        self.model = get_rudolph_model(model_name,  fp16=False, device=device)
        self.model_name = model_name
        self.model.train()
        self.args = args
        self.save_hyperparameters()
    def forward(self,
            input_ids,
            lt_loss_weight=0.1,
            img_loss_weight=0.8,
            rt_loss_weight=0.1,
            return_loss=True):
        total_seq_length = self.args.l_text_seq_length + self.args.image_seq_length*self.args.image_seq_length + self.args.r_text_seq_length
        masks = torch.ones(self.args.bs, self.args.r_text_seq_length, dtype=torch.int32)
        attention_mask = get_attention_mask(self.args.bs, self.args.l_text_seq_length, self.args.image_tokens_per_dim,
                                                        self.args.r_text_seq_length, self.device)
        loss, loss_values = self.model.forward(input_ids,
                                               attention_mask,
                                               lt_loss_weight=lt_loss_weight,
                                               img_loss_weight=img_loss_weight,
                                               rt_loss_weight=rt_loss_weight,
                                               return_loss=True)
        return loss
    def training_step(self, batch):
        text, images = batch[0], batch[1]
        image_input_ids = self.vae.get_codebook_indices(images).to(self.device)
        r_text = text.to(self.device)
        l_text = torch.zeros((self.args.bs, self.args.l_text_seq_length),  dtype=torch.long).to(self.device)
        input_ids = torch.cat((l_text, image_input_ids, r_text), dim=1)
        loss = self.forward(input_ids,
                            lt_loss_weight=self.args.lt_loss_weight,
                            img_loss_weight=self.args.img_loss_weight,
                            rt_loss_weight=self.args.rt_loss_weight,
                            return_loss=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss}
    def on_train_epoch_end(self, *args, **kwargs):
        pass
    def freeze(self,
          model,
          freeze_emb=True,
          freeze_ln=True,
          freeze_attn=False,
          freeze_ff=True,
          freeze_other=True,
    ):
          for name, p in model.named_parameters():
              name = name.lower()
              if 'ln' in name or 'norm' in name:
                  p.requires_grad = not freeze_ln
              elif 'embeddings' in name:
                  p.requires_grad = not freeze_emb
              elif 'mlp' in name:
                  p.requires_grad = not freeze_ff
              elif 'attention' in name:
                  p.requires_grad = not freeze_attn
              else:
                  p.requires_grad = not freeze_other
          return model

    def configure_optimizers(self):
        if self.args.freeze:
            self.model = self.freeze(self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        #bnb.optim.Adam8bit(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
          optimizer,
          max_lr=self.args.lr,
          final_div_factor=500,
          steps_per_epoch=self.args.train_steps,
          epochs=self.args.epochs
        )
        return optimizer
