from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

model_name = '350M'
device = 'cuda'
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    mode="min"
)
path = {
    'food': {'file_path': 'files/food/', 'csv_path':'files/food/food.csv'},
    'market':{'file_path': 'files/market/', 'csv_path':None}
}

class Args():
    def __init__(self, model):
        self.device = model.get_param('device')
        self.l_text_seq_length = model.get_param('l_text_seq_length')
        self.r_text_seq_length = model.get_param('r_text_seq_length')
        self.image_tokens_per_dim = model.get_param('image_tokens_per_dim')
        self.image_seq_length = model.get_param('image_seq_length')
        self.epochs = 4
        self.checkpont_save_path = 'files/checkpoints/'
        self.model_save_path = 'files/models/'
        self.model_name = 'awesomemodel.pt'
        self.save_every = 500
        self.bs = 2
        self.clip = 1.0
        self.lr = 2e-5
        self.freeze = False
        self.wandb = False
        self.train_steps = 10
        self.lt_loss_weight = 0.01
        self.img_loss_weight = 1
        self.rt_loss_weight = 7
        self.image_size = self.image_tokens_per_dim * 8
