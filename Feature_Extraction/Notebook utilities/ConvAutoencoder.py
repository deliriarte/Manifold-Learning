import torch
import pytorch_lightning as pl  
from torch import nn
import torch.nn.functional as F
from callbacks import MetricsCallback, LitProgressBar
import torch.optim as optim


class ConvAutoEncoder(pl.LightningModule):
    
    def __init__(self, encoded_space_dim,  hyper_parameters : dict = None, *args, **kwargs):
        super().__init__()
    
        if hyper_parameters is None:
            self.hyper_parameters = { #Default values
                'optimizer' : 'Adam',
                'learning_rate' : 1e-3,
                'dropout' : 0,
                'conv1' : 8,
                'conv2' : 16, 
                'conv3' : 32,
                'fc' : 16, 
                'weight':1e-5
            }
            self.hyper_parameters.update(**kwargs)
        else:
            self.hyper_parameters = hyper_parameters    
    
        self.save_hyperparameters() #store hyper_parameters in checkpoints

        self.dropout = self.hyper_parameters['dropout']
        self.encoded_space_dim = encoded_space_dim
        self.w_decay = self.hyper_parameters['weight']
        self.conv1 = self.hyper_parameters['conv1']
        self.conv2 = self.hyper_parameters['conv2']
        self.conv3 = self.hyper_parameters['conv3']
        self.fc = self.hyper_parameters['fc']
        
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv1d(1,self.conv1,5, stride=3).double(),
            nn.Dropout(p=self.dropout),
            
            # Second convolutional layer
            nn.Conv1d(self.conv1, self.conv2, 3, stride=3).double(),
            nn.LeakyReLU(),

            
            # Third convolutional layer
            nn.Conv1d(self.conv2, self.conv3, 4, stride=2).double(),
            nn.Dropout(p=self.dropout),
            nn.LeakyReLU(),

        
        ### Flatten layer
            nn.Flatten(start_dim=1),

        ### Linear section
            # First linear layer
            nn.Linear(2905*self.conv3, self.fc).double(),
            # Second linear layer
            nn.Linear(self.fc, self.encoded_space_dim).double()
        )
 

        ### Linear section
        self.decoder_cnn = nn.Sequential(
            # First linear layer
            nn.Linear(self.encoded_space_dim, self.fc).double(),
           
            # Second linear layer
            nn.Linear(self.fc, 2905*self.conv3).double(),
           
        ### Unflatten
            nn.Unflatten(dim=1, unflattened_size=(self.conv3, 2905)),


        ### Convolutional section
            # First transposed convolution
            nn.ConvTranspose1d(self.conv3, self.conv2, 4, stride=2,output_padding=1).double(),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),

            # Second transposed convolution
            nn.ConvTranspose1d(self.conv2, self.conv1, 3, stride=3, padding =0, output_padding=2).double(),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout),
            
            # Third transposed convolution
            nn.ConvTranspose1d(self.conv1, 1, 5, stride=3, padding =0, output_padding=1).double(),
           
            )
        
    def forward(self, x : "torch.tensor"):
        embedding = self.encoder_cnn(x)
        
        return embedding
    
    def training_step(self, batch, batch_idx):
        x = batch #ignore labels
        internal_repr = self.encoder_cnn(x)
        
        x_hat = self.decoder_cnn(internal_repr)

        loss = F.mse_loss(x_hat, x, reduction='mean')
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hyper_parameters['optimizer'])(self.parameters(), lr=self.hyper_parameters['learning_rate'], weight_decay = self.hyper_parameters['weight']) #, weight_decay=1e-5)
        return optimizer

    def validation_step(self, batch, batch_idx, log_name = 'val_loss'):
        x = batch
        internal_repr = self.encoder_cnn(x)
        
        x_hat = self.decoder_cnn(internal_repr)

        loss = F.mse_loss(x_hat, x)
        
        self.log(log_name, loss)

        return loss

    def test_step(self, batch, batch_idx):
        
        return self.validation_step(batch, batch_idx, log_name='test_loss')