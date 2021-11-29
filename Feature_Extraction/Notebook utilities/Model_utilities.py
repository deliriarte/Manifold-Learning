import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import ticker
from Data_Preprocessing import to_matrix
import torch
from pathlib import Path
import numpy as np
import json
from skimage.metrics import structural_similarity as ssim

def sim_error(cnn_autoencoder, train_dataset, latent_space,original):
    #Compute the structural similarity between two images 
    s = []
    with torch.no_grad():
        encoded_imgs = cnn_autoencoder.encoder_cnn(train_dataset.fc.double())
        rec_img = cnn_autoencoder.decoder_cnn(encoded_imgs).cpu().detach().numpy()

    for i in range(len(train_dataset.fc)):
        s.append(ssim(to_matrix(train_dataset.fc[i]), to_matrix(rec_img[i])))
    return np.mean(s), np.std(s)

MODEL_SAVE_FOLDER = Path("SavedModels")

def checkpoint_path(name : str):
    """Given a model `name`, return a path to its checkpoint"""
    return MODEL_SAVE_FOLDER / Path(name + '.ckpt')
def learn_curve_path(name : str):
    """Given a model `name`, return a path to its saved learning curves data"""
    return MODEL_SAVE_FOLDER / Path(name + '.csv') 


def save_state(name : str, trainer : "pl.Trainer", metrics : "MetricsCallback"):
    """Save the model as a checkpoint, and also its learning curves data, under the specified `name`."""
    trainer.save_checkpoint(checkpoint_path(name))
    
    df = pd.DataFrame(metrics)
    df.to_csv(learn_curve_path(name), index=False)
    
def load_state(model_class : "pl.Module", name : str):
    """Load the state named `name` for the model with class `model_class`.
    
    Returns
    -------
    instance : pl.Module
        Model instantiated from the checkpoint
    metrics : pd.DataFrame
        DataFrame with the learning curves from the loaded training process.
    """
    
    instance = model_class.load_from_checkpoint(checkpoint_path(name))
    
    metrics = pd.read_csv(learn_curve_path(name))
    
    return instance, metrics



def plot_reconstruction_error(savefig : False,
                              metrics : "pd.DataFrame",
                              ax : "plt.ax" = None,
                              title = "CNN AutoEncoder Learning Curve"):
    """Plot the learning curves from the columns of `metrics` on a given matplotlib `ax`
    (new if not provided) with the specified `title`.
    Parameters
    ----------
    metrics : pd.DataFrame
        DataFrame with two columns "train_loss" and "val_loss" containing the learning curves data.
    ax : plt.ax, optional
        Matplotlib axes used as output, by default None
    title : str, optional
        Title of the figure, by default "CNN AutoEncoder Learning Curve"
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,7))

    xs = np.arange(len(metrics) - 1) + 1

    ax.plot(xs, metrics['train_loss'][1:], c='k', label="Train")
    ax.plot(xs, metrics['val_loss'][1:], c='r', label="Validation")

    ax.set_title(title)

    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_xlabel("Epoch")
    ax.legend()

    formatter = ticker.ScalarFormatter(useMathText=True) #Put the multiplier (e.g. *1e-2) at the top left side, above the plot, making everything more "compact"
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 

    ax.yaxis.set_major_formatter(formatter)

    ax.patch.set_facecolor('white')
    plt.tight_layout()
    
    if savefig:
        plt.savefig('RESULTS\Losses')
        
        

def plot_samples(cnn_autoencoder_strk, total_dataset, latent_space, worst = False):   
    rec = []
    for i in range(len(total_dataset.fc)):
        with torch.no_grad():
            encoded_imgs = cnn_autoencoder_strk.encoder_cnn(total_dataset.fc[i].unsqueeze(1).double())
            rec_img = cnn_autoencoder_strk.decoder_cnn(encoded_imgs).cpu().detach().numpy()

        rer = np.mean((total_dataset.fc[i].cpu().detach().numpy() - rec_img.squeeze(1))**2)
        rec.append(rer)
        np.savetxt("RESULTS/Features"+str(latent_space), np.asarray(encoded_imgs))
        
    n = 4
    idx = np.argsort(rec)
    if worst: idx = idx[::-1]
    np.mean(rec)


    plt.figure(figsize=(8, 4))
    for i in range(n):
        # display original
        with torch.no_grad():
            encoded_imgs = cnn_autoencoder_strk.encoder_cnn(total_dataset.fc[idx[i]].unsqueeze(1).double())
            rec_img = cnn_autoencoder_strk.decoder_cnn(encoded_imgs).cpu().detach().numpy()

            ax = plt.subplot(2, n, i + 1)
            plt.imshow(to_matrix(total_dataset.fc[idx[i]]), cmap = 'jet')
            plt.title(str(round(rec[idx[i]], 4)))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(to_matrix(rec_img), cmap = 'jet')
            plt.title("reconstructed")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.savefig("RESULTS/Samples_"+str(latent_space))
    plt.show()
    return idx, rec
    
    
def reconstruction_error(cnn_autoencoder, train_dataset, latent_space,original):
    with torch.no_grad():
        encoded_imgs = cnn_autoencoder.encoder_cnn(train_dataset.fc.double())
        rec_img = cnn_autoencoder.decoder_cnn(encoded_imgs).cpu().detach().numpy()

    rer = np.mean((original - rec_img.squeeze(1))**2, 1)
    mse=np.mean(rer)
    sd = np.std(rer, ddof=1)
    return mse, sd

def save_features(cnn_autoencoder, train_dataset, latent_space):
    with torch.no_grad():        
        encoded_imgs = cnn_autoencoder.encoder_cnn(train_dataset.fc)
        np.savetxt("RESULTS/Features"+str(latent_space), np.asarray(encoded_imgs))
        
        
        
        
def save_state_cv(name, trainer, fold_metrics, encoded_space):
    """Save the model as a checkpoint, and also its learning curves data, under the specified `name`."""
    trainer.save_checkpoint(checkpoint_path(name))
    
    val_loss_epoch = []
    train_loss_epoch = []

    for fold in fold_metrics:
        val_loss_epoch_fold = []
        train_loss_epoch_fold = []

        for epoch in range(1,len(fold)):
            val_loss_epoch_fold.append(fold[epoch]['val_loss'])
            train_loss_epoch_fold.append(fold[epoch]['train_loss'])

        val_loss_epoch.append(val_loss_epoch_fold)
        train_loss_epoch.append(train_loss_epoch_fold)
    
    np.savetxt('Val_losses_'+str(encoded_space),val_loss_epoch)
    np.savetxt('Train_losses_'+str(encoded_space),train_loss_epoch)
    
    
def load_state_cv(model_class : "pl.Module", name : str, encoded_space: int):
    """Load the state named `name` for the model with class `model_class`.
    
    Returns
    -------
    instance : pl.Module
        Model instantiated from the checkpoint
    metrics : pd.DataFrame
        DataFrame with the learning curves from the loaded training process.
    """
    
    instance = model_class.load_from_checkpoint(checkpoint_path(name))
    
    val_loss = np.loadtxt('Val_losses_'+str(encoded_space_dim))
    train_loss = np.loadtxt('Train_losses_'+str(encoded_space_dim))

    return instance, val_loss, train_loss
