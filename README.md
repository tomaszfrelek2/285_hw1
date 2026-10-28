# HW1

This project trains a variational autoencoder on the Kaggle
[chest X-ray dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
and uses it to generate new X-ray images. The generated images are scored with
Inception Score and FID, using code from the
[PyTorch-Ignite tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/).

## Files
All code-realated files are in `vae_cxr` folder
- **`vae.py`**: the VAE model. The encoder turns an image into a small latent code, and the decoder turns a code back into an image. It also has the loss function (reconstruction loss + KL divergence).
- **`data.py`**: loads the X-ray images, converts them to grayscale 64×64, and saves a cached copy so they only need to be processed once.
- **`train.py`**: trains the model. It logs the losses, saves checkpoints, and makes the loss-curve plots and example images.
- **`evaluate.py`**: loads the trained model, generates images, and computes the Inception Score and FID.
- **`figures.py`**: helper functions for the plots and image grids.
- **`run_all.sh`**: runs everything in order (download the data, train, evaluate).
- **`colab.ipynb`**: the same steps as a Google Colab notebook.

## How to run

This code was run via Google colab
You'll need to create a kaggle API token first, then you can run all the cells in colab.ipynb 


## Results
All the results (plots, sample images, scores, model weights, etc) are in `results/runs2/vae64/`.
