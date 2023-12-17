import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

from data_loading import DataLoading
from dense_nonlin_reg import dense_regression_model

if __name__ == "__main__":
    model = dense_regression_model()

    logger = pl_loggers.TensorBoardLogger(save_dir = r"/Users/jantheiss/Desktop/Python/plant_regression/logs")
    trainer = pl.Trainer(devices = 1, accelerator = "auto", max_epochs = 100, logger = logger, callbacks = EarlyStopping(monitor = "val_loss", min_delta = 0.01, patience = 5, verbose = True))

    data = DataLoading() 
    train_loader, val_loader, test_loader = data.load_data()  

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders = test_loader)