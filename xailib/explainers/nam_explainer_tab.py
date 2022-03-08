from xailib.xailib_transparent_by_design import Explainer, Explanation
from nam.utils import plot_mean_feature_importance
from nam.utils import plot_nams
from nam.config import defaults
from nam.data import FoldedDataset, NAMDataset
from nam.models import NAM, get_num_units
from nam.trainer import LitNAM
from nam.utils import *
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from nam.trainer import Trainer



class NamExplanation(Explanation):
    def __init__(self, exp, feature_names: list, bias: float):
        super().__init__()
        self.exp = exp
        self.feature_names = feature_names
        self.bias = bias

    def getFeaturesImportance(self):
        return self.exp

    def getExemplars(self):
            return None

    def getCounterExemplars(self):
        return None

    def getRules(self):
        return None

    def getCounterfactualRules(self):
        return None


class NamTransparentByDesign(Explainer):

    def __init__(self, X, y, feature_names, target_name, config=None, **kwargs):
        """
            NAM: Neural Additive Models - Interpretable Machine Learning with Neural Nets

        @param X: Input features
        @param y: values to predict
        @param feature_names: names of input features
        @param target_name: name of the target feature
        @param config: [opt] config object to use instead of default
        @param kwargs: parameters to alter in the default config
        """
        config = defaults() if (config is None) else config
        self.config = config
        config.update(**kwargs)
        print(config)
        self.feature_names = feature_names
        self.target_name = target_name
        df = pd.DataFrame(X, columns=feature_names)
        df[target_name] = y
        dataset = NAMDataset(config, data_path=df, features_columns=feature_names,
                             targets_column=target_name)
        self.dataset = dataset
        self.model = NAM(
            config=config,
            name="NAM_SIM",
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(config, dataset.features),
        )

    def fit(self, dataloaders=None):
        config = self.config
        model = self.model
        tb_logger = TensorBoardLogger(config.logdir, name=f'{model.name}')
        checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir +
                                                       "/{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=config.save_top_k,
                                              mode='min')

        litmodel = LitNAM(config, model)
        trainer = pl.Trainer(logger=tb_logger,
                             max_epochs=config.num_epochs, )
        #    checkpoint_callback=checkpoint_callback)
        if dataloaders is None:
            dataloaders = self.dataset.train_dataloaders()
        train_dl, valid_dl = dataloaders
        trainer.fit(litmodel,
                    train_dataloaders=train_dl,
                    val_dataloaders=valid_dl)

        last_bias = 0
        for name, p in self.model.named_parameters():
            if name.split('.')[-1] == 'bias':
                last_bias = p
        self.bias = last_bias

    def explain(self, x):
        pred, fnn_out = self.predict(x, return_fnn_out=True)
        return NamExplanation(fnn_out, self.feature_names, self.bias)

    def predict(self, x: torch.tensor, return_fnn_out=False):
        pred, fnn_out = self.model(x)
        if return_fnn_out:
            return pred, fnn_out
        return pred

    def predict_proba(self, x: torch.tensor):
        pred, fnn_out = self.model(x)
        return pred

    def explain_global(self, num_cols=2):
        plot_mean_feature_importance(self.model, self.dataset)
        plot_nams(self.model, self.dataset, num_cols=num_cols)


