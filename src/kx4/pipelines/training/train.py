#%%
import argparse
import os
import tempfile

import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data as data
import yaml
from dlkit import models
from dlkit.criterions import Criterion
from estimator import Estimator
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm, trange
from utils import geometry, modules
from utils.dataloader import NpLaparoDataset
from utils.transformer import ScaleTransformer


#%%
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_num", type=int, help="d`ataset number")
    parser.add_argument("model", help="model name")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-e", "--n_epochs", default=100, type=int, help="epochs")
    parser.add_argument("-d", "--device", default=0, type=int)
    parser.add_argument("-try", "--n_trials", default=3, type=int)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use the model that is pretrained by myself",
    )
    parser.add_argument("-w", "--n_workers", default=4, type=int)
    parser.add_argument("-exp", "--exp_name", default="Debug")
    parser.add_argument("--a1", default=1.0, type=float, help="Coefficient of l_2d")
    parser.add_argument("--a2", default=1.0, type=float, help="Coefficient of l_orient")
    parser.add_argument("--a3", default=1.0, type=float, help="Coefficient of l_phi")
    parser.add_argument("--a4", default=1.0, type=float, help="Coefficient of l_gamma")
    parser.add_argument(
        "--optuna", action="store_true", help="Hyperparameter optimization [False]"
    )
    return parser


cfg = config().parse_args()

writer = None


class Trainer:
    """Train a model."""

    def __init__(self, model, exp_name=cfg.exp_name):
        """
        Args:
            model (nn.Module): Model name.
            exp_name (object, optional): Experiment name. Defaults to cfg.exp_name.
        """

        self.device = torch.device(
            "cuda:{}".format(cfg.device) if torch.cuda.is_available() else "cpu"
        )
        self.params = ["trans3d", "trans2d", "orient", "roll", "joint"]
        self.scaler = ScaleTransformer(cfg.ds_num)
        self.model = model
        self.model.to(self.device)
        self.loss_weights = {
            "trans3d": 1.0,
            "trans2d": cfg.a1,
            "orient": cfg.a2,
            "roll": cfg.a3,
            "joint": cfg.a4,
        }

        #         TODO: Read from model_config.yaml
        #         summary(self.model, (3, 224, 224))

        torch.backends.cudnn.benchmark = True
        mlflow.set_experiment(cfg.exp_name)

    def run(self, trial):
        """
        Return:
            Best epoch loss through one trial

        Execute learning with fixed hyper parameters.
        """

        min_loss = np.Inf

        with mlflow.start_run():
            for key, value in vars(cfg).items():
                mlflow.log_param(key, value)

            lr = self._get_lr(trial)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            criterions = self._get_criterion(trial)

            with trange(cfg.n_epochs) as epoch_bar:
                for epoch in epoch_bar:
                    for phase in ["train", "val"]:
                        epoch_bar.set_description(
                            "[{}] Epoch {}".format(phase.title().rjust(5), epoch + 1)
                        )
                        epoch_loss = self._train(epoch, phase, criterions, optimizer)
                        epoch_bar.set_postfix(loss=epoch_loss)

                        # Log weights when the minimum loss is updated
                        if phase == "val" and epoch_loss < min_loss:
                            min_loss = epoch_loss

                            mlflow.pytorch.log_model(self.model, "best_model")
                            mlflow.log_artifacts(output_dir, artifact_path="best_model")
                            mlflow.log_metric("best epoch", epoch + 1)

            # Save weights
            torch.save(model.state_dict(), output_dir + "/weight.pth")
            mlflow.pytorch.log_model(self.model, "model")

            self._test()

            mlflow.log_artifacts(output_dir, artifact_path="model")

        return min_loss

    def _train(self, epoch, phase, criterions, optimizer=None):

        if phase == "train":
            self.model.train()

        else:
            self.model.eval()

        sum_train_loss = torch.zeros(6).to(self.device)
        sum_val_loss = torch.zeros(6).to(self.device)
        sum_loss_dict = {"train": sum_train_loss, "val": sum_val_loss}
        epoch_loss_dict = {"train": None, "val": None}

        for inputs, targets in dataloader_dict[phase]:

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.set_grad_enabled(phase == "train"):
                optimizer.zero_grad()

                outputs = self.model(inputs)

                pred = self._split_param(outputs)
                label = self._split_param(targets)

                loss = 0.0
                for param, criterion in criterions.items():
                    param_loss = criterion(pred[param], label[param])
                    loss += self.loss_weights[param] * param_loss

                    sum_loss_dict[phase][self.params.index(param) + 1] += param_loss

                sum_loss_dict[phase][0] += loss

                errors = calc_errors(pred, label, self.scaler)

                # Update weights
                if phase == "train":
                    loss.backward()
                    optimizer.step()

        # Calculate the loss through one epoch
        epoch_loss_dict[phase] = sum_loss_dict[
            phase
        ].detach().cpu().numpy().copy() / len(dataloader_dict[phase])

        logparam_list = ["all", *self.params]
        for i, param_name in enumerate(logparam_list):
            self._log_scalar(
                "Loss_{}/{}".format(param_name.title(), phase),
                epoch_loss_dict[phase][i],
                epoch,
            )

        return epoch_loss_dict[phase][0]

    def _test(self):
        print("\n\nStart Testing...\n")

        run_uri = mlflow.get_artifact_uri() + "/model"
        estimator = Estimator(
            mode=2, ds_num=cfg.ds_num, run_uri=run_uri, transform=transform
        )
        target_df = val_ds.dataframe.drop(columns=["alpha", "beta"])
        target_df["z"] = -target_df["z"]
        value_list = []
        for i in trange(len(target_df)):
            im_path = "./Database/ds_{:03d}/val/img_{:05d}.jpg".format(
                cfg.ds_num, i + 1
            )
            im = Image.open(im_path)
            value = estimator(im)
            value_list.append(value)

        columns = ["x", "y", "z", "x_2d", "y_2d", "nx", "ny", "nz", "gamma", "phi"]
        pred_df = pd.DataFrame(value_list, columns=columns)
        pred_df.to_csv(output_dir + "/pred_{:03d}.csv".format(cfg.ds_num))

    #         error_df = (target_df - pred_df).rename(columns=lambda x: "e_" + x)
    #         result = pred_df.join(error_df)
    #         save_path = mlflow.get_artifact_uri()

    #     analyzer(result, target_df, save_path, conf.ds_num, resize_shape)

    def _get_lr(self, trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        mlflow.log_param("lr", lr)

        return lr

    def _get_criterion(self, trial):
        trans3d = ["mae", "mse", "huber"]
        trans2d = ["mae", "mse", "huber"]
        orient = ["mae", "mse", "huber", "cos"]
        roll = ["mae", "mse", "huber"]
        joint = ["mae", "mse", "huber"]
        criterion_list = [trans3d, trans2d, orient, roll, joint]
        criterions = dict()
        for param, criterion in zip(self.params, criterion_list):
            c = trial.suggest_categorical(param, criterion)
            mlflow.log_param(param, c)
            criterions[param] = Criterion(mode=c)

        return criterions

    def _split_param(self, values):
        position = self.scaler.inverse_transform(values.detach().cpu())[:, :3]
        value_dict = {
            "trans3d": values[:, :3],
            "trans2d": geometry.project_onto_plane(  # TODO: Read from config
                torch.from_numpy(position),
                a_ratio=self.scaler.aspect,
                fov=self.scaler.fov,
                is_batch=True,
            ).to(self.device),
            "orient": values[:, 3:6],
            "roll": values[:, 6:8],
            "joint": values[:, 8].view(-1, 1),
        }

        return value_dict

    def _log_scalar(self, name, value, step):
        """
        Log a scalar value to both MLflow and TensorBoard
        """
        writer.add_scalar(name, value, step)
        mlflow.log_metric(name, value, step)


# +
def calc_errors(pred, label, rescaler):
    """
    Return:
    [x, y, z, x_2d, y_2d, nx, ny, nz, gamma_s, gamma_c, phi, trans3d, trans2d, orient, roll, joint]
    """

    # Denormalize outputs
    transformed_pred = rescaler.inverse_transform(pred, size=resize_shape)
    transformed_label = rescaler.inverse_transform(label, size=resize_shape)

    #     print("\n-------------\n", transformed_pred, "\n+++++++++++++\n", transformed_label)

    trans2d_pred = geometry.project_onto_plane(
        transformed_pred[:, :3],
        a_ratio=rescaler.aspect,
        fov=rescaler.fov,
        is_batch=True,
    )
    trans2d_label = geometry.project_onto_plane(
        transformed_label[:, :3],
        a_ratio=rescaler.aspect,
        fov=rescaler.fov,
        is_batch=True,
    )

    roll_pred = geometry.sc2deg(transformed_pred[:, 6], transformed_pred[:, 7])
    roll_label = geometry.sc2deg(transformed_label[:, 6], transformed_label[:, 7])

    # Calculate errors
    transformed_error = transformed_pred - transformed_label
    trans3d_error = transformed_error[:, :3].norm(dim=1, keepdim=True).mean()
    trans2d_error = (
        (trans2d_pred - trans2d_label).norm(dim=1, keepdim=True) * resize_shape[0]
    ).mean()
    similarity = torch.nn.CosineSimilarity(dim=1)
    orient_error = (
        torch.rad2deg(similarity(transformed_pred[:, 3:6], transformed_label[:, 3:6]))
        .abs()
        .mean()
    )
    roll_error = geometry.scale180(roll_pred - roll_label).mean()
    joint_error = geometry.scale180(transformed_error[:, 8]).mean()


#     print(
#         "error\n",
#         transformed_error,
#         "\nem\n",
#         transformed_error.mean(dim=0),
#         "\n3\n",
#         trans3d_error,
#         "\n2\n",
#         trans2d_error,
#         "\no\n",
#         orient_error,
#         "\nr\n",
#         roll_error,
#         "\nj\n",
#         joint_error,
#     )
# -


def get_model_config():
    with open("./dlkit/model_config.yaml") as m_config:
        m_cfg = yaml.load(m_config)
        m = m_cfg[cfg.model]["name"]
        input_size = m_cfg[cfg.model]["input_size"]
        return m, input_size


# +
# TODO: Read from model_config.yaml and ds_config.yaml
with open("./Database/ds_{:03d}/ds_config.yaml".format(cfg.ds_num)) as f:
    ds_config = yaml.load(f, Loader=yaml.SafeLoader)

with open("./dlkit/model_config.yaml") as f:
    model_config = yaml.load(f, Loader=yaml.SafeLoader)
    resize_shape = model_config[cfg.model]["input_size"]

if cfg.model == "res50":
    cmean = [0.485, 0.456, 0.406]
    cstd = [0.229, 0.224, 0.225]
    model = models.ResNet50()

elif cfg.model == "res50_448":
    cmean = ds_config["color"]["mean"]
    cstd = ds_config["color"]["std"]
    model = models.ResNet50_448()

transform = Compose([Resize(resize_shape), ToTensor(), Normalize(cmean, cstd)])
# -

# ## Setting a dataset and dataeset loader

train_ds = NpLaparoDataset(phase="train", ds_num=cfg.ds_num, input_size=resize_shape)
val_ds = NpLaparoDataset(phase="val", ds_num=cfg.ds_num, input_size=resize_shape)

# +
# train_ds = LaparoDataset(phase="train", transform=transform, ds_num=cfg.ds_num)
# val_ds = LaparoDataset(phase="val", transform=transform, ds_num=cfg.ds_num)
# -

train_loader = data.DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=cfg.n_workers,
)
val_loader = data.DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=cfg.n_workers,
)
dataloader_dict = {"train": train_loader, "val": val_loader}

if __name__ == "__main__":
    # Train a model
    print(
        "\nNow Starting...\n\n"
        "\n========================\n"
        " HYPER PARAMETERS    \n"
        "------------------------\n"
        " Device     >> {}\n"
        " Dataset    >> {}\n"
        " Model      >> {}\n"
        " Batch size >> {}\n"
        " Epochs     >> {}\n"
        "========================\n".format(
            cfg.device, cfg.ds_num, cfg.model.title(), cfg.batch_size, cfg.n_epochs
        )
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = tmp_dir + "/logs"
        writer = SummaryWriter(output_dir)

        # Load weights when using a pre-trained model
        if cfg.pretrained:
            exp_dir = input("Experiment Directory: ")
            weight = torch.load(
                os.getcwd() + "/" + exp_dir + "/artifacts/models/weight.pth"
            )
            model.load_state_dict(weight)

        trainer = Trainer(model=model)

        while True:
            try:
                study = optuna.create_study(
                    study_name=cfg.exp_name + "/" + cfg.model,
                    storage="sqlite:///mlruns/{}_{}.db".format(cfg.exp_name, cfg.model),
                    load_if_exists=True,
                )
                study.optimize(trainer.run, n_trials=cfg.n_trials)
                break

            except RuntimeError:
                break

    message = """
    ========================
        TRANING FINISHED
    ========================
    """

    modules.line_notify(message)
    print(message)

# %%
