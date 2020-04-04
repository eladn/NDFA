import torch
from torch.utils.data.dataloader import DataLoader

from execution_parameters import ModelExecutionParams
from code_tasks.code_tasks import load_task
from dataset_properties import DataFold
from nn_utils import fit


def main():
    exec_params: ModelExecutionParams = ModelExecutionParams.factory(
        load_from_args=True, load_from_yaml=True, verify_confclass=True)
    device = torch.device("cuda" if exec_params.use_gpu_if_available and torch.cuda.is_available() else "cpu")

    task = load_task(exec_params.experiment_setting.task)
    model = task.build_model(exec_params.experiment_setting.model_hyper_params)

    if exec_params.should_load_model:
        raise NotImplementedError()  # TODO: implement!

    if exec_params.perform_training:
        train_dataset = task.create_dataset(
            model_hps=exec_params.experiment_setting.model_hyper_params,
            dataset_props=exec_params.experiment_setting.dataset,
            datafold=DataFold.Train,
            dataset_path=exec_params.train_data_path)
        train_loader = DataLoader(
            train_dataset, batch_size=exec_params.experiment_setting.train_hyper_params.batch_size, shuffle=True)
        eval_loader = None
        if exec_params.perform_evaluation:
            eval_dataset = task.create_dataset(
                model_hps=exec_params.experiment_setting.model_hyper_params,
                dataset_props=exec_params.experiment_setting.dataset,
                datafold=DataFold.Validation,
                dataset_path=exec_params.train_data_path)
            eval_loader = DataLoader(
                eval_dataset, batch_size=exec_params.experiment_setting.train_hyper_params.batch_size * 2)
        fit(
            nr_epochs=exec_params.experiment_setting.train_hyper_params.nr_epochs,
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=eval_loader,
            optimizer=None,  # TODO: implement!
            criterion=None  # TODO: implement!
        )

    if exec_params.perform_evaluation:  # TODO: consider adding `and not exec_params.perform_training`
        raise NotImplementedError()  # TODO: implement!

    if exec_params.perform_prediction:
        raise NotImplementedError()  # TODO: implement!


if __name__ == '__main__':
    main()
