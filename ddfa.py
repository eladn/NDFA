import torch

from execution_parameters import ModelExecutionParams
from code_tasks.code_tasks import load_task
from nn_utils import fit


def main():
    exec_params: ModelExecutionParams = ModelExecutionParams.factory(
        load_from_args=True, load_from_yaml=True, verify_confclass=True)
    device = torch.device("cuda" if exec_params.use_gpu_if_available and torch.cuda.is_available() else "cpu")

    task = load_task(exec_params.experiment_setting.task)
    model = task.build_model(exec_params.experiment_setting.model_hyper_params)

    if exec_params.should_load_model:
        pass  # TODO: implement!

    if exec_params.perform_training:
        fit(
            nr_epochs=exec_params.experiment_setting.train_hyper_params.nr_epochs,
            model=model,
            device=device,
            train_loader=None,  # TODO: implement!
            valid_loader=None,  # TODO: implement!
            optimizer=None,  # TODO: implement!
            criterion=None  # TODO: implement!
        )

    if exec_params.perform_evaluation:  # TODO: consider adding `and not exec_params.perform_training`
        pass  # TODO: implement!

    if exec_params.perform_prediction:
        pass  # TODO: implement!

    # TODO: perform training and/or evaluation and/or prediction ...
    raise NotImplementedError()  # TODO: implement!


if __name__ == '__main__':
    main()
