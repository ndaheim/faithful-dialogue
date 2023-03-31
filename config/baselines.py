import sys

import numpy as np

sys.setrecursionlimit(2500)

# ------------------------------ Sisyphus -------------------------------------

import sisyphus.toolkit as tk
from ukp.huggingface.search import *
from ukp.huggingface.training import *
from ukp.huggingface.task_arithmetic import MakeAndApplyTaskVectorsJob, \
 MakeTaskVectorsJob, MakeAndApplyTaskVectorsCapeJob, MakeAndApplyTaskVectorsCapeWithFisherJob, MakeAndApplyTaskVectorsEWRJob
from ukp.huggingface.evaluation import *
from i6_core.text import PipelineJob


Path = tk.Path

code_root = gs.CODE_ROOT

def train_model(method, model_name_or_path, dataset, dataset_config_name, model_description, per_device_train_batch_size=4, gradient_accumulation_steps=8,
                dataset_train_split="train", dataset_val_split="validation", num_epochs=10, time_rqmt=24, mem_rqmt=24, gpu_mem=10, learning_rate=6.25e-5,
                per_device_eval_batch_size=8):
    config = {
        'model_name_or_path': model_name_or_path,
        'predict_with_generate': True,
        'method': method,
        'learning_rate': learning_rate,
        'per_device_train_batch_size': per_device_train_batch_size,
        # 'per_device_eval_batch_size': per_device_eval_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'cache_dir': gs.CACHE_DIR,
    }
    train_data_config = {
        'dataset_name': os.path.join(code_root, f'dialog/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_train_split': dataset_train_split,
        'dataset_val_split': dataset_val_split,
    }

    train_job = HuggingfaceTrainingJob(
        code_root=code_root,
        config=config,
        train_data_config=train_data_config,
        num_epochs=num_epochs,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem
    )
    train_job.add_alias(f"cleaned_repo/train_job-{dataset}_{dataset_config_name}_{model_description}")
    tk.register_output(f'{dataset}_{dataset_config_name}_{model_description}', train_job.out_best_model)
    return train_job

def evaluate_model(method, model_name_or_path, dataset, dataset_config_name, model_description, per_device_eval_batch_size=8,
                   dataset_test_split="test", time_rqmt=2, mem_rqmt=24, gpu_mem=10, calculate_q2=False, generation_beam_size=None):
    config = {
        'model_name_or_path': model_name_or_path,
        'predict_with_generate': True,
        'method': method,
        'per_device_eval_batch_size': per_device_eval_batch_size,
    }
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'dialog/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': dataset_test_split,
    }

    if generation_beam_size is not None:
        config["generation_beam_size"] = generation_beam_size

    search_job = HuggingfaceSearchJob(
        code_root=code_root,
        model_path=model_name_or_path,
        config=config,
        search_data_config=search_data_config,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem,
    )

    if dataset == "dstc11":
        scoring_job = CalculateMetricsForMultiDocJob(
            code_root,
            search_data_config["dataset_name"],
            search_data_config['dataset_test_split'],
            search_job.out_search_file,
            time_rqmt=2,
        )
    else:
        scoring_job = CalculateMetricsJob(
            code_root,
            search_data_config["dataset_name"],
            search_data_config['dataset_test_split'],
            search_job.out_search_file,
            time_rqmt=2
        )
    tk.register_output(f'results/{dataset}/{dataset_config_name}_{method}_{model_description}.metrics.json', scoring_job.out_results_file)

def calculate_fisher_information(method, model_name_or_path, dataset, dataset_config_name, model_description, per_device_eval_batch_size=8,
                   dataset_test_split="validation", time_rqmt=2, mem_rqmt=24, gpu_mem=10):
    assert "fisher" in method
    config = {
        'model_name_or_path': model_name_or_path,
        'predict_with_generate': True,
        'method': method,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'track_fim': True,
    }
    search_data_config = {
        'dataset_name': os.path.join(code_root, f'dialog/datasets/{dataset}.py'),
        'dataset_config_name': dataset_config_name,
        'dataset_test_split': dataset_test_split,
    }

    search_job = HuggingfaceSearchJob(
        code_root=code_root,
        model_path=model_name_or_path,
        config=config,
        search_data_config=search_data_config,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem
    )
    tk.register_output(f'results/{dataset}/{dataset_config_name}_{method}_{model_description}_fisher', search_job.out_checkpoints_dir)

    return search_job.out_checkpoints_dir


def run_models(method=None, model_name_or_path=None, train_dataset=None, train_dataset_config_name=None, test_datasets=None, baseline_model_name_or_path=None,
                        test_dataset_config_name=None, model_description=None, anti_expert_model_name_or_path=None, expert_model_name_or_path=None,
                        expert_dataset_name=None, anti_expert_dataset_name=None, expert_dataset_config_name=None, anti_expert_dataset_config_name=None,
                        dataset_test_split="test", dataset_train_split="train", dataset_val_split="validation", anti_expert_fisher_path=None, baseline_fisher_path=None,
                        per_device_eval_batch_size=8, per_device_train_batch_size=4, gradient_accumulation_steps=8, train_time_rqmt=24,
                        time_rqmt=2, mem_rqmt=24, gpu_mem_train=10, gpu_mem_test=10, gpu_mem_fisher=10, num_epochs=10, fisher_estimation_method="fisher_approx_document_grounded_generation",
                        calculate_fisher_norms=False, num_expert_epochs=5):
    # Train all models
    if baseline_model_name_or_path is None:
        baseline_model = train_model(
            method,
            model_name_or_path,
            train_dataset,
            train_dataset_config_name,
            model_description,
            gpu_mem=gpu_mem_train,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataset_train_split=dataset_train_split,
            dataset_val_split=dataset_val_split,
            num_epochs=num_epochs,
            time_rqmt=train_time_rqmt
        ).out_best_model
    else: 
        baseline_model = baseline_model_name_or_path

    if anti_expert_model_name_or_path is None:
        anti_expert_model = train_model(
            method,
            baseline_model,
            anti_expert_dataset_name,
            anti_expert_dataset_config_name,
            f"{model_description}_anti_expert",
            gpu_mem=gpu_mem_train,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataset_train_split=dataset_train_split,
            dataset_val_split=dataset_val_split,
            num_epochs=num_expert_epochs
        ).out_models[num_expert_epochs]
    else:
        anti_expert_model = anti_expert_model_name_or_path

    if expert_model_name_or_path is None:
        expert_model = train_model(
            method,
            baseline_model,
            expert_dataset_name,
            expert_dataset_config_name,
            f"{model_description}_expert",
            gpu_mem=gpu_mem_train,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataset_train_split=dataset_train_split,
            dataset_val_split=dataset_val_split,
            num_epochs=num_expert_epochs
        ).out_models[num_expert_epochs]
    else:
        expert_model = expert_model_name_or_path

    # Noisy Channel

    channel_model = train_model(
        "channel_model",
        baseline_model,
        train_dataset,
        "response_generation",
        f"{model_description}_channel_model",
        gpu_mem=gpu_mem_train,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataset_train_split=dataset_train_split,
        dataset_val_split=dataset_val_split,
        num_epochs=num_expert_epochs
    ).out_models[num_expert_epochs]

    response_generation_model = train_model(
        "response_generation",
        baseline_model,
        train_dataset,
        "response_generation",
        f"{model_description}_response_generation_model",
        gpu_mem=gpu_mem_train,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataset_train_split=dataset_train_split,
        dataset_val_split=dataset_val_split,
        num_epochs=num_expert_epochs
    ).out_models[num_expert_epochs]

    noisy_channel_model = CreateNoisyChannelCheckpointJob(
        code_root,
        baseline_model,
        channel_model,
        response_generation_model,
        0.5,
        0.2,
        1.0,
    ).out_model_path

    for test_dataset in test_datasets:

        # noisy channel

        evaluate_model(
            "noisy_channel_reranking",
            noisy_channel_model,
            test_dataset,
            test_dataset_config_name,
            model_description,
            gpu_mem=gpu_mem_test,
            dataset_test_split=dataset_test_split,
            per_device_eval_batch_size=1, # not implemented for batched inference yet
            calculate_q2=True,
            time_rqmt=6
        )

        # baseline

        evaluate_model(
            method,
            baseline_model,
            test_dataset,
            test_dataset_config_name,
            model_description,
            gpu_mem=gpu_mem_test,
            dataset_test_split=dataset_test_split,
            calculate_q2=True
        )

        # Task Arithmetic + Cape

        for scaling_factor in [1.0]:
            scaling_factor = round(scaling_factor, 2)

            # Task Arithmetic
            new_model = MakeAndApplyTaskVectorsJob(
                code_root,
                baseline_model,
                [],
                [anti_expert_model],
                scaling_factors_experts=[],
                scaling_factors_anti_experts=[scaling_factor]
            ).out_model_path

            evaluate_model(
                method,
                new_model,
                test_dataset,
                test_dataset_config_name,
                model_description+"_task_arithmetic_"+str(scaling_factor),
                gpu_mem=gpu_mem_test,
                dataset_test_split=dataset_test_split,
                calculate_q2=scaling_factor == 1.0
            )

            if not train_dataset_config_name == expert_dataset_config_name:

                # Cape
                new_model = MakeAndApplyTaskVectorsCapeJob(
                    code_root,
                    baseline_model,
                    expert_model,
                    anti_expert_model,
                    operation="negation",
                    scaling_factor=scaling_factor
                ).out_model_path

                evaluate_model(
                    method,
                    new_model,
                    test_dataset,
                    test_dataset_config_name,
                    model_description+"_task_arithmetic_cape_"+str(scaling_factor),
                    gpu_mem=gpu_mem_test,
                    dataset_test_split=dataset_test_split
                )

            # DExperts
            new_model = CreateDensityRatioCheckpointJob(
                code_root,
                baseline_model,
                anti_expert_model,
                expert_model,
                0.5,
                0.2
            ).out_model_path

            evaluate_model(
                "document_grounded_generation_density_ratio",
                new_model,
                test_dataset,
                test_dataset_config_name,
                model_description+"_DExperts",
                gpu_mem=int(gpu_mem_test*1.5),
                dataset_test_split=dataset_test_split,
                calculate_q2=False
            )

        # EWR
        if baseline_fisher_path is None:
            fisher_base = calculate_fisher_information(
                fisher_estimation_method,
                baseline_model,
                train_dataset,
                train_dataset_config_name,
                model_description,
                dataset_test_split=dataset_val_split if not dataset_val_split == dataset_test_split else dataset_train_split,
                time_rqmt=4,
                gpu_mem=gpu_mem_fisher
            )
        else:
            fisher_base = baseline_fisher_path

        task_vector = MakeTaskVectorsJob(
            code_root,
            baseline_model,
            anti_expert_model,
            operation="negation"
        ).out_model_path
        
        if anti_expert_fisher_path is None:
            fisher_task_vector = calculate_fisher_information(
                fisher_estimation_method,
                task_vector,
                train_dataset,
                train_dataset_config_name,
                model_description,
                dataset_test_split=dataset_val_split if not dataset_val_split == dataset_test_split else dataset_train_split,
                time_rqmt=4,
                gpu_mem=gpu_mem_fisher
            )
        else:
            fisher_task_vector = anti_expert_fisher_path

        if not train_dataset_config_name == expert_dataset_config_name:

            task_vector_cape = MakeTaskVectorsJob(
                code_root,
                expert_model,
                anti_expert_model,
                operation="negation"
            ).out_model_path

            fisher_task_vector_cape = calculate_fisher_information(
                fisher_estimation_method,
                task_vector_cape,
                train_dataset,
                train_dataset_config_name,
                model_description,
                dataset_test_split=dataset_val_split if not dataset_val_split == dataset_test_split else dataset_train_split,
                time_rqmt=4,
                gpu_mem=gpu_mem_fisher
            )

            fisher_expert = calculate_fisher_information(
                fisher_estimation_method,
                expert_model,
                train_dataset,
                train_dataset_config_name,
                model_description,
                dataset_test_split=dataset_val_split if not dataset_val_split == dataset_test_split else dataset_train_split,
                time_rqmt=4,
                gpu_mem=gpu_mem_fisher
            )


        for scaling_factor in [0.15]:
            scaling_factor = round(scaling_factor, 3)

            new_model = MakeAndApplyTaskVectorsEWRJob(
                code_root,
                baseline_model,
                [],
                [anti_expert_model],
                fisher_base,
                [],
                [fisher_task_vector],
                scaling_factors_experts=[],
                scaling_factors_anti_experts=[scaling_factor]
            ).out_model_path

            evaluate_model(
                method,
                new_model,
                test_dataset,
                test_dataset_config_name,
                model_description+"_ewr_"+str(scaling_factor),
                gpu_mem=gpu_mem_test,
                dataset_test_split=dataset_test_split,
                calculate_q2=scaling_factor == 0.15
            )
            
            if not train_dataset_config_name == expert_dataset_config_name:
                new_model = MakeAndApplyTaskVectorsCapeWithFisherJob(
                    code_root,
                    expert_model,
                    expert_model,
                    anti_expert_model,
                    fisher_expert,
                    fisher_task_vector_cape,
                    operation="negation",
                    scaling_factor=scaling_factor
                ).out_model_path

                evaluate_model(
                    method,
                    new_model,
                    test_dataset,
                    test_dataset_config_name,
                    model_description+"_task_arithmetic_cape_ewr_"+str(scaling_factor),
                    gpu_mem=gpu_mem_test,
                    dataset_test_split=dataset_test_split
                )

def run_ctrl(method=None, model_name_or_path=None, train_dataset=None, train_dataset_config_name=None, test_datasets=None, baseline_model_name_or_path=None,
                        test_dataset_config_name=None, model_description=None, anti_expert_model_name_or_path=None, expert_model_name_or_path=None,
                        expert_dataset_name=None, anti_expert_dataset_name=None, expert_dataset_config_name=None, anti_expert_dataset_config_name=None,
                        dataset_test_split="test", dataset_train_split="train", dataset_val_split="validation", anti_expert_fisher_path=None, baseline_fisher_path=None,
                        per_device_eval_batch_size=8, per_device_train_batch_size=4, gradient_accumulation_steps=8, train_time_rqmt=24,
                        time_rqmt=2, mem_rqmt=24, gpu_mem_train=10, gpu_mem_test=10, gpu_mem_fisher=10, num_epochs=10, fisher_estimation_method="fisher_approx_document_grounded_generation",
                        calculate_fisher_norms=False, num_expert_epochs=5):
    # Train all models
    if baseline_model_name_or_path is None:
        baseline_model = train_model(
            method,
            model_name_or_path,
            train_dataset,
            train_dataset_config_name,
            model_description,
            gpu_mem=gpu_mem_train,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataset_train_split=dataset_train_split,
            dataset_val_split=dataset_val_split,
            num_epochs=num_epochs,
            time_rqmt=train_time_rqmt
        ).out_best_model
    else: 
        baseline_model = baseline_model_name_or_path

    if anti_expert_model_name_or_path is None:
        anti_expert_model = train_model(
            method,
            baseline_model,
            anti_expert_dataset_name,
            anti_expert_dataset_config_name,
            f"{model_description}_anti_expert",
            gpu_mem=gpu_mem_train,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataset_train_split=dataset_train_split,
            dataset_val_split=dataset_val_split,
            num_epochs=num_expert_epochs
        ).out_models[num_expert_epochs]
    else:
        anti_expert_model = anti_expert_model_name_or_path

    if expert_model_name_or_path is None:
        expert_model = train_model(
            method,
            baseline_model,
            expert_dataset_name,
            expert_dataset_config_name,
            f"{model_description}_expert",
            gpu_mem=gpu_mem_train,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=per_device_eval_batch_size,
            dataset_train_split=dataset_train_split,
            dataset_val_split=dataset_val_split,
            num_epochs=num_expert_epochs
        ).out_models[num_expert_epochs]
    else:
        expert_model = expert_model_name_or_path

    for test_dataset in test_datasets:

        # baseline

        evaluate_model(
            method,
            baseline_model,
            test_dataset,
            test_dataset_config_name,
            model_description,
            gpu_mem=gpu_mem_test,
            dataset_test_split=dataset_test_split,
            calculate_q2=True
        )

        # Task Arithmetic

        for scaling_factor in [1.0]:
            scaling_factor = round(scaling_factor, 2)

            # Task Arithmetic
            new_model = MakeAndApplyTaskVectorsJob(
                code_root,
                baseline_model,
                [],
                [anti_expert_model],
                scaling_factor_experts=[],
                scaling_factor_anti_experts=[scaling_factor]
            ).out_model_path

            evaluate_model(
                method,
                new_model,
                test_dataset,
                test_dataset_config_name,
                model_description+"_task_arithmetic_"+str(scaling_factor),
                gpu_mem=gpu_mem_test,
                dataset_test_split=dataset_test_split,
                calculate_q2=True
            )

        # EWR
        if baseline_fisher_path is None:
            fisher_base = calculate_fisher_information(
                fisher_estimation_method,
                baseline_model,
                train_dataset,
                train_dataset_config_name,
                model_description,
                dataset_test_split=dataset_val_split if not dataset_val_split == dataset_test_split else dataset_train_split,
                time_rqmt=4,
                gpu_mem=gpu_mem_fisher
            )
        else:
            fisher_base = baseline_fisher_path

        task_vector = MakeTaskVectorsJob(
            code_root,
            baseline_model,
            anti_expert_model,
            operation="negation"
        ).out_model_path
        
        if anti_expert_fisher_path is None:
            fisher_task_vector = calculate_fisher_information(
                fisher_estimation_method,
                task_vector,
                train_dataset,
                train_dataset_config_name,
                model_description,
                dataset_test_split=dataset_val_split if not dataset_val_split == dataset_test_split else dataset_train_split,
                time_rqmt=4,
                gpu_mem=gpu_mem_fisher
            )
        else:
            fisher_task_vector = anti_expert_fisher_path


        for scaling_factor in [0.15]:
            scaling_factor = round(scaling_factor, 3)

            new_model = MakeAndApplyTaskVectorsEWRJob(
                code_root,
                baseline_model,
                [],
                [anti_expert_model],
                fisher_base,
                [],
                [fisher_task_vector],
                scaling_factors_experts=[],
                scaling_factors_anti_experts=[scaling_factor]
            ).out_model_path

            evaluate_model(
                method,
                new_model,
                test_dataset,
                test_dataset_config_name,
                model_description+"_task_arithmetic_fisher"+str(scaling_factor),
                gpu_mem=gpu_mem_test,
                dataset_test_split=dataset_test_split,
                calculate_q2=True
            )

async def task_arithmetic():
    config = {
        "method": "document_grounded_generation",
        "model_name_or_path": "google/flan-t5-base",
        "model_description": "flan_t5_base_baseline",
        "train_dataset": "wow",
        "test_datasets": ["wow"],
        "train_dataset_config_name": "response_generation",
        "test_dataset_config_name": "response_generation",
        "expert_dataset_name": "wow",
        "anti_expert_dataset_name": "faithdial",
        "expert_dataset_config_name": "cape_expert",
        "anti_expert_dataset_config_name": "hallucinated_response",
        "dataset_train_split": "train",
        "dataset_val_split": "validation",
        "dataset_test_split": "test",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "per_device_eval_batch_size": 8,
        "gpu_mem_train": 12,
        "gpu_mem_test": 10,
        "num_epochs": 10,
        "num_expert_epochs": 5,
        "gpu_mem_fisher": 12
    }

    run_models(**config)

    config = {
        "method": "document_grounded_generation_ctrl",
        "model_name_or_path": "google/flan-t5-base",
        "model_description": "flan_t5_base_baseline",
        "train_dataset": "wow",
        "test_datasets": ["wow"],
        "train_dataset_config_name": "ctrl",
        "test_dataset_config_name": "ctrl",
        "expert_dataset_name": "wow",
        "anti_expert_dataset_name": "faithdial",
        "expert_dataset_config_name": "cape_expert",
        "anti_expert_dataset_config_name": "hallucinated_response_ctrl",
        "dataset_train_split": "train",
        "dataset_val_split": "validation",
        "dataset_test_split": "test",
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "per_device_eval_batch_size": 8,
        "gpu_mem_train": 12,
        "gpu_mem_test": 10,
        "num_epochs": 10,
        "num_expert_epochs": 5,
        "gpu_mem_fisher": 12
    }

    run_ctrl(**config)

async def async_main():
    await task_arithmetic()

async def py():
    await async_main()
