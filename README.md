# Faithful Dialogue

This repository serves two main purposes:
- it contains multiple models, datasets, and metrics for *document-grounded dialogue response generation*, allowing for their systematic comparison;
- it implements the methods from our paper [Elastic Weight Removal for Faithful and Abstractive Dialogue Generation](https://arxiv.org/pdf/2303.17574.pdf).

## Contents
  1. [Environment Set-up](#environment-set-up)
  2. [Running the code](#running-the-code)
  3. [Code Structure](#code-structure)
  4. [Contributing](#contributing)
  5. [Citation](#citation)

Here are all the datasets, models, and metrics that are implemented. The list **continues to grow**!

### Models
  | Model | method identifier | description |
  | ----- | ----------------- | ----------- |
  | Baseline | `document_grounded_generation`| (context, knowledge) -> response | 
  | [CTRL](https://arxiv.org/pdf/2107.06963) | `document_grounded_generation_ctrl`| (context, knowledge, control tokens) -> response |
  | [Quark](https://proceedings.neurips.cc/paper_files/paper/2022/file/b125999bde7e80910cbdbd323087df8f-Paper-Conference.pdf) | `document_grounded_generation_Quark`| CTRL tokens as quantized reward for samples drawn from the model during training |
  | [Noisy Channel Reranking](https://arxiv.org/pdf/2210.17418) | `noisy_channel_reranking`| reranking controllably with a faithfulness and fluency expert |
  | [DExperts](https://aclanthology.org/2021.acl-long.522.pdf) | `document_grounded_generation_density_ratio`| faithfulness expert and anti-expert combined at inference time |
  | [Task Arihmetic](https://arxiv.org/pdf/2212.04089) | `document_grounded_generation` | task vector subtracted from base model using hallucination anti-expert
  | [CaPE](https://arxiv.org/pdf/2110.07166.pdf) | `document_grounded_generation` | task vector subtracted from base model using hallucination anti-expert and faithfulness expert
  | [EWR](https://arxiv.org/pdf/2303.17574.pdf) | `document_grounded_generation` | task vector subtracted from base model using hallucination anti-expert weighted by Fisher Information

Note that TA, CaPE and EWR just change the model parameters via interpolation, but not the model architecture!

### Datasets
  1. [Wizard-of-Wikipedia](https://arxiv.org/pdf/1811.01241.pdf)
  2. [FaithDial](https://webdocs.cs.ualberta.ca/~zaiane/postscript/TACL2023.pdf)
  3. [DSTC9](https://arxiv.org/pdf/2006.03533)
  4. [DSTC11](https://github.com/alexa/dstc11-track5)

### Metrics
  1. [BLEU](https://aclanthology.org/P02-1040.pdf)
  2. [BERTScore](https://arxiv.org/pdf/1904.09675.pdf)
  3. [Faithfulness Critic](https://webdocs.cs.ualberta.ca/~zaiane/postscript/TACL2023.pdf)
  4. [Q^2](https://arxiv.org/pdf/2104.08202)
  5. Knowledge F1
  6. [Density & Coverage](https://arxiv.org/pdf/1804.11283)

## Environment Set-up
To ensure that the experiments are consistent and comparable, we use the [sisyphus](https://github.com/rwth-i6/sisyphus) workflow manager.

Sisyphus requires 3 folders (depending on the cluster set-up, you might want to symlink them to other folders, for example to use partitions optimised for large files):
  1. `alias`: It's possible to identify aliases for each job to identify it quickly (as a default, a hash is appended to the jobclass name as an identifier), and sisyphus adds a symlink to the job under the alias.
  2. `output`: `tk.register_output("name", job_class.file)` registers an output under the filename `name` in the output folder that symlinks to `job_class.file`
  3. `work`: All jobs will be placed here under their hash.

## Running the code

Using the code is as easy as invoking a sisyphus *config* by, for example using: ```sis --config config/baselines.py m``` which starts the manager that guides you through starting jobs and schedules all jobs depending on them automatically once they are finished!

If you want to write custom configs, you can use the existing `Job` objects that define an experiment to be scheduled. For example, training a model involves multiple subjobs, such as downloading data, saving it on disk, and then training the actual model.
These are defined in the `recipe` folder. For example, for training, you may use `HuggingfaceTrainingJob` found under `recipe/ukp/huggingface/training.py`.

The `TrainingJob` relies on configs that define all necessary information: method, model, datasets, hyperparameters.
For example, to train a document-grounded generation model using flan-t5-base on WoW, you can use the following:

```
    config = {
        'model_name_or_path': 'google/flan-t5-base',
        'predict_with_generate': True,
        'method': 'document_grounded_generation',
        'learning_rate': learning_rate,
        'per_device_train_batch_size': per_device_train_batch_size,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'cache_dir': '/your/path/to/.cache/',
    }
    train_data_config = {
        'dataset_name':  '/path/to/this/repo/dialog/datasets/wow.py',
        'dataset_config_name': 'response_generation',
        'dataset_train_split': 'train',
        'dataset_val_split': 'validation',
    }

    baseline_model = HuggingfaceTrainingJob(
        code_root='/path/to/this/repo/',
        config=config,
        train_data_config=train_data_config,
        num_epochs=num_epochs,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem
    ).out_best_model
    train_job.add_alias('wow_baseline_flant5base')
    tk.register_output('wow_baseline_flant5base', baseline_model)
```
This way, sisyphus will automatically take care of creating all files (your configs are stored in the job folder, for example), starting the job, and picking the best model according to validation loss!
Also, hyperparameters for job scheduling, like the time, cpu memory, gpu memory, are all taken care of!

Now, it's easy to use the model for inference by taking the object and running:
```
    search_data_config = {
        'dataset_name': '/path/to/this/repo/dialog/datasets/wow.py',
        'dataset_config_name': 'response_generation',
        'dataset_test_split': 'test',
    }
    
    search_job = HuggingfaceSearchJob(
        code_root='/path/to/this/repo/',
        model_path=baseline_model,
        config=config,
        search_data_config=search_data_config,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem,
    )
    tk.register_output('wow_baseline_flant5_base_generations.json, search_job.out_search_file)
    
    scoring_job = CalculateMetricsJob(
        code_root,
        search_data_config["dataset_name"],
        search_data_config['dataset_test_split'],
        search_job.out_search_file,
        time_rqmt=2,
    )
    tk.register_output(f'/your/path/to/results/description.metrics.json', scoring_job.out_results_file)
    
```

Using these basic Job objects, one can easily define all training runs. For example, for reproducing EWR with a hallucination anti-expert, you need to specify 1) the Fisher of the baseline, 2) the task vector, 3) the Fisher of the task vector, 4) merging as follows, using our helper methods defined in `config/baselines.py`:
```
    fisher_base = calculate_fisher_information(
        'fisher_approx_document_grounded_generation',
        baseline_model,
        '/path/to/this/repo/dialog/datasets/wow.py',,
        'response_generation',
        'wow_baseline_flant5base',
        dataset_test_split='validation',
        time_rqmt=4,
        gpu_mem=gpu_mem_fisher
    )
    
    train_data_config = {
        'dataset_name':  '/path/to/this/repo/dialog/datasets/faithdial.py',
        'dataset_config_name': 'hallucinated_response',
        'dataset_train_split': 'train',
        'dataset_val_split': 'validation',
    }
    
    config['model_name_or_path'] = baseline_model # Initialize the anti_expert using the baseline model!
    
    train_job_anti_expert = HuggingfaceTrainingJob(
        code_root='/path/to/this/repo/',
        config=config,
        train_data_config=train_data_config,
        num_epochs=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        gpumem=gpu_mem
    )
    train_job_anti_expert.add_alias('wow_baseline_flant5base')
    tk.register_output('wow_baseline_flant5base_anti_expert', train_job_anti_expert.out_models[5])
    
    task_vector = MakeTaskVectorsJob(
        code_root,
        baseline_model,
        anti_expert,
        operation="negation"
    ).out_model_path

    fisher_task_vector = calculate_fisher_information(
        'fisher_approx_document_grounded_generation',
        task_vector,
        '/path/to/this/repo/dialog/datasets/wow.py',
        'response_generation,
        'wow_baseline_flant5base_task_vector',
        dataset_test_split='validation',
        time_rqmt=4,
        gpu_mem=gpu_mem_fisher
    )
    
    new_model = MakeAndApplyTaskVectorsEWRJob(
        code_root,
        baseline_model, # baseline
        [], # expert models
        [anti_expert_model], # anti-expert models
        fisher_base, # fisher baseline
        [], # expert fishers
        [fisher_task_vector], # anti expert fishers
        scaling_factors_experts=[],
        scaling_factors_anti_experts=[scaling_factor]
    ).out_model_path
```
Then, one can use the same search job to start generating with a newly created model that is more faithful than its baseline!

## Code Structure
The code is mainly based on the concept of ''methods'' that are found in the `/code/dialog/methods/` folder which wrap all of the functionality needed to reproduce a certain method:
  1. Defining and loading Trainer and Data Collator classes
  2. Loading all datasets
  3. Defining the model class, which is either an out-of-the-box Hugging Face implementation OR found in `/code/dialog/models`
  4. Defining and applying the preprocessing methods, defined in `/code/dialog/methods/preprocessing`
  5. Defining special tokens
  
For example, the baseline for document-grounded response generation uses the `DocumentGroundedGenerationMethod`, which uses the default `Seq2SeqTrainer` and `DataCollator`, as well as the `AutoModelForSeq2SeqLM` and the `DocumentGroundedPreprocessor` class, which wraps tokenization according to a standard format.
For CTRL, the only required modification is then to specify the special tokens that are used, as well as the preprocessor that adds CTRL tokens (see the `DocumentGroundedPreprocessorWithCTRLTokens` class).

To understand how the method classes are structured it's probably best to check `code/dialog/methods/base.py` which defines a base class from which all methods inherit.

The main entry point for the code is `/code/dialog/main.py` that handles loading method classes, models, and running the Trainers.

## Contributing

We are happy about any contribution! Contributions are possible in many ways, for example:
  1. New methods:
    - All methods are implemented in `code/dialog/methods/.` For a new response generation method, one can add a new class that inherits from `Method` to `code/dialog/methods/generation.py` which defines all necessary components, namely the preprocessor, the model class, trainer, and data collator. Feel free to reuse some of the ones that are already implemented, as well!
  2. Datasets:
    - Document-Grounded Datasets inherit form `DocumentGroundedDataset` in `/code/dialog/datasets/base.py`, which handles common functionalities, like defining a common data format, and dataset configs, for example for experts and anti-experts. So to add a dataset, all you need to do is inherit from this class and define methods that a) download the data and b) map it to the common format.

## Citation
If you use our method EWR or this repository, please cite our paper:
```
@misc{daheim2023elastic,
      title={Elastic Weight Removal for Faithful and Abstractive Dialogue Generation}, 
      author={Nico Daheim and Nouha Dziri and Mrinmaya Sachan and Iryna Gurevych and Edoardo M. Ponti},
      year={2023},
      eprint={2303.17574},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
