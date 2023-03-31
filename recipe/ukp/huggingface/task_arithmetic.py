import copy
import os
import shutil
import subprocess as sp
import copy
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from sisyphus import *

import i6_core.util as util
from i6_core.returnn.config import instanciate_delayed

Path = setup_path(__package__)


class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                if os.path.exists(pretrained_checkpoint):
                    pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
                else:
                    pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict()
                if os.path.exists(finetuned_checkpoint):
                    finetuned_state_dict = torch.load(finetuned_checkpoint, map_location='cpu')
                else:
                    finetuned_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(finetuned_checkpoint.split("/")[:-1])).state_dict()                
                
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            if os.path.exists(pretrained_checkpoint):
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
            else:
                pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        return new_state_dict

class TaskVectorWithFisher(TaskVector):
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, pretrained_fisher=None, finetuned_fisher=None, vector=None, fisher_floor=1e-6):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.pretrained_fisher_path = pretrained_fisher
        self.finetuned_fisher_path = finetuned_fisher
        self.pretrained_fisher = torch.load(pretrained_fisher, map_location='cpu')
        self.finetuned_fisher = torch.load(finetuned_fisher, map_location='cpu')
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                if os.path.exists(pretrained_checkpoint):
                    pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
                else:
                    pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict()
                if os.path.exists(finetuned_checkpoint):
                    finetuned_state_dict = torch.load(finetuned_checkpoint, map_location='cpu')
                else:
                    finetuned_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(finetuned_checkpoint.split("/")[:-1])).state_dict()  

                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVectorWithFisher(vector=new_vector, pretrained_fisher=self.pretrained_fisher_path, finetuned_fisher=self.finetuned_fisher_path)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVectorWithFisher(vector=new_vector, pretrained_fisher=self.pretrained_fisher_path, finetuned_fisher=self.finetuned_fisher_path)

    def apply_to(self, pretrained_checkpoint, scaling_coef_pretrained=1.0, scaling_coef_finetuned=1.0, fisher_floor=1e-6):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            if os.path.exists(pretrained_checkpoint):
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
            else:
                pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict() 
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = ((scaling_coef_pretrained*torch.maximum(self.pretrained_fisher[key], torch.tensor(fisher_floor))*pretrained_state_dict[key]) \
                                        + (scaling_coef_finetuned*torch.maximum(self.finetuned_fisher[key], torch.tensor(fisher_floor))*self.vector[key])) / \
                                        (scaling_coef_pretrained*torch.maximum(self.pretrained_fisher[key], torch.tensor(fisher_floor)) + scaling_coef_finetuned*torch.maximum(self.finetuned_fisher[key], torch.tensor(fisher_floor)))
        return new_state_dict

class MakeTaskVectorsJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        finetuned_model_name_or_path,
        operation="negation",
        scaling_factor=1.0,
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=0,
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.finetuned_model_name_or_path = finetuned_model_name_or_path
        self.scaling_factor = scaling_factor

        self.operation = operation

        self.out_model_path = self.output_path("model", directory=True)

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def apply(self, pretrained_checkpoint, task_vector, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            if os.path.exists(pretrained_checkpoint):
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
            else:
                pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict()
            for key in pretrained_state_dict:
                if key not in task_vector.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = task_vector.vector[key]
        return new_state_dict

    def run(self):
        model_file_name = "pytorch_model.bin"
        pretrained_model_file = os.path.join(self.pretrained_model_name_or_path, model_file_name)
        finetuned_model_file = os.path.join(self.finetuned_model_name_or_path, model_file_name)
        task_vector = TaskVector(pretrained_checkpoint=pretrained_model_file, finetuned_checkpoint=finetuned_model_file)
        # Negate the task vector
        if self.operation == "negation":
            task_vector = -task_vector
        # Apply the task vector
        new_model = self.apply(pretrained_model_file, task_vector, scaling_coef=self.scaling_factor)

        if os.path.exists(self.pretrained_model_name_or_path):
            for file in os.listdir(self.pretrained_model_name_or_path):
                if file != model_file_name:
                    input_path = os.path.join(self.pretrained_model_name_or_path, file)
                    output_path = os.path.join(self.out_model_path, file)
                    if not os.path.isdir(input_path):
                        shutil.copy(input_path, output_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            model.save_pretrained(self.out_model_path)
            tokenizer.save_pretrained(self.out_model_path)

        output_path = os.path.join(self.out_model_path, model_file_name)

        torch.save(new_model, output_path)

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
    # yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class MakeAndApplyTaskVectorsCapeJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        expert_model_name_or_path,
        anti_expert_model_name_or_path,
        operation="addition",
        scaling_factor=1.0,
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=0,
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.anti_expert_model_name_or_path = anti_expert_model_name_or_path
        self.operation = operation
        self.scaling_factor = scaling_factor

        self.out_model_path = self.output_path("model", directory=True)

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def run(self):
        model_file_name = "pytorch_model.bin"
        pretrained_model_file = os.path.join(self.pretrained_model_name_or_path, model_file_name)
        expert_model_file = os.path.join(self.expert_model_name_or_path, model_file_name)
        anti_expert_model_file = os.path.join(self.anti_expert_model_name_or_path, model_file_name)
        task_vector = TaskVector(expert_model_file, anti_expert_model_file)
        # Negate the task vector
        neg_task_vector = -task_vector
        # Apply the task vector
        new_model = neg_task_vector.apply_to(pretrained_model_file, scaling_coef=self.scaling_factor)

        if os.path.exists(self.pretrained_model_name_or_path):
            for file in os.listdir(self.pretrained_model_name_or_path):
                if file != model_file_name:
                    input_path = os.path.join(self.pretrained_model_name_or_path, file)
                    output_path = os.path.join(self.out_model_path, file)
                    if not os.path.isdir(input_path):
                        shutil.copy(input_path, output_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            model.save_pretrained(self.out_model_path)
            tokenizer.save_pretrained(self.out_model_path)

        output_path = os.path.join(self.out_model_path, model_file_name)

        torch.save(new_model, output_path)

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
    # yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class MakeAndApplyTaskVectorsCapeWithFisherJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        expert_model_name_or_path,
        anti_expert_model_name_or_path,
        pretrained_fisher_name_or_path,
        finetuned_fisher_name_or_path,
        operation="addition",
        scaling_factor=1.0,
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=0,
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.anti_expert_model_name_or_path = anti_expert_model_name_or_path
        self.pretrained_fisher_name_or_path = pretrained_fisher_name_or_path
        self.finetuned_fisher_name_or_path = finetuned_fisher_name_or_path
        self.operation = operation
        self.scaling_factor = scaling_factor

        self.out_model_path = self.output_path("model", directory=True)

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def run(self):
        model_file_name = "pytorch_model.bin"
        pretrained_model_file = os.path.join(self.pretrained_model_name_or_path, model_file_name)
        expert_model_file = os.path.join(self.expert_model_name_or_path, model_file_name)
        anti_expert_model_file = os.path.join(self.anti_expert_model_name_or_path, model_file_name)
        pretrained_fisher_file = os.path.join(self.pretrained_fisher_name_or_path, "fisher_estimates", model_file_name)
        finetuned_fisher_model_file = os.path.join(self.finetuned_fisher_name_or_path, "fisher_estimates", model_file_name)

        task_vector = CapeTaskVectorWithFisher(pretrained_model_file, expert_model_file, anti_expert_model_file, pretrained_fisher_file, finetuned_fisher_model_file)
        # Negate the task vector
        if self.operation == "negation":
            task_vector = -task_vector
        # Apply the task vector
        new_model = task_vector.apply_to(pretrained_model_file, scaling_coef=self.scaling_factor)

        if os.path.exists(self.pretrained_model_name_or_path):
            for file in os.listdir(self.pretrained_model_name_or_path):
                if file != model_file_name:
                    input_path = os.path.join(self.pretrained_model_name_or_path, file)
                    output_path = os.path.join(self.out_model_path, file)
                    if not os.path.isdir(input_path):
                        shutil.copy(input_path, output_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            model.save_pretrained(self.out_model_path)
            tokenizer.save_pretrained(self.out_model_path)

        output_path = os.path.join(self.out_model_path, model_file_name)

        torch.save(new_model, output_path)

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
    # yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)

class CapeTaskVectorWithFisher(TaskVectorWithFisher):
    def __init__(self, pretrained_checkpoint=None, expert_checkpoint=None, anti_expert_checkpoint=None, pretrained_fisher=None, finetuned_fisher=None, vector=None, fisher_floor=1e-6):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.pretrained_fisher_path = pretrained_fisher
        self.finetuned_fisher_path = finetuned_fisher
        self.pretrained_fisher = torch.load(pretrained_fisher, map_location='cpu')
        self.finetuned_fisher = torch.load(finetuned_fisher, map_location='cpu')
        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and expert_checkpoint is not None and anti_expert_checkpoint is not None
            with torch.no_grad():
                if os.path.exists(pretrained_checkpoint):
                    pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
                else:
                    pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict()
                if os.path.exists(expert_checkpoint):
                    expert_state_dict = torch.load(expert_checkpoint, map_location='cpu')
                else:
                    expert_checkpoint = AutoModelForSeq2SeqLM.from_pretrained("/".join(expert_checkpoint.split("/")[:-1])).state_dict()
                if os.path.exists(anti_expert_checkpoint):
                    anti_expert_state_dict = torch.load(anti_expert_checkpoint, map_location='cpu')
                else:
                    anti_expert_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(anti_expert_checkpoint.split("/")[:-1])).state_dict()

                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = anti_expert_state_dict[key] - expert_state_dict[key]
    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return CapeTaskVectorWithFisher(vector=new_vector, pretrained_fisher=self.pretrained_fisher_path, finetuned_fisher=self.finetuned_fisher_path)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return CapeTaskVectorWithFisher(vector=new_vector, pretrained_fisher=self.pretrained_fisher_path, finetuned_fisher=self.finetuned_fisher_path)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, fisher_floor=1e-6, fisher_ceil=1e6):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            if os.path.exists(pretrained_checkpoint):
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location='cpu')
            else:
                pretrained_state_dict = AutoModelForSeq2SeqLM.from_pretrained("/".join(pretrained_checkpoint.split("/")[:-1])).state_dict()            #pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                scaling_coef_pretrained = 1.0 - scaling_coef
                scaling_coef_finetuned = scaling_coef
                new_state_dict[key] = ((scaling_coef_pretrained*torch.clamp(self.pretrained_fisher[key], torch.tensor(fisher_floor), torch.tensor(fisher_ceil))*pretrained_state_dict[key]) \
                                        + (scaling_coef_finetuned*torch.clamp(self.finetuned_fisher[key], torch.tensor(fisher_floor), torch.tensor(fisher_ceil))*self.vector[key])) / \
                                        (scaling_coef_pretrained*torch.clamp(self.pretrained_fisher[key], torch.tensor(fisher_floor), torch.tensor(fisher_ceil)) + scaling_coef_finetuned*torch.clamp(self.finetuned_fisher[key], torch.tensor(fisher_floor), torch.tensor(fisher_ceil)))

        return new_state_dict

class MakeAndApplyTaskVectorsEWRJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        expert_model_name_or_paths,
        anti_expert_model_name_or_paths,
        pretrained_fisher_path,
        expert_fisher_paths,
        anti_expert_fisher_paths,
        operation="addition",
        scaling_factor_pretrained=1.0,
        scaling_factors_experts=None,
        scaling_factors_anti_experts=None,
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=1,
        gpumem=10,
        **kwargs
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.expert_model_name_or_paths = expert_model_name_or_paths
        self.anti_expert_model_name_or_paths = anti_expert_model_name_or_paths
        self.pretrained_fisher_path = pretrained_fisher_path
        self.expert_fisher_paths = expert_fisher_paths
        self.anti_expert_fisher_paths = anti_expert_fisher_paths
        self.operation = operation
        self.scaling_factors_experts = scaling_factors_experts
        self.scaling_factors_anti_experts = scaling_factors_anti_experts

        self.out_model_path = self.output_path("model", directory=True)

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
            "gpumem": gpumem,
        }

    def run(self):
        model_file_name = "pytorch_model.bin"
        pretrained_model_file = os.path.join(self.pretrained_model_name_or_path, model_file_name)
        pretrained_fisher_file = os.path.join(self.pretrained_fisher_path, "fisher_estimates", model_file_name)
        
        fishers, task_vectors = [], []
        for model_path, fisher_path in zip(self.expert_model_name_or_paths, self.expert_fisher_paths):
            finetuned_model_file = os.path.join(model_path, model_file_name)
            finetuned_fisher_file = os.path.join(fisher_path, "fisher_estimates", model_file_name)
            task_vector = TaskVectorWithFisher(pretrained_model_file, finetuned_model_file, pretrained_fisher_file, finetuned_fisher_file)
            task_vectors.append(task_vector)
            fishers.append(finetuned_fisher_file)

        for model_path, fisher_path in zip(self.anti_expert_model_name_or_paths, self.anti_expert_fisher_paths):
            finetuned_model_file = os.path.join(model_path, model_file_name)
            finetuned_fisher_file = os.path.join(fisher_path, "fisher_estimates", model_file_name)
            task_vector = TaskVectorWithFisher(pretrained_model_file, finetuned_model_file, pretrained_fisher_file, finetuned_fisher_file)
        # Negate the task vector
            neg_task_vector = -task_vector
            task_vectors.append(neg_task_vector)
            fishers.append(finetuned_fisher_file)

        # Apply the task vector
        model = torch.load(pretrained_model_file)
        fisher = torch.load(pretrained_fisher_file)
        new_model = {}
        Z = {}
        for key in model.keys():
            new_model[key] = torch.maximum(fisher[key], torch.tensor(1e-6)).to(model[key].device) * model[key]
            Z[key] = torch.maximum(fisher[key], torch.tensor(1e-6)).to(model[key].device)
        for task_vector, fisher, scaling_factor in zip(task_vectors, fishers, self.scaling_factors_experts + self.scaling_factors_anti_experts):
            fisher = torch.load(fisher)
            for key in model.keys():
                new_model[key] += scaling_factor * torch.maximum(fisher[key], torch.tensor(1e-6)).to(new_model[key].device) * task_vector.vector[key].to(new_model[key].device)
                Z[key] += scaling_factor * torch.maximum(fisher[key], torch.tensor(1e-6)).to(Z[key].device)

        for key in model.keys():
            new_model[key] /= Z[key]

        if os.path.exists(self.pretrained_model_name_or_path):
            for file in os.listdir(self.pretrained_model_name_or_path):
                if file != model_file_name:
                    input_path = os.path.join(self.pretrained_model_name_or_path, file)
                    output_path = os.path.join(self.out_model_path, file)
                    if not os.path.isdir(input_path):
                        shutil.copy(input_path, output_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            model.save_pretrained(self.out_model_path)
            tokenizer.save_pretrained(self.out_model_path)

        output_path = os.path.join(self.out_model_path, model_file_name)

        torch.save(new_model, output_path)

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
    # yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)


class MakeAndApplyTaskVectorsJob(Job):

    def __init__(
        self,
        code_root,
        pretrained_model_name_or_path,
        expert_model_name_or_paths,
        anti_expert_model_name_or_paths,
        operation="addition",
        scaling_factor_pretrained=1.0,
        scaling_factors_experts=[],
        scaling_factors_anti_experts=[],
        time_rqmt=1,
        mem_rqmt=24,
        cpu_rqmt=1,
        gpu_rqmt=1,
        **kwargs
    ):
        self.code_root = code_root
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.expert_model_name_or_paths = expert_model_name_or_paths
        self.anti_expert_model_name_or_paths = anti_expert_model_name_or_paths
        self.operation = operation
        self.scaling_factors_experts = scaling_factors_experts
        self.scaling_factors_anti_experts = scaling_factors_anti_experts

        self.out_model_path = self.output_path("model", directory=True)

        self.rqmt = {
            "gpu": gpu_rqmt,
            "cpu": cpu_rqmt,
            "mem": mem_rqmt,
            "time": time_rqmt,
        }

    def run(self):
        model_file_name = "pytorch_model.bin"
        pretrained_model_file = os.path.join(self.pretrained_model_name_or_path, model_file_name)
        
        task_vectors = []
        for model_path in self.expert_model_name_or_paths:
            finetuned_model_file = os.path.join(model_path, model_file_name)
            task_vector = TaskVector(pretrained_model_file, finetuned_model_file)
            task_vectors.append(task_vector)

        for model_path in self.anti_expert_model_name_or_paths:
            finetuned_model_file = os.path.join(model_path, model_file_name)
            task_vector = TaskVector(pretrained_model_file, finetuned_model_file)
        # Negate the task vector
            neg_task_vector = -task_vector
            task_vectors.append(neg_task_vector)

        # Apply the task vector
        new_model = torch.load(pretrained_model_file)
        new_state_dict = {}

        for task_vector, scaling_factor in zip(task_vectors, self.scaling_factors_experts + self.scaling_factors_anti_experts):
            for key in new_model.keys():
                new_model[key] += scaling_factor * task_vector.vector[key].to(new_model[key].device)

        if os.path.exists(self.pretrained_model_name_or_path):
            for file in os.listdir(self.pretrained_model_name_or_path):
                if file != model_file_name:
                    input_path = os.path.join(self.pretrained_model_name_or_path, file)
                    output_path = os.path.join(self.out_model_path, file)
                    if not os.path.isdir(input_path):
                        shutil.copy(input_path, output_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
            model.save_pretrained(self.out_model_path)
            tokenizer.save_pretrained(self.out_model_path)

        output_path = os.path.join(self.out_model_path, model_file_name)

        torch.save(new_model, output_path)

    def create_files(self):
        util.create_executable("run.sh", self._get_run_cmd())

    def tasks(self):
    # yield Task("create_files", mini_task=True)
        yield Task("run", resume="run", rqmt=self.rqmt)

    @classmethod
    def hash(cls, kwargs):
        hash_kwargs = copy.deepcopy(kwargs)
        excluded_keys = ['time_rqmt', 'mem_rqmt', 'cpu_rqmt', 'gpu_rqmt']
        for key in excluded_keys:
            if key in hash_kwargs:
                del hash_kwargs[key]

        return super().hash(hash_kwargs)