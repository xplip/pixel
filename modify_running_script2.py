import json                                                                                                                                                                                                          
import os                                                                                                                                                                                                            
from dataclasses import asdict, dataclass, field                                                                                                                                                                     
from typing import Any, Dict, List, Optional, Union                                                                                                                                                                  
                                                                                                                                                                                                                     
import yaml                                                                                                                                                                                                          
from transformers import HfArgumentParser                                                                                                                                                                            
                                                                                                                                                                                                                     
from configs.config_maps import MODEL_PROTOTYPE_CONFIGS, TRAINING_CONFIGS                                                                                                                                            
                                                                                                                                                                                                                     
                                                                                                                                                                                                                     
@dataclass                                                                                                                                                                                                           
class SubmititTrainingArguments:                                                                                                                                                                                     
    job_dir: str = field(metadata={"help": "Job dir"})  
    interactive_job_name: str = field(default=None, metadata={"help": "Job name for interactive pod"})                                                                                                                                                               
    prototype_config_name: Optional[str] = field(                                                                                                                                                                    
        default=None, metadata={"help": "Name of the model prototype config to train"}                                                                                                                               
    )                                                                                                                                                                                                                
    training_config_name: Optional[str] = field(default=None, metadata={"help": "Name of the training config to use"})                                                                                               
    ngpus: Optional[int] = field(default=1, metadata={"help": "Number of gpus to request on each node"})                                                                                                             
    nodes: Optional[int] = field(default=1, metadata={"help": "Number of nodes to request"})                                                                                                                         
    # timeout: Optional[int] = field(default=4320, metadata={"help": "Duration of the job"})
    # partition: Optional[str] = field(default="debug", metadata={"help": "Partition where to submit"})
    # exclude_nodes: Optional[str] = field(default=None, metadata={"help": "Slurm nodes to exclude"})
    # comment: Optional[str] = field(default=None, metadata={"help": "Comment to pass to scheduler"})
                                                                                                                                                                                                            
    def __post_init__(self):                                                                                                                                                                                         
        if self.prototype_config_name and self.prototype_config_name not in MODEL_PROTOTYPE_CONFIGS:                                                                                                                 
            raise ValueError(                                                                                                                                                                                        
                f"Specified prototype model config not available. "                                                                                                                                                  
                f"Available options: {list(MODEL_PROTOTYPE_CONFIGS.keys())}"                                                                                                                                         
            )                                                                                                                                                                                                        
        if self.interactive_job_name is not None:  
            self.job_dir = os.path.join(self.job_dir, self.interactive_job_name)     
        else:                                                                                                                                                                                                           
            self.job_dir = os.path.join(self.job_dir, "%j")                                                                                                                                                              
                                                                                                                                                                                                                     
    def __str__(self):                                                                                                                                                                                               
        self_as_dict = asdict(self)                                                                                                                                                                                  
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]                                                                                                                                      
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"                                                                                                                                               
                                                                                                                                                                                                                     
    __repr__ = __str__                                                                                                                                                                                               
                                                                                                                                                                                                                     
def process_remaining_strings(remaining_strings: Union[str, List[str]]):                                                                                                                                             
    def parse_string(s: str):                                                                                                                                                                                        
        s = s.strip().replace("--", "")                                                                                                                                                                              
        if " " in s:                                                                                                                                                                                                 
            k, v = s.split(" ")                                                                                                                                                                                      
        elif "=" in s:                                                                                                                                                                                               
            k, v = s.split("=")                                                                                                                                                                                      
        else:                                                                                                                                                                                                        
            k, v = s, "True"      
        return {k: yaml.safe_load(v)}                                                                                                                                                                                
                                                                                                                                                                                                                     
    if isinstance(remaining_strings, str):
        remaining_strings_dict = parse_string(remaining_strings)
    else:
        remaining_strings_dict = {}
        [remaining_strings_dict.update(parse_string(rs)) for rs in remaining_strings]
    return remaining_strings_dict

def get_config_dict(**kwargs):
    job_name = kwargs.pop("interactive_job_name")
    print("job_name", job_name)
    if job_name is None:
        config_dict = {"output_dir": kwargs.pop("job_dir"), "run_name": "pixel-%j"}
    else:
        config_dict = {"output_dir": kwargs.pop("job_dir"), "run_name": "pixel-{}".format(job_name)}

    submitit_kwargs = ["ngpus", "nodes"]
    [kwargs.pop(k) for k in submitit_kwargs]

    # Model config
    prototype_config_name = kwargs.pop("prototype_config_name")
    if prototype_config_name:
        print(MODEL_PROTOTYPE_CONFIGS)
        print(MODEL_PROTOTYPE_CONFIGS[prototype_config_name])
        model_config = load_json(MODEL_PROTOTYPE_CONFIGS[prototype_config_name])
        [kwargs.pop(k) for k in model_config.keys() if k in kwargs]
        config_dict.update(model_config)

        config_dict["run_name"] = f"{prototype_config_name}-%j"

    # Training config
    training_config_name = kwargs.pop("training_config_name")
    if training_config_name:
        print(TRAINING_CONFIGS)
        training_config = load_json(TRAINING_CONFIGS[training_config_name])
        [kwargs.pop(k) for k in training_config.keys() if k in kwargs]
        config_dict.update(training_config)

    config_dict.update(kwargs)

    return config_dict

def load_json(json_name: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(json_name, "r", encoding="utf-8") as f: 
        data = json.load(f)
    return data
     
class Trainer:
    def __init__(
        self,
        config_dict: Dict[str, Any],
    ):
        config_dict["do_eval"] = False
        self.config_dict = config_dict

    def __call__(self):
        import scripts.training.run_pretraining as trainer
        print("successful trainer")

        # self._setup()
        trainer.main(self.config_dict)
        print("successful finish trainer")


    # def _setup(self):
    #     import submitit

    #     job_env = submitit.JobEnvironment()

    #     self.config_dict["output_dir"] = os.path.join(
    #         self.config_dict["output_dir"].replace("%j", str(job_env.job_id)), "outputs"
    #     )
    #     self.config_dict["run_name"] = self.config_dict["run_name"].replace("%j", str(job_env.job_id))

    #     os.environ["LOCAL_RANK"] = str(job_env.local_rank)
    #     if "SLURM_PROCID" in os.environ:
    #         os.environ["RANK"] = os.environ["SLURM_PROCID"]
    #     else:
    #         os.environ["RANK"] = str(job_env.global_rank)
    #     os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
    #     print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    trainer_parser = HfArgumentParser(SubmititTrainingArguments)
    args, remaining_strings = trainer_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args_dict = asdict(args)
    # Get run configuration
    if remaining_strings:
        remaining_strings_dict = process_remaining_strings(remaining_strings)
        args_dict.update(remaining_strings_dict)

    config_dict = get_config_dict(**args_dict)
    print(config_dict)
    print(config_dict["run_name"])
    # print("successful config_dict")
    # trainer = Trainer(config_dict)       
    # trainer() 

if __name__ == "__main__":
    main()