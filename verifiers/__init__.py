# 将信息导入到包的命名空间中
from .envs.environment import Environment
from .envs.code_env import CodeEnv
from .envs.doublecheck_env import DoubleCheckEnv
from .envs.math_env import MathEnv
from .envs.simple_env import SimpleEnv
from .envs.tool_env import ToolEnv
from .trainers.grpo_env_trainer import GRPOEnvTrainer
from .utils.data_utils import extract_boxed_answer, extract_hash_answer, preprocess_dataset
from .utils.model_utils import get_model, get_tokenizer, get_model_and_tokenizer
from .utils.config_utils import get_default_grpo_config
from .utils.logging_utils import setup_logging, print_prompt_completions_sample

__version__ = "0.1.0"

# Setup default logging configuration
# 在整个包中都可以使用相同的日志配置，方便调试和监控
setup_logging()

# 注册导入信息
__all__ = [
    "Environment",
    "CodeEnv",
    "DoubleCheckEnv",
    "MathEnv",
    "SimpleEnv",
    "ToolEnv",
    "GRPOEnvTrainer",
    "get_model",
    "get_tokenizer",
    "get_model_and_tokenizer",
    "get_default_grpo_config",
    "extract_boxed_answer",
    "extract_hash_answer",
    "preprocess_dataset",
    "setup_logging",
    "print_prompt_completions_sample",
]
