from datetime import datetime

import verifiers as vf

# model_name = "Qwen/Qwen2.5-Math-1.5B"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = vf.MathEnv(dataset="gsm8k")
dataset = vf_env.get_dataset()
# 获取奖励函数
rubric = vf_env.get_rubric()

run_name = "gsm8k_" + model_name.split("/")[-1].lower()
# 获取当前时间
now = datetime.now()

# 格式化为只包含到分钟的字符串
current_time_str = now.strftime("%m_%d_%H_%M")
output_dir = 'outputs' + run_name + current_time_str
training_args = vf.get_default_grpo_config(run_name=run_name, num_gpus=8)
trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=dataset,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model(output_dir)
