from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('Qwen/Qwen2.5-Math-1.5B-Instruct', cache_dir='/root/autodl-tmp/base_model/', revision='master')
