#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-4B-Instruct-2507',cache_dir='/mnt/workspace/LLM')
model_dir = snapshot_download('BAAI/bge-small-zh-v1___5',cache_dir='/mnt/workspace/LLM')