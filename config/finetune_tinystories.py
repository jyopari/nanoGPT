import time

out_dir = 'out-tinystories'
eval_interval = 5
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'ngpt-tinystories'
wandb_run_name = 'ft-moe-' + str(time.time())

dataset = 'tinystories'
init_from = 'gpt2' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 100

# finetune at constant LR
learning_rate = 4e-3
decay_lr = False

is_moe = True
add_lora = False

compile = True
