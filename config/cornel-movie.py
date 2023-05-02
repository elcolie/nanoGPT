import time

out_dir = 'out-cornel-movie'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'cornel-movie'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'cornel-movie'
# init_from = 'gpt2-xl' # this is the largest GPT-2 model
init_from = 'scratch'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 100

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

# on macbook also add
device = 'mps'  # run on mps, cpu
compile = False # do not torch compile the model