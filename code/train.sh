conf=$1
scan_id=$2
cuda_devices=$3

CUDA_VISIBLE_DEVICES=$cuda_devices python -m torch.distributed.launch  --master_port $((cuda_devices+22112)) --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf $conf  --scan_id $scan_id