cd code
python -m torch.distributed.launch --master_port 31236 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_mlp.conf --scan_id=1 --is_continue --timestamp _pretrained
python -m torch.distributed.launch --master_port 31237 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_mlp.conf --scan_id=2 --is_continue --timestamp _pretrained
python -m torch.distributed.launch --master_port 31238 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_mlp.conf --scan_id=3 --is_continue --timestamp _pretrained
python -m torch.distributed.launch --master_port 31239 --nproc_per_node 1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_mlp.conf --scan_id=4 --is_continue --timestamp _pretrained
cd ..

cd scannet_eval
python3 evaluate.py
cd ..