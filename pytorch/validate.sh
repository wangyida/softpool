CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_pcn/network.pth --dataset shapenet --methods pcn
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_pointcnn/network.pth --dataset shapenet --methods pointcnn
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_folding/network.pth --dataset shapenet --methods folding
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_dense/network.pth --dataset shapenet --methods grnet
CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet/network.pth --dataset shapenet --methods softpool msn
