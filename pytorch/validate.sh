CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_pcn/network.pth --dataset shapenet --methods pcn
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_pointcnn/network.pth --dataset shapenet --methods pointcnn
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_folding/network.pth --dataset shapenet --methods folding
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_grnet/network.pth --dataset shapenet --methods grnet
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 8 --num_points 2048 --model log/ijcv_shapenet_softpool8/network.pth --dataset shapenet --methods softpool
mv pcds/clouds1 pcds/clouds7
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_softpool/network.pth --dataset shapenet --methods softpool
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_msn/network.pth --dataset shapenet --methods msn
