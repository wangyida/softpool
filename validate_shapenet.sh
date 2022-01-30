# CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_pcn/network.pth --dataset shapenet --methods pcn
mv pcds/all_clouds6 pcds/pcn
# CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_pointcnn/network.pth --dataset shapenet --methods pointcnn
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_folding/network.pth --dataset shapenet --methods folding
mv pcds/all_clouds3 pcds/folding
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_grnet/network.pth --dataset shapenet --methods grnet
mv pcds/all_clouds4 pcds/grnet
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 8 --num_points 2048 --model log/ijcv_shapenet_softpool/network.pth --dataset shapenet --methods softpool
mv pcds/all_clouds1 pcds/softpool
# CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_msn/network.pth --dataset shapenet --methods msn
mv pcds/all_clouds2 pcds/msn
# CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_3dgcn/network.pth --dataset shapenet --methods pointgcn
mv pcds/all_clouds7 pcds/3dgcn
mv pcds/input pcds/z_input
mv pcds/gt pcds/z_gt
mv pcds/all_mesh pcds/z_mesh
