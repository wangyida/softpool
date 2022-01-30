CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_fusion_pcn/network.pth --dataset fusion --methods pcn
mv pcds/all_clouds6 pcds/pcn
# CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_fusion_pointcnn/network.pth --dataset fusion --methods pointcnn
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_fusion_folding/network.pth --dataset fusion --methods folding
mv pcds/all_clouds3 pcds/folding
# CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_fusion_grnet/network.pth --dataset fusion --methods grnet
# mv pcds/all_clouds4 pcds/grnet
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_fusion_softpool/network.pth --dataset fusion --methods softpool
mv pcds/all_clouds1 pcds/softpool
CUDA_VISIBLE_DEVICES=1 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_fusion_msn/network.pth --dataset fusion --methods msn
mv pcds/all_clouds2 pcds/msn
# CUDA_VISIBLE_DEVICES=0 python3 val.py --n_regions 1 --num_points 2048 --model log/ijcv_shapenet_3dgcn/network.pth --dataset shapenet --methods pointgcn
# mv pcds/all_clouds7 pcds/3dgcn
mv pcds/input pcds/z_input
mv pcds/gt pcds/z_gt
mv pcds/all_mesh pcds/z_mesh
