#!/bin/bash
#SBATCH -p batch # partition (this is the queue your job will be added to)
#SBATCH -n 4 # number of cores (here 2 cores requested)
#SBATCH -c 4
#SBATCH --time=24:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:4 # generic resource required (here requires 1 GPU)
#SBATCH --mem=48GB # memory pool for all cores (here set to 8 GB)
#SBATCH -M acvt
source /home/a1718443/tensorflow/bin/activate
cd /home/a1718443/code_seg/src 
python multi_gpu_train_refine.py --root-dir /fast/users/a1718443/coco/ \
                                 --batch-size 2 \
                                 --crop-size 512 \
                                 --init-weights /fast/users/a1718443/logs/model.ckpt-175999 \
				 --solver-state /fast/users/a1718443/logs/nas_refine/ \
                                 --learning-rate 2e-5 \
                                 --exclude-scope None \
				 --save-step 6000 \
                                 --ohem True \
				 --step-size 100000 \
                                 --save-dir /fast/users/a1718443/logs/nas_refine \
                                 --gpus-list 0,1,2,3 \
                                 2>&1 | tee /home/a1718443/code_seg/src/logs/train_4.log



