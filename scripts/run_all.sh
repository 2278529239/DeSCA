#!/usr/bin/env bash

###############################################################################
# DeSCA
# Run all experiments reported in the paper
#
# Usage:
#   bash scripts/run_all.sh
###############################################################################

set -e

GPU=0

echo "=================================================================="
echo "DeSCA Experiments"
echo "GPU = ${GPU}"
echo "=================================================================="

###############################################################################
# EAC + DeSCA
###############################################################################

echo ""
echo "==================== EAC + DeSCA ===================="

python main.py --conf conf/PEMS03/eac_DeSCA.json   --gpuid ${GPU} --seed 43
python main.py --conf conf/AIR/eac_DeSCA.json    --gpuid ${GPU} --seed 46
python main.py --conf conf/PEMS04/eac_DeSCA.json --gpuid ${GPU} --seed 43


###############################################################################
# STBP + DeSCA
###############################################################################

echo ""
echo "==================== STBP + DeSCA ===================="

python mainSTBP_DeSCA.py --conf conf/PEMS03/STBP_DeSCA.json --gpuid ${GPU} --seed 43
python mainSTBP_DeSCA.py --conf conf/AIR/STBP_DeSCA.json   --gpuid ${GPU} --seed 44
python mainSTBP_DeSCA.py --conf conf/PEMS04/STBP_DeSCA.json --gpuid ${GPU} --seed 43


###############################################################################
# DCRNN + DeSCA
###############################################################################

echo ""
echo "==================== DCRNN + DeSCA ===================="

python main.py --conf conf/PEMS03/DCRNN_DeSCA.json   --gpuid ${GPU} --seed 43
python main.py --conf conf/AIR/DCRNN_DeSCA.json    --gpuid ${GPU} --seed 43
python main.py --conf conf/PEMS04/DCRNN_DeSCA.json --gpuid ${GPU} --seed 43


###############################################################################
# PDFormer + DeSCA
###############################################################################

echo ""
echo "==================== PDFormer + DeSCA ===================="

python main.py --conf conf/PEMS03/PDFormer_DeSCA.json   --gpuid ${GPU} --seed 43
python main.py --conf conf/AIR/PDFormer_DeSCA.json    --gpuid ${GPU} --seed 85
python main.py --conf conf/PEMS04/PDFormer_DeSCA.json --gpuid ${GPU} --seed 43


echo ""
echo "=================================================================="
echo "All DeSCA experiments completed."
echo "=================================================================="