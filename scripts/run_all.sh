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

python main_pre.py --conf conf/PEMS/eac.json   --gpuid ${GPU} --seed 43
python main_pre.py --conf conf/AIR/eac.json    --gpuid ${GPU} --seed 46
python main_pre.py --conf conf/PEMS04/eac.json --gpuid ${GPU} --seed 43


###############################################################################
# STBP + DeSCA
###############################################################################

echo ""
echo "==================== STBP + DeSCA ===================="

python mainSTBP.py --conf conf/PEMS/STBP_PEMS.json --gpuid ${GPU} --seed 43
python mainSTBP.py --conf conf/AIR/STBP_AIR.json   --gpuid ${GPU} --seed 44
python mainSTBP.py --conf conf/PEMS04/STBP_04.json --gpuid ${GPU} --seed 43


###############################################################################
# DCRNN+ + DeSCA
###############################################################################

echo ""
echo "==================== DCRNN+ + DeSCA ===================="

python main_pre.py --conf conf/PEMS/DCRNNplus.json   --gpuid ${GPU} --seed 43
python main_pre.py --conf conf/AIR/DCRNNplus.json    --gpuid ${GPU} --seed 43
python main_pre.py --conf conf/PEMS04/DCRNNplus.json --gpuid ${GPU} --seed 43


###############################################################################
# PDFormer+ + DeSCA
###############################################################################

echo ""
echo "==================== PDFormer+ + DeSCA ===================="

python main_pre.py --conf conf/PEMS/PDFormerplus.json   --gpuid ${GPU} --seed 43
python main_pre.py --conf conf/AIR/PDFormerplus.json    --gpuid ${GPU} --seed 85
python main_pre.py --conf conf/PEMS04/PDFormerplus.json --gpuid ${GPU} --seed 43


echo ""
echo "=================================================================="
echo "All DeSCA experiments completed."
echo "=================================================================="