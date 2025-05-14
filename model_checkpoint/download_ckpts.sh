#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for SAM 2 checkpoints
# SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
# sam2_hiera_t_url="${SAM2_BASE_URL}/sam2_hiera_tiny.pt"
# sam2_hiera_s_url="${SAM2_BASE_URL}/sam2_hiera_small.pt"
# sam2_hiera_b_plus_url="${SAM2_BASE_URL}/sam2_hiera_base_plus.pt"
# sam2_hiera_l_url="${SAM2_BASE_URL}/sam2_hiera_large.pt"

# Download each of the four checkpoints using wget
# echo "Downloading sam2_hiera_tiny.pt checkpoint..."
# $CMD $sam2_hiera_t_url || { echo "Failed to download checkpoint from $sam2_hiera_t_url"; exit 1; }

# echo "Downloading sam2_hiera_small.pt checkpoint..."
# $CMD $sam2_hiera_s_url || { echo "Failed to download checkpoint from $sam2_hiera_s_url"; exit 1; }

# echo "Downloading sam2_hiera_base_plus.pt checkpoint..."
# $CMD $sam2_hiera_b_plus_url || { echo "Failed to download checkpoint from $sam2_hiera_b_plus_url"; exit 1; }

# echo "Downloading sam2_hiera_large.pt checkpoint..."
# $CMD $sam2_hiera_l_url || { echo "Failed to download checkpoint from $sam2_hiera_l_url"; exit 1; }

# Define the URLs for SAM checkpoints
SAM_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything"
sam_vit_b_url="${SAM_BASE_URL}/sam_vit_b_01ec64.pth"
sam_vit_h_url="${SAM_BASE_URL}/sam_vit_h_4b8939.pth"
sam_vit_l_url="${SAM_BASE_URL}/sam_vit_l_0b3195.pth"

# SAM checkpoints
echo "Downloading sam_vit_b checkpoint..."
$CMD $sam_vit_b_url || { echo "Failed to download checkpoint from $sam_vit_b_url"; exit 1; }

echo "Downloading sam_vit_h checkpoint..."
$CMD $sam_vit_h_url || { echo "Failed to download checkpoint from $sam_vit_h_url"; exit 1; }

echo "Downloading sam_vit_l checkpoint..."
$CMD $sam_vit_l_url || { echo "Failed to download checkpoint from $sam_vit_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."