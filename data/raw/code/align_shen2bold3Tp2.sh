#!/bin/bash

set -e
set -u

for sub in 01 02 03 04 05 06 09 10 14 15 16 17 18 19 20; do
  mkdir -p sub-${sub}/atlases/bold3Tp2
  $FSLDIR/bin/applywarp \
    -i atlases/shen/fconn_atlas_150_1mm.nii.gz \
    -r src/aligned/src/tnt/sub-${sub}/bold3Tp2/brain.nii.gz \
    -o sub-${sub}/atlases/bold3Tp2/shen_fconn_atlas_150.nii.gz \
    --premat=src/aligned/src/tnt/templates/grpbold3Tp2/xfm/mni2tmpl_12dof.mat \
    -w src/aligned/src/tnt/sub-${sub}/bold3Tp2/in_grpbold3Tp2/tmpl2subj_warp.nii.gz \
    --interp=nn \
    -m src/aligned/src/tnt/sub-${sub}/bold3Tp2/brain_mask.nii.gz
done
