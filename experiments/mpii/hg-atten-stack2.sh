#!/usr/bin/env sh
expID=mpii/hg_atten
dataset=mpii
gpuID=1
nGPU=1
batchSize=6
LR=2.5e-4
netType=hg-atten
nStack=2
nResidual=1
nThreads=8
minusMean=true
nClasses=16
nEpochs=250
snapshot=2
nFeats=256

CUDA_VISIBLE_DEVICES=$gpuID th main.lua \
	-dataset $dataset \
	-expID $expID \
	-batchSize $batchSize \
	-nGPU $nGPU \
	-LR $LR \
	-momentum 0.0 \
	-weightDecay 0.0 \
	-netType $netType \
	-nStack $nStack \
	-nResidual $nResidual \
	-nThreads $nThreads \
	-minusMean $minusMean \
	-nClasses $nClasses \
	-nEpochs $nEpochs \
	-snapshot $snapshot \
	-nFeats $nFeats \
	# -resume checkpoints/$expID
