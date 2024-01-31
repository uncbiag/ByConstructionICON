# VoxelMorph++ with ThoraxCBCT

Example submission for the OncoReg Type 3 Challenge using VoxelMorph++ (Voxelmorph with keypoint supervision and multi-channel instance optimisation) with data from the ThoraxCBCT Type 2 Challenge Task.  
Participants may use this as a template for the containerisation of their submissions. 
To submit to the challenge every solution must be packaged as a Docker container including training and inference in case of a DL solution or any required scripts to output displacementfields with a classical solution.

Make sure to include a requirements.txt file and preferably base your solution on torch 2.1.2+cu121 as this is tested on our setup.
For easy containerisation you may use the Dockerfile provided as this is guarenteed to work with our infrastructure. Make sure to include all files requiring copying in the Dockerfile.

Please include logging in you submission, as can be seen in this example. 

It is advised to copy our train.sh / test.sh structure and you may also look at our data loading process as it is easy adaptable from ThoraxCBCT to OncoReg.

## How To

Download our ThoraxCBCT dataset including training and validation data, keypoints, masks and the **ThoraxCBCT_dataset.json**:  
https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg

Build the docker:

```
docker build -t vxmpp /PATH_TO/OncoReg/examples/VoxelMorph++/
```

Run docker and start training (insert path to ThoraxCBCT data):

```
docker run --gpus all --entrypoint ./train.sh
-v /PATH_TO_DATA_DIR/:/oncoreg/data
-v ./model/:/oncoreg/model/
vxmpp ThoraxCBCT
```

Run inference (insert path to ThoraxCBCT data):

```
docker run --gpus all --entrypoint ./test.sh
-v /PATH_TO_DATA_DIR/:/oncoreg/data
-v ./model/:/oncoreg/model/
-v ./results/:/oncoreg/results/
vxmpp ThoraxCBCT Val
```

## Usage without Docker

If you want to use our example without docker containerisation you can do so by creating an environment directly via the requirements.txt and then run the training and inference scripts after adjusting the paths at the beginning of the scripts:
```
python train_vxmpp_supervised.py <task>
python inference_vxmpp.py <task> <Val/Ts>
```




### References
Heinrich, Mattias P., and Lasse Hansen. "Voxelmorph++ going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation." International Workshop on Biomedical Image Registration. Cham: Springer International Publishing, 2022. https://link.springer.com/chapter/10.1007/978-3-031-11203-4_10  

For more information on VoxelMorph++ see https://github.com/mattiaspaul/VoxelMorphPlusPlus.
