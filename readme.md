# ConstrICON with ThoraxCBCT

A dockerization of "Inverse Consistency By Construction for Multistep Deep Registration" (MICCAI 2023) for the OncoReg challenge. In the original paper we train our model using the same configuration on four datasets. This docker container trains that model on the provided dataset.json using that configuration.

training loss should start near 2, rapidly drop below 1, and then slowly decrease.

With the settings provided in this repo, training should take about 8 hours, with an early checkpoint produced at the 45 minute mark. If 8 hours is too long, please adjust 
```
num_iterations = 7*4900
```
in `ByConstructionICON/train_constricon_supervised.py` at line 43. If more than 8 hours is available, the value 7 can be increased to use this time. From our experiments we expect mtre performance to continue improving up to 2 days of training, although train loss will plateau earlier. (`num_iterations = 24 * 2 * 4900`).

Our inference script performs 50 steps of instance optimization. This should take a few minutes per image pair.

## How To

Download the ThoraxCBCT dataset including training and validation data, keypoints, masks and the **ThoraxCBCT_dataset.json**:  
https://cloud.imi.uni-luebeck.de/s/xQPEy4sDDnHsmNg

Build the docker:

```
sudo docker build -t constricon ByConstructionICON/
```

Run docker and start training (insert path to ThoraxCBCT data):

```
sudo docker run --gpus all --entrypoint ./train.sh \
-v /playpen/tgreer/docker_shit/Release_06_12_23/:/oncoreg/data \
-v ./model/:/oncoreg/model/ \
constricon ThoraxCBCT 
```

Run inference (insert path to ThoraxCBCT data):

```
sudo docker run --gpus all --entrypoint ./test.sh \
-v /playpen/tgreer/docker_shit/Release_06_12_23/:/oncoreg/data \
-v ./model/:/oncoreg/model/ \
-v ./results/:/oncoreg/results/ \
constricon ThoraxCBCT Val
```

## Usage without Docker

If you want to use this repository without docker containerisation you can do so by creating an environment directly via the requirements.txt and then run the training and inference scripts after adjusting the paths at the beginning of the scripts:
```
python train_constricon_supervised.py <task>
python inference_constricon.py <task> <Val/Ts>
```




### References

Inverse Consistency by Construction for Multistep Deep Registration
Hastings Greer, Lin Tian, Francois-Xavier Vialard, Roland Kwitt, Sylvain Bouix, Raul San Jose Estepar, Richard Rushmore, Marc Niethammer 
MICCAI 2023
