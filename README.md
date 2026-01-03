# Prerequisites
The program uses [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) as the package manager.

It can be installed with the following command:

```
<bash> curl -LsSf https://astral.sh/uv/install.sh | sh
```
```
<PS> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Running the Program
```
uv run python parser/video_parser.py
```


## How Does It Work
1. The program records a video of the athlete in front of the camera.
2. The user will be prompted to label each frame according to the table below.

|Label|Meaning|
|------------|-----------------|
|`0`|Not in the set position. |
|`1`|In the set position. |
|`2`|Exit. |


## Modifications

The following constants may be adjusted for various purposes.

|Constant|Usage|Default|
|------------|-----------------|--------------------------|
|`VID_LENGTH`|The number of frames your camera is opened for. |200|
|`VID_DIR`|Path to the directory storing videos recorded by `video_parser.py` |C:\Data\SprinterData\TrainingVideos|
|`DATASET_DIR`|Path to the directory storing datasets pickled by `video_parser.py`|C:\Data\SprinterData\Datasets|
