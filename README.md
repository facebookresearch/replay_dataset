# 🔁 Replay Dataset

**[[project]](https://replay-dataset.github.io/)** | **[[paper TBD]](https://arxiv.org/abs/)**

Replay is a dataset of 68 scenes of social interactions between people, such as playing boarding games, exercising, or unwrapping presents.
Each scene is about 5 minutes long and filmed with 12 cameras, static and dynamic.
Audio is captured separately by 12 binaural microphones and additional near-range microphones for each actor and for each egocentric video.
All sensors are temporally synchronized, undistorted, geometrically calibrated, and color calibrated.
This repository contains download scripts and classes to access the data.
Examples of usage and format description are coming soon.

## Download

The compressed dataset takes **244 GB of space**. We distribute it in chunks up to 30 GB.
The links to all dataset files are present in this repository in [links/links.json](links/links.json).
We provide an automated way of downloading and decompressing the data.

First, run the install script that will take care of dependencies:

```
pip install -e .
```

Then run the script (make sure to change `<DESTINATION_FOLDER>`):

```
python ./cop3d/download_dataset.py --download_folder <DESTINATION_FOLDER> --checksum_check
```

The script has multiple parameters, e.g. `--download_categories audio,videos,masks` will download all modalities (the default behaviour).
You can select only a subset of those, e.g. you can skip `audio` files.
Metadata will be always downloaded.
Another flag, `--clear_archives_after_unpacking`, will remove the redundant archives.
Run `python ./cop3d/download_dataset.py -h` for the full list of options.


## API Quick Start and Tutorials

Make sure the setup is done and the dataset is downloaded as per above.

We provide `ReplayDataset` class to access the data and are working on the tutorials on using it with Implicitron.
For now, please check the [unit test](tests/test_replay_dataset.py).

## Dataset Format

TBD

## Reference

TBD

## License

The data are released under the [CC BY-NC 4.0 license](LICENSE).

