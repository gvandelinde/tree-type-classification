# Tree Type Classification

## To retrieve Google Street View images of trees in Delft:

1. Download the shapefile dataset from: [https://drive.google.com/file/d/1rnScsLg924qSM6s8JTADw9yU_b943MRj/view?usp=sharing](https://drive.google.com/file/d/1rnScsLg924qSM6s8JTADw9yU_b943MRj/view?usp=sharing) and place its contents (bomen.dbf, bomen.prj, bomen.shp, bomen.shx) in folder `RetrieveGSV/RoadPointsGeneration/TreehealthDataset/`.
2. Fill your Google Street View API key in the `config.toml` file
3. Run `python RetrieveGSV/RoadPointsGeneration/getStreetPointsDelft.py` from the root of the project to generate a CSV of road locations and bearings to be retrieved.
4. Run `python RetrieveGSV/getGSVFieldDelft.py` from the root of the project to retrieve Google Street View images for each of the locations.
5. The output images are saved to `RetrieveGSV/images/` into seperate folders according to their health condition


## (OPTIONAL) How to filter out images without trees using the custom no-tree classifier:

1. (5 mins) Train your classifier by running "Model/is_tree_train.ipynb"
2. "is_tree_model.keras" file is saved
3. Filter out images by running `python Model/is_tree_classify.py` from root directory

## How to augment images

1. Run `python Model/blur_trees.py` from root directory and generate blurred tree images
2. Run  `python Model/data_augmentation.py` from current directory and generate augmented images and append to blurred_tree
