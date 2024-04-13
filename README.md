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

1. Run `python Model/blur_trees.py` from root directory and generate blurred tree images to blurred_trees
2. Run `python Model/data_augmentation.py` from root directory and append augmented images to blurred_trees

## How to train and classify images

Option 1: Full dataset has been generated and in Kaggle
   1. Open "Model/delfttreescnn.ipynb" in Kaggle
   2. Copy notebook and run all cells.

Option 2: When creating dataset from scratch
   1. Run `python Model/delfttreescnn.py` from root directory
   
**(IMPORTANT)**

Two models are compared in our implementation:
1. model_name_1 = 'ResNet50'
2. model_name_2 = 'CustomCNN'
   
Comment out either trained_model_1 or trained_model_2 depending on whether you want to use the ResNet50 architecture or our custom architecture.

This is in the last two lines of code as shown below:

trained_model_1 = ...,
trained_model_2 = ...
