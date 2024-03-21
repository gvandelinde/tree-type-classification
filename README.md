# Tree Type Classification

## To retrive Google Street View images of trees in Delft:

1. Download the shapefile dataset from: [https://drive.google.com/file/d/1rnScsLg924qSM6s8JTADw9yU_b943MRj/view?usp=sharing](https://drive.google.com/file/d/1rnScsLg924qSM6s8JTADw9yU_b943MRj/view?usp=sharing) and place its contents (bomen.dbf, bomen.prj, bomen.shp, bomen.shx) in folder `RetrieveGSV/RoadPointsGeneration/TreehealthDataset/`.
2. Fill your Google Street View API key in the `config.toml` file
3. Run `python RetrieveGSV/RoadPointsGeneration/getStreetPointsDelft.py` to generate a CSV of road locations and bearings to be retrieved.
4. Run `python RetrieveGSV/getGSVFieldDelft.py` to retrieve Google Street View images for each of the locations.
5. The output images are saved to `RetrieveGSV/images/` into seperate folders according to their health condition
