# Plant-disease-detecting-ML-model
RULE 1:
  DO NOT TOUCH ML MODEL (AT ANY COST)
DEV NOTE: This model is under development! more classes will be added and im planning to make this a multi-lable classifcation model

///////////////////////////////////
   STEPS TO FOLLOW TO RUN SERVER
//////////////////////////////////
1) Download main.py to desired folder(make sure that directory has imports, predict.py, test_heatmap.py, model.h5).
2) Before running the server, Make sure you change the ip(your lap/Pc ip) of heatmap in return of main.py.
3) In cmd change the directory to the target folder and run this command: {uvicorn main:app --host {your ip} --port 8000}
4) Your server is ready!
   
///////////////////////////////////
   STEPS TO TRAIN THE MODEL
///////////////////////////////////
1) Get the datasets for the given classes (healthy, early blight and leaf curl virus), if needed the dataset will be uploaded soon
2) Save the files main.py, predict.py ,test_heatmap.py and the model.H5 file in a designated folder together
3) Create a dataset/train folder in the designated folder and upload the datasets of the seperate classes in the train file in seperate folders
4) Set the epoch and the validation size as per requirements and run the train.py

///////////////////////////////////
   STEPS FOR TESTING
///////////////////////////////////
1) Once the training is the complete check if the base accuracy and validation accuracy are in a good state (both >90/80%) [if else lower the epoch and clean the dataset]
2) Run the predict.py to obtain the prediction and confidence level
3) Run test_heatmap.py to obtain the prediction + heatmap
4) Your model is ready!

   
   
