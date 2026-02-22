# Plant-disease-detecting-ML-model
RULE 1:
  DO NOT TOUCH ML MODEL (AT ANY COST)

///////////////////////////////////
   STEPS TO FOLLOW TO RUN SERVER
//////////////////////////////////
1) Download main.py to desired folder(make sure that directory has imports, predict.py, test_map.py, model/h5).
2) Before running the server, Make sure you change the ip(your lap/Pc ip) of heatmap in return.
3) In cmd change the directory to the target file and run this command:
//////////////////////////////////////////////////
   uvicorn main:app --host <your ip> --port 8000
//////////////////////////////////////////////////
4) Your server is ready!
