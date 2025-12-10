# Feature-Importance
14 features were systematically blurred to assess its importance and rank it in order
The generated large model has 1022 channels. To find out how the model performs when every 
individual feature was masked, separately tests were performed. For e.g. if the intention is to see how pre441 feature performs, 
if the pre441 feature is fed as input while predicting the accuracy, in the place of pre441 features the pre441 mean matrix is substituted 441 times. 
Similarly, to assess the importance of each feature, every feature was blurred individually at a time using their respective noise matrices and 
prediction accuracy was calculated. 
