
from imageai.Prediction import ImagePrediction
import os
execution_path = os.getcwd()

prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(os.path.join(execution_path, "densenet_trained_model.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "test.jpeg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
	print(eachPrediction , " : " , eachProbability)