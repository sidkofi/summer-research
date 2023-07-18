import os
import re
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from load_matrix_from_csv import load_matrix_from_csv

# SET
data_path = "/home/sidkofi/Documents/summer research/data"
matrix_type = "400x2000"

# automatically get all user data
featureFolder = os.path.join(data_path)
subfolders = os.listdir(featureFolder)
subfolders = [
    subfolder
    for subfolder in subfolders
    if os.path.isdir(os.path.join(featureFolder, subfolder))
]
subfolders.sort()
person_ids = list(map(int, subfolders))

session_ids = list(range(1, 9))
sessions = [
    "tfMRI_EMOTION_LR",
    "tfMRI_GAMBLING_LR",
    "tfMRI_LANGUAGE_LR",
    "tfMRI_MOTOR_LR",
    "tfMRI_RELATIONAL_LR",
    "tfMRI_SOCIAL_LR",
    "tfMRI_WM_LR",
    "rfMRI_REST2_LR",
]
numPeople = len(person_ids)

# initialize data and labels
matrix_temp = matrix_type.replace("x", "")
feature_vals = re.findall(r"\d+", matrix_type)
vector_size = int(feature_vals[0]) * int(feature_vals[1])
allData = np.empty((0, vector_size))
allLabels = []
idx = 0

# load in each matrix
for person_id in person_ids:
    for session_id in session_ids:
        sess = sessions[session_id - 1]
        file_path = os.path.join(
            data_path,
            str(person_id),
            matrix_type,
            "matrices",
            sess
            + "."
            + str(person_id)
            + ".normLap.fb.coeffs.norm1.rt."
            + matrix_type
            + ".csv",
        )
        data = load_matrix_from_csv(file_path)

        featureMatrix = np.array(data)

        # vectorize feature matrix
        featureVector = featureMatrix.flatten()

        allData = np.vstack((allData, featureVector))
        allLabels.append(sess)
        idx += 1

allData = normalize(allData)
allLabels = np.array(allLabels)
# Begin model training
# determine number of people and sample size
numSamples = allData.shape[0]
correctCounts = np.zeros(len(sessions))

np.random.seed(123)  # random seed for reproducibility

t = SVC(kernel="linear")

# get labels from sessions
taskLabels = [session.split("_")[1] for session in sessions]
task_correct_counts = np.zeros(len(sessions))

for i in range(numPeople):
    # select person for testing
    testIndices = np.arange(
        (i * numSamples) // numPeople, ((i + 1) * numSamples) // numPeople
    )
    testData = allData[testIndices, :]
    testLabels = allLabels[testIndices]

    # Use the remaining data for training
    trainIndices = np.setdiff1d(np.arange(numSamples), testIndices)
    trainData = allData[trainIndices, :]
    trainLabels = allLabels[trainIndices]

    # train and test model
    svmModel = t.fit(trainData, trainLabels)
    predictions = svmModel.predict(testData)

    # get overall accuracy
    accuracy = np.sum(predictions == testLabels) / len(testLabels)
    print(f"Overall Accuracy for Person {person_ids[i]}: {accuracy * 100:.2f}%")

    # get accuracy for each task
    for j, session in enumerate(sessions):
        taskIndices = testLabels == session
        taskPredictions = predictions[taskIndices]
        task_correct = np.sum(taskPredictions == testLabels[taskIndices])
        task_accuracy = task_correct / len(taskPredictions)
        task_correct_counts[j] += task_correct
        task_result = "Yes" if task_correct > 0 else "No"
        print(f"Task: {session} - Correctly predicted: {task_result}")

# plot correctly predicted tasks
plt.figure()
plt.bar(range(len(task_correct_counts)), task_correct_counts)
plt.xlabel("Task")
plt.ylabel("Number of Correct Predictions")
plt.title("Number of Correct Predictions for Each Task")
plt.xticks(range(len(task_correct_counts)), taskLabels, rotation=45)
plt.ylim([0, numPeople])
plt.show()

total_accuracy = np.sum(task_correct_counts) / numSamples
print(f"Total accuracy for all: {total_accuracy * 100:.2f}%")
