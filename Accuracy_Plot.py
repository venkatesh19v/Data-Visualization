import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv('MODEL_ACCURACY.csv')

# List of valid emotion classes
valid_emotions = ['angry', 'sad', 'disgust', 'happy', 'fear', 'surprise', 'neutral']

# Function to calculate accuracy for a specific emotion
def calculate_emotion_accuracy(y_true, y_pred, emotion):
    correct = 0
    total = 0
    for true, pred in zip(y_true, y_pred):
        true = str(true).lower()
        pred = str(pred).lower()
        if true == emotion:
            total += 1
            if pred == emotion:
                correct += 1
    if total == 0:
        return 0
    return correct / total

# Calculate accuracy for each model and emotion
accuracy_data = {}
for model in ['DeepFace Library', 'DeepFace after Training', 'Gemini']:
    accuracy_data[model] = {}
    y_true = df['Actual Emotion '].str.lower()
    y_pred = df[model].str.lower()
    for emotion in valid_emotions:
        accuracy = calculate_emotion_accuracy(y_true, y_pred, emotion)
        accuracy_data[model][emotion] = accuracy

# Plot the accuracy for each emotion and model
emotions = valid_emotions
models = ['DeepFace Library', 'DeepFace after Training', 'Gemini']
accuracy_values = [accuracy_data[model] for model in models]
for model, accuracy in accuracy_scores.items():
    print(f'{model} accuracy: {accuracy:.2f}')
x = range(len(emotions))
plt.figure(figsize=(10, 6))
for i, model in enumerate(models):
    plt.bar([j + i * 0.25 for j in x], [accuracy_values[i][emotion] for emotion in emotions], width=0.25, label=model)

plt.xticks([j + 0.375 for j in x], emotions)
plt.xlabel('Emotion')
plt.ylabel('Accuracy')
plt.title('Emotion Classification Accuracy')
plt.legend()
plt.show()
