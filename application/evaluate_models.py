#%%
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import utils
import pickle
from tensorflow import keras
import model
#%%
df = utils.load_and_transform_dataset(encode_target_variable=False)                            
# %%
df[utils.text_var].shape
#%%
# load model
try:
    loaded_model = keras.models.load_model(os.path.abspath(os.path.join(__file__, r"..", utils._saved_model_root, utils._model_directory)))
except:
    loaded_model = model.create_and_save_model()

# load enc
with open(os.path.abspath(os.path.join(__file__, r"..", utils.label_encoder_file)), 'rb') as loaded_enc:
    enc = pickle.load(loaded_enc)
#%%
x = df[utils.text_var]
y = df[utils.label_var]

# same random state
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=utils.test_size,
                                                    random_state=utils.random_state)   
#%%
df_evaluate = df.iloc[x_test.index].copy()
df_evaluate = df_evaluate.reset_index(drop=True)
#%%
list_of_predictions = loaded_model.predict(df_evaluate[utils.text_var].values)
# for the evaluation of the metrics, we must pick a unique value
# in this case, I chose the most probable answer according to the model predict
df_evaluate["y_predict"] = np.argmax(list_of_predictions, axis=1)

# # mapping the categories back to create the confusion matrix
df_evaluate["y_predict"] = enc.inverse_transform(df_evaluate["y_predict"])
#%%
output_list = utils.transform_predictions_to_strings(list_of_predictions, threshold=utils.threshold)
# %%
# normal confusion matrix
cm = confusion_matrix(df_evaluate[utils.label_var], df_evaluate['y_predict'], labels=enc.classes_, normalize=None)
# row-based cm
cm_true = confusion_matrix(df_evaluate[utils.label_var], df_evaluate['y_predict'], labels=enc.classes_, normalize='true')
# column-based cm
cm_columns = confusion_matrix(df_evaluate[utils.label_var], df_evaluate['y_predict'], labels=enc.classes_, normalize='pred')
# display cm's
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8))

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=enc.classes_)

disp_true = ConfusionMatrixDisplay(confusion_matrix=cm_true,
                              display_labels=enc.classes_)

disp_columns = ConfusionMatrixDisplay(confusion_matrix=cm_columns,
                              display_labels=enc.classes_)

disp.plot(ax=ax1)
disp_true.plot(ax=ax2)
disp_columns.plot(ax=ax3)

ax1.set_title('Total')
ax2.set_title('% Rows')
ax3.set_title('% Columns')

plt.tight_layout
plt.show()
# %%
# calculate metrics
precision, recall, fscore, support = precision_recall_fscore_support(df_evaluate[utils.label_var], df_evaluate['y_predict'], average='macro')
metric_accuracy_score = accuracy_score(df_evaluate[utils.label_var], df_evaluate['y_predict'])
metric_balanced_accuracy_score = balanced_accuracy_score(df_evaluate[utils.label_var], df_evaluate['y_predict'])
print(f"Precision: {precision}, Recall: {recall}, FScore: {fscore}, Acc_score: {metric_accuracy_score}, Balanced Acc_score: {metric_balanced_accuracy_score}")
# %%
