#%%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import utils
import pickle
from tensorflow import keras
#%%
text_var = utils.text_var
label_var = utils.label_var

df = utils.load_and_transform_dataset(encode_target_variable=False)                            
# %%
df[text_var].shape
#%%
# load model
model = keras.models.load_model(os.path.join(utils._saved_model_root, 'basic_serving_model'))

# load enc
with open('basic_implementation_label_encoder.pkl', 'rb') as loaded_enc:
    enc = pickle.load(loaded_enc) 
#%%
x = df[text_var]
y = df[label_var]

# same random state
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=0)   
#%%
df_evaluate = df.iloc[x_test.index].copy()
df_evaluate = df_evaluate.reset_index(drop=True)
#%%
list_of_predictions = model.predict(df_evaluate[text_var].values)
# for the evaluation of the metrics, we must pick a unique value
# in this case, I chose the most probable answer according to the model predict
df_evaluate["y_predict"] = np.argmax(list_of_predictions, axis=1)

# # mapping the categories back to create the confusion matrix
df_evaluate["y_predict"] = enc.inverse_transform(df_evaluate["y_predict"])
#%%
output_list = utils.transform_predictions_to_strings(list_of_predictions, threshold=utils.threshold)
# %%
# normal confusion matrix
cm = confusion_matrix(df_evaluate[label_var], df_evaluate['y_predict'], labels=enc.classes_, normalize=None)
# row-based cm
cm_true = confusion_matrix(df_evaluate[label_var], df_evaluate['y_predict'], labels=enc.classes_, normalize='true')
# column-based cm
cm_columns = confusion_matrix(df_evaluate[label_var], df_evaluate['y_predict'], labels=enc.classes_, normalize='pred')
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
precision, recall, fscore, support = precision_recall_fscore_support(df_evaluate[label_var], df_evaluate['y_predict'], average='macro')
metric_accuracy_score = accuracy_score(df_evaluate[label_var], df_evaluate['y_predict'])
metric_balanced_accuracy_score = balanced_accuracy_score(df_evaluate[label_var], df_evaluate['y_predict'])
print(f"Precision: {precision}, Recall: {recall}, FScore: {fscore}, Acc_score: {metric_accuracy_score}, Balanced Acc_score: {metric_balanced_accuracy_score}")
# %%
