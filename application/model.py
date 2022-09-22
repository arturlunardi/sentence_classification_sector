import os
from sklearn.model_selection import train_test_split
import utils

def create_and_save_model():
    dataset = utils.load_and_transform_dataset(encode_target_variable=True)

    x = dataset[utils.text_var]
    y = dataset[utils.label_var]

    # split for train and test
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=utils.test_size,
                                                        random_state=utils.random_state)
    model = utils.model_builder()

    model.fit(x_train,
            y_train,
            validation_data = (x_test, y_test),
            steps_per_epoch = x_train.shape[0], 
            validation_steps= x_test.shape[0],
    )

    model.save(os.path.abspath(os.path.join(__file__, r"..", utils._saved_model_root, utils._model_directory)), overwrite=True)

    return model


if __name__ == '__main__':
    create_and_save_model()
