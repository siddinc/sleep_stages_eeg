from tensorflow.keras.optimizers import Adam
from model_fn import build_model
from utility_fn import build_dataset, preprocess_dataset
from tensorflow.keras.models import Model, load_model, save_model

if __name__ == "__main__":
  # EPOCHS = 10
  # BATCH_SIZE = 16
  
  # dataset = build_dataset('../datasets/data.csv')
  # x_train, x_test, y_train, y_test = preprocess_dataset(dataset)

  # STEPS_PER_EPOCH = x_train.shape[0] // BATCH_SIZE
  # VALIDATION_STEPS = x_test.shape[0] // BATCH_SIZE

  # opt = Adam(lr=0.0001, amsgrad=True)
  # model = build_model(height, width, classes=5)
  # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

  # r = model.fit(
  #   x_train, y_train,
  #   validation_data=(x_test, y_test),
  #   epochs=EPOCHS, batch_size=BATCH_SIZE,
  #   steps_per_epoch=STEPS_PER_EPOCH,
  #   validation_steps=VALIDATION_STEPS,
  # )
  model = load_model('../models/model_e15-bs16-a92.02-va92.40.h5')
  model.summary()