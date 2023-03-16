import pickle
from corr_study.datasetApi import *
from corr_study.correlation import *
from corr_study.voxels import *
from corr_study.mobileclassifier import *

dataset = Dataset("corr_study/dataset/")

train, val, test, len_train, len_val, len_test = create_tf_dataset(dataset, [["t3high", Weather.Clear, Time.Sunset, Sensor.LT],
                                                                             ["t3medium", Weather.Clear, Time.Sunset, Sensor.LT],
                                                                             ["t3low", Weather.Clear, Time.Sunset, Sensor.LT],], 4)
m = get_miniUNet((1024,1024,2))
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,alpha=.9)
m.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.FalseNegatives()], run_eagerly = True)

callbacks = [tf.keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True),
             tf.keras.callbacks.ModelCheckpoint("network.h5", monitor="val_loss", save_best_only=True)]

history = m.fit(train, steps_per_epoch=int(np.ceil(len_train/4)), epochs=30, validation_data=val, validation_steps=int(np.ceil(len_val/4)), callbacks=callbacks)

with open('hist.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)