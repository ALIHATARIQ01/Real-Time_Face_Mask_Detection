import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

with open("preprocessed_data.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

base = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(2, activation='softmax')(x)

model = Model(inputs=base.input, outputs=preds)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=32)
model.save("mask_detector.keras")


print("✅ Model trained and saved.")

