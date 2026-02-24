def train(model, x_train, y_train, epochs, batch_size, callbacks=None):
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
