def evaluate_model(model, test_ds):
    results = model.evaluate(test_ds)
    print("Evaluation:", results)