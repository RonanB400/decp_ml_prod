def amount_prediction(X, amount_pipeline, amount_model):
    """
    Predict amount using preprocessed data and trained models
    
    Args:
        X: Input data to predict
        amount_pipeline: Preprocessing pipeline for amount prediction
        amount_model: Trained Keras model for amount prediction
    
    Returns:
        y: Predicted probabilities for price ranges
    """
    try:
        # Preprocessing of the market data
        X_preproc = amount_pipeline.transform(X)
        
        # Prediction of price range probabilities
        y = amount_model.predict(X_preproc)

        return y
        
    except Exception as e:
        import traceback
        print(f"Error in amount_prediction: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise


