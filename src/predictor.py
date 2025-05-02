def predict_incident(model, origin, destination, departure_time):
    input_data = {
        "ORIGIN": [origin],
        "DEST": [destination],
        "DEP_TIME": [departure_time]
    }
    import pandas as pd
    df = pd.DataFrame(input_data)
    return model.predict(df)[0], model.predict_proba(df)[0][1]