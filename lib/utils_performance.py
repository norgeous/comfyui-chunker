import time
import math

def get_ts():
    return int(time.time() * 1000) # current time in milliseconds

def predict(data, next_count):
    avg = sum(data) / len(data)
    return [avg] * next_count

# Example Usage:
# result = predict_next([10, 20, 30], 2)
# print(result) # [40, 50]
def predict2(data, next_count):
    n = len(data)
    # if n < 2: return []
    if n == 1: return data * next_count
    
    # Generate x values [1, 2, ..., n]
    x = [i + 1 for i in range(n)]
    
    def get_linear_fit(X, Y):
        n_len = len(X)
        sum_x = sum(X)
        sum_y = sum(Y)
        sum_xy = sum(val_x * val_y for val_x, val_y in zip(X, Y))
        sum_xx = sum(val_x**2 for val_x in X)
        
        # Calculate slope (m) and intercept (b) for y = mx + b
        denominator = (n_len * sum_xx - sum_x**2)
        # Avoid division by zero if all x values are the same
        slope = (n_len * sum_xy - sum_x * sum_y) / denominator if denominator != 0 else 0
        intercept = (sum_y - slope * sum_x) / n_len
        
        def predict(v):
            return slope * v + intercept
            
        # Calculate Sum of Squared Errors
        error = sum((val_y - predict(val_x))**2 for val_x, val_y in zip(X, Y))
        
        return {"predict": predict, "error": error}

    # Linear Model
    linear = get_linear_fit(x, data)
    
    # Exponential Model (only if all data points are positive)
    exponential = {"error": float('inf')}
    if all(v > 0 for v in data):
        ln_y = [math.log(v) for v in data]
        log_model = get_linear_fit(x, ln_y)
        
        def exp_predict(v):
            return math.exp(log_model["predict"](v))
            
        exp_error = sum((val_y - exp_predict(val_x))**2 for val_x, val_y in zip(x, data))
        exponential = {"predict": exp_predict, "error": exp_error}

    # Select the model with the lowest error
    best_model = exponential if exponential["error"] < linear["error"] else linear
    
    # Generate predictions for the next_count intervals
    predictions = [round(best_model["predict"](n + i + 1)) for i in range(next_count)]
    
    return predictions
