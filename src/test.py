import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle
import pandas as pd
from io import BytesIO
from tqdm import tqdm
from src.model import PlantModel

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = PlantModel(num_ancillary_features=163, num_output_features=6)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_bytes, transform):
    image = Image.open(BytesIO(image_bytes))
    image = transform(image)
    return image.unsqueeze(0)

def predict(model, test_data, yScaler, xScaler, device):
    input_cols = [col for col in test_data.columns if not col.startswith('X') and col not in['id', 'file_path', 'jpeg_bytes']]
    log_transformed_features = [1,2,3,4,5]
    predictions = []
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    test_data[input_cols] = xScaler.transform(test_data[input_cols])
    
    with torch.no_grad():
        for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Predicting"):
            image_bytes = row['jpeg_bytes']

            ancillary_data = row[input_cols].values.astype(np.float32)
            
            image = preprocess_image(image_bytes, transform)
            image = image.to(device)
            ancillary_data = torch.tensor(ancillary_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            output = model(image, ancillary_data)
            output = output.cpu().numpy()
            output = yScaler.inverse_transform(output)
            
            for feature in log_transformed_features:
                output[:, feature] = np.power(10, output[:, feature])
            predictions.append(output[0])
    
    return predictions

def test(test_data_path, model_path="checkpoints/best_model.pth",xScaler_path="data/X_Scaler.pkl", yScaler_path="data/yScaler.pkl",output_path="data/y_test.csv",device="mps"):
    columns = ['X4', 'X11','X18','X50','X26','X3112']
    #log10Columns=['X11','X18','X50','X26','X3112']
    # Set the device
    device = torch.device(device)

    # Load the trained model
    model = load_model(model_path, device)

    # Load the yScaler
    with open(yScaler_path, "rb") as f:
        yScaler = pickle.load(f)
        
    with open(xScaler_path, "rb") as f:
        XScaler = pickle.load(f)


    # Load the test data
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    # Perform predictions
    predicted_outputs = predict(model, test_data, yScaler,XScaler, device)

    # Create a DataFrame with the predicted outputs
    output_df = pd.DataFrame(predicted_outputs, columns=columns)
    output_df.insert(0, "id", test_data["id"])

    # Save the output DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)
    print("Predictions saved to {output_path}'")