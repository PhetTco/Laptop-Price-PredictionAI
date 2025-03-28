from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model and data
model = joblib.load('best_laptop_price_model.pkl')
laptop_data = pd.read_csv('laptop.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input data from the form
        brand = request.form['brand']
        processor_name = request.form['processor_name']
        processor_brand = request.form['processor_brand']
        ram_expandable = request.form['ram_expandable']
        ram = float(request.form['ram'])
        ram_type = request.form['ram_type']
        ghz = float(request.form['ghz'])
        display_type = request.form['display_type']
        display = float(request.form['display'])
        gpu = request.form['gpu']
        gpu_brand = request.form['gpu_brand']
        ssd = float(request.form['ssd'])
        hdd = int(request.form['hdd'])
        battery_life = float(request.form['battery_life'])

        # Prepare the data for prediction
        input_data = pd.DataFrame([[brand, processor_name, processor_brand, ram_expandable, ram, ram_type, 
                                    ghz, display_type, display, gpu, gpu_brand, ssd, hdd, battery_life]], 
                                  columns=['Brand', 'Processor_Name', 'Processor_Brand', 'RAM_Expandable', 'RAM', 'RAM_TYPE', 
                                           'Ghz', 'Display_type', 'Display', 'GPU', 'GPU_Brand', 'SSD', 'HDD', 'Battery_Life'])

        # Make the prediction
        predicted_price = model.predict(input_data)[0]

        # Function to recommend laptops based on similarity in specs and price
        def recommend_laptops(predicted_price, tolerance=5000, top_n=10):
            lower_bound = predicted_price - tolerance
            upper_bound = predicted_price + tolerance

            # Filter laptops within the price range
            recommended_laptops = laptop_data[(laptop_data['Price'] >= lower_bound) & (laptop_data['Price'] <= upper_bound)]

            # Calculate the similarity between the input specs and each laptop in the dataset
            def calculate_similarity(laptop):
                score = 0
                if laptop['Brand'] == brand:
                    score += 1
                if laptop['Processor_Name'] == processor_name:
                    score += 1
                if laptop['Processor_Brand'] == processor_brand:
                    score += 1
                if laptop['RAM_Expandable'] == ram_expandable:
                    score += 1
                if laptop['RAM'] == ram:
                    score += 1
                if laptop['RAM_TYPE'] == ram_type:
                    score += 1
                if laptop['Ghz'] == ghz:
                    score += 1
                if laptop['Display_type'] == display_type:
                    score += 1
                if laptop['Display'] == display:
                    score += 1
                if laptop['GPU'] == gpu:
                    score += 1
                if laptop['GPU_Brand'] == gpu_brand:
                    score += 1
                if laptop['SSD'] == ssd:
                    score += 1
                if laptop['HDD'] == hdd:
                    score += 1
                if laptop['Battery_Life'] == battery_life:
                    score += 1
                return score

            # Apply the similarity function to each laptop in the recommended list
            if len(recommended_laptops) > 0:
                recommended_laptops['similarity_score'] = recommended_laptops.apply(calculate_similarity, axis=1)
                recommended_laptops = recommended_laptops.sort_values(by=['similarity_score', 'Price'], ascending=[False, True]).head(top_n)
                return recommended_laptops[['Brand', 'Processor_Name', 'RAM', 'SSD', 'Price', 'Name']]
            else:
                return pd.DataFrame()  # Return empty dataframe if no laptops found in range

        # Get recommended laptops
        recommended = recommend_laptops(predicted_price)

        # Pass predicted price and recommended laptops to the result page
        return render_template('result.html', price=predicted_price, recommendations=recommended.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)