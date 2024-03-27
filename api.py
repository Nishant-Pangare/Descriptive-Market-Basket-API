import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from flask import Flask, request, jsonify

# API Initialization
app = Flask(__name__)

# Data Loading and Preprocessing Function
def load_and_preprocess_data(file_path, invoice_column, country_column):
    data = pd.read_excel(file_path)
    data['Description'] = data['Description'].str.strip()
    data.dropna(axis=0, subset=[invoice_column], inplace=True)
    data[invoice_column] = data[invoice_column].astype('str')
    data = data[~data[invoice_column].str.contains('C')]  # Remove credit transactions
    return data

# Association Rule Mining Function
def generate_association_rules(data, country, invoice_column, item_column, min_support, min_threshold):
    basket = (data[data['Country'] == country]
              .groupby([invoice_column, item_column])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index(invoice_column))
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)
    frq_items = apriori(basket_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frq_items, metric="lift", min_threshold=min_threshold)

    # Convert frozensets to lists for JSON serialization
    rules['antecedents'] = rules['antecedents'].apply(list)
    rules['consequents'] = rules['consequents'].apply(list)

    # Simplify output for better readability
    rules = rules[['antecedents', 'consequents', 'lift', 'confidence']]
    rules.rename(columns={'antecedents': 'Products Bought', 'consequents': 'Product Recommended'}, inplace=True)
    return rules.to_dict('records')

# API Endpoint
@app.route('/generate_rules', methods=['POST'])
def generate_rules():
    data = request.json  # Retrieve data from API request
    try:
        file_path = data['filePath']
        country = data['country']
        invoice_column = data['invoiceColumn']
        item_column = data['itemColumn']
        min_support = data['minSupport']
        min_threshold = data['minThreshold']

        data = load_and_preprocess_data(file_path, invoice_column, 'Country')  # Assuming Country column is fixed
        rules = generate_association_rules(data, country, invoice_column, item_column, min_support, min_threshold)

        # Return success message and rules
        return jsonify({'message': 'Association rules generated successfully.', 'rules': rules})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the API
if __name__ == '__main__':
    app.run(debug=True,port=3000)  # Adjust for production deployment
