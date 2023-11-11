# Import the module
import json

#json to dictinary
'''
# String with JSON format
data_JSON =  """
{
	"size": "Medium",
	"price": 15.67,
	"toppings": ["Mushrooms", "Extra Cheese", "Pepperoni", "Basil"],
	"client": {
		"name": "Jane Doe",
		"phone": "455-344-234",
		"email": "janedoe@email.com"
	}
}
"""

# Convert JSON string to dictionary
data_dict = json.loads(data_JSON)
print(data_dict)

print(data_dict["size"])
print(data_dict["price"])
print(data_dict["toppings"])
print(data_dict["client"])

# dictionary to json

# Python Dictionary
client = {
    "name": "Nora",
    "age": 56,
    "id": "45355",
    "eye_color": "green",
    "wears_glasses": False
}

# Get a JSON formatted string
client_JSON = json.dumps(client, indent= 4)
print(client_JSON)
print(type(client_JSON))
'''
'''
with open('orders.json') as f:
    data = json.load(f)
    #print(data)
    print(data["orders"][0]["toppings"])  # for topping first order
    print(len(data["orders"]))
'''

# Open the orders.json file
with open("orders.json") as file:
    # Load its content and make a new dictionary
    data = json.load(file)

    # Delete the "client" key-value pair from each order
    for order in data["orders"]:
        del order["client"]

# Open (or create) an orders_new.json file 
# and store the new version of the data.
with open("orders_new.json", 'w') as file:
    json.dump(data, file, indent = 4) 
