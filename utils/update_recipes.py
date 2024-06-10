import json

with open('assets/recipe.json', 'r') as f:
    recipes = json.load(f)

new_recipes = {}
for key, value in recipes.items():
    ingredients = '_'.join(sorted(value['ingredients']) + [value['location']] )
    final_product = key
    new_recipes[ingredients] = final_product
    

with open('assets/new_recipe.json', 'w') as f:
    json.dump(new_recipes, f, indent=4)
    