# IngredientClassifier
A neural network that classifies ingredient names into one of five categories: Fruits, Vegetables, Grains, Protein, and Other.

## About
This project is a neural network that is trained through restaurant ingredient data provided (`FoodGroup.csv`) by NutriKarma, UnMesh LLC. The dataset can be provided upon request. 
The algorithm takes in a string -- an ingedient name -- and converts each character between a value from 0-1. The network takes in each character-value as an input node and processes it outputing a value from 0-4, indicating whether the ingredient is in the categories of 0 - Vegetables, 1 - Fruits, 2 - Grains, 3 - Protein, and 4 - Other.
