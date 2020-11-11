import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class IngredientGroupDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)  #6347

    def __getitem__(self, index):
        ingredient = self.annotations.iloc[index, 1]
        ingredient = ingredient.lower()
        ingredientAsNum = np.zeros(30)
        idx_of_letter = 0
        for char in ingredient:
            if idx_of_letter < 30:         #only do first 30 letters  of ingredient
                idx = ord(char) - 96       #ascii val formatted to be 1-28 (inclusive)
                if (not(1 <= idx and idx <=28)):   #if char is a weird character just eval it to 0
                    idx = 0
                idx = round(idx / 28, 7)            #normalizing data to 0-1 scale
                ingredientAsNum[idx_of_letter] = idx
                idx_of_letter+=1

        ingredient = torch.from_numpy(ingredientAsNum)

        food_group = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            ingredient = self.transform(ingredient)

        return (ingredient, food_group)

