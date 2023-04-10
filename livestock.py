import streamlit as st
import pandas as pd

# Load data

livestockPopn = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/1 Livestock population.csv")
milkAnimal = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/2 Milking animals.csv")
meatProd = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/3 Meat production.csv")
eggProd = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/4 Egg production.csv")
woolProd = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/5 Wool production.csv")
yakPopn = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/6 YakNakChauri population.csv")
rabbitPopn = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/7 Rabbit population.csv")
horsesPopn = pd.read_csv("/Users/saramshkhadka/Downloads/Thesis/Livestock data analysis/8 HorsesAsses population.csv")


# Display data
st.write(livestockPopn)
