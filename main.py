import streamlit as st
from langchain.llms import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

default_template = """
    You are a professional chef creating a new recipe.

    Take the dish name below delimited by triple backticks.
    dish: ```{dish}```

    Generate the recipe with this format:
    1. Recipe Name
    2. Ingredients
    3. Cooking Instructions
    4. Any additional notes or tips you'd like to include.
    The output should be a well-formatted recipe.
    """

def generate_recipe(repo_id, api_key, template, dish_name):
  hub_llm = HuggingFaceEndpoint(
    repo_id=repo_id, model_kwargs={"min_length": 512, "max_length": 1024}, token=api_key
  )
  prompt = PromptTemplate(
    input_variables=["dish"],
    template=template
  )
  hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
  return hub_chain.run(dish_name)

st.title("Recipe Generator")

with st.form("recipe_form"):
    api_key = st.text_input("Enter your API key:", type="password")
    repo_id = st.text_input("Enter Huggingface LLM Repo ID:", "mistralai/Mistral-7B-v0.1")
    template = st.text_area("Enter template:", default_template)
    dish_name = st.text_input("Enter the dish name:")

    if st.form_submit_button("Generate Recipe"):
        if not (api_key and repo_id and template and dish_name):
            st.error("Please fill in all fields.")
        else:
            recipe = generate_recipe(repo_id, api_key, template, dish_name)
            st.info(recipe)
