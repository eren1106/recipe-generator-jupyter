import streamlit as st
from langchain.llms import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv()

hub_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-v0.1", model_kwargs={"min_length": 512, "max_length": 1024}
)

template_string = """
    You are a professional chef creating a new recipe.

    Take the dish name below delimited by triple backticks.
    dish: ```{dish}```

    Please provide the following details for your recipe of that dish:

    1. Recipe Name: [recipe_name]
    2. Ingredients (separated by commas): [ingredients]
    3. Cooking Instructions:

    [cooking_instructions]

    4. Optional: Any additional notes or tips you'd like to include.

    The output should be a well-formatted recipe.
    """

prompt = PromptTemplate(
  input_variables=["dish"],
  template=template_string
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

st.title("Recipe Generator")

dish_name = st.text_input("Enter the dish name:")
if st.button("Generate Recipe"):
    recipe = hub_chain.run(dish_name)
    st.info(recipe)
