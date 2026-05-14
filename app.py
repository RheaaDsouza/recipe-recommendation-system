import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
# from langchain_community.llms import Replicate
from langchain_community.chat_models import ChatOllama
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory # Standard import
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Comment Replicate API out as its a paid service. Use local llama model instead.
# REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Setup Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j and llm connection
@st.cache_resource
def connection():
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD,
        database=NEO4J_USERNAME,
    )
    llm = ChatOllama(model="llama3.2", temperature=0) 

    # llm = Replicate(
    #     model="meta/meta-llama-3-8b-instruct",
    #     replicate_api_token=REPLICATE_API_TOKEN,
    #     model_kwargs={
    #         "temperature": 0.01,
    #         "max_new_tokens": 1024,
    #         "top_p": 0.9,
    #     }
    # )

    return graph, llm

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

if "last_recipe_context" not in st.session_state:
    st.session_state.last_recipe_context = ""

# Cypher query generation
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query to find recipes in the Knowledge Graph.
Schema: (r:Recipe)-[:USES]->(i:Ingredient)-[:BELONGS_TO]->(c:Category)

User Pantry: {pantry_list}
User Allergies: {allergies}

Rules:
1. If an ingredient is in the User Pantry AND the User Allergies, it is UNSAFE. Remove it from the pantry list immediately.
2. A recipe is UNSAFE if it uses ANY ingredient matching the User Allergies.
3. Match recipes using the SAFE pantry items.

Follow this query structure:
WITH [item IN {pantry_list} WHERE NOT item IN {allergies}] AS safe_pantry

MATCH (r:Recipe)-[:USES]->(i:Ingredient)
// Fuzzy match for pantry items
WHERE ANY(p IN safe_pantry WHERE toLower(i.name) CONTAINS toLower(p))

AND NOT EXISTS {{
  MATCH (r)-[:USES]->(bad:Ingredient)
  WHERE ANY(a IN {allergies} WHERE toLower(bad.name) CONTAINS toLower(a))
}}

WITH r, safe_pantry, collect(i) AS matched_ingredients, 
     sum(1.0 / (toFloat(coalesce(i.shelf_life_days, 999)) + 1.0)) AS urgency_score
     
RETURN r.title AS title, 
       r.directions AS directions, 
       urgency_score, 
       [x IN matched_ingredients WHERE coalesce(x.shelf_life_days, 999) < 7 | x.name] AS expiring_items
ORDER BY urgency_score DESC LIMIT 3;

Question: {question}
Cypher Query:"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE, 
    input_variables=["pantry_list", "allergies", "question"]
)


QA_TEMPLATE = """
You are a professional Chef assistant focused on reducing food waste.
Recommend recipes that uses maximum ingredients the user has 
available in their pantry, as given by the database below.

User's Pantry: {pantry_list}
Database Results: {context}

Instructions:
1. For each recipe found, emphasize why it was chosen (e.g., "This recipe helps you use up your [expiring items] which are about to spoil").
2. Format the response with clear headers and numbered steps.
3. If no recipes use the urgent items, explain that but suggest a recipe that uses the most pantry items possible.

---
### **Recipe Name** (Match Percentage)
**Why this match?** [Explain urgency based on shelf life]

**Ingredients Used:**
- [List here]

 Missing: [list any ingredients explicitly required by the recipe that the user does not have in their pantry]

**Preparation Steps:**
1. [Steps]
---

Critical instruction:
Do not invent new recipes. Only recommend recipes that are present in the 'Database Results' context above. 
If the Database Results are empty or do not list a recipe, state clearly that no matches 
were found in the database.


Question: {question}
Final Chef's Answer:"""

qa_prompt = PromptTemplate(
    template=QA_TEMPLATE, 
    input_variables=["pantry_list", "context", "question"]
)

# Prompt template to handle follow up questions from the user on the already
# recommended recipes.
FOLLOWUP_TEMPLATE = """
You are a professional chef assistant.
The user was previously shown these recipe recommendations:

{last_context}

Conversation so far:
{chat_history}

Using ONLY the recipes above, answer the user's follow-up question.
Do not invent new recipes or ingredients not mentioned above.
If the question cannot be answered from the context, say so clearly.

Question: {question}
Chef's Answer:"""

followup_prompt = PromptTemplate(
    template=FOLLOWUP_TEMPLATE,
    input_variables=["last_context", "chat_history", "question"]
)


# Prompt template that classifies the intent
INTENT_TEMPLATE = """
You are a request classifier. Reply with ONLY one word, nothing else. No punctuation.

Reply "SEARCH" if the user wants new recipe recommendations.
Reply "FOLLOWUP" if the user is asking about recipes already shown to them.

Rules:
- If no recipes have been shown yet, always reply SEARCH
- Questions about substitutions, steps, scaling, nutrition, missing ingredients = FOLLOWUP
- Any request for new or different recipes = SEARCH

Recipes already shown to user: {has_context}
User message: {question}

One word reply:"""

intent_prompt = PromptTemplate(
    template=INTENT_TEMPLATE,
    input_variables=["has_context", "question"]
)

# This function returns 'SEARCH' or 'FOLLOWUP'.
def classify_intent(question: str) -> str:
    _, llm = connection()
    has_context = "Yes" if st.session_state.get("last_recipe_context") else "No"
    chain = intent_prompt | llm
    result = chain.invoke({"has_context": has_context, "question": question})
    text = result.content if hasattr(result, "content") else str(result)
    # models sometimes add punctuation or extra words
    return "FOLLOWUP" if "FOLLOWUP" in text.strip().upper() else "SEARCH"

# Get recipes from the Knowledge graph
def get_recipes(question):
    graph, llm = connection()
    
    pantry_list = st.session_state.get("pantry_list", [])
    allergies   = st.session_state.get("allergies", [])

    # Normalize inputs
    clean_pantry = [x.lower() for x in pantry_list]
    clean_allergies = [x.lower() for x in allergies]

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        cypher_prompt=cypher_prompt.partial(
            pantry_list=clean_pantry,
            allergies=clean_allergies
        ),
        qa_prompt=qa_prompt.partial(
            pantry_list=", ".join(st.session_state.pantry_list)
        ),
        return_intermediate_steps=True,
        validate_cypher=True,
        allow_dangerous_requests=True
    )

    response = chain.invoke({"query": question})
    
    # Persist retrieved recipes so follow-up calls can reference them
    st.session_state.last_recipe_context = str(
        response["intermediate_steps"][1]["context"]
    )

    return response["result"]

# This function answers follow-ups from cached recipe context
def followup_recipes(question: str) -> str:
    _, llm = connection()

    last_context = st.session_state.get("last_recipe_context", "")

    if not last_context:
        return (
            "I don't have any previously recommended recipes to refer to. "
            "Please ask for recipe recommendations first."
        )

    history = st.session_state.memory.load_memory_variables({}).get("chat_history", "")

    chain = followup_prompt | llm
    result = chain.invoke({
        "last_context": last_context,
        "chat_history": history,
        "question": question
    })

    return result.content if hasattr(result, "content") else str(result)

# Streamlit UI 
def run_streamlit():
    st.set_page_config(layout="wide", page_title="Pantry to Plate - Recipe Recommender")

    st.title("Pantry to Plate ")
    st.subheader("Recipe Recommendation System")
    st.info("""This chatbot interface is designed to help you find recipes 
        that would prioritize the ingredients you have in your pantry.""")

    with st.sidebar:
        st.sidebar.header("How this works")
        st.markdown("""
        1. Add ingredients you have
        2. Add allergies if any
        3. Get recommendations that prioritize:
            - Using ingredients that expire soon
            - Maximizing the use of ingredients you have
        """)

        st.sidebar.header(""" Example Questions - 
            Recommend recipes based on my provided ingredients and preferences""")
        st.divider()

        st.header("Your Pantry Ingredients")
        pantry_input = st.text_area("Ingredients you have:", value="Eggs, Milk, Chicken, Spinach")
        # Store as a list for the prompt
        st.session_state.pantry_list = [x.strip() for x in pantry_input.split(",") if x.strip()]
        
        allergies_input = st.text_input("Allergies (e.g. Peanuts, Gluten):", value="")
        # Clean allergies list
        st.session_state.allergies = [x.strip().lower() for x in allergies_input.split(",") if x.strip()]
        
        if st.button("Clear History"):
            st.session_state.memory.clear()
            st.rerun()

    # Chat interface
    for msg in st.session_state.memory.chat_memory.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("What should I cook with my igredients?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the Knowledge Graph..."):
                intent = classify_intent(prompt)
                if intent == "FOLLOWUP":
                    output = followup_recipes(prompt)
                else:
                    output = get_recipes(prompt)

                st.markdown(output)

                # Save to memory
                st.session_state.memory.save_context({"input": prompt}, {"output": output})

if __name__ == "__main__":
    run_streamlit()