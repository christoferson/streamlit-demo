import streamlit as st
import pandas as pd
import numpy as np
import random

# Set page title
st.title("Animal Dataset with Row Selection")
st.write("Select rows from the table to view detailed information.")

# Function to generate random animal data
def generate_animal_data(num_animals=50):
    # Lists for random data generation
    animal_types = ["Dog", "Cat", "Bird", "Fish", "Rabbit", "Hamster", "Turtle", "Snake", "Lizard", "Ferret"]
    dog_breeds = ["Labrador", "German Shepherd", "Bulldog", "Poodle", "Beagle", "Rottweiler", "Husky", "Corgi", "Dachshund", "Golden Retriever"]
    cat_breeds = ["Persian", "Siamese", "Maine Coon", "Bengal", "Ragdoll", "Sphynx", "British Shorthair", "Abyssinian", "Scottish Fold", "Burmese"]
    colors = ["Black", "White", "Brown", "Gray", "Orange", "Spotted", "Striped", "Tan", "Cream", "Mixed"]
    names = ["Max", "Bella", "Charlie", "Luna", "Cooper", "Lucy", "Milo", "Daisy", "Rocky", "Lola", 
             "Buddy", "Sadie", "Jack", "Stella", "Bear", "Molly", "Duke", "Chloe", "Teddy", "Penny",
             "Oliver", "Ruby", "Leo", "Rosie", "Zeus", "Lily", "Bentley", "Coco", "Oscar", "Zoe"]

    data = {
        "Name": random.choices(names, k=num_animals),
        "Type": random.choices(animal_types, k=num_animals),
        "Age": [random.randint(1, 15) for _ in range(num_animals)],
        "Weight (kg)": [round(random.uniform(0.5, 50.0), 1) for _ in range(num_animals)],
        "Color": random.choices(colors, k=num_animals),
        "Adoption Status": random.choices(["Available", "Adopted", "Pending", "Reserved"], k=num_animals),
        "Health Score": [random.randint(1, 10) for _ in range(num_animals)]
    }

    # Add breed based on animal type
    breeds = []
    for animal_type in data["Type"]:
        if animal_type == "Dog":
            breeds.append(random.choice(dog_breeds))
        elif animal_type == "Cat":
            breeds.append(random.choice(cat_breeds))
        else:
            breeds.append("N/A")

    data["Breed"] = breeds

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add a unique ID column
    df["ID"] = [f"A{i+1:03d}" for i in range(num_animals)]

    # Reorder columns to put ID first
    cols = ["ID"] + [col for col in df.columns if col != "ID"]
    df = df[cols]

    return df

# Callback function for row selection
def handle_selection():
    # This function will be called when a row is selected
    pass

# Generate random animal data
if "animal_data" not in st.session_state:
    st.session_state.animal_data = generate_animal_data(50)

# Add a button to regenerate data
if st.button("Generate New Animal Data"):
    st.session_state.animal_data = generate_animal_data(50)

# Display the dataframe with selection enabled
st.subheader("Animal Database")
selection = st.dataframe(
    st.session_state.animal_data,
    use_container_width=True,
    column_config={
        "ID": st.column_config.TextColumn("ID", width="small"),
        "Name": st.column_config.TextColumn("Name", width="medium"),
        "Type": st.column_config.TextColumn("Type", width="medium"),
        "Breed": st.column_config.TextColumn("Breed", width="medium"),
        "Age": st.column_config.NumberColumn("Age (years)", width="small"),
        "Weight (kg)": st.column_config.NumberColumn("Weight (kg)", format="%.1f kg", width="medium"),
        "Color": st.column_config.TextColumn("Color", width="medium"),
        "Adoption Status": st.column_config.SelectboxColumn(
            "Adoption Status",
            options=["Available", "Adopted", "Pending", "Reserved"],
            width="medium"
        ),
        "Health Score": st.column_config.ProgressColumn(
            "Health Score",
            min_value=0,
            max_value=10,
            format="%d",
            width="medium"
        ),
    },
    hide_index=True,
    on_select=handle_selection,
    selection_mode="single-row",
    key="animal_table"
)

# Display selected row details
st.subheader("Selected Animal Details")

# Check if rows are selected
if selection and hasattr(selection, "selection") and selection.selection.rows:
    # Get the selected row index
    selected_row_index = selection.selection.rows[0]

    # Get the selected row data
    selected_animal = st.session_state.animal_data.iloc[selected_row_index]

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {selected_animal['Name']}")
        st.markdown(f"**ID:** {selected_animal['ID']}")
        st.markdown(f"**Type:** {selected_animal['Type']}")
        st.markdown(f"**Breed:** {selected_animal['Breed']}")
        st.markdown(f"**Color:** {selected_animal['Color']}")

    with col2:
        st.markdown(f"**Age:** {selected_animal['Age']} years")
        st.markdown(f"**Weight:** {selected_animal['Weight (kg)']} kg")
        st.markdown(f"**Adoption Status:** {selected_animal['Adoption Status']}")
        st.markdown(f"**Health Score:** {selected_animal['Health Score']}/10")

    # Add a visual representation of the health score
    st.progress(selected_animal['Health Score']/10)

    # Add some fictional notes about the animal
    notes = [
        "Very friendly and good with children.",
        "Needs special diet due to allergies.",
        "Loves to play fetch and go for walks.",
        "Prefers a quiet home environment.",
        "Gets along well with other animals.",
        "Needs regular grooming.",
        "Very active and energetic.",
        "Shy at first but warms up quickly.",
        "Trained to do basic commands.",
        "Loves to cuddle and be petted."
    ]

    # Use the animal ID as a seed for consistent notes per animal
    random.seed(int(selected_animal['ID'][1:]))
    st.markdown("### Notes")
    st.markdown(random.choice(notes))
    # Reset the random seed
    random.seed()

else:
    st.info("Please select an animal from the table above to view details.")