import json

with open(r'd:\Machine learning\day16-handling-mixed-variables\Untitled.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

# Intro cell
intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Ohk, here we are basically teaching how to handle Mixed Variables!\n",
        "\n",
        "**Mixed Variables** are columns in your dataset that contain both numbers and letters (categories) mixed together. For example, a ticket might be `A/5 21171`, or a Cabin might be `C85`. \n",
        "\n",
        "Machine learning models hate these! They want clean numbers or clear categories, not a messy combination. So, our goal today is to **split** these mixed variables into their numerical parts and categorical parts."
    ]
}

new_cells.append(intro)

for i, cell in enumerate(nb['cells']):
    new_cells.append(cell)
    # i == 1 is the cell that loads the CSV
    if i == 1:
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let's look at our data. Notice the `Cabin`, `Ticket`, and `number` columns. They have a mix of letters and numbers."
            ]
        })
    # i == 4 is the plot
    elif i == 4:
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Step 1: Handling the 'number' column\n",
                "The `number` column is a simple mixed variable. It contains numbers (like 5, 3, 6) and letters (like A). Let's extract the numerical part first. We do this by forcing everything into a number using `pd.to_numeric()`. This turns letters into `NaN` (Not a Number)."
            ]
        })
    # i == 5 is the numerical extraction
    elif i == 5:
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Now let's extract the categorical part (the letters). We can logically say: 'If the numerical part we just created is NaN, then the original value must have been a letter!'"
            ]
        })
    # i == 8 is the Ticket checking, right before Cabin extracting
    elif i == 9:
        new_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Step 2: Handling the 'Cabin' column (Using Regex)\n",
                "The `Cabin` column is trickier (e.g. `C85`). It has a letter and a number smashed together. We can use string functions to extract them.\n",
                "- `str.extract('(\\\\d+)')` uses a Regular Expression to mean 'find one or more digits (numbers)'.\n",
                "- `str[0]` means 'grab the very first character (which is the letter)'."
            ]
        })

nb['cells'] = new_cells

with open(r'd:\Machine learning\day16-handling-mixed-variables\Untitled.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
