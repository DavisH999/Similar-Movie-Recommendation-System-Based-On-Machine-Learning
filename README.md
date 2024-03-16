
# Movie Recommendation System

## Overview
This Movie Recommendation System utilizes machine learning to recommend movies based on user input. It leverages the K-Nearest Neighbors algorithm to find movies similar to the one entered by the user. The recommendation is based on movie metadata, including titles, genres, and release dates.

## Features
- **User Interaction:** Prompts users to input a movie title and guides them through the recommendation process.
- **Data Preprocessing:** Extracts and processes movie genres, converts release dates to timestamps, and applies one-hot encoding for genre categorization.
- **Dynamic Matching:** Uses regular expressions to match user input with movie titles in the dataset, accounting for case sensitivity and partial matches.
- **Machine Learning Model:** Implements a K-Nearest Neighbors classifier to find movies similar to the user's input.
- **Recommendation Display:** Shows titles and overviews of recommended movies based on the user's input.

## Requirements
Before you run this system, ensure you have the following packages installed:
- Python 3.x
- Pandas
- NumPy
- scikit-learn
- ast
- re

You can install these packages using pip:
```
pip install pandas numpy scikit-learn
```

## Dataset
This system uses the TMDB 5000 Movie Dataset. Ensure you have the `tmdb_5000_movies.csv` file in a directory named `Datasets/IMDB/` relative to the script's location.

## Usage
To run the Movie Recommendation System, execute the following command in your terminal:
```
python movie_recommendation_system.py
```
Follow the on-screen prompts to enter a movie title and navigate through the recommendations.

## How It Works
1. **Data Extraction and Transformation:** The system reads the dataset, extracts relevant features, and preprocesses the data for the machine learning model.
2. **User Interaction:** Through the command line, the user is prompted to input a movie title.
3. **Title Matching:** The system matches the input with existing titles in the dataset using regular expressions.
4. **Recommendation Generation:** The K-Nearest Neighbors algorithm finds movies similar to the user's input based on genre and release date.
5. **Output:** The system displays recommended movies, including their titles and overviews.

## Contributions
Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
