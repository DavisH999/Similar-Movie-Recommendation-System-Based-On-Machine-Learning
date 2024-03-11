import ast
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def extract_items(string):
    items = ast.literal_eval(string)
    alist = []
    for item in items:
        name = item.get('name')
        if name:
            alist.append(name)
    return '|'.join(alist)


deprecated_inefficient_code = """
    def insert_new_cols(df, col_name):
        total_unique_set = set()
        for item in df[col_name]:
            total_unique_set.update(item)

        for new_col_name in total_unique_set:
            if new_col_name not in df.columns:
                df[new_col_name] = 0
        return df


    def set_genre_flags(row, col_name):
        items = row[col_name]
        for item in items:
            row[item] = 1
        return row
    """


def convert_to_timestamp(date):
    if pd.notnull(date):
        return date.timestamp()
    return None


def regex_search(movie_name, movie_list):
    # treats escape character as literal character and  ignore cases and
    pattern = re.compile(re.escape(movie_name), re.IGNORECASE)
    matched_set = set()
    for movie_n in movie_list:
        if pattern.search(movie_n):
            matched_set.add(movie_n)
    return matched_set


def exit_system():
    print('Thank you for using similar movie recommendation system! See you next time!')
    exit()


def process_user_input(movie_titles):
    print('Welcome to Movie Recommendation System')
    u_input = input('Please enter one movie, we will recommend five similar movies based on the movie you entered.\n')
    result_list = regex_search(u_input, movie_titles)
    if result_list:
        for res in result_list:
            while True:
                user_input = input(f'Do you mean "{res}"? ' +
                                   f'Enter "y" to confirm, '
                                   f'"n" to show the next result, or '
                                   f'"q" to exit the recommendation system.\n').lower().strip()
                if user_input == 'y':
                    return res
                elif user_input == 'n':
                    break  # Breaks out of inner loop
                elif user_input == 'q':
                    exit_system()
                    return
                else:
                    print('Please enter a valid response.')
        else:
            print('No matches were confirmed.')
    else:
        print(f'Unfortunately, there are no movies matching your input "{u_input}". ' +
              'Please consider using alternative keywords and try again.')
    exit_system()


def lookup_rows(recommended_movie_titles, df_movies):
    matched_rows = df_movies[df_movies['title'].isin(recommended_movie_titles)]
    return matched_rows


def print_title_overview(row):
    print('TITLE: <' + row['title'] + '>. \nOVERVIEW: ' + row['overview'] + '\n')


def show_recommendations(rows):
    print("\nBased on your input movies, we recommend following movies and hope you enjoy them.\n")
    rows.apply(print_title_overview, axis=1)


def main():
    # E.T.L. PART
    df_movies = pd.read_csv('Datasets/IMDB/tmdb_5000_movies.csv')
    # back up initial df
    df_entire_df = df_movies.copy()
    # extract genres
    df_movies['genres'] = df_movies['genres'].apply(extract_items)
    df_movies_ml = df_movies[['title', 'release_date', 'genres']]
    # one-hot encoding, this one line code is adapted from
    # "https://medium.com/@luzhenna/one-hot-encode-categorical-features-for-dummies-with-get-dummies-60e1cb9197"
    df_one_hot = df_movies_ml['genres'].str.get_dummies(sep='|')
    df_movies_ml = df_movies_ml.join(df_one_hot)
    # convert the str of release_date to timestamp
    df_movies_ml['release_date'] = pd.to_datetime(df_movies_ml['release_date'])
    df_movies_ml['release_date'] = df_movies_ml['release_date'].apply(convert_to_timestamp)
    # remove duplicate row with same titles
    df_movies_ml = df_movies_ml.drop_duplicates(subset=['title'])
    # drop cols
    df_movies_ml = df_movies_ml.drop(columns=['genres'])
    # drop na
    df_movies_ml = df_movies_ml.dropna()

    # MACHINE LEARNING PART
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_movies_ml['release_date'] = scaler.fit_transform(df_movies_ml[['release_date']])
    X = df_movies_ml.drop(columns=['title'])
    y = df_movies_ml['title']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model = KNeighborsClassifier(n_neighbors=6)
    model.fit(X_train.values, y_train.values)
    # read user input and try to match them with re
    matched_title = process_user_input(df_movies_ml['title'])
    # find neighbors
    title_col = df_movies_ml[df_movies_ml['title'] == matched_title].drop(columns=['title'])
    distances, indices = model.kneighbors(X=title_col.values, return_distance=True)
    # print(distances, indices)
    # Exclude the first result if it's the input movie itself
    if distances[0][0] == 0:
        recommended_indices = indices[0][1:6]  # take [1, 6)
    else:
        recommended_indices = indices[0][0:5]  # take [0, 5)
    # find rows
    recommended_movie_titles = y_train.iloc[recommended_indices].values
    rows = lookup_rows(recommended_movie_titles, df_entire_df)
    # print result title and overview
    show_recommendations(rows)
    exit_system()


if __name__ == '__main__':
    main()
