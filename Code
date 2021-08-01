import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def common_ingredients(df):
    """
    This function takes in a dataframe as input.
    It returns the list of the three most common
    ingredients found within the skin care products
    that are part of the given data frame. The most
    common ingredients are determined by the number
    of times the ingredient appears within the skin
    care products.
    """
    common = {}
    col1 = df['Ingredients']
    for value in col1:
        ing = value.split(',')
        for i in ing:
            if i not in common:
                common[i] = 1
            else:
                common[i] += 1
    sorted_common = OrderedDict(
        sorted(
            common.items(), key=lambda kv: kv[1], reverse=True
            )
    )
    important_ing = []
    important_ing.append(list(sorted_common.keys())[0].strip())
    important_ing.append(list(sorted_common.keys())[1].strip())
    important_ing.append(list(sorted_common.keys())[2].strip())
    return important_ing


def create_plot(df, important_ing):
    """
    This function takes in a data frame and a list
    of the three most common ingredients. It graphs
    a bar graph that represents the the number of
    common ingredients the products have and their
    average ranking.
    """
    total = 0
    newcol = []
    used_term = []
    ing = df['Ingredients']
    for val in ing:
        word = val.split(',')
        for i in word:
            i = i.strip()
            if i not in used_term:
                if i == important_ing[0].strip():
                    total += 1
                    used_term.append(i)
                elif i == important_ing[1].strip():
                    total += 1
                    used_term.append(i)
                elif i == important_ing[2].strip():
                    total += 1
                    used_term.append(i)
        newcol.append(total)
        total = 0
        used_term = []
    df['Total'] = newcol
    sns.set()
    g = sns.catplot(x='Total', y='Rank', kind='bar', ci=None,  data=df)
    g.set(ylim=(4.1, 4.22))
    plt.xlabel("Number of Ingredients")
    plt.ylabel("Average Rating of Products")
    plt.title(
        label="Average Rating of Products over Total Number of "
        + "Common Ingredients",
        fontsize=14,
        pad='20.0'
    )
    plt.savefig('/home/plot.png', bbox_inches='tight')


def ingredient_for_skin_type(df, skin_type):
    """
    This function takes in a data frame and a string
    representing name of the skin_type as arguments.
    It returns a list of the five most common ingredients
    found within the skin care products meant for the
    specific skin type. It also returns the number of
    times the ingredient appeared within the skin care
    products.
    """
    df = df[df[skin_type] == 1]
    df = df[['Ingredients']]
    all_ingredients = {}
    for line in df["Ingredients"]:
        choices = line.split(',')
        for each in choices:
            if each not in all_ingredients:
                all_ingredients[each] = 0
            all_ingredients[each] += 1
    all_ingredients = dict(
        sorted(
            all_ingredients.items(), key=itemgetter(1), reverse=True
        )
    )
    return list(all_ingredients.keys())[0:5]


def predict_skintype(df, skin_type):
    """
    This function takes in a data frame and a string
    representing skin_type as arguments. It trains a
    logistic regression model to predict the skin type
    the product is meant for based on the ingredients.
    It returns the accuracy of the machine learning model
    for the test data.
    """
    df = df[[skin_type, 'Ingredients']]
    features = df.loc[:, df.columns != skin_type]
    features = pd.get_dummies(features)
    labels = df[skin_type]
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = LogisticRegression()
    model.fit(features_train, labels_train)
    return model.score(features_train, labels_train)


def common_ingredient_skin_type(data):
    '''
    This function takes in a dataframe as input. It
    returns a sorted tuple of ingredients and the
    number of times that ingredient appeared within
    the skin care products that are part of the
    dataframe. The most common ingredients are
    determined by the number of times the ingredient
    appeared within the skin care products.
    '''
    common = {}
    common_lst = []
    for each in data['Ingredients']:
        ing_lst = each.lower().split(', ')
        for ing in ing_lst:
            if ing not in common:
                common[ing] = 0
            common[ing] += 1
    for each in common:
        each_pair = each, common[each]
        common_lst.append(each_pair)
    sorted_common_lst = sorted(common_lst, key=itemgetter(1), reverse=True)
    return sorted_common_lst[4:9]


def list_ingredient(data, ing):
    '''
    This function takes in a dataframe and a list of
    tuples as inputs. It returns the indexes of the
    skin care products within the dataframe that
    had at least one of the common ingredients.
    '''
    ing_set = set()
    common_list = [i[0] for i in ing]
    for index, row in data.iterrows():
        ingredient_lst = row['Ingredients'].lower().split(', ')
        for each_ingredient in ingredient_lst:
            if each_ingredient in common_list:
                ing_set.add((index))
    return ing_set


def add_to_df(data, lst):
    '''
    This function takes in a dataframe and a list of
    indicies of the skin care products within the
    dataframe. It returns a new dataframe that contains
    only those products and only the columns Name,
    Rank, and Ingredients.
    '''
    filtered_df = pd.DataFrame()
    for i in lst:
        filtered_df.loc[i, 'Name'] = data.loc[i, 'Name']
        filtered_df.loc[i, 'Rank'] = data.loc[i, 'Rank']
        filtered_df.loc[i, 'Ingredients'] = data.loc[i, 'Ingredients']
    return filtered_df.reset_index()


def add_common_ingredient_dummy(data, common):
    '''
    This function takes in a dataframe and a tuple of common
    ingredients. It returns a dataframe with columns added that
    represent the appearnce of common ingredients in each of
    the skincare products.
    '''
    only_ing = [common[i][0] for i in range(len(common))]
    for index, row in data.iterrows():
        ingredient_list = row['Ingredients'].lower().split(', ')
        for each in only_ing:
            if each in ingredient_list:
                data.loc[index, each] = 1
            else:
                data.loc[index, each] = 0
    return data.drop('Ingredients', 1)


def summ_common_ingredients(data):
    '''
    This function takes in a dataframe. It returns
    a new dataframe that has an added column
    representing the total number of common
    ingredients the product contains.
    '''
    data['Total Common Ingredients'] = data.iloc[:, -6:-1].sum(axis=1)
    return data


def regression_predict(data):
    '''
    This function takes in a data frame. It returns
    the Test X and y and predicted y by building a
    linear regression model.
    '''
    filtered_data = data.loc[:, ['Rank', 'Total Common Ingredients']]
    filtered_data.reset_index()
    X = filtered_data.loc[:, 'Total Common Ingredients'].values
    y = filtered_data.loc[:, 'Rank'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = X_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    return X_test, y_test, y_pred


def main():
    df = pd.read_csv('/home/cosmetics 2.csv')
    mask1 = df['Ingredients'] != '#NAME?'
    mask2 = df["Ingredients"].str.contains("Visit") == False
    mask3 = df['Ingredients'] != 'unknown'
    mask4 = df['Ingredients'] != 'No Info'
    ndf = df[mask1 & mask2 & mask3 & mask4].reset_index()
    ndf.to_csv(r'/home/clean_data.csv', index=False, header=True)
    df = pd.read_csv('/home/clean_data.csv')
    common_ingredients(df)
    important_ing = common_ingredients(df)
    create_plot(df, important_ing)
    print("Common Ingredients for Sensitive Skin Type: " +
          str(ingredient_for_skin_type(df, "Sensitive")))
    print("Common Ingredients for Normal Skin Type: " +
          str(ingredient_for_skin_type(df, "Normal")))
    print("Common Ingredients for Oily Skin Type: " +
          str(ingredient_for_skin_type(df, "Oily")))
    print("Common Ingredients for Combination Skin Type: " +
          str(ingredient_for_skin_type(df, "Combination")))
    print("Common Ingredients for Dry Skin Type: " +
          str(ingredient_for_skin_type(df, "Dry")))

    print("Sensitive: " + str(predict_skintype(df, "Sensitive")))
    print("Normal: " + str(predict_skintype(df, "Normal")))
    print("Oily: " + str(predict_skintype(df, "Oily")))
    print("Combination: " + str(predict_skintype(df, "Combination")))
    print("Dry: " + str(predict_skintype(df, "Dry")))

    mask1 = df['Combination'] == 1
    mask2 = df['Dry'] == 1
    mask3 = df['Normal'] == 1
    mask4 = df['Oily'] == 1
    mask5 = df['Sensitive'] == 1
    ndf1 = df[mask1]
    ndf2 = df[mask2]
    ndf3 = df[mask3]
    ndf4 = df[mask4]
    ndf5 = df[mask5]

    common_ings1 = common_ingredient_skin_type(ndf1)
    in_ls1 = list_ingredient(ndf1, common_ings1)
    df_com = add_to_df(ndf1, in_ls1)
    add_common_ingredient_dummy(df_com, common_ings1)
    summ_common_ingredients(df_com)
    X1 = regression_predict(df_com)[0]
    y1 = regression_predict(df_com)[1]
    y12 = regression_predict(df_com)[2]

    common_ings2 = common_ingredient_skin_type(ndf2)
    in_ls2 = list_ingredient(ndf2, common_ings2)
    df_dry = add_to_df(ndf2, in_ls2)
    add_common_ingredient_dummy(df_dry, common_ings2)
    summ_common_ingredients(df_dry)
    X2 = regression_predict(df_dry)[0]
    y2 = regression_predict(df_dry)[1]
    y22 = regression_predict(df_dry)[2]

    common_ings3 = common_ingredient_skin_type(ndf3)
    in_ls3 = list_ingredient(ndf3, common_ings3)
    df_nor = add_to_df(ndf3, in_ls3)
    add_common_ingredient_dummy(df_nor, common_ings3)
    summ_common_ingredients(df_nor)
    X3 = regression_predict(df_nor)[0]
    y3 = regression_predict(df_nor)[1]
    y32 = regression_predict(df_nor)[2]

    common_ings4 = common_ingredient_skin_type(ndf4)
    in_ls4 = list_ingredient(ndf4, common_ings4)
    df_oil = add_to_df(ndf4, in_ls4)
    add_common_ingredient_dummy(df_oil, common_ings4)
    summ_common_ingredients(df_oil)
    X4 = regression_predict(df_oil)[0]
    y4 = regression_predict(df_oil)[1]
    y42 = regression_predict(df_oil)[2]

    common_ings5 = common_ingredient_skin_type(ndf5)
    in_ls5 = list_ingredient(ndf5, common_ings5)
    df_sen = add_to_df(ndf5, in_ls5)
    add_common_ingredient_dummy(df_sen, common_ings5)
    summ_common_ingredients(df_sen)
    X5 = regression_predict(df_sen)[0]
    y5 = regression_predict(df_sen)[1]
    y52 = regression_predict(df_sen)[2]

    """
    This is creating a graph to represent the results of
    the linear regression model for each skin type.
    """
    fig, axs = plt.subplots(2, 3)
    fig.tight_layout()
    plt.subplots_adjust(
        left=0.125, bottom=0.15, right=0.9, top=0.9, wspace=0.3, hspace=0.3
    )
    fig.text(
        0.5, 0.04, 'Number of Common Ingredient', ha='center', fontsize=10
    )
    fig.text(
        0.04, 0.5, 'Rank', va='center', rotation='vertical', fontsize=10
    )
    fig.delaxes(axs[1, 2])
    fig.suptitle(
        'Relationship Between Number of Common Ingredients and'
        + 'Ranking in Different Skin Type',
        fontsize=10
    )
    axs[0][0].scatter(X1, y1, c='bisque', s=8)
    axs[0][0].plot(X1, y12, c='sandybrown', linewidth=0.8)
    axs[0][0].set_title('Combination Skin', fontsize=7)
    axs[0][1].scatter(X2, y2, c='darksalmon', s=8)
    axs[0][1].plot(X2, y22, c='rosybrown', linewidth=0.8)
    axs[0][1].set_title('Dry Skin', fontsize=7)
    axs[0][2].scatter(X3, y3, c='yellowgreen', s=8)
    axs[0][2].plot(X3, y32, c='olivedrab', linewidth=0.8)
    axs[0][2].set_title('Normal Skin', fontsize=7)
    axs[1][0].scatter(X4, y4, c='mediumaquamarine', s=8)
    axs[1][0].plot(X4, y42, c='teal', linewidth=0.8)
    axs[1][0].set_title('Oil Skin', fontsize=7)
    axs[1][1].scatter(X5, y5, c='thistle', s=8)
    axs[1][1].plot(X5, y52, c='mediumorchid', linewidth=0.8)
    axs[1][1].set_title('Sensitive Skin', fontsize=7)
    fig.savefig('/home/q3.png')


if __name__ == '__main__':
    main()
