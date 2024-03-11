# Eric Jackman DSC311 Assignment 1
import pandas as pd
import matplotlib.pyplot as plt


def question_1():
    print("Question 1:")

    df = pd.read_csv("table.csv")  # Store csv file in pandas dataframe
    return df


def question_2(df):
    print("Question 2:")

    data = []  # List of list for data in frequency table
    total = 14  # Number of weight values
    absCum = 0  # Absolute cumulative frequency total
    relCum = 0  # Relative cumulative frequency total

    absFreqDict = df["Weight"].value_counts().to_dict()  # Stores absolute frequencies for weight column in a dictionary

    for weight, freq in absFreqDict.items():
        absRow = freq  # Absolute frequency for current row
        absCum += absRow  # Add row's absolute frequency to total
        relRow = absRow / total  # Relative frequency for current row
        relCum += relRow  # Add row's relative frequency to total
        row = [weight, absRow, "{:.2%}".format(relRow), absCum,
               "{:.2%}".format(relCum)]  # List for row of new dataframe

        data.append(row)  # Add row to list of lists

    freqDF = pd.DataFrame(data, columns=["Weight", "Abs_Freq", "Rel_Freq", "Abs_Cum_Freq", "Rel_Cum_Freq"])
    return freqDF.to_string(index=False)


def question_3(df):
    print("Question 3: See plots")

    plt.hist(df["Weight"])  # Create histogram for weight column
    plt.title("Weight Frequency")
    plt.show()

    gender = df["Gender"].value_counts().to_dict()
    plt.bar(gender.keys(), gender.values())  # Create bar graph for gender frequency
    plt.title("Gender Frequency")
    plt.show()

    maleDF = df[df["Gender"] == "M"]  # Create dataframe for just males
    femaleDF = df[df["Gender"] == "F"]  # Create dataframe for just females
    plt.subplot(1, 2, 1)  # First index is subplot array
    plt.boxplot(maleDF["Weight"])  # Create boxplot for male weights
    plt.title("Male Weight")
    plt.subplot(1, 2, 2)  # First index is subplot array
    plt.boxplot(femaleDF["Weight"])  # Create boxplot for female weights
    plt.title("Female Weight")
    plt.show()


def question_4(df):
    print("Question 4: See plots")

    plt.hist(df["Height"])  # Create histogram for height column
    plt.title("Height Frequency")
    plt.show()


def question_5(df):
    print("Question 5:")

    weightMin = df["Weight"].min()  # Find min of weight column
    weightMax = df["Weight"].max()  # Find max of weight column
    weightMean = df["Weight"].mean()  # Find mean of weight column
    weightMode = df["Weight"].mode().iat[0]  # Find mode of weight column
    weightFirst = df["Weight"].quantile(0.25)  # Find first quartile of weight column
    weightMed = df["Weight"].median()  # Find median of weight column
    weightThird = df["Weight"].quantile(0.75)  # Find third quartile of weight column

    weightAmp = weightMax - weightMin  # Find the amplitude of the weight column
    weightIQR = weightThird - weightFirst  # Find inter-quartile range of weight column
    weightMAD = df["Weight"].mad()  # Find mean absolute deviation of weight column
    weightSD = df["Weight"].std()  # Find standard deviation of weight column

    data = {"Location_stats": ["Min", "Max", "Average", "Mode", "First_qt", "Second_qt", "Third_qt"],
            "Weight": ["{:.2f}".format(weightMin), "{:.2f}".format(weightMax), "{:.2f}".format(weightMean),
                       "{:.2f}".format(weightMode), "{:.2f}".format(weightFirst),
                       "{:.2f}".format(weightMed), "{:.2f}".format(weightThird)]}
    locationDF = pd.DataFrame(data)  # Create dataframe from dictionary

    data2 = {"Dispersion_stats": ["Amp", "IQR", "MAD", "SD"],
             "Weight": ["{:.2f}".format(weightAmp), "{:.2f}".format(weightIQR), "{:.2f}".format(weightMAD),
                        "{:.2f}".format(weightSD), ]}
    dispersionDF = pd.DataFrame(data2)  # Create dataframe from dictionary

    return locationDF.to_string(index=False) + "\n\n" + dispersionDF.to_string(index=False)


def question_6(df):
    print("Question 6:")

    tempMean = df["Max_temp"].mean()  # Find mean of temp column
    tempMed = df["Max_temp"].median()  # Find median of temp column
    tempMode = df["Max_temp"].mode().iat[1]  # Find mode of temp column

    weightMean = df["Weight"].mean()  # Find mean of weight column
    weightMed = df["Weight"].median()  # Find median of weight column
    weightMode = df["Weight"].mode().iat[0]  # Find mode of weight column

    heightMean = df["Height"].mean()  # Find mean of height column
    heightMed = df["Height"].median()  # Find median of height column
    heightMode = df["Height"].mode().iat[0]  # Find mode of height column

    data = {"Cent_tend": ["Mean", "Median", "Mode"],  # Create dictionary for central tendency stats
            "Max_temp": ["{:.2f}".format(tempMean), "{:.2f}".format(tempMed), "{:.2f}".format(tempMode)],
            "Weight": ["{:.2f}".format(weightMean), "{:.2f}".format(weightMed), "{:.2f}".format(weightMode)],
            "Height": ["{:.2f}".format(heightMean), "{:.2f}".format(heightMed), "{:.2f}".format(heightMode)]}
    centendDF = pd.DataFrame(data)  # Create dataframe from dictionary
    return centendDF.to_string(index=False)


def question_7(df):
    print("Question 7: See plots and below")

    plt.scatter(df["Weight"], df["Height"])  # Create scatter plot for weight and height
    plt.xlabel("Weight")
    plt.ylabel("Height")
    plt.title("Weight vs Height")
    plt.show()

    weightMean = df["Weight"].mean()  # Find mean of weight column
    heightMean = df["Height"].mean()  # Find mean of height column
    coVar = 0  # Covariance of weight and height
    n = 14  # Number of rows
    c = 1 / (n - 1)  # First half of covariance equation
    for k in range(n - 1):  # Second half of covariance equation
        coVar += (df["Weight"].iat[k] - weightMean) * (df["Height"].iat[k] - heightMean)
    coVar = coVar * c  # Solve for covariance

    weightSD = df["Weight"].std()  # Find the standard deviation of weight column
    heightSD = df["Height"].std()  # Find the standard deviation of height column

    linCor = coVar / (weightSD * heightSD)  # Solve for linear correlation between weight and height
    return "\tCovariance of weight and height: {:.2f}\n\tLinear Correlation of weight and height: {:.2f}".format(coVar,
                                                                                                                 linCor)


def question_8(df):
    print("Question 8:")

    cont = pd.crosstab(df["Gender"], df["Company"], margins=True)  # Create contingency table
    print(cont)


if __name__ == "__main__":
    fullTableDF = question_1()  # Store dataframe and call methods
    print(fullTableDF.to_string(index=False))
    print(question_2(fullTableDF))
    question_3(fullTableDF)
    question_4(fullTableDF)
    print(question_5(fullTableDF))
    print(question_6(fullTableDF))
    print(question_7(fullTableDF))
    print(question_8(fullTableDF))
