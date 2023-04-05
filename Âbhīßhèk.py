import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc
import warnings

plt.rc('figure', figsize=(8, 8))

def file_reader(fle):
    df=pd.read_csv(fle,header=2)
    df=df.fillna(0)
    df.replace('..',0,inplace=True)
    df1=df
    df1=df1.set_index('Country Name') 
    dft=df1.transpose()
    return df,dft

df,supplimentary=file_reader("API_EN.URB.MCTY.TL.ZS_DS2_en_csv_v2_4912379.csv")

df.head()

# Transposed DataFrame
supplimentary.head()

supplimentary.describe()

supplimentary['Andorra'].head()

supplimentary['Angola'].head()

df["Country Name"].unique()

frame0=df.drop(["Country Name"	,"Country Code", 'Indicator Name', 'Indicator Code'], axis=1)
frame3=frame0.astype(float)
frame3.describe()

k=np.arange(50)
fig, ax = plt.subplots()



ax.plot( np.arange(50), frame3['1960'].values[50:100], label='1960')
ax.plot( np.arange(50), frame3['1980'].values[50:100], label='1980')
ax.plot( np.arange(50), frame3['2000'].values[50:100], label='2000')
ax.plot( np.arange(50), frame3['2020'].values[50:100], label='2020')


    
ax.set_xlabel('countries')
ax.set_ylabel("population in urban agglomerations")
ax.set_title('Population in urban agglomerations of more than\n 1 million (% of total population)for the first 50 countries')

# add a legend
ax.legend()

# display the plot
plt.show()

valuess=frame3['1990'].values[10:20]
labelss = df['Country Name'].values[10:20]

figg, axx = plt.subplots()
axx.pie(valuess, labels=labelss, autopct='%1.1f%%', radius=1, startangle=90,
       wedgeprops={'width': 0.4, 'edgecolor': 'w'})

# Add a white circle to create the donut
centre_circle = plt.Circle((0,0),0.7,color='white', fc='white')
figg = plt.gcf()
figg.gca().add_artist(centre_circle)

# Add title
plt.title('Population in urban agglomerations of more than 1 million\n (% of total population)for 10th to twentieth country in 1990')

# Show the plot
plt.show()

df2,frt=file_reader('API_BX.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_5348264.csv')

frame1=df2.drop(["Country Name"	,"Country Code", 'Indicator Name', 'Indicator Code'], axis=1)
frame4=frame1.astype(float)
frame4.describe()

warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")


x = df['Country Name'].values[80:100]
y1 = frame4['1990'].values[80:100]
y2 = frame4['2016'].values[80:100]
y3 = frame4['2021'].values[80:100]

# Create the stacked bar chart
fig, ax = plt.subplots()
ax.bar(x, y1, label='1990')
ax.bar(x, y2, bottom=y1, label='2016')
ax.bar(x, y3, bottom=y1+y2, label='2021')
ax.set_xticklabels(x, rotation=90)

# Add legend and labels
ax.legend()
ax.set_title('Foreign direct investment, net inflows (% of GDP)')
ax.set_xlabel('countries')
ax.set_ylabel('1990,2016,2021')


# Show the plot
plt.show()

x=df['Country Name'].values[30:50]
y1=frame3['1990'].values[30:50]
y2=frame3['2013'].values[30:50]
y3=frame3['2020'].values[30:50]
plt.bar(x, y1, width=0.2, align='center', label='1990')
plt.bar([i + 0.2 for i in range(len(x))], y2, width=0.2, align='center', label='2013')
plt.bar([i + 0.4 for i in range(len(x))], y3, width=0.2, align='center', label='2021')


plt.xlabel('countries')
plt.ylabel('Years 1990 2013 2021')
plt.title('Population in urban agglomerations of more than 1 million\n (% of total population)for 10th to twentieth country in 1990')
plt.xticks(rotation=90)
plt.legend()


plt.show()

data =frame4.iloc[30:40,30:40]

# create a heatmap using the 'hot' colormap
heatmap = plt.imshow(data, cmap='coolwarm')

# add a colorbar to the plot
plt.colorbar(heatmap)

# set the x and y axis labels
plt.xlabel('1990-2021')
plt.ylabel('countries')
plt.xticks(np.arange(0.5, 10.5), range(1990, 2000), rotation=90)
plt.yticks(np.arange(0.5, 10.5),df2['Country Name'].values[30:40] )

# set the plot title
plt.title('aggregate heatmap of Foreign direct investment\n net inflows (% of GDP) for 10 countries')

# show the plot
plt.show()

plt.violinplot(frame4.iloc[0:10], showmedians=True)  # create violin plot
plt.xlabel("1990-2021")  # add x-label
plt.title("Foreign direct investment, net inflows (% of GDP)")  # add title
plt.show()  # display the plot