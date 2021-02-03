#Ashlin Darius Govindasamy Pokemon Type Detector ML Tree Method
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing



from sklearn.tree import DecisionTreeClassifier
%matplotlib inline

df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/635/1677/pokemon_alopez247.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210202%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210202T190157Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=079b1a84983bdefd0074d021d158d65ec2b360660e765de007203e605665d0f10773da073756aff23ec2a5085a70729139d8795cd4eb8363299547d187a11be61739d56a5fb96ac88126500f7d4fd852d36267a79b6855b7787f156ed092a4ff9039210f28fd1d5167ecf217057245c1de9d03165cee9107ab2dcc12d57d9f14b5ac9e52e18a0922375db2dddd3d725a901829be0d56856d774f94cd08e0f4b6d6f7abb41a333b8d3e8844cded458c2fb4653d85063483a12c2ee8c65ba82e2019a10a4dd8421a4d706747e40210203b16a19502991772449523f2a6464cbdb619ff184aad9519783ed8ea85b2542d1b5345212c92c712ff79108b7d4edbba80')
df.head()

X = df[['HP', 'Attack', 'Defense', 'Sp_Atk', 'Sp_Def', 'Speed', 'Generation', 'Height_m', 'Weight_kg', 'Catch_Rate']]
y = df['Type_1']

 

    
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree.fit(X_trainset,y_trainset)
predTree = Tree.predict(X_testset)

print (predTree [0:10])

print (y_testset [0:10])



from  io import StringIO
 
import matplotlib.image as mpimg
from sklearn import tree
%matplotlib inline 

dot_data = StringIO()
filename = "pokemonprediction.png"
my_data = df



featureNames = df.columns[0:10]
targetNames =  df["Type_1"].unique().tolist()
out=tree.export_graphviz(Tree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

print("Tree Diagram Created") 
