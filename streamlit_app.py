import streamlit as st
st.title('Divisha Kushwaha')
st.write('This app classifies the image and figures out the number put by the user (0-9)')
index=st.slider('Choose a number', 0, 359)
st.write("Selected number:",index)

from sklearn.datasets import load_digits
import matplotlib.pyplot as mpt
digit=load_digits()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(digit.data,digit.target,test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
model.score(x_test,y_test)
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
y_pred = model.predict(x_test)

fig,ax=mpt.subplots()
mpt.imshow(x_test[index].reshape(8,8),cmap='gray')
st.pyplot(fig)
mpt.axis('off')
ax.axis('off')
from sklearn.metrics import accuracy_score, confusion_matrix
st.divider() # horizontal line

if st.button("Show full confusion matrix"):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    fig2, ax2 = mpt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    accuracy = (y_pred == y_test).mean()
    st.info(f"Overall accuracy: {accuracy * 100:.2f}%")
    st.write("Predicted Value:",y_pred[index])
