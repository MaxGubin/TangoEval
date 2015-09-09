# Different service functions for data preparation/processing

import numpy as np
import pandas as bp
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Dataset processing
def LoadDataset(name, test_size):
    """
    Load a dataset into memory. Splits into training/testing according to split
    factor. 
    Returns:
    X_train, X_test, y_train, y_test
    """
    data=bp.read_csv(name, header=None)
    return train_test_split(data[range(3,31)].values, data[0].values,
            test_size=test_size)

# Wrapper to present tables
class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. 
        if titles != None, adds titles.
    """
    def SetHeaders(self, headers):
        self.headers = headers
        return self
                            
    def _repr_html_(self):
        html = ["<table>"]
        if self.headers:
            html.append("<tr>")
            for title in self.headers:
                html.append("<th>{0}</th>".format(title))
            html.append("</tr>")
                
        for row in self:
            html.append("<tr>")
            for col in row:
                html.append("<td>{0}</td>".format(col))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)

def drawPRCurves(curve_tuples):
    """
    Renders pr curves, every tuple is recall, precision, label
    """
    plt.clf()
    for curve in curve_tuples:
        plt.plot(curve[0], curve[1], label = curve[2])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="lower left")
    plt.show()

def classifierMetrics(classifier, X_test, y_test):
    """Returns a string with info about a classifier
    """
    return (' mean_precision:' + str(classifier.score(X_test, y_test)) +
            ' roc_auc_score :' + str(roc_auc_score(y_test,
                classifier.predict_proba(X_test)[:,1])))
