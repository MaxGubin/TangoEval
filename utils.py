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

# Names of columns in dataformats from 09/26.
cvs_names_0926 = ["label", "user_actid", "cand_actid", "same_country",
        "same_locale", "gender_mm", "gender_mf", "gender_mu", "gender_fm",
        "gender_ff", "gender_fu", "gender_um", "gender_uf", "gender_uu",
        "gender_m", "gender_f", "gender_u", "picture_daily",
        "update_profile_daily", "wink_daily", "picture_30days",
        "num_face","age", "beauty", "bright", "sharp", "white", "indian",
        "asian", "distance", "fav_ratio", "swipe_source", "user_tango_age",
        "cand_tango_age", "user_fav_ratio", "cand_user_agediff"]

# Names of features
feature_names_0926 = ["same_country",
        "same_locale", "gender_mm", "gender_mf", "gender_mu", "gender_fm",
        "gender_ff", "gender_fu", "gender_um", "gender_uf", "gender_uu",
        "gender_m", "gender_f", "gender_u", "picture_daily",
        "update_profile_daily", "wink_daily", "picture_30days",
        "num_face","age", "beauty", "bright", "sharp", "white", "indian",
        "asian", "distance", "fav_ratio", "swipe_source", "user_tango_age",
        "cand_tango_age", "user_fav_ratio", "cand_user_agediff"]
label_names_0926 = ["label"]

def LoadDataset_0926(name, test_sample, 
                        feature_columns=feature_names_0926,
                        label_columns=label_names_0926):
    """ Loads data in the new format with more fields and 
    """
    swipeSourceDict = {"POPULAR": 0.0,
            "GEO": 1.0, "ONLINENOW": 2.0, "ICF": 3.0, "RSM": 4.0, "PUK": 5.0}
    def swipeSourceConverter(source):
        if source not in swipeSourceDict:
            print "Unknown source:", source
        return swipeSourceDict.get(source)
    data = bp.read_csv(name, header = None,
            names = cvs_names_0926, converters = {
                "swipe_source": swipeSourceConverter 
                })

    # Sample based on the viewer id.
    set_of_users = set(data["user_actid"])
    test_user_ids = set(np.random.choice(np.array([i for i in set_of_users]),
        int(len(set_of_users)*test_sample)))
    data["test"] = data["user_actid"].map(lambda x: x in test_user_ids)
    return (data[data["test"]==False][feature_columns].values,
            data[data["test"]==True][feature_columns].values,
            data[data["test"]==False][label_columns].values.ravel(),
            data[data["test"]==True][label_columns].values.ravel())

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
