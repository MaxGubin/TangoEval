# Exports sklearn random forest model into a java module.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cPickle
from optparse import OptionParser
from sklearn.tree import _tree

def writeComment(pythonObject, outstream):
    """Writes a java comment about the python object
    """
    outstream.write("/**\n" + str(pythonObject) + "*/\n")

def exportReturn(decisionTree, index, depth , outstream):
    outstream.write("\t"*depth)
    outstream.write("return ")
    outstream.write(str(decisionTree.value[index]))
    outstream.write(";\n")

def exportNode(decisionTree, index, depth, outstream):
    child_left = decisionTree.children_left[index]
    child_right = decisionTree.children_right[index]
    if child_left == _tree.TREE_LEAF:
        exportReturn(decisionTree, index, depth , outstream)
        return
    indent = "\t"*depth
    outstream.write("{indent}if (features[{value}] <= {threshold})".
            format(indent=indent,
                value=decisionTree.feature[index],
                threshold=decisionTree.threshold[index]))
    outstream.write(" {\n")
    exportNode(decisionTree, child_left, depth + 1, outstream)
    outstream.write(indent + "} else {\n")
    exportNode(decisionTree, child_right, depth + 1, outstream)
    outstream.write(indent + "}\n")

def exportTreeAsFunction(decisionTree, funcName, outstream):
    """
    Exports a tree as a Java function.
    """
    writeComment(decisionTree, outstream)
    outstream.write(" private float " + funcName + "(float[] features) {\n")
    exportNode(decisionTree.tree_, 0, 1, outstream)
    outstream.write("}\n")

    

def exportForest(forest, className, outstream):
    writeComment(forest, outstream);
    outstream.write("class " + className + "{\n")
    functionList = []
    for ind, dtree in enumerate(forest.estimators_):
        functionName = "decisionTree_"+str(ind)
        exportTreeAsFunction(dtree, functionName, outstream)
        functionList.append(functionName)
    outstream.write("public float[] Predict(float[] features) {")
    outstream.write(" float[] result = {};")
    for treeFunction in functionList:
        outstream.write(" float " + treeFunction + "Res = " + treeFunction + 
                "(features);")
        outstream.write(" for (int i = 0; i < treeFunctionRes.length; ++i){\n")
        outstream.write("   result[i] += treeFunctionRes[i];\n")
        outstream.write(" };\n")
    outstream.write(" return result;")
    outstream.write("}\n}\n")
    



def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input_file",
            help = "Saved in pickle format sklearn random forest")
    parser.add_option("-o", "--output", dest="output_file",
            help = "Name of output file")
    parser.add_option("-c", "--class_name", dest="class_name",
            help = "Name of the class that is generated", 
            default="RandomForestClassifier")
    (options, args) = parser.parse_args()
    print "Loading model...."
    randomForest = cPickle.load(open(options.input_file))
    print "Outputing java code...."
    exportForest(randomForest, options.class_name, open(options.output_file,"w"))


if __name__ == "__main__":
        main()

